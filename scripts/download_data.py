"""Bulk historical data downloader for all configured markets.

Downloads 5-year (or as many years as market_config specifies) OHLCV history
for every symbol in strategy_config.yaml, across all timeframes you need for
the ICT bot (M15 primary + H4 HTF bias + D1 daily context).

The script is idempotent:
  - If a cache file already exists it performs an incremental refresh (append
    only the bars that are missing since the last cached bar).
  - Pass --force to wipe and re-download everything from scratch.

Usage
-----
    # Download / refresh all 16 markets (MT5 or yfinance fallback)
    python -m scripts.download_data

    # Only specific symbols
    python -m scripts.download_data --symbols EURUSD GBPUSD XAUUSD NAS100

    # Specific timeframes
    python -m scripts.download_data --timeframes M15 H4 D1

    # Force full re-download (ignore existing cache)
    python -m scripts.download_data --force

    # Quiet mode - only show summary table
    python -m scripts.download_data --quiet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import (  # noqa: E402
    load_csv,
    load_from_mt5,
    load_from_mt5_since,
    load_from_yfinance,
)
from src.utils.logging_utils import setup_logging  # noqa: E402

import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Timeframes to keep in the local cache (ordered: primary → HTF → daily)
DEFAULT_TIMEFRAMES = ["M15", "H4", "D1"]

# yfinance interval mapping used when MT5 is unavailable
YF_INTERVAL_MAP = {
    "M1": "1m",  "M5": "5m",  "M15": "15m", "M30": "30m",
    "H1": "1h",  "H4": "4h",  "D1": "1d",
}

# yfinance period caps (intraday data is heavily restricted)
YF_PERIOD_MAP = {
    "1m": "7d",  "2m": "7d",  "5m": "60d", "15m": "60d",
    "30m": "60d","60m": "730d","4h": "730d","1d": "max",
}

# MT5 bar counts for a 5-year download (approximate; broker may return fewer)
# M15: 5yr * 252 trading days * 24h/d * 4 bars/h ≈ 121k bars
BARS_5YR: Dict[str, int] = {
    "M1":  600_000,
    "M5":  150_000,
    "M15":  90_000,
    "M30":  45_000,
    "H1":   22_000,
    "H4":    6_500,
    "D1":    1_300,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _age_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    last = df.index[-1]
    age = datetime.utcnow() - last.to_pydatetime()
    h = int(age.total_seconds() // 3600)
    return f"{h}h ago" if h < 48 else f"{age.days}d ago"


def _bar_count_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "0"
    return f"{len(df):,}"


def _date_range_str(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    return f"{df.index.min().date()} to {df.index.max().date()}"


# Explicit yfinance ticker overrides for symbols that don't follow the SYMBOL=X pattern.
# Crypto: use ASSET-USD format. Metals: use futures codes. Indices: use index codes.
YF_TICKER_MAP: Dict[str, str] = {
    # Crypto
    "BTCUSD":  "BTC-USD",
    "ETHUSD":  "ETH-USD",
    "LTCUSD":  "LTC-USD",
    "XRPUSD":  "XRP-USD",
    # Metals (continuous futures)
    "XAUUSD":  "GC=F",
    "XAGUSD":  "SI=F",
    # Equity indices
    "NAS100":  "NQ=F",
    "SPX500":  "ES=F",
    "US30":    "YM=F",
    "US500":   "ES=F",
    "UK100":   "^FTSE",
    "GER40":   "^GDAXI",
}


def _yf_bars(symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    """Best-effort yfinance download with appropriate period/interval."""
    interval = YF_INTERVAL_MAP.get(timeframe.upper(), "1d")
    # For intraday use maximum allowed period; for D1 use years directly
    if interval in ("1d",):
        period = f"{years * 365}d"
    elif interval in ("1h", "4h", "60m"):
        period = "730d"
    else:
        period = YF_PERIOD_MAP.get(interval, "60d")
    # Resolve to the correct yfinance ticker (crypto/indices differ from forex)
    yf_ticker = YF_TICKER_MAP.get(symbol.upper())
    if yf_ticker is None:
        # Forex fallback: EURUSD -> EURUSD=X
        if len(symbol) == 6 and "=" not in symbol:
            yf_ticker = f"{symbol}=X"
        else:
            yf_ticker = symbol
    return load_from_yfinance(yf_ticker, period=period, interval=interval)


# ---------------------------------------------------------------------------
# Core per-symbol downloader
# ---------------------------------------------------------------------------

def download_symbol(
    symbol: str,
    timeframe: str,
    cache_dir: Path,
    years: int = 5,
    force: bool = False,
    quiet: bool = False,
) -> Tuple[str, int, str, str]:
    """Download or refresh one symbol/timeframe combination.

    Returns (status_label, bar_count, date_range, source_used).
    """
    cache = cache_dir / f"{symbol}_{timeframe}.csv"

    # ------------------------------------------------------------------ #
    #  Incremental path — cache already exists                            #
    # ------------------------------------------------------------------ #
    if cache.exists() and not force:
        df_existing = load_csv(str(cache))
        if df_existing.empty:
            # Corrupt cache — fall through to full download
            logger.warning("Cache %s is empty — re-downloading", cache)
        else:
            last_bar = df_existing.index[-1].to_pydatetime()
            age_hours = (datetime.utcnow() - last_bar).total_seconds() / 3600

            # Already up-to-date (< 1 bar of the timeframe)
            tf_hours = {"M1": 1/60, "M5": 5/60, "M15": 0.25, "M30": 0.5,
                        "H1": 1, "H4": 4, "D1": 24}.get(timeframe.upper(), 1)
            if age_hours <= tf_hours * 2:
                if not quiet:
                    logger.info("  %-8s %-5s → already up-to-date (%s bars, %s)",
                                symbol, timeframe, _bar_count_str(df_existing), _age_str(df_existing))
                return ("up-to-date", len(df_existing),
                        _date_range_str(df_existing), "cache")

            if not quiet:
                logger.info("  %-8s %-5s -> incremental update (cache %.0fh old)",
                            symbol, timeframe, age_hours)
            new_bars = load_from_mt5_since(symbol, timeframe, since=last_bar)

            if not new_bars.empty:
                new_bars = new_bars[new_bars.index > df_existing.index[-1]]
            if new_bars.empty:
                # MT5 offline — yfinance can't incrementally append (limited history)
                if not quiet:
                    logger.info("    -> no new bars from MT5, keeping cached data")
                return ("cache-only", len(df_existing),
                        _date_range_str(df_existing), "cache")

            df_updated = pd.concat([df_existing, new_bars]).sort_index()
            df_updated = df_updated[~df_updated.index.duplicated(keep="last")]
            df_updated.to_csv(cache, index_label="time")
            added = len(df_updated) - len(df_existing)
            if not quiet:
                logger.info("    → appended %d bars (total %s)", added, _bar_count_str(df_updated))
            return ("refreshed", len(df_updated),
                    _date_range_str(df_updated), "MT5-incremental")

    # ------------------------------------------------------------------ #
    #  Full download path — no cache or --force                           #
    # ------------------------------------------------------------------ #
    count = BARS_5YR.get(timeframe.upper(), 90_000)
    # Honour per-market training_start_years if < 5
    count = int(count * (years / 5))

    if not quiet:
        logger.info("  %-8s %-5s -> full download (%d bars requested)", symbol, timeframe, count)

    source = "MT5"
    df = load_from_mt5(symbol, timeframe, count)

    if df.empty:
        source = "yfinance"
        logger.warning("    MT5 unavailable - falling back to yfinance (limited history)")
        df = _yf_bars(symbol, timeframe, years)

    if df.empty:
        logger.error("    No data obtained for %s %s", symbol, timeframe)
        return ("failed", 0, "n/a", source)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index_label="time")
    if not quiet:
        logger.info("    -> saved %s bars (%s) via %s",
                    _bar_count_str(df), _date_range_str(df), source)
    return ("downloaded", len(df), _date_range_str(df), source)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download / refresh historical OHLCV for all markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Override the symbol list (default: all from strategy_config.yaml)",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=None,
        help=f"Timeframes to download (default: {DEFAULT_TIMEFRAMES})",
    )
    parser.add_argument(
        "--config-dir", default="config",
        help="Directory containing *.yaml config files",
    )
    parser.add_argument(
        "--cache-dir", default="data",
        help="Directory to store/read CSV cache files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe existing cache and re-download from scratch",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-bar progress; only print the summary table",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Max parallel download threads (default: auto = min(32, symbols × timeframes))",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    cfg_dir   = Path(args.config_dir)
    cache_dir = Path(args.cache_dir)

    # ---- Load configs -------------------------------------------------------
    with open(cfg_dir / "strategy_config.yaml") as f:
        strat_cfg = yaml.safe_load(f)["strategy"]

    with open(cfg_dir / "market_config.yaml") as f:
        market_cfg = yaml.safe_load(f).get("markets", {})

    symbols    = args.symbols    or strat_cfg.get("symbols", ["EURUSD"])
    timeframes = args.timeframes or DEFAULT_TIMEFRAMES

    # ---- Banner -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  ICT Trade Bot — Market Data Downloader")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Symbols   : {len(symbols)}  ({', '.join(symbols)})")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print(f"  Cache dir : {cache_dir.resolve()}")
    print(f"  Mode      : {'FORCE RE-DOWNLOAD' if args.force else 'incremental refresh'}")
    print("=" * 70 + "\n")

    # ---- Download loop (parallel) -------------------------------------------
    # Downloads are network/I-O bound → ThreadPoolExecutor gives full concurrency
    # without spawning separate processes.
    tasks = []
    for sym in symbols:
        years = market_cfg.get(sym, {}).get("training_start_years", 5)
        for tf in timeframes:
            tasks.append((sym, tf, years))

    max_workers = args.workers if args.workers > 0 else min(32, len(tasks))
    max_workers = max(1, max_workers)
    logger.info("Starting %d download tasks with %d worker thread(s)",
                len(tasks), max_workers)

    # We keep an ordered dict so the summary prints in symbol→timeframe order.
    result_map: Dict[tuple, tuple] = {}
    t0 = time.time()

    def _worker(sym, tf, years):
        return download_symbol(
            symbol=sym,
            timeframe=tf,
            cache_dir=cache_dir,
            years=years,
            force=args.force,
            quiet=args.quiet,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_key = {
            pool.submit(_worker, sym, tf, years): (sym, tf)
            for sym, tf, years in tasks
        }
        for future in as_completed(future_to_key):
            sym, tf = future_to_key[future]
            try:
                status, bars, date_range, source = future.result()
            except Exception as exc:
                logger.exception("Unexpected error for %s %s: %s", sym, tf, exc)
                status, bars, date_range, source = "error", 0, "n/a", "n/a"
            result_map[(sym, tf)] = (status, bars, date_range, source)

    # Reconstruct results in original order for the summary table.
    results = []
    for sym, tf, years in tasks:
        status, bars, date_range, source = result_map[(sym, tf)]
        results.append((sym, tf, status, bars, date_range, source))

    elapsed = time.time() - t0

    # ---- Summary table -------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"  SUMMARY  ({elapsed:.1f}s)")
    print("=" * 70)
    header = f"  {'Symbol':<10} {'TF':<6} {'Status':<14} {'Bars':>8}  {'Range':<27} {'Source'}"
    print(header)
    print("  " + "-" * 66)

    status_icon = {
        "downloaded":  "[+]",
        "refreshed":   "[~]",
        "up-to-date":  "[=]",
        "cache-only":  "[-]",
        "failed":      "[!]",
        "error":       "[!]",
    }

    ok = 0
    failed = 0
    total_bars = 0
    for sym, tf, status, bars, date_range, source in results:
        icon = status_icon.get(status, "[?]")
        print(f"  {icon} {sym:<9} {tf:<6} {status:<14} {bars:>8,}  {date_range:<27} {source}")
        if status in ("downloaded", "refreshed", "up-to-date", "cache-only"):
            ok += 1
            total_bars += bars
        else:
            failed += 1

    print("  " + "-" * 66)
    print(f"  Done: {ok} ok, {failed} failed | {total_bars:,} total bars cached\n")

    if failed:
        print("  [!] Some downloads failed - check logs above.")
        print("      If MT5 is available, ensure it is running and connected.\n")

    sys.exit(1 if failed == len(results) else 0)


if __name__ == "__main__":
    main()
