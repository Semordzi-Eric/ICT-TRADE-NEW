"""Historical price data loader.

Priority chain:
    1. Local CSV cache  (always read first if it exists)
    2. Incremental refresh via MT5  (append only new bars to the cache)
    3. Full download via MT5        (when cache is missing)
    4. yfinance fallback            (when MT5 is unavailable)

After the first full download the cache is always used and only the delta
since the last cached bar is fetched online — so startup is near-instant
and you never re-download history you already have.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# yfinance ticker map — symbols that don't follow the SYMBOL=X forex pattern
# ---------------------------------------------------------------------------

#: Maps internal broker symbols to their Yahoo Finance equivalents.
#: Forex pairs NOT listed here will use the automatic ``SYMBOL=X`` mapping.
YF_TICKER_MAP: dict = {
    # Crypto (use ASSET-USD dash format)
    "BTCUSD":  "BTC-USD",
    "ETHUSD":  "ETH-USD",
    "LTCUSD":  "LTC-USD",
    "XRPUSD":  "XRP-USD",
    # Metals (continuous front-month futures)
    "XAUUSD":  "GC=F",
    "XAGUSD":  "SI=F",
    # Equity indices (CME futures)
    "NAS100":  "NQ=F",
    "SPX500":  "ES=F",
    "US30":    "YM=F",
    "US500":   "ES=F",
    # European indices
    "UK100":   "^FTSE",
    "GER40":   "^GDAXI",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_csv(
    path: str,
    parse_dates_col: str = "time",
    resample_to: Optional[str] = None,
) -> pd.DataFrame:
    """Load OHLCV from a CSV with a parseable timestamp column."""
    df = pd.read_csv(path)
    if parse_dates_col not in df.columns:
        for c in df.columns:
            if c.lower() in ("time", "timestamp", "date", "datetime"):
                parse_dates_col = c
                break
    df[parse_dates_col] = pd.to_datetime(df[parse_dates_col], format="mixed")
    df = df.set_index(parse_dates_col).sort_index()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    if resample_to:
        df = resample_ohlcv(df, resample_to)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample bar data to a new timeframe (e.g. ``'5T'``, ``'1H'``)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(rule).agg(agg).dropna()


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------

def load_from_mt5(
    symbol: str,
    timeframe: str,
    count: int = 100_000,
) -> pd.DataFrame:
    """Pull bars from a running MT5 terminal."""
    try:
        from ..live.mt5_client import HAS_MT5, MT5Client
    except ImportError:
        return pd.DataFrame()
    if not HAS_MT5:
        return pd.DataFrame()
    client = MT5Client()
    if not client.connect():
        return pd.DataFrame()
    try:
        return client.fetch_rates(symbol, timeframe, count)
    finally:
        client.disconnect()


def load_from_mt5_since(
    symbol: str,
    timeframe: str,
    since: datetime,
) -> pd.DataFrame:
    """Pull only bars newer than *since* from MT5 (incremental refresh).

    Uses ``copy_rates_range`` so we only transfer what we don't have yet.
    Falls back to an empty DataFrame when MT5 is unavailable.
    """
    try:
        from ..live.mt5_client import HAS_MT5, MT5Client
        import MetaTrader5 as _mt5
    except ImportError:
        return pd.DataFrame()
    if not HAS_MT5:
        return pd.DataFrame()

    client = MT5Client()
    if not client.connect():
        return pd.DataFrame()
    try:
        # Fetch from (last_bar + 1 second) to now
        date_from = since + timedelta(seconds=1)
        date_to = datetime.utcnow() + timedelta(hours=1)  # slight buffer
        tf_key = {
            "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5",
            "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
            "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
            "D1": "TIMEFRAME_D1",
        }.get(timeframe.upper())
        if tf_key is None:
            return pd.DataFrame()
        tf_val = getattr(_mt5, tf_key)
        rates = _mt5.copy_rates_range(symbol, tf_val, date_from, date_to)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df = df.set_index("time").rename(columns={"tick_volume": "volume"})
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]
    finally:
        client.disconnect()


def load_from_yfinance(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Free fallback when MT5 isn't available.

    Note: yfinance intraday history limits:
      - 1m / 2m: last 7 days only
      - 5m / 15m / 30m: last 60 days only
      - 1h: last 730 days
      - 1d+: unlimited
    For walk-forward training (12-month windows) you NEED MT5 or a paid
    data feed. yfinance is suitable for quick smoke-tests only.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed")
        return pd.DataFrame()

    # Resolve to the correct Yahoo Finance ticker.
    # Use the explicit map first (crypto / metals / indices),
    # then fall back to SYMBOL=X for 6-character forex pairs.
    yf_ticker = YF_TICKER_MAP.get(ticker.upper())
    if yf_ticker is None:
        if len(ticker) == 6 and "=" not in ticker and "-" not in ticker:
            yf_ticker = f"{ticker}=X"
            logger.info("Mapping symbol %s -> %s for yfinance", ticker, yf_ticker)
        else:
            yf_ticker = ticker

    df = yf.download(yf_ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        logger.warning("yfinance: no data for %s (period=%s interval=%s)", yf_ticker, period, interval)
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df.index.name = "time"
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    symbol: str,
    timeframe: str = "M5",
    count: int = 100_000,
    cache_dir: str = "data",
    use_cache: bool = True,
    min_bars_warn: int = 20_000,
    incremental: bool = True,
    max_cache_age_hours: float = 4.0,
) -> pd.DataFrame:
    """End-to-end loader: cache → incremental MT5 refresh → full download → yfinance.

    Flow
    ----
    1. If a local CSV exists and ``use_cache=True``:
       a. Load it.
       b. If ``incremental=True`` and the newest bar is older than
          ``max_cache_age_hours``, fetch only the missing bars from MT5 and
          append them to the cache file.
    2. If no cache exists (or ``use_cache=False``):
       a. Full pull from MT5.
       b. If MT5 unavailable, fall back to yfinance.
       c. Write result to cache.

    Args:
        symbol:              Instrument name (e.g. ``'EURUSD'``).
        timeframe:           Bar timeframe (``'M5'``, ``'M15'``, ``'H1'``, etc.).
        count:               Number of bars to pull on a full (non-incremental) download.
        cache_dir:           Directory that holds ``SYMBOL_TF.csv`` files.
        use_cache:           Set ``False`` to force a full re-download.
        min_bars_warn:       Warn when the dataset is smaller than this.
        incremental:         Append only new bars instead of re-downloading everything.
        max_cache_age_hours: Skip the incremental refresh when the cache is
                             already fresh enough (avoids an MT5 round-trip on
                             every import).
    """
    cache = Path(cache_dir) / f"{symbol}_{timeframe}.csv"

    # ------------------------------------------------------------------
    # Branch A: cache exists
    # ------------------------------------------------------------------
    if use_cache and cache.exists():
        logger.info("Loading cached data from %s", cache)
        df = load_csv(str(cache))

        if incremental and not df.empty:
            last_bar: datetime = df.index[-1].to_pydatetime()  # type: ignore[union-attr]
            age_hours = (datetime.utcnow() - last_bar).total_seconds() / 3600

            if age_hours > max_cache_age_hours:
                logger.info(
                    "%s_%s cache is %.1fh old — fetching incremental update from MT5",
                    symbol, timeframe, age_hours,
                )
                new_bars = load_from_mt5_since(symbol, timeframe, since=last_bar)
                if not new_bars.empty:
                    # Drop overlap (last bar may be incomplete in MT5 history)
                    new_bars = new_bars[new_bars.index > df.index[-1]]
                    if not new_bars.empty:
                        df = pd.concat([df, new_bars]).sort_index()
                        # Rewrite cache with updated data
                        df.to_csv(cache, index_label="time")
                        logger.info(
                            "Appended %d new bars to %s (total %d)",
                            len(new_bars), cache, len(df),
                        )
                    else:
                        logger.info("%s_%s is already up-to-date", symbol, timeframe)
                else:
                    logger.info(
                        "MT5 incremental fetch returned nothing for %s_%s "
                        "(MT5 may be offline — using cached data as-is)",
                        symbol, timeframe,
                    )
            else:
                logger.info(
                    "%s_%s cache is %.1fh old — within %.1fh threshold, skipping refresh",
                    symbol, timeframe, age_hours, max_cache_age_hours,
                )

        if len(df) < min_bars_warn:
            logger.warning(
                "%s_%s: only %d bars — walk-forward training needs ~%d+. "
                "Use MT5 for a full history.",
                symbol, timeframe, len(df), min_bars_warn,
            )
        return df

    # ------------------------------------------------------------------
    # Branch B: no cache — full download
    # ------------------------------------------------------------------
    df = load_from_mt5(symbol, timeframe, count)
    if df.empty:
        # Map MT5 timeframe → yfinance interval.
        # NOTE: yfinance intraday limits apply (60d max for sub-1h intervals).
        yf_interval_map = {
            "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "1h", "D1": "1d",
        }
        # Use 1h / daily to maximise available history from yfinance.
        interval = yf_interval_map.get(timeframe.upper(), "5m")
        period = "730d" if interval in ("1h", "1d") else "60d"
        logger.warning(
            "MT5 unavailable for %s. Falling back to yfinance (%s / %s). "
            "History will be limited — use MT5 for production training.",
            symbol, interval, period,
        )
        df = load_from_yfinance(symbol, period=period, interval=interval)

    if not df.empty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index_label="time")
        logger.info("Cached %d rows to %s", len(df), cache)
        if len(df) < min_bars_warn:
            logger.warning(
                "%s_%s: only %d bars — walk-forward training needs ~%d+. "
                "Use MT5 for a full history.",
                symbol, timeframe, len(df), min_bars_warn,
            )
    return df
