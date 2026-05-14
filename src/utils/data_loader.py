"""Historical price data loader.

Tries MT5 first, falls back to yfinance, then to a local CSV cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


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
    df[parse_dates_col] = pd.to_datetime(df[parse_dates_col])
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

    # Map forex symbols: EURUSD -> EURUSD=X
    yf_ticker = ticker
    if len(ticker) == 6 and "=" not in ticker:
        yf_ticker = f"{ticker}=X"
        logger.info("Mapping symbol %s -> %s for yfinance", ticker, yf_ticker)

    df = yf.download(yf_ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        # Try original if mapping failed
        if yf_ticker != ticker:
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
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


def load_data(
    symbol: str,
    timeframe: str = "M5",
    count: int = 100_000,
    cache_dir: str = "data",
    use_cache: bool = True,
    min_bars_warn: int = 20_000,
) -> pd.DataFrame:
    """End-to-end loader: cache → MT5 → yfinance, with CSV cache write-through.

    Args:
        min_bars_warn: log a warning if the returned dataset is smaller than
            this threshold (walk-forward training needs ~70k+ M15 bars).
    """
    cache = Path(cache_dir) / f"{symbol}_{timeframe}.csv"
    if use_cache and cache.exists():
        logger.info("Loading cached data from %s", cache)
        df = load_csv(str(cache))
        if len(df) < min_bars_warn:
            logger.warning(
                "%s_%s: only %d bars — walk-forward training needs ~%d+. "
                "Use MT5 for a full history.",
                symbol, timeframe, len(df), min_bars_warn,
            )
        return df

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
