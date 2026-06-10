"""Liquidity detection: sweeps, equal highs/lows, prior session levels.

A liquidity sweep is a price wick that pierces a prior swing high/low and
closes back inside the prior range — the classic 'stop run'.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


@dataclass
class LiquiditySweep:
    """Represents a detected liquidity sweep event."""
    index: int
    timestamp: pd.Timestamp
    direction: str           # 'bullish' (sweep of low) or 'bearish' (sweep of high)
    sweep_level: float       # the price that was swept
    pierce_depth: float      # how far past the level price went, in price units
    pierce_atr: float        # depth normalised by ATR
    close_back_inside: bool


@dataclass
class EqualLevel:
    """Represents an equal-high or equal-low cluster."""
    indices: List[int]
    level: float
    kind: str                # 'EQH' or 'EQL'
    count: int


def _atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing approximated by EMA with alpha=1/period
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _swing_highs_lows(candles: pd.DataFrame, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Return boolean masks for swing highs and swing lows.

    Uses scipy.signal.argrelextrema when available (vectorised, O(n));
    falls back to the original O(n²) loop if scipy is absent.
    """
    high = candles["high"].values
    low = candles["low"].values
    n = len(candles)
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)

    if _HAS_SCIPY:
        sh_idx = argrelextrema(high, np.greater_equal, order=lookback)[0]
        sl_idx = argrelextrema(low, np.less_equal, order=lookback)[0]
        # Keep only strict local extrema (no flat-top duplicates)
        for idx in sh_idx:
            window = high[max(0, idx - lookback): idx + lookback + 1]
            if (window == high[idx]).sum() == 1:
                is_sh[idx] = True
        for idx in sl_idx:
            window = low[max(0, idx - lookback): idx + lookback + 1]
            if (window == low[idx]).sum() == 1:
                is_sl[idx] = True
    else:
        # Fallback: original O(n²) loop
        for i in range(lookback, n - lookback):
            window_h = high[i - lookback: i + lookback + 1]
            window_l = low[i - lookback: i + lookback + 1]
            if high[i] == window_h.max() and (window_h == high[i]).sum() == 1:
                is_sh[i] = True
            if low[i] == window_l.min() and (window_l == low[i]).sum() == 1:
                is_sl[i] = True

    return pd.Series(is_sh, index=candles.index), pd.Series(is_sl, index=candles.index)


def detect_liquidity_sweeps(
    candles: pd.DataFrame,
    lookback: int = 400,
    threshold_atr: float = 0.1,
    swing_lookback: int = 5,
    atr_period: int = 14,
) -> List[LiquiditySweep]:
    """Detect liquidity sweeps where price pierces a prior swing then closes back.

    Args:
        candles: OHLCV DataFrame indexed by timestamp.
        lookback: how many bars back to look for swept swing levels.
        threshold_atr: minimum pierce depth in ATR units to count as a sweep.
        swing_lookback: lookback used to identify swings.
        atr_period: ATR smoothing period.

    Returns:
        List of LiquiditySweep events.
    """
    if len(candles) < lookback + swing_lookback + atr_period:
        return []

    atr = _atr(candles, atr_period).values
    is_sh, is_sl = _swing_highs_lows(candles, swing_lookback)
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    open_ = candles["open"].values

    sh_idx = np.where(is_sh.values)[0]
    sl_idx = np.where(is_sl.values)[0]

    sweeps: List[LiquiditySweep] = []
    for i in range(lookback, len(candles)):
        cur_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        if cur_atr <= 0:
            continue

        # Bearish sweep: high pierces a recent swing high, close back below it.
        left_sh = np.searchsorted(sh_idx, i - lookback)
        right_sh = np.searchsorted(sh_idx, i)
        recent_sh = sh_idx[left_sh:right_sh]
        if recent_sh.size:
            level = high[recent_sh].max()
            if high[i] > level and close[i] < level:
                depth = high[i] - level
                if depth >= threshold_atr * cur_atr:
                    sweeps.append(
                        LiquiditySweep(
                            index=i,
                            timestamp=candles.index[i],
                            direction="bearish",
                            sweep_level=float(level),
                            pierce_depth=float(depth),
                            pierce_atr=float(depth / cur_atr),
                            close_back_inside=True,
                        )
                    )

        # Bullish sweep: low pierces a recent swing low, close back above it.
        left_sl = np.searchsorted(sl_idx, i - lookback)
        right_sl = np.searchsorted(sl_idx, i)
        recent_sl = sl_idx[left_sl:right_sl]
        if recent_sl.size:
            level = low[recent_sl].min()
            if low[i] < level and close[i] > level:
                depth = level - low[i]
                if depth >= threshold_atr * cur_atr:
                    sweeps.append(
                        LiquiditySweep(
                            index=i,
                            timestamp=candles.index[i],
                            direction="bullish",
                            sweep_level=float(level),
                            pierce_depth=float(depth),
                            pierce_atr=float(depth / cur_atr),
                            close_back_inside=True,
                        )
                    )
    return sweeps


def find_previous_session_highs_lows(
    candles: pd.DataFrame,
    sessions: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Return a DataFrame with prior session high and low for each bar.

    Args:
        candles: must have a tz-naive or tz-aware DatetimeIndex.
        sessions: e.g. ``{'london': ('08:00','17:00'), 'newyork': ('13:00','22:00')}``.

    Returns:
        DataFrame indexed like `candles` with columns
        ``f"{session}_prev_high"`` and ``f"{session}_prev_low"``.
    """
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("candles must have a DatetimeIndex")

    out = pd.DataFrame(index=candles.index)
    dates = candles.index.normalize()

    for sess_name, (start_s, end_s) in sessions.items():
        start_t = time.fromisoformat(start_s)
        end_t = time.fromisoformat(end_s)
        in_session = (
            (candles.index.time >= start_t) & (candles.index.time <= end_t)
        )
        df = candles.assign(_date=dates, _in=in_session)
        sess_only = df[df["_in"]]
        if sess_only.empty:
            out[f"{sess_name}_prev_high"] = np.nan
            out[f"{sess_name}_prev_low"] = np.nan
            continue
        agg = sess_only.groupby("_date").agg(high=("high", "max"), low=("low", "min"))
        # Shift one day so we always read the *previous* session.
        agg_prev = agg.shift(1)
        mapped_high = pd.Series(dates).map(agg_prev["high"]).values
        mapped_low = pd.Series(dates).map(agg_prev["low"]).values
        out[f"{sess_name}_prev_high"] = mapped_high
        out[f"{sess_name}_prev_low"] = mapped_low

    return out


def detect_equal_highs_lows(
    candles: pd.DataFrame,
    tolerance_atr: float = 0.1,
    swing_lookback: int = 5,
    atr_period: int = 14,
    cluster_window: int = 50,
) -> List[EqualLevel]:
    """Detect equal-highs and equal-lows (relative-equal liquidity).

    Two swing highs are 'equal' if they differ by less than
    ``tolerance_atr * ATR``. Same for lows.
    """
    if len(candles) < cluster_window:
        return []

    atr = _atr(candles, atr_period).values
    is_sh, is_sl = _swing_highs_lows(candles, swing_lookback)
    high = candles["high"].values
    low = candles["low"].values
    sh_idx = np.where(is_sh.values)[0]
    sl_idx = np.where(is_sl.values)[0]

    levels: List[EqualLevel] = []

    def _cluster(indices: np.ndarray, prices: np.ndarray, kind: str) -> None:
        used = set()
        # Use the median ATR over the visible bars as a safe fallback.
        atr_median = float(np.nanmedian(atr[atr > 0])) if np.any(atr > 0) else 1.0
        for i, idx_a in enumerate(indices):
            if idx_a in used:
                continue
            cluster = [idx_a]
            raw_atr = atr[idx_a]
            safe_atr = float(raw_atr) if np.isfinite(raw_atr) and raw_atr > 0 else atr_median
            tol = tolerance_atr * safe_atr
            for idx_b in indices[i + 1 :]:
                if idx_b - idx_a > cluster_window:
                    break
                if abs(prices[idx_b] - prices[idx_a]) <= tol:
                    cluster.append(idx_b)
                    used.add(idx_b)
            if len(cluster) >= 2:
                level_vals = prices[cluster]
                levels.append(
                    EqualLevel(
                        indices=[int(c) for c in cluster],
                        level=float(np.mean(level_vals)),
                        kind=kind,
                        count=len(cluster),
                    )
                )

    _cluster(sh_idx, high, "EQH")
    _cluster(sl_idx, low, "EQL")
    return levels
