"""Feature engineering pipeline.

Produces a 35-feature row-aligned DataFrame from raw OHLCV + detection outputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..detection.fvg import FVGZone, calculate_fvg_imbalance
from ..detection.liquidity import EqualLevel, LiquiditySweep, _atr
from ..detection.orderblock import OrderBlock
from ..detection.session import add_session_features
from ..detection.structure import StructureEvent, calculate_structure_strength


# The canonical 35 feature columns produced by `build_feature_pipeline`.
FEATURE_COLUMNS: List[str] = [
    # Structure (5)
    "structure_strength",
    "bars_since_bos",
    "bars_since_choch",
    "bos_direction",
    "choch_direction",
    # Liquidity (5)
    "bars_since_liq_sweep",
    "liq_sweep_direction",
    "liq_sweep_pierce_atr",
    "near_eqh_dist_atr",
    "near_eql_dist_atr",
    # FVG (5)
    "active_bull_fvg",
    "active_bear_fvg",
    "fvg_dist_atr",
    "fvg_gap_atr",
    "fvg_imbalance",
    # Order block (5)
    "active_bull_ob",
    "active_bear_ob",
    "ob_dist_atr",
    "ob_strength",
    "ob_move_atr",
    # Time / session (5)
    "sess_London",
    "sess_NY",
    "sess_Asian",
    "sess_London_NY",
    "minute_of_day",
    # Volatility (5)
    "atr_norm",
    "atr_pct_change",
    "range_zscore",
    "body_to_range",
    "wick_imbalance",
    # Microstructure (5)
    "ret_1",
    "ret_5",
    "vwap_dist",
    "cvd_momentum",
    "volume_zscore",
]


def _bars_since(events_idx: List[int], n: int) -> np.ndarray:
    """For each bar 0..n-1, how many bars since the most recent event index (>=0).

    Returns ``n`` if no event has occurred yet.
    """
    out = np.full(n, n, dtype=float)
    last = -1
    ev = sorted(set(events_idx))
    j = 0
    for i in range(n):
        while j < len(ev) and ev[j] <= i:
            last = ev[j]
            j += 1
        if last >= 0:
            out[i] = i - last
    return out


def _last_property(
    events: List, n: int, value_fn, default: float = 0.0
) -> np.ndarray:
    """Forward-fill a per-event property across bars."""
    out = np.full(n, default, dtype=float)
    cur = default
    ev_sorted = sorted(events, key=lambda e: getattr(e, "index", 0))
    j = 0
    for i in range(n):
        while j < len(ev_sorted) and getattr(ev_sorted[j], "index") <= i:
            cur = float(value_fn(ev_sorted[j]))
            j += 1
        out[i] = cur
    return out


def build_feature_pipeline(
    candles: pd.DataFrame,
    detections: Dict[str, object],
    normalize: bool = True,
    correlation_threshold: float = 1.0,
) -> pd.DataFrame:
    """Combine OHLCV + detection outputs into a 35-feature DataFrame.

    Args:
        candles: OHLCV DataFrame indexed by timestamp.
        detections: dict with keys ``'fvg'``, ``'order_blocks'``, ``'liquidity_sweeps'``,
            ``'equal_levels'``, ``'bos'``, ``'choch'``.
        normalize: z-score numeric columns (skips one-hots and bools).
        correlation_threshold: drop columns whose absolute correlation with an earlier
            kept column exceeds this. Default 1.0 = disabled. Only apply on
            training data splits to avoid data leakage.

    Returns:
        DataFrame with no NaNs, indexed like `candles`.
    """
    n = len(candles)
    if n == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    fvg: List[FVGZone] = detections.get("fvg", [])
    obs: List[OrderBlock] = detections.get("order_blocks", [])
    sweeps: List[LiquiditySweep] = detections.get("liquidity_sweeps", [])
    eqs: List[EqualLevel] = detections.get("equal_levels", [])
    bos: List[StructureEvent] = detections.get("bos", [])
    choch: List[StructureEvent] = detections.get("choch", [])

    # Use ffill-only (no bfill) to avoid lookahead bias in early bars.
    atr = _atr(candles, 14).ffill().fillna(0.0)
    atr_safe = atr.replace(0, np.nan).ffill().fillna(1e-6).values
    close = candles["close"].values
    high = candles["high"].values
    low = candles["low"].values
    open_ = candles["open"].values

    feats = pd.DataFrame(index=candles.index)

    # --- Structure (5) ---
    feats["structure_strength"] = calculate_structure_strength(candles, bos, choch).values
    feats["bars_since_bos"] = _bars_since([e.index for e in bos], n)
    feats["bars_since_choch"] = _bars_since([e.index for e in choch], n)
    feats["bos_direction"] = _last_property(
        bos, n, lambda e: 1.0 if e.direction == "bullish" else -1.0, 0.0
    )
    feats["choch_direction"] = _last_property(
        choch, n, lambda e: 1.0 if e.direction == "bullish" else -1.0, 0.0
    )

    # --- Liquidity (5) ---
    feats["bars_since_liq_sweep"] = _bars_since([s.index for s in sweeps], n)
    feats["liq_sweep_direction"] = _last_property(
        sweeps, n, lambda s: 1.0 if s.direction == "bullish" else -1.0, 0.0
    )
    feats["liq_sweep_pierce_atr"] = _last_property(sweeps, n, lambda s: s.pierce_atr, 0.0)

    def _calc_nearest(eq_list, out_array, swing_lookback=5):
        if not eq_list:
            return
        
        activation_indices = np.array([max(eq.indices) + swing_lookback for eq in eq_list])
        levels = np.array([eq.level for eq in eq_list])
        
        sort_idx = np.argsort(activation_indices)
        activation_indices = activation_indices[sort_idx]
        levels = levels[sort_idx]
        
        for i in range(n):
            num_active = np.searchsorted(activation_indices, i, side='right')
            if num_active > 0:
                active_levels = levels[:num_active]
                out_array[i] = np.min(np.abs(active_levels - close[i])) / atr_safe[i]

    nearest_eqh = np.full(n, 10.0)
    nearest_eql = np.full(n, 10.0)
    _calc_nearest([e for e in eqs if e.kind == "EQH"], nearest_eqh)
    _calc_nearest([e for e in eqs if e.kind == "EQL"], nearest_eql)
    
    feats["near_eqh_dist_atr"] = nearest_eqh
    feats["near_eql_dist_atr"] = nearest_eql

    # --- FVG (5) ---
    bull_fvg_active = np.zeros(n)
    bear_fvg_active = np.zeros(n)
    fvg_dist = np.full(n, 10.0)
    fvg_gap = np.zeros(n)
    fvg_imb = np.zeros(n)
    for z in fvg:
        end = z.mitigation_index if z.mitigation_index is not None else n
        sl = slice(z.index, end)
        if z.direction == "bullish":
            bull_fvg_active[sl] = 1
        else:
            bear_fvg_active[sl] = 1
        # Vectorised distance calculation — avoids O(n*m) inner loop.
        dist = np.minimum(np.abs(close[sl] - z.top), np.abs(close[sl] - z.bottom))
        d_atr = dist / np.maximum(atr_safe[sl], 1e-6)
        better = d_atr < fvg_dist[sl]
        fvg_dist[sl] = np.where(better, d_atr, fvg_dist[sl])
        fvg_gap[sl] = np.where(better, z.gap_atr, fvg_gap[sl])
        # Use the proper imbalance scorer (gap_atr × body_strength)
        imb_score = calculate_fvg_imbalance(candles, z)
        fvg_imb[sl] = np.where(better, imb_score, fvg_imb[sl])
    feats["active_bull_fvg"] = bull_fvg_active
    feats["active_bear_fvg"] = bear_fvg_active
    feats["fvg_dist_atr"] = fvg_dist
    feats["fvg_gap_atr"] = fvg_gap
    feats["fvg_imbalance"] = fvg_imb

    # --- Order block (5) ---
    bull_ob_active = np.zeros(n)
    bear_ob_active = np.zeros(n)
    ob_dist = np.full(n, 10.0)
    ob_strength = np.zeros(n)
    ob_move_atr = np.zeros(n)
    for ob in obs:
        end = ob.mitigation_index if ob.mitigation_index is not None else n
        sl = slice(ob.index, end)
        if ob.direction == "bullish":
            bull_ob_active[sl] = 1
        else:
            bear_ob_active[sl] = 1
        # Vectorised distance calculation — avoids O(n*m) inner loop.
        dist = np.minimum(np.abs(close[sl] - ob.top), np.abs(close[sl] - ob.bottom))
        d_atr = dist / np.maximum(atr_safe[sl], 1e-6)
        better = d_atr < ob_dist[sl]
        ob_dist[sl] = np.where(better, d_atr, ob_dist[sl])
        ob_strength[sl] = np.where(better, ob.strength, ob_strength[sl])
        ob_move_atr[sl] = np.where(better, ob.move_atr, ob_move_atr[sl])
    feats["active_bull_ob"] = bull_ob_active
    feats["active_bear_ob"] = bear_ob_active
    feats["ob_dist_atr"] = ob_dist
    feats["ob_strength"] = ob_strength
    feats["ob_move_atr"] = ob_move_atr

    # --- Time / Session (5) ---
    sess_feats = add_session_features(candles)
    for col in ["sess_London", "sess_NY", "sess_Asian", "sess_London_NY"]:
        if col in sess_feats.columns:
            feats[col] = sess_feats[col].values
        else:
            feats[col] = 0
    feats["minute_of_day"] = candles.index.hour * 60 + candles.index.minute

    # --- Volatility (5) ---
    atr_vals = atr.values
    feats["atr_norm"] = atr_vals / np.maximum(close, 1e-6)
    feats["atr_pct_change"] = pd.Series(atr_vals).pct_change(5).fillna(0).values
    rng = high - low
    rng_mean = pd.Series(rng).rolling(20, min_periods=1).mean().values
    rng_std = pd.Series(rng).rolling(20, min_periods=1).std().fillna(1e-6).values
    feats["range_zscore"] = (rng - rng_mean) / np.maximum(rng_std, 1e-6)
    body = np.abs(close - open_)
    feats["body_to_range"] = body / np.maximum(rng, 1e-6)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    feats["wick_imbalance"] = (upper_wick - lower_wick) / np.maximum(rng, 1e-6)

    # --- Microstructure (5) ---
    close_s = pd.Series(close)
    feats["ret_1"] = close_s.pct_change(1).fillna(0).values
    feats["ret_5"] = close_s.pct_change(5).fillna(0).values
    if "volume" in candles.columns:
        vol = candles["volume"].values.astype(float)
        vol_mean = pd.Series(vol).rolling(50, min_periods=1).mean().values
        vol_std = pd.Series(vol).rolling(50, min_periods=1).std().fillna(1e-6).values
        feats["volume_zscore"] = (vol - vol_mean) / np.maximum(vol_std, 1e-6)
        
        # VWAP (Daily Anchored)
        typ_price = (high + low + close) / 3
        daily_groups = pd.Series(vol, index=candles.index).groupby(candles.index.date)
        daily_vol = daily_groups.cumsum().values
        daily_pv = pd.Series(typ_price * vol, index=candles.index).groupby(candles.index.date).cumsum().values
        vwap = daily_pv / np.maximum(daily_vol, 1e-6)
        feats["vwap_dist"] = (close - vwap) / np.maximum(atr_vals, 1e-6)
        
        # CVD (Cumulative Volume Delta) Approximation
        buy_pct = (close - low) / np.maximum(rng, 1e-6)
        sell_pct = (high - close) / np.maximum(rng, 1e-6)
        delta = (buy_pct - sell_pct) * vol
        cvd_mom = pd.Series(delta).rolling(10, min_periods=1).sum().values
        cvd_std = pd.Series(delta).rolling(50, min_periods=1).std().fillna(1e-6).values
        feats["cvd_momentum"] = cvd_mom / np.maximum(cvd_std, 1e-6)
    else:
        feats["volume_zscore"] = 0.0
        feats["vwap_dist"] = 0.0
        feats["cvd_momentum"] = 0.0

    # Order columns canonically and ensure no NaNs / infs.
    feats = feats[FEATURE_COLUMNS].copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if normalize:
        feats = _zscore(feats)

    if correlation_threshold < 1.0:
        feats = _drop_correlated(feats, correlation_threshold)

    return feats


def _zscore(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """Z-score numeric columns using a rolling window to prevent lookahead bias."""
    out = df.copy()
    for col in out.columns:
        v = out[col].values
        unique = np.unique(v)
        if set(unique).issubset({0.0, 1.0, -1.0}):
            continue
        roll = out[col].rolling(window, min_periods=1)
        mean = roll.mean().values
        std = roll.std().fillna(1e-9).values
        out[col] = (v - mean) / np.maximum(std, 1e-9)
    return out


def _drop_correlated(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose absolute correlation with an earlier kept column exceeds threshold."""
    if df.shape[1] < 2:
        return df
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return df.drop(columns=to_drop)
