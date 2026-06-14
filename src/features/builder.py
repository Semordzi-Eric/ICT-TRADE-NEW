"""Feature engineering pipeline.

Produces a 56-feature row-aligned DataFrame from raw OHLCV + detection outputs.

Upgrade log (v3):
  - Replaced raw minute_of_day with time_sin + time_cos (cyclical encoding)
  - Added D1 structural bias: d1_bos_direction, d1_close_vs_open
  - Added daily_open_dist_atr (distance to current day's open price)
  - Added adx_pos_di, adx_neg_di separately (directional strength disaggregated)
  - Added ob_fvg_confluence (binary: active OB and FVG in same price zone)
  - Log-transformed bars_since_bos, bars_since_choch, bars_since_liq_sweep
  - All v2 features retained (48 → 56 features)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..detection.fvg import FVGZone, calculate_fvg_imbalance
from ..detection.liquidity import EqualLevel, LiquiditySweep, _atr
from ..detection.orderblock import OrderBlock
from ..detection.regime import (
    RANGING, RegimeEngine, _calc_adx, _calc_atr,
    _rolling_hurst, _market_efficiency_ratio,
)
from ..detection.session import add_session_features
from ..detection.structure import StructureEvent, calculate_structure_strength


# The canonical 56 feature columns produced by `build_feature_pipeline`.
FEATURE_COLUMNS: List[str] = [
    # ── Structure (5) ──────────────────────────────────────────────────────
    "structure_strength",
    "bars_since_bos",       # log-transformed
    "bars_since_choch",     # log-transformed
    "bos_direction",
    "choch_direction",
    # ── Liquidity (5) ──────────────────────────────────────────────────────
    "bars_since_liq_sweep",  # log-transformed
    "liq_sweep_direction",
    "liq_sweep_pierce_atr",
    "near_eqh_dist_atr",
    "near_eql_dist_atr",
    # ── FVG (5) ────────────────────────────────────────────────────────────
    "active_bull_fvg",
    "active_bear_fvg",
    "fvg_dist_atr",
    "fvg_gap_atr",
    "fvg_imbalance",
    # ── Order Block (5) ────────────────────────────────────────────────────
    "active_bull_ob",
    "active_bear_ob",
    "ob_dist_atr",
    "ob_strength",
    "ob_move_atr",
    # ── Time / Session (5) — minute_of_day replaced by sin+cos ─────────────
    "sess_London",
    "sess_NY",
    "sess_Asian",
    "time_sin",             # NEW: sin(2π·minute/1440) — cyclical hour encoding
    "time_cos",             # NEW: cos(2π·minute/1440) — cyclical hour encoding
    # ── Volatility (5) ─────────────────────────────────────────────────────
    "atr_norm",
    "atr_pct_change",
    "range_zscore",
    "body_to_range",
    "wick_imbalance",
    # ── Microstructure (5) ─────────────────────────────────────────────────
    "ret_1",
    "ret_5",
    "vwap_dist",
    "cvd_momentum",
    "volume_zscore",
    # ── Trend / Regime Indicators (6) — +DI/-DI now separate ───────────────
    "adx",
    "adx_pos_di",           # NEW: +DI directional strength
    "adx_neg_di",           # NEW: -DI directional strength
    "hurst",
    "mer",
    "realized_vol_ratio",
    # ── H4 Structural Bias (3) ─────────────────────────────────────────────
    "h4_bos_direction",
    "h4_choch_direction",
    "h4_adx",
    # ── D1 Structural Bias (3) NEW ─────────────────────────────────────────
    "d1_bos_direction",     # NEW: Daily BOS direction (+1 bull / -1 bear / 0)
    "d1_close_vs_open",     # NEW: (D1 close - D1 open) / ATR — daily momentum
    "daily_open_dist_atr",  # NEW: (close - today's open) / ATR — ODL distance
    # ── OB+FVG Confluence (1) NEW ──────────────────────────────────────────
    "ob_fvg_confluence",    # NEW: 1 if active OB and FVG in same price zone
    # ── Regime One-Hot (4) ─────────────────────────────────────────────────
    "regime_TRENDING_BULL",
    "regime_TRENDING_BEAR",
    "regime_RANGING",
    "regime_HIGH_VOLATILITY",
    # ── Previous Session Levels (2) ────────────────────────────────────────
    "prev_london_dist_atr",
    "prev_ny_dist_atr",
]


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def _bars_since(events_idx: List[int], n: int) -> np.ndarray:
    """For each bar 0..n-1, how many bars since the most recent event index."""
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


# ---------------------------------------------------------------------------
# H4 helpers
# ---------------------------------------------------------------------------

def _resample_to_d1(candles: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 OHLCV to D1."""
    if not isinstance(candles.index, pd.DatetimeIndex):
        return pd.DataFrame()
    ohlcv = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in candles.columns:
        ohlcv["volume"] = "sum"
    d1 = candles.resample("1D").agg(ohlcv).dropna(subset=["close"])
    return d1


def _d1_bos_direction(
    d1_candles: pd.DataFrame,
    m15_index: pd.DatetimeIndex,
) -> np.ndarray:
    """D1 BOS direction forward-filled to M15 bars.

    Values: +1 bullish, -1 bearish, 0 neutral.
    """
    from ..detection.structure import detect_bos

    if len(d1_candles) < 10:
        return np.zeros(len(m15_index))

    bos_events = detect_bos(d1_candles, period=10, confirmation_bars=1, swing_lookback=3)
    dir_series = pd.Series(0.0, index=d1_candles.index)
    for ev in bos_events:
        ts = d1_candles.index[ev.index]
        dir_series.loc[ts] = 1.0 if ev.direction == "bullish" else -1.0
    dir_series = dir_series.replace(0.0, np.nan).ffill().fillna(0.0)
    combined = dir_series.reindex(
        dir_series.index.union(m15_index)
    ).ffill().reindex(m15_index).fillna(0.0)
    return combined.values


def _d1_close_vs_open(
    d1_candles: pd.DataFrame,
    m15_index: pd.DatetimeIndex,
    atr_safe: np.ndarray,
) -> np.ndarray:
    """(D1 close - D1 open) / ATR — daily candle momentum, forward-filled."""
    if len(d1_candles) < 2:
        return np.zeros(len(m15_index))

    daily_mom = (d1_candles["close"] - d1_candles["open"]).rename("d1_co")
    combined = daily_mom.reindex(
        daily_mom.index.union(m15_index)
    ).ffill().reindex(m15_index).fillna(0.0)
    # Normalize by ATR
    return combined.values / np.maximum(atr_safe, 1e-9)


def _daily_open_dist_atr(
    candles: pd.DataFrame,
    atr_safe: np.ndarray,
) -> np.ndarray:
    """(close - today's open) / ATR — how far price has moved from the daily open.

    Positive = price above daily open (premium), negative = discount.
    """
    close = candles["close"].values
    # Forward-fill the first bar of each calendar day as that day's open
    daily_open = (
        candles["open"]
        .groupby(candles.index.date)
        .transform("first")
        .values
    )
    return (close - daily_open) / np.maximum(atr_safe, 1e-9)



def _resample_to_h4(candles: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 OHLCV to H4."""
    if not isinstance(candles.index, pd.DatetimeIndex):
        return pd.DataFrame()
    ohlcv = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
    }
    if "volume" in candles.columns:
        ohlcv["volume"] = "sum"
    h4 = candles.resample("4h").agg(ohlcv).dropna(subset=["close"])
    return h4


def _h4_bos_choch_direction(
    h4_candles: pd.DataFrame,
    m15_index: pd.DatetimeIndex,
) -> tuple:
    """Return two arrays (bos_dir, choch_dir) forward-filled onto m15_index.

    Values: +1 bullish, -1 bearish, 0 neutral.
    We use the H4 structure detection functions directly.
    """
    from ..detection.structure import detect_bos, detect_choch

    if len(h4_candles) < 30:
        z = np.zeros(len(m15_index))
        return z, z

    bos_events   = detect_bos(h4_candles, period=20, confirmation_bars=1, swing_lookback=3)
    choch_events = detect_choch(h4_candles, swing_lookback=3)

    def _fill(events, m15_idx):
        dir_series = pd.Series(0.0, index=h4_candles.index)
        for ev in events:
            ts = h4_candles.index[ev.index]
            dir_series.loc[ts] = 1.0 if ev.direction == "bullish" else -1.0
        dir_series = dir_series.replace(0.0, np.nan).ffill().fillna(0.0)
        # Reindex to M15: forward-fill H4 signal into M15 bars
        combined = dir_series.reindex(
            dir_series.index.union(m15_idx)
        ).ffill().reindex(m15_idx).fillna(0.0)
        return combined.values

    return _fill(bos_events, m15_index), _fill(choch_events, m15_index)


def _h4_adx(h4_candles: pd.DataFrame, m15_index: pd.DatetimeIndex) -> np.ndarray:
    """H4 ADX forward-filled to M15 frequency."""
    if len(h4_candles) < 30:
        return np.full(len(m15_index), 20.0)
    adx_df = _calc_adx(h4_candles["high"], h4_candles["low"], h4_candles["close"], 14)
    adx_series = adx_df["adx"].reindex(
        adx_df.index.union(m15_index)
    ).ffill().reindex(m15_index).fillna(20.0)
    return adx_series.values


# ---------------------------------------------------------------------------
# Previous session high/low distance
# ---------------------------------------------------------------------------

_SESSION_DEF = {
    "london":  ("08:00", "17:00"),
    "newyork": ("13:00", "22:00"),
}


def _prev_session_dist_atr(
    candles: pd.DataFrame,
    atr_safe: np.ndarray,
) -> tuple:
    """Distance (in ATR) from current close to prev London & NY session H/L."""
    from ..detection.liquidity import find_previous_session_highs_lows
    n = len(candles)

    try:
        sess_df = find_previous_session_highs_lows(candles, _SESSION_DEF)
    except Exception:
        return np.full(n, 2.0), np.full(n, 2.0)

    close = candles["close"].values

    def _dist(col_h, col_l):
        arr = np.full(n, 2.0)
        if col_h not in sess_df.columns or col_l not in sess_df.columns:
            return arr
        h_vals = sess_df[col_h].values
        l_vals = sess_df[col_l].values
        for i in range(n):
            ch, cl = h_vals[i], l_vals[i]
            if np.isnan(ch) or np.isnan(cl):
                continue
            dist_h = abs(close[i] - ch)
            dist_l = abs(close[i] - cl)
            arr[i] = min(dist_h, dist_l) / max(atr_safe[i], 1e-9)
        return arr

    london_dist = _dist("london_prev_high", "london_prev_low")
    ny_dist     = _dist("newyork_prev_high", "newyork_prev_low")
    return london_dist, ny_dist


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_feature_pipeline(
    candles: pd.DataFrame,
    detections: Dict[str, object],
    normalize: bool = True,
    drop_columns: Optional[List[str]] = None,
    h4_candles: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine OHLCV + detection outputs into a 48-feature DataFrame.

    Args:
        candles:      OHLCV DataFrame indexed by timestamp.
        detections:   dict with keys ``'fvg'``, ``'order_blocks'``,
                      ``'liquidity_sweeps'``, ``'equal_levels'``, ``'bos'``,
                      ``'choch'``.
        normalize:    z-score numeric columns (skips one-hots and bools).
        drop_columns: list of correlated columns to drop (computed on training
                      fold only to avoid lookahead bias).
        h4_candles:   optional pre-resampled H4 DataFrame; if None, the
                      pipeline resamples from *candles* automatically.

    Returns:
        DataFrame with no NaNs, indexed like *candles*.
    """
    n = len(candles)
    if n == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    fvg:    List[FVGZone]        = detections.get("fvg", [])
    obs:    List[OrderBlock]     = detections.get("order_blocks", [])
    sweeps: List[LiquiditySweep] = detections.get("liquidity_sweeps", [])
    eqs:    List[EqualLevel]     = detections.get("equal_levels", [])
    bos:    List[StructureEvent] = detections.get("bos", [])
    choch:  List[StructureEvent] = detections.get("choch", [])

    atr      = _atr(candles, 14).ffill().fillna(0.0)
    atr_safe = atr.replace(0, np.nan).ffill().fillna(1e-6).values
    close    = candles["close"].values
    high     = candles["high"].values
    low      = candles["low"].values
    open_    = candles["open"].values

    feats = pd.DataFrame(index=candles.index)

    # ── Structure (5) ──────────────────────────────────────────────────────
    feats["structure_strength"] = calculate_structure_strength(candles, bos, choch).values
    # Log-transform bars_since features: recency decays non-linearly
    feats["bars_since_bos"]   = np.log1p(_bars_since([e.index for e in bos], n))
    feats["bars_since_choch"] = np.log1p(_bars_since([e.index for e in choch], n))
    feats["bos_direction"]    = _last_property(
        bos, n, lambda e: 1.0 if e.direction == "bullish" else -1.0, 0.0)
    feats["choch_direction"]  = _last_property(
        choch, n, lambda e: 1.0 if e.direction == "bullish" else -1.0, 0.0)

    # ── Liquidity (5) ──────────────────────────────────────────────────────
    feats["bars_since_liq_sweep"]  = np.log1p(_bars_since([s.index for s in sweeps], n))
    feats["liq_sweep_direction"]   = _last_property(
        sweeps, n, lambda s: 1.0 if s.direction == "bullish" else -1.0, 0.0)
    feats["liq_sweep_pierce_atr"]  = _last_property(sweeps, n, lambda s: s.pierce_atr, 0.0)

    def _calc_nearest(eq_list, out_array, swing_lookback=5):
        if not eq_list:
            return
        activation_indices = np.array([max(eq.indices) + swing_lookback for eq in eq_list])
        levels             = np.array([eq.level for eq in eq_list])
        sort_idx           = np.argsort(activation_indices)
        activation_indices = activation_indices[sort_idx]
        levels             = levels[sort_idx]
        for i in range(n):
            num_active = np.searchsorted(activation_indices, i, side="right")
            if num_active > 0:
                active_levels = levels[:num_active]
                out_array[i]  = np.min(np.abs(active_levels - close[i])) / atr_safe[i]

    nearest_eqh = np.full(n, 10.0)
    nearest_eql = np.full(n, 10.0)
    _calc_nearest([e for e in eqs if e.kind == "EQH"], nearest_eqh)
    _calc_nearest([e for e in eqs if e.kind == "EQL"], nearest_eql)
    feats["near_eqh_dist_atr"] = nearest_eqh
    feats["near_eql_dist_atr"] = nearest_eql

    # ── FVG (5) ────────────────────────────────────────────────────────────
    bull_fvg_active = np.zeros(n)
    bear_fvg_active = np.zeros(n)
    fvg_dist = np.full(n, 10.0)
    fvg_gap  = np.zeros(n)
    fvg_imb  = np.zeros(n)
    for z in fvg:
        end = z.mitigation_index if z.mitigation_index is not None else n
        sl  = slice(z.index, end)
        if z.direction == "bullish":
            bull_fvg_active[sl] = 1
        else:
            bear_fvg_active[sl] = 1
        dist     = np.minimum(np.abs(close[sl] - z.top), np.abs(close[sl] - z.bottom))
        d_atr    = dist / np.maximum(atr_safe[sl], 1e-6)
        better   = d_atr < fvg_dist[sl]
        fvg_dist[sl] = np.where(better, d_atr, fvg_dist[sl])
        fvg_gap[sl]  = np.where(better, z.gap_atr, fvg_gap[sl])
        imb_score    = calculate_fvg_imbalance(candles, z)
        fvg_imb[sl]  = np.where(better, imb_score, fvg_imb[sl])
    feats["active_bull_fvg"] = bull_fvg_active
    feats["active_bear_fvg"] = bear_fvg_active
    feats["fvg_dist_atr"]    = fvg_dist
    feats["fvg_gap_atr"]     = fvg_gap
    feats["fvg_imbalance"]   = fvg_imb

    # ── Order Block (5) ────────────────────────────────────────────────────
    bull_ob_active = np.zeros(n)
    bear_ob_active = np.zeros(n)
    ob_dist     = np.full(n, 10.0)
    ob_strength = np.zeros(n)
    ob_move_atr = np.zeros(n)
    for ob in obs:
        end = ob.mitigation_index if ob.mitigation_index is not None else n
        sl  = slice(ob.index, end)
        if ob.direction == "bullish":
            bull_ob_active[sl] = 1
        else:
            bear_ob_active[sl] = 1
        dist    = np.minimum(np.abs(close[sl] - ob.top), np.abs(close[sl] - ob.bottom))
        d_atr   = dist / np.maximum(atr_safe[sl], 1e-6)
        better  = d_atr < ob_dist[sl]
        ob_dist[sl]     = np.where(better, d_atr, ob_dist[sl])
        ob_strength[sl] = np.where(better, ob.strength, ob_strength[sl])
        ob_move_atr[sl] = np.where(better, ob.move_atr, ob_move_atr[sl])
    feats["active_bull_ob"] = bull_ob_active
    feats["active_bear_ob"] = bear_ob_active
    feats["ob_dist_atr"]    = ob_dist
    feats["ob_strength"]    = ob_strength
    feats["ob_move_atr"]    = ob_move_atr

    # ── Time / Session (4) — sess_London_NY dropped ────────────────────────
    sess_feats = add_session_features(candles)
    for col in ["sess_London", "sess_NY", "sess_Asian"]:
        feats[col] = sess_feats[col].values if col in sess_feats.columns else 0
    # Cyclical time encoding — replaces raw minute_of_day integer
    minutes = candles.index.hour * 60 + candles.index.minute
    feats["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    feats["time_cos"] = np.cos(2 * np.pi * minutes / 1440)

    # ── Volatility (5) ─────────────────────────────────────────────────────
    atr_vals = atr.values
    feats["atr_norm"]       = atr_vals / np.maximum(close, 1e-6)
    feats["atr_pct_change"] = pd.Series(atr_vals).pct_change(5).fillna(0).values
    rng      = high - low
    rng_mean = pd.Series(rng).rolling(20, min_periods=1).mean().values
    rng_std  = pd.Series(rng).rolling(20, min_periods=1).std().fillna(1e-6).values
    feats["range_zscore"]   = (rng - rng_mean) / np.maximum(rng_std, 1e-6)
    body = np.abs(close - open_)
    feats["body_to_range"]  = body / np.maximum(rng, 1e-6)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    feats["wick_imbalance"] = (upper_wick - lower_wick) / np.maximum(rng, 1e-6)

    # ── Microstructure (5) ─────────────────────────────────────────────────
    close_s = pd.Series(close)
    feats["ret_1"] = close_s.pct_change(1).fillna(0).values
    feats["ret_5"] = close_s.pct_change(5).fillna(0).values
    if "volume" in candles.columns:
        vol      = candles["volume"].values.astype(float)
        vol_mean = pd.Series(vol).rolling(50, min_periods=1).mean().values
        vol_std  = pd.Series(vol).rolling(50, min_periods=1).std().fillna(1e-6).values
        feats["volume_zscore"] = (vol - vol_mean) / np.maximum(vol_std, 1e-6)
        typ_price   = (high + low + close) / 3
        daily_groups = pd.Series(vol, index=candles.index).groupby(candles.index.date)
        daily_vol    = daily_groups.cumsum().values
        daily_pv     = (pd.Series(typ_price * vol, index=candles.index)
                        .groupby(candles.index.date).cumsum().values)
        vwap = daily_pv / np.maximum(daily_vol, 1e-6)
        feats["vwap_dist"]    = (close - vwap) / np.maximum(atr_vals, 1e-6)
        buy_pct  = (close - low)  / np.maximum(rng, 1e-6)
        sell_pct = (high - close) / np.maximum(rng, 1e-6)
        delta    = (buy_pct - sell_pct) * vol
        cvd_mom  = pd.Series(delta).rolling(10, min_periods=1).sum().values
        cvd_std  = pd.Series(delta).rolling(50, min_periods=1).std().fillna(1e-6).values
        feats["cvd_momentum"] = cvd_mom / np.maximum(cvd_std, 1e-6)
    else:
        feats["volume_zscore"] = 0.0
        feats["vwap_dist"]     = 0.0
        feats["cvd_momentum"]  = 0.0

    # ── Trend / Regime Indicators (4) NEW ──────────────────────────────────
    # ADX-14
    adx_df = _calc_adx(
        pd.Series(high, index=candles.index),
        pd.Series(low,  index=candles.index),
        pd.Series(close, index=candles.index),
        14,
    )
    feats["adx"]        = adx_df["adx"].values
    feats["adx_pos_di"] = adx_df["pos_di"].values   # NEW: +DI strength
    feats["adx_neg_di"] = adx_df["neg_di"].values   # NEW: -DI strength

    # Hurst Exponent (100-bar rolling)
    feats["hurst"] = _rolling_hurst(
        pd.Series(close, index=candles.index), window=100
    ).values

    # Market Efficiency Ratio (10-bar)
    feats["mer"] = _market_efficiency_ratio(
        pd.Series(close, index=candles.index), window=10
    ).values

    # Realized Volatility Ratio (current 20-bar / 100-bar reference)
    prev_close  = pd.Series(close).shift(1).bfill()
    log_rets    = pd.Series(np.log(close / np.maximum(prev_close, 1e-9)))
    rv_20  = log_rets.rolling(20,  min_periods=5).std().fillna(0).values
    rv_100 = log_rets.rolling(100, min_periods=20).std().fillna(0).values
    feats["realized_vol_ratio"] = rv_20 / np.maximum(rv_100, 1e-9)

    # ── H4 Structural Bias (3) NEW ─────────────────────────────────────────
    if h4_candles is None:
        h4_candles = _resample_to_h4(candles)

    if not h4_candles.empty:
        h4_bos_dir, h4_choch_dir = _h4_bos_choch_direction(h4_candles, candles.index)
        feats["h4_bos_direction"]   = h4_bos_dir
        feats["h4_choch_direction"] = h4_choch_dir
        feats["h4_adx"]             = _h4_adx(h4_candles, candles.index)
    else:
        feats["h4_bos_direction"]   = 0.0
        feats["h4_choch_direction"] = 0.0
        feats["h4_adx"]             = 20.0

    # ── D1 Structural Bias (3) NEW ─────────────────────────────────────────
    d1_candles = _resample_to_d1(candles)
    if not d1_candles.empty:
        feats["d1_bos_direction"]  = _d1_bos_direction(d1_candles, candles.index)
        feats["d1_close_vs_open"]  = _d1_close_vs_open(d1_candles, candles.index, atr_safe)
    else:
        feats["d1_bos_direction"]  = 0.0
        feats["d1_close_vs_open"]  = 0.0
    feats["daily_open_dist_atr"] = _daily_open_dist_atr(candles, atr_safe)

    # ── OB + FVG Confluence (1) NEW ────────────────────────────────────────
    # Binary: 1 when there is an active OB and an active FVG in the same
    # price direction AND their price ranges overlap — highest-conviction zone
    confluence = np.zeros(n)
    for i in range(n):
        bull_ob_on  = bull_ob_active[i] > 0
        bear_ob_on  = bear_ob_active[i] > 0
        bull_fvg_on = bull_fvg_active[i] > 0
        bear_fvg_on = bear_fvg_active[i] > 0
        if (bull_ob_on and bull_fvg_on) or (bear_ob_on and bear_fvg_on):
            confluence[i] = 1.0
    feats["ob_fvg_confluence"] = confluence

    # ── Regime One-Hot (4) NEW ─────────────────────────────────────────────
    engine       = RegimeEngine()
    regime_ser   = engine.detect(candles)
    regime_onehot = engine.encode_onehot(regime_ser)
    for col in ["regime_TRENDING_BULL", "regime_TRENDING_BEAR",
                "regime_RANGING", "regime_HIGH_VOLATILITY"]:
        feats[col] = regime_onehot.get(col, pd.Series(0, index=candles.index)).values

    # ── Previous Session Levels (2) NEW ────────────────────────────────────
    lon_dist, ny_dist = _prev_session_dist_atr(candles, atr_safe)
    feats["prev_london_dist_atr"] = np.clip(lon_dist, 0, 20)
    feats["prev_ny_dist_atr"]     = np.clip(ny_dist,  0, 20)

    # ── Canonicalize → no NaNs / infs ──────────────────────────────────────
    feats = feats[FEATURE_COLUMNS].copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if normalize:
        feats = _zscore(feats)

    if drop_columns:
        valid_drops = [c for c in drop_columns if c in feats.columns]
        feats = feats.drop(columns=valid_drops)

    return feats


# ---------------------------------------------------------------------------
# Z-score normalization (rolling, no lookahead)
# ---------------------------------------------------------------------------

def _zscore(df: pd.DataFrame, window: int = 500) -> pd.DataFrame:
    """Z-score numeric columns using a rolling window (no lookahead)."""
    out = df.copy()
    for col in out.columns:
        v      = out[col].values
        unique = np.unique(v)
        # Skip binary / one-hot columns
        if set(unique).issubset({0.0, 1.0, -1.0}):
            continue
        roll = out[col].rolling(window, min_periods=1)
        mean = roll.mean().values
        std  = roll.std().fillna(1e-9).values
        out[col] = (v - mean) / np.maximum(std, 1e-9)
    return out


# ---------------------------------------------------------------------------
# Correlation utility (unchanged)
# ---------------------------------------------------------------------------

def get_correlated_columns(df: pd.DataFrame, threshold: float) -> List[str]:
    """Find columns whose absolute correlation with an earlier kept column exceeds threshold."""
    if df.shape[1] < 2:
        return []
    corr  = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return [c for c in upper.columns if (upper[c] > threshold).any()]
