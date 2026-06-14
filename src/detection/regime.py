"""Market Regime Detection Engine.

Classifies each bar into one of four regimes:
    TRENDING_BULL   — ADX > threshold AND price rising
    TRENDING_BEAR   — ADX > threshold AND price falling
    RANGING         — ADX <= threshold AND low Hurst (mean-reverting)
    HIGH_VOLATILITY — ATR > 1.5× rolling average (volatility expansion)

Usage::

    from src.detection.regime import RegimeEngine, TRENDING_BULL, RANGING

    engine = RegimeEngine()
    regime_series = engine.detect(candles)          # pd.Series of str labels
    features_df   = engine.features(candles)        # DataFrame of numeric signals
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Regime label constants
TRENDING_BULL   = "TRENDING_BULL"
TRENDING_BEAR   = "TRENDING_BEAR"
RANGING         = "RANGING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"

_ALL_REGIMES = [TRENDING_BULL, TRENDING_BEAR, RANGING, HIGH_VOLATILITY]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RegimeEngine:
    """Detects market regime using ADX, Hurst Exponent, and ATR-ratio.

    Args:
        adx_period:          ATR/DM smoothing period (standard = 14).
        adx_trend_threshold: ADX value above which we declare a trend (25).
        hurst_window:        bars used for Hurst exponent rolling calc (100).
        hurst_trend_min:     H > this → trending (0.55). H < 0.45 → mean-revert.
        vol_expansion_ratio: current_atr / rolling_atr_ref ratio above which
                             we declare HIGH_VOLATILITY (1.5).
        vol_ref_window:      reference window for ATR normalisation (100).
        adx_smooth:          number of bars to forward-fill regime label
                             (smooths rapid flipping).
    """

    def __init__(
        self,
        adx_period:          int   = 14,
        adx_trend_threshold: float = 25.0,
        hurst_window:        int   = 100,
        hurst_trend_min:     float = 0.55,
        vol_expansion_ratio: float = 1.5,
        vol_ref_window:      int   = 100,
        smooth_bars:         int   = 4,
    ) -> None:
        self.adx_period          = adx_period
        self.adx_threshold       = adx_trend_threshold
        self.hurst_window        = hurst_window
        self.hurst_trend_min     = hurst_trend_min
        self.vol_expansion_ratio = vol_expansion_ratio
        self.vol_ref_window      = vol_ref_window
        self.smooth_bars         = smooth_bars

    # ------------------------------------------------------------------
    def detect(self, candles: pd.DataFrame) -> pd.Series:
        """Return a pd.Series of regime labels aligned to *candles*.

        The series is forward-filled for ``smooth_bars`` to reduce flipping.
        """
        f = self.features(candles)
        labels = _classify(f, self.adx_threshold, self.hurst_trend_min,
                           self.vol_expansion_ratio)
        # Smooth: carry last confirmed label forward for smooth_bars
        if self.smooth_bars > 1:
            labels = labels.ffill(limit=self.smooth_bars)
        labels = labels.fillna(RANGING)
        return labels

    # ------------------------------------------------------------------
    def features(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of raw regime indicator values.

        Columns:
            adx              — ADX(14) value 0-100
            adx_pos_di       — +DI line
            adx_neg_di       — -DI line
            hurst            — rolling Hurst exponent (0-1)
            atr_ratio        — current ATR / reference ATR
            price_trend      — sign of EMA(20)-EMA(50) difference (+1/-1/0)
        """
        close = candles["close"]
        high  = candles["high"]
        low   = candles["low"]

        adx_df   = _calc_adx(high, low, close, self.adx_period)
        hurst    = _rolling_hurst(close, self.hurst_window)
        atr      = _calc_atr(high, low, close, self.adx_period)
        ref_atr  = atr.rolling(self.vol_ref_window, min_periods=1).mean()
        atr_ratio = atr / ref_atr.replace(0, np.nan).ffill().fillna(1.0)

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        diff  = ema20 - ema50
        price_trend = np.sign(diff).astype(float)

        return pd.DataFrame(
            {
                "adx":        adx_df["adx"],
                "adx_pos_di": adx_df["pos_di"],
                "adx_neg_di": adx_df["neg_di"],
                "hurst":      hurst,
                "atr_ratio":  atr_ratio,
                "price_trend": price_trend,
            },
            index=candles.index,
        ).bfill().fillna(0.0)

    # ------------------------------------------------------------------
    def encode_onehot(self, regime_series: pd.Series) -> pd.DataFrame:
        """One-hot encode the regime series into 4 binary columns."""
        out = pd.DataFrame(0, index=regime_series.index,
                           columns=[f"regime_{r}" for r in _ALL_REGIMES])
        for r in _ALL_REGIMES:
            out[f"regime_{r}"] = (regime_series == r).astype(float)
        return out


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def _classify(
    f:          pd.DataFrame,
    adx_thresh: float,
    hurst_min:  float,
    vol_ratio:  float,
) -> pd.Series:
    """Vectorised regime classification from feature DataFrame."""
    high_vol  = f["atr_ratio"] > vol_ratio
    trending  = (f["adx"] > adx_thresh) & ~high_vol
    ranging   = (~high_vol) & (~trending)

    labels = pd.Series(RANGING, index=f.index, dtype=object)
    labels = labels.where(~high_vol,  HIGH_VOLATILITY)
    labels = labels.where(~trending,  # temporary; overwrite with direction
                          np.where(f["price_trend"] >= 0, TRENDING_BULL, TRENDING_BEAR))
    # Re-apply high-vol on top (highest priority)
    labels[high_vol] = HIGH_VOLATILITY
    return labels


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _calc_atr(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low  - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _calc_adx(
    high:  pd.Series,
    low:   pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Wilder's ADX with +DI/-DI.  Returns DataFrame with columns adx, pos_di, neg_di."""
    atr = _calc_atr(high, low, close, period)

    up_move   = high.diff()
    down_move = -low.diff()

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_dm_s = pd.Series(pos_dm, index=high.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period).mean()
    neg_dm_s = pd.Series(neg_dm, index=high.index).ewm(
        alpha=1.0 / period, adjust=False, min_periods=period).mean()

    atr_safe = atr.replace(0, np.nan).ffill().fillna(1e-9)
    pos_di = 100 * pos_dm_s / atr_safe
    neg_di = 100 * neg_dm_s / atr_safe

    dx = (np.abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9)) * 100
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"adx": adx, "pos_di": pos_di, "neg_di": neg_di},
                        index=high.index)


def _rolling_hurst(
    price: pd.Series,
    window: int = 100,
    min_periods: int = 40,
) -> pd.Series:
    """Rolling Hurst Exponent via R/S analysis.

    H ≈ 0.5  → random walk (efficient)
    H > 0.5  → trending / persistent
    H < 0.5  → mean-reverting
    """
    log_returns = np.log(price / price.shift(1)).fillna(0.0).values
    n = len(log_returns)
    hurst_vals = np.full(n, 0.5)

    for i in range(window - 1, n):
        seg = log_returns[i - window + 1: i + 1]
        if len(seg) < min_periods:
            continue
        hurst_vals[i] = _hurst_rs(seg)

    # Back-fill warm-up period with first computed value
    first_valid = next((v for v in hurst_vals if v != 0.5), 0.5)
    for i in range(window - 1):
        hurst_vals[i] = first_valid

    return pd.Series(hurst_vals, index=price.index, name="hurst")


def _hurst_rs(series: np.ndarray) -> float:
    """Hurst exponent via R/S for a 1D array of log-returns."""
    n = len(series)
    if n < 20:
        return 0.5
    # Use 4 sub-window sizes (powers-of-2 style)
    sizes = [max(10, n // 8), max(15, n // 4), max(20, n // 2), n]
    rs_vals, used_sizes = [], []
    for size in sizes:
        if size < 10 or size > n:
            continue
        chunk = series[:size]
        mean_r  = np.mean(chunk)
        deviate = np.cumsum(chunk - mean_r)
        r = deviate.max() - deviate.min()
        s = np.std(chunk, ddof=1)
        if s > 0 and r > 0:
            rs_vals.append(np.log(r / s))
            used_sizes.append(np.log(size))
    if len(rs_vals) < 2:
        return 0.5
    # Hurst = slope of log(R/S) vs log(n)
    slope = np.polyfit(used_sizes, rs_vals, 1)[0]
    return float(np.clip(slope, 0.01, 0.99))


def _market_efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """Kaufman's Market Efficiency Ratio: net change / sum of absolute changes.

    MER ≈ 1.0 → highly trending  |  MER ≈ 0.0 → choppy/ranging
    """
    net   = close.diff(window).abs()
    noise = close.diff(1).abs().rolling(window, min_periods=1).sum()
    ratio = net / noise.replace(0, np.nan)
    return ratio.fillna(0.0).rename("mer")
