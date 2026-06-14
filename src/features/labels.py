"""Label generation for ML training.

A 'setup' is a candidate trade with an entry price, SL, TP, and direction.
We walk forward up to ``max_holding_bars`` and decide which barrier was hit first.

Upgrade v2:
  - Default RR reduced from 2.0 → 1.5 (raises base win rate ~35% → ~42%)
  - Default max_holding_bars reduced from 24 → 12 bars (3 hours on M15)
  - Both changes reduce the noise-to-signal ratio in the classification problem
  - A ``confidence_score`` is added based on how quickly the barrier was hit
    (fast win → higher confidence; slow time-stop → lower confidence)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Setup:
    """One candidate trade."""
    index:       int       # bar index of entry (entry assumed at close[index])
    direction:   str       # 'long' or 'short'
    entry:       float
    stop_loss:   float
    take_profit: float


def create_labels(
    setups: List[Setup],
    candles: pd.DataFrame,
    max_holding_bars: int = 12,          # was 24 — 12 bars = 3 hours on M15
    rr_ratio: float = 1.5,              # was 2.0 — tighter TP raises base win-rate
    min_risk_atr_fraction: float = 0.1,  # filter degenerate setups with tiny risk
) -> pd.DataFrame:
    """Build label DataFrame from a list of setups using triple-barrier logic.

    Returns a DataFrame with columns:
        - ``binary``:           1 if TP hit before SL, else 0
        - ``multi_class``:      'win' / 'loss' / 'breakeven'
        - ``r_multiple``:       realized R (continuous)
        - ``bars_held``:        how many bars until exit
        - ``confidence``:       0.0-1.0; fast wins/losses score higher
        - ``index``, ``direction``, ``entry``, ``stop_loss``, ``take_profit``
    """
    high  = candles["high"].values
    low   = candles["low"].values
    close = candles["close"].values
    n     = len(candles)

    # ATR for degenerate-risk filter
    from ..detection.liquidity import _atr as _calc_atr
    atr_vals = _calc_atr(candles, 14).ffill().fillna(1e-6).values

    rows = []
    for s in setups:
        if s.index >= n - 1:
            continue
        risk = abs(s.entry - s.stop_loss)
        if risk <= 0:
            continue
        # Filter setups with risk < 10% of ATR (degenerate entries)
        atr_at_entry = atr_vals[s.index] if s.index < len(atr_vals) else 1e-6
        if risk < min_risk_atr_fraction * atr_at_entry:
            continue

        end     = min(s.index + max_holding_bars, n - 1)
        outcome = "breakeven"
        binary  = 0
        r       = 0.0
        bars    = end - s.index

        for j in range(s.index + 1, end + 1):
            if s.direction == "long":
                if low[j] <= s.stop_loss:
                    outcome, binary = "loss", 0
                    r    = -1.0
                    bars = j - s.index
                    break
                if high[j] >= s.take_profit:
                    outcome, binary = "win", 1
                    r    = (s.take_profit - s.entry) / risk
                    bars = j - s.index
                    break
            else:  # short
                if high[j] >= s.stop_loss:
                    outcome, binary = "loss", 0
                    r    = -1.0
                    bars = j - s.index
                    break
                if low[j] <= s.take_profit:
                    outcome, binary = "win", 1
                    r    = (s.entry - s.take_profit) / risk
                    bars = j - s.index
                    break
        else:
            # Time-stop: realized R from final close
            if s.direction == "long":
                r = (close[end] - s.entry) / risk
            else:
                r = (s.entry - close[end]) / risk
            if r > 0.1:
                outcome, binary = "win",  1
            elif r < -0.1:
                outcome, binary = "loss", 0
            else:
                outcome, binary = "breakeven", 0

        # Confidence: higher when barrier is hit faster
        # 1.0 = hit on bar 1, 0.0 = time-stop at max_holding_bars
        max_bars = max(end - s.index, 1)
        if outcome != "breakeven":
            confidence = float(1.0 - (bars - 1) / max_bars)
        else:
            confidence = 0.0   # time-stops have zero confidence

        rows.append(
            {
                "index":       s.index,
                "direction":   s.direction,
                "entry":       s.entry,
                "stop_loss":   s.stop_loss,
                "take_profit": s.take_profit,
                "binary":      binary,
                "multi_class": outcome,
                "r_multiple":  float(r),
                "bars_held":   int(bars),
                "confidence":  round(confidence, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold optimization utility
# ---------------------------------------------------------------------------

def optimize_entry_threshold(
    probabilities: np.ndarray,
    r_multiples:   np.ndarray,
    rr_ratio:      float = 1.5,
    min_trades:    int   = 30,
    search_range:  Tuple[float, float] = (0.50, 0.90),
    n_steps:       int   = 80,
) -> dict:
    """Find the probability threshold that maximises Profit Factor on OOF data.

    This replaces the hard-coded min_model_probability=0.65 with a data-driven
    threshold calibrated to the actual model outputs.

    Args:
        probabilities:  1-D array of model output probabilities (0-1).
        r_multiples:    1-D array of realised R values aligned to probabilities.
                        Positive R = win, negative R = loss.
        rr_ratio:       take-profit R for win sizing (default 1.5).
        min_trades:     minimum trades above threshold before PF is computed.
        search_range:   (low, high) bounds for threshold search.
        n_steps:        grid resolution.

    Returns:
        dict with keys: ``optimal_threshold``, ``optimal_pf``, ``win_rate``,
                        ``n_trades``, ``expected_value_per_trade``.
    """
    probabilities = np.asarray(probabilities)
    r_multiples   = np.asarray(r_multiples)

    thresholds  = np.linspace(search_range[0], search_range[1], n_steps)
    best_pf     = 0.0
    best_t      = float(search_range[0])
    best_info   = {}

    for t in thresholds:
        mask = probabilities >= t
        n    = mask.sum()
        if n < min_trades:
            continue
        r_sub = r_multiples[mask]
        wins  = r_sub[r_sub > 0]
        loss  = r_sub[r_sub < 0]
        if len(loss) == 0:
            continue
        gross_profit = float(wins.sum())
        gross_loss   = float(-loss.sum())
        pf           = gross_profit / max(gross_loss, 1e-9)
        if pf > best_pf:
            best_pf   = pf
            best_t    = float(t)
            best_info = {
                "optimal_threshold":       best_t,
                "optimal_pf":              round(best_pf, 4),
                "win_rate":                round(float(len(wins) / n), 4),
                "n_trades":                int(n),
                "expected_value_per_trade": round(
                    float(r_sub.mean()) * rr_ratio, 4
                ),
            }

    if not best_info:
        best_info = {
            "optimal_threshold":       0.65,
            "optimal_pf":              0.0,
            "win_rate":                0.0,
            "n_trades":                0,
            "expected_value_per_trade": 0.0,
        }
    return best_info
