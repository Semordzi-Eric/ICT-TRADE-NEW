"""Label generation for ML training.

A 'setup' is a candidate trade with an entry price, SL, TP, and direction.
We walk forward up to ``max_holding_bars`` and decide which barrier was hit first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Setup:
    """One candidate trade."""
    index: int                # bar index of entry (entry assumed at close[index])
    direction: str            # 'long' or 'short'
    entry: float
    stop_loss: float
    take_profit: float


def create_labels(
    setups: List[Setup],
    candles: pd.DataFrame,
    max_holding_bars: int = 24,
) -> pd.DataFrame:
    """Build label DataFrame from a list of setups using triple-barrier logic.

    Returns a DataFrame with columns:
        - ``binary``: 1 if TP hit before SL, else 0
        - ``multi_class``: 'win' / 'loss' / 'breakeven'
        - ``r_multiple``: realized R (continuous)
        - ``bars_held``: how many bars until exit
        - ``index``, ``direction``, ``entry``, ``stop_loss``, ``take_profit``
    """
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    n = len(candles)

    rows = []
    for s in setups:
        if s.index >= n - 1:
            continue
        risk = abs(s.entry - s.stop_loss)
        if risk <= 0:
            continue
        end = min(s.index + max_holding_bars, n - 1)
        outcome = "breakeven"
        binary = 0
        r = 0.0
        bars = end - s.index
        for j in range(s.index + 1, end + 1):
            if s.direction == "long":
                if low[j] <= s.stop_loss:
                    outcome, binary = "loss", 0
                    r = -1.0
                    bars = j - s.index
                    break
                if high[j] >= s.take_profit:
                    outcome, binary = "win", 1
                    r = (s.take_profit - s.entry) / risk
                    bars = j - s.index
                    break
            else:  # short
                if high[j] >= s.stop_loss:
                    outcome, binary = "loss", 0
                    r = -1.0
                    bars = j - s.index
                    break
                if low[j] <= s.take_profit:
                    outcome, binary = "win", 1
                    r = (s.entry - s.take_profit) / risk
                    bars = j - s.index
                    break
        else:
            # Time-stop: realised R from final close
            if s.direction == "long":
                r = (close[end] - s.entry) / risk
            else:
                r = (s.entry - close[end]) / risk
            if r > 0.1:
                outcome, binary = "win", 1
            elif r < -0.1:
                outcome, binary = "loss", 0
            else:
                outcome, binary = "breakeven", 0

        rows.append(
            {
                "index": s.index,
                "direction": s.direction,
                "entry": s.entry,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "binary": binary,
                "multi_class": outcome,
                "r_multiple": float(r),
                "bars_held": int(bars),
            }
        )
    return pd.DataFrame(rows)
