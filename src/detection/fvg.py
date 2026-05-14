"""Fair Value Gap (FVG) detection.

A bullish FVG forms on a 3-bar sequence (i-2, i-1, i) when ``low[i] > high[i-2]``,
leaving a price gap that is unfilled by candle ``i-1``. Mirror logic for bearish.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .liquidity import _atr  # reuse ATR helper


@dataclass
class FVGZone:
    """An unmitigated (or partially mitigated) fair value gap."""
    index: int                      # bar index where FVG was created (the i bar)
    timestamp: pd.Timestamp
    direction: str                  # 'bullish' or 'bearish'
    top: float
    bottom: float
    mitigation_level: float         # 50% of the gap (CE — Consequent Encroachment)
    gap_size: float
    gap_atr: float
    mitigated: bool = False
    mitigation_index: Optional[int] = None


def detect_fvg(
    candles: pd.DataFrame,
    min_gap_atr: float = 1.5,
    atr_period: int = 14,
    track_mitigation: bool = True,
) -> List[FVGZone]:
    """Detect 3-bar fair value gaps and (optionally) flag mitigation.

    Args:
        candles: OHLC DataFrame indexed by timestamp.
        min_gap_atr: minimum gap size in ATR units.
        atr_period: ATR smoothing period.
        track_mitigation: if True, walk forward and mark zones as mitigated
            once price closes through the gap.

    Returns:
        List of FVGZone objects in chronological order.
    """
    if len(candles) < atr_period + 3:
        return []

    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    atr = _atr(candles, atr_period).values
    n = len(candles)

    zones: List[FVGZone] = []
    for i in range(2, n):
        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr <= 0:
            continue
        # Bullish FVG: low[i] > high[i-2]
        if low[i] > high[i - 2]:
            gap = low[i] - high[i - 2]
            if gap >= min_gap_atr * cur_atr:
                zones.append(
                    FVGZone(
                        index=i,
                        timestamp=candles.index[i],
                        direction="bullish",
                        top=float(low[i]),
                        bottom=float(high[i - 2]),
                        mitigation_level=float((low[i] + high[i - 2]) / 2.0),
                        gap_size=float(gap),
                        gap_atr=float(gap / cur_atr),
                    )
                )
        # Bearish FVG: high[i] < low[i-2]
        elif high[i] < low[i - 2]:
            gap = low[i - 2] - high[i]
            if gap >= min_gap_atr * cur_atr:
                zones.append(
                    FVGZone(
                        index=i,
                        timestamp=candles.index[i],
                        direction="bearish",
                        top=float(low[i - 2]),
                        bottom=float(high[i]),
                        mitigation_level=float((low[i - 2] + high[i]) / 2.0),
                        gap_size=float(gap),
                        gap_atr=float(gap / cur_atr),
                    )
                )

    if track_mitigation:
        for z in zones:
            for j in range(z.index + 1, n):
                if z.direction == "bullish" and low[j] <= z.bottom:
                    z.mitigated = True
                    z.mitigation_index = j
                    break
                if z.direction == "bearish" and high[j] >= z.top:
                    z.mitigated = True
                    z.mitigation_index = j
                    break
    return zones


def calculate_fvg_imbalance(candles: pd.DataFrame, fvg_zone: FVGZone) -> float:
    """Compute an imbalance score for an FVG.

    The score combines:
      * relative gap size (vs. local ATR)
      * the buying/selling pressure of the displacement candle (close-to-range)

    Returns a float in [0, ~3].
    """
    i = fvg_zone.index
    if i >= len(candles):
        return 0.0
    bar = candles.iloc[i]
    rng = float(bar["high"] - bar["low"])
    if rng <= 0:
        body_strength = 0.0
    elif fvg_zone.direction == "bullish":
        body_strength = float((bar["close"] - bar["low"]) / rng)
    else:
        body_strength = float((bar["high"] - bar["close"]) / rng)
    return float(fvg_zone.gap_atr * (0.5 + body_strength))
