"""Market-structure detection: Break of Structure (BOS) and Change of Character (CHoCH)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .liquidity import _swing_highs_lows


@dataclass
class StructureEvent:
    index: int
    timestamp: pd.Timestamp
    kind: str            # 'BOS' or 'CHOCH'
    direction: str       # 'bullish' or 'bearish'
    swing_index: int
    swing_price: float
    break_price: float


def detect_bos(
    candles: pd.DataFrame,
    period: int = 20,
    confirmation_bars: int = 2,
    swing_lookback: int = 5,
) -> List[StructureEvent]:
    """Detect Break-of-Structure events.

    A bullish BOS occurs when close prints above the most recent confirmed swing
    high and stays above for ``confirmation_bars`` bars. Mirror logic for bearish.
    """
    if len(candles) < period + swing_lookback + confirmation_bars:
        return []

    is_sh, is_sl = _swing_highs_lows(candles, swing_lookback)
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    n = len(candles)

    events: List[StructureEvent] = []
    last_emitted_dir: Optional[str] = None
    last_emitted_idx = -1

    for i in range(period, n - confirmation_bars):
        recent_sh_idx = np.where(is_sh.values[max(0, i - period) : i])[0]
        recent_sl_idx = np.where(is_sl.values[max(0, i - period) : i])[0]
        offset = max(0, i - period)

        if recent_sh_idx.size:
            sh_global = recent_sh_idx + offset
            sh_idx = sh_global[np.argmax(high[sh_global])]
            sh_price = high[sh_idx]
            if close[i] > sh_price and all(
                close[i + k] > sh_price for k in range(1, confirmation_bars + 1)
            ):
                if not (last_emitted_dir == "bullish" and i - last_emitted_idx < period):
                    events.append(
                        StructureEvent(
                            index=i,
                            timestamp=candles.index[i],
                            kind="BOS",
                            direction="bullish",
                            swing_index=int(sh_idx),
                            swing_price=float(sh_price),
                            break_price=float(close[i]),
                        )
                    )
                    last_emitted_dir = "bullish"
                    last_emitted_idx = i

        if recent_sl_idx.size:
            sl_global = recent_sl_idx + offset
            sl_idx = sl_global[np.argmin(low[sl_global])]
            sl_price = low[sl_idx]
            if close[i] < sl_price and all(
                close[i + k] < sl_price for k in range(1, confirmation_bars + 1)
            ):
                if not (last_emitted_dir == "bearish" and i - last_emitted_idx < period):
                    events.append(
                        StructureEvent(
                            index=i,
                            timestamp=candles.index[i],
                            kind="BOS",
                            direction="bearish",
                            swing_index=int(sl_idx),
                            swing_price=float(sl_price),
                            break_price=float(close[i]),
                        )
                    )
                    last_emitted_dir = "bearish"
                    last_emitted_idx = i

    return events


def detect_choch(candles: pd.DataFrame, swing_lookback: int = 5) -> List[StructureEvent]:
    """Detect Change-of-Character events (trend reversals).

    Walks the timeline and tracks the most recent confirmed swing in each direction.
    A CHoCH is registered when, in an established up-trend (HH/HL), price closes
    below the most recent HL — and vice versa for a down-trend.
    """
    if len(candles) < swing_lookback * 4:
        return []

    is_sh, is_sl = _swing_highs_lows(candles, swing_lookback)
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    n = len(candles)

    events: List[StructureEvent] = []
    trend: Optional[str] = None       # 'up' or 'down'
    last_swing_high: Optional[Tuple[int, float]] = None
    last_swing_low: Optional[Tuple[int, float]] = None
    prev_swing_high: Optional[Tuple[int, float]] = None
    prev_swing_low: Optional[Tuple[int, float]] = None

    for i in range(n):
        if is_sh.values[i]:
            prev_swing_high = last_swing_high
            last_swing_high = (i, float(high[i]))
            if (
                prev_swing_high
                and last_swing_high[1] > prev_swing_high[1]
                and last_swing_low
            ):
                trend = "up"
        if is_sl.values[i]:
            prev_swing_low = last_swing_low
            last_swing_low = (i, float(low[i]))
            if (
                prev_swing_low
                and last_swing_low[1] < prev_swing_low[1]
                and last_swing_high
            ):
                trend = "down"

        # CHoCH check on close
        if trend == "up" and last_swing_low and close[i] < last_swing_low[1]:
            events.append(
                StructureEvent(
                    index=i,
                    timestamp=candles.index[i],
                    kind="CHOCH",
                    direction="bearish",
                    swing_index=last_swing_low[0],
                    swing_price=last_swing_low[1],
                    break_price=float(close[i]),
                )
            )
            trend = "down"
        elif trend == "down" and last_swing_high and close[i] > last_swing_high[1]:
            events.append(
                StructureEvent(
                    index=i,
                    timestamp=candles.index[i],
                    kind="CHOCH",
                    direction="bullish",
                    swing_index=last_swing_high[0],
                    swing_price=last_swing_high[1],
                    break_price=float(close[i]),
                )
            )
            trend = "up"

    return events


def calculate_structure_strength(
    candles: pd.DataFrame,
    bos_list: List[StructureEvent],
    choch_list: List[StructureEvent],
    window: int = 100,
) -> pd.Series:
    """Per-bar 'structure score': +ve when bullish events dominate recent history."""
    n = len(candles)
    score = np.zeros(n)
    events = sorted(bos_list + choch_list, key=lambda e: e.index)
    for ev in events:
        sign = 1.0 if ev.direction == "bullish" else -1.0
        weight = 1.5 if ev.kind == "CHOCH" else 1.0
        end = min(ev.index + window, n)
        decay = np.linspace(1.0, 0.0, end - ev.index)
        score[ev.index : end] += sign * weight * decay
    return pd.Series(score, index=candles.index, name="structure_strength")
