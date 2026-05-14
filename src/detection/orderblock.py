"""Order Block detection.

A bullish order block is the last down-close candle before a strong upward
displacement move. A bearish order block is the last up-close candle before
a strong downward displacement move.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .liquidity import _atr


@dataclass
class OrderBlock:
    """Detected order block zone."""
    index: int                # index of the OB candle itself
    timestamp: pd.Timestamp
    direction: str            # 'bullish' or 'bearish'
    top: float
    bottom: float
    move_size: float          # absolute price displacement that confirmed it
    move_atr: float
    strength: float           # 0..~5
    mitigated: bool = False
    mitigation_index: Optional[int] = None


def detect_order_blocks(
    candles: pd.DataFrame,
    min_move_atr: float = 1.5,
    lookback: int = 100,
    atr_period: int = 14,
    displacement_bars: int = 3,
    track_mitigation: bool = True,
) -> List[OrderBlock]:
    """Identify order blocks created by displacement moves.

    Args:
        candles: OHLC DataFrame.
        min_move_atr: required displacement (close-to-close over `displacement_bars`)
            in ATR units.
        lookback: max bars retained when scanning history.
        atr_period: ATR smoothing period.
        displacement_bars: number of bars over which to measure displacement.
        track_mitigation: if True, mark an OB as mitigated when price
            re-enters its range.

    Returns:
        List of OrderBlock objects.
    """
    if len(candles) < atr_period + displacement_bars + 2:
        return []

    open_ = candles["open"].values
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    atr = _atr(candles, atr_period).values
    n = len(candles)

    blocks: List[OrderBlock] = []
    start = max(displacement_bars + 1, atr_period)
    for i in range(start, n - 1):
        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr <= 0:
            continue
        end = min(i + displacement_bars, n - 1)
        move = close[end] - close[i]
        # Bullish OB: down candle at i, then strong up displacement.
        if close[i] < open_[i] and move >= min_move_atr * cur_atr:
            ob_top = float(high[i])
            ob_bottom = float(low[i])
            blocks.append(
                OrderBlock(
                    index=i,
                    timestamp=candles.index[i],
                    direction="bullish",
                    top=ob_top,
                    bottom=ob_bottom,
                    move_size=float(abs(move)),
                    move_atr=float(abs(move) / cur_atr),
                    strength=calculate_ob_strength(
                        candles.iloc[i], candles.iloc[end], cur_atr
                    ),
                )
            )
        # Bearish OB: up candle at i, then strong down displacement.
        elif close[i] > open_[i] and -move >= min_move_atr * cur_atr:
            ob_top = float(high[i])
            ob_bottom = float(low[i])
            blocks.append(
                OrderBlock(
                    index=i,
                    timestamp=candles.index[i],
                    direction="bearish",
                    top=ob_top,
                    bottom=ob_bottom,
                    move_size=float(abs(move)),
                    move_atr=float(abs(move) / cur_atr),
                    strength=calculate_ob_strength(
                        candles.iloc[i], candles.iloc[end], cur_atr
                    ),
                )
            )

    # Trim to most recent `lookback` blocks
    if len(blocks) > lookback:
        blocks = blocks[-lookback:]

    if track_mitigation:
        for ob in blocks:
            for j in range(ob.index + 1, n):
                if ob.direction == "bullish" and low[j] <= ob.bottom:
                    ob.mitigated = True
                    ob.mitigation_index = j
                    break
                if ob.direction == "bearish" and high[j] >= ob.top:
                    ob.mitigated = True
                    ob.mitigation_index = j
                    break
    return blocks


def calculate_ob_strength(
    order_block_candle: pd.Series,
    move_candle: pd.Series,
    atr_value: float,
) -> float:
    """Heuristic strength score combining displacement size and OB tightness."""
    if atr_value <= 0:
        return 0.0
    move = abs(float(move_candle["close"]) - float(order_block_candle["close"]))
    rng = float(order_block_candle["high"] - order_block_candle["low"])
    tightness = 1.0 / (rng / atr_value + 1e-6)
    return float(move / atr_value * (0.5 + min(tightness, 2.0) / 4.0))
