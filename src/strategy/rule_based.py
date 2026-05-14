"""Rule-based ICT entry logic — the baseline strategy.

Two setup families:

1. **Liquidity Sweep + FVG Bounce.** Price sweeps a recent swing low (bullish)
   or high (bearish), forms an FVG in the opposite direction within a few bars,
   and we enter on FVG retest.

2. **Order Block Mitigation.** Price retraces into a fresh, unmitigated order
   block in the direction of the prevailing structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..detection.fvg import FVGZone
from ..detection.liquidity import LiquiditySweep, _atr
from ..detection.orderblock import OrderBlock
from ..detection.structure import StructureEvent
from ..features.labels import Setup


@dataclass
class Signal:
    """A trade signal emitted by the rule-based strategy."""
    index: int
    timestamp: pd.Timestamp
    direction: str            # 'long' or 'short'
    entry: float
    stop_loss: float
    take_profit: float
    setup_type: str           # 'sweep_fvg' or 'ob_mitigation'
    rationale: str
    risk_atr: float


def _structure_bias(
    bos: List[StructureEvent], choch: List[StructureEvent], i: int
) -> str:
    """Most-recent BOS/CHoCH gives directional bias. ``'neutral'`` if none."""
    recent = [e for e in bos + choch if e.index <= i]
    if not recent:
        return "neutral"
    last = max(recent, key=lambda e: e.index)
    return "long" if last.direction == "bullish" else "short"


def generate_signals(
    candles: pd.DataFrame,
    detections: Dict[str, object],
    risk_config: dict,
    proximity_atr: float = 0.5,
    sweep_to_fvg_max_bars: int = 6,
) -> List[Signal]:
    """Walk the bars and generate rule-based signals.

    Args:
        candles: OHLC DataFrame.
        detections: dict from the detection layer.
        risk_config: dict with keys ``stop_loss_atr_multiplier``, ``take_profit_r_multiple``.
        proximity_atr: how close (in ATR units) price must be to a zone to trigger.
        sweep_to_fvg_max_bars: a sweep is only paired with an FVG that forms within
            this window after it.
    """
    sl_mult = float(risk_config.get("stop_loss_atr_multiplier", 1.5))
    rr = float(risk_config.get("take_profit_r_multiple", 2.0))

    fvg: List[FVGZone] = detections.get("fvg", [])
    obs: List[OrderBlock] = detections.get("order_blocks", [])
    sweeps: List[LiquiditySweep] = detections.get("liquidity_sweeps", [])
    bos: List[StructureEvent] = detections.get("bos", [])
    choch: List[StructureEvent] = detections.get("choch", [])

    atr = _atr(candles, 14).ffill().fillna(0.0).values
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    n = len(candles)

    signals: List[Signal] = []
    sweeps_sorted = sorted(sweeps, key=lambda s: s.index)
    fvg_sorted = sorted(fvg, key=lambda f: f.index)

    # Setup 1: Sweep + FVG retest
    for sw in sweeps_sorted:
        candidate_fvgs = [
            f for f in fvg_sorted
            if 0 < f.index - sw.index <= sweep_to_fvg_max_bars
            and f.direction == ("bullish" if sw.direction == "bullish" else "bearish")
        ]
        if not candidate_fvgs:
            continue
        zone = candidate_fvgs[0]
        # Walk forward from zone creation looking for the retest entry.
        for i in range(zone.index + 1, min(n, zone.index + 30)):
            if zone.mitigation_index is not None and i >= zone.mitigation_index:
                break
            cur_atr = atr[i] if np.isfinite(atr[i]) else 0.0
            if cur_atr <= 0:
                continue
            # Bullish FVG retest: low taps zone, close confirms above bottom (body support).
            if (
                zone.direction == "bullish"
                and low[i] <= zone.top
                and close[i] >= zone.bottom
            ):
                entry = float(close[i])
                sl = float(min(zone.bottom - sl_mult * cur_atr, low[i] - sl_mult * cur_atr))
                risk = entry - sl
                if risk <= 0:
                    break
                tp = float(entry + rr * risk)
                signals.append(
                    Signal(
                        index=i,
                        timestamp=candles.index[i],
                        direction="long",
                        entry=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        setup_type="sweep_fvg",
                        rationale=f"bull sweep@{sw.index} + bull FVG@{zone.index}",
                        risk_atr=float(risk / cur_atr),
                    )
                )
                break
            # Bearish FVG retest: high taps zone, close confirms below top (body resistance).
            if (
                zone.direction == "bearish"
                and high[i] >= zone.bottom
                and close[i] <= zone.top
            ):
                entry = float(close[i])
                sl = float(max(zone.top + sl_mult * cur_atr, high[i] + sl_mult * cur_atr))
                risk = sl - entry
                if risk <= 0:
                    break
                tp = float(entry - rr * risk)
                signals.append(
                    Signal(
                        index=i,
                        timestamp=candles.index[i],
                        direction="short",
                        entry=entry,
                        stop_loss=sl,
                        take_profit=tp,
                        setup_type="sweep_fvg",
                        rationale=f"bear sweep@{sw.index} + bear FVG@{zone.index}",
                        risk_atr=float(risk / cur_atr),
                    )
                )
                break

    # Setup 2: Order block mitigation with 1-bar displacement confirmation.
    # We do NOT enter on the exact mitigation bar — we wait for the next bar
    # to close in the OB direction, confirming a reaction rather than a pierce-through.
    for ob in obs:
        if ob.mitigation_index is None:
            continue
        confirm_i = ob.mitigation_index + 1   # confirmation bar
        if confirm_i >= n:
            continue
        bias = _structure_bias(bos, choch, confirm_i)
        # Trade only with structure
        if ob.direction == "bullish" and bias != "long":
            continue
        if ob.direction == "bearish" and bias != "short":
            continue
        cur_atr = atr[confirm_i] if np.isfinite(atr[confirm_i]) else 0.0
        if cur_atr <= 0:
            continue
        if ob.direction == "bullish":
            # Confirmation: close above OB bottom (price rejected lower)
            if close[confirm_i] < ob.bottom:
                continue
            entry = float(close[confirm_i])
            sl = float(ob.bottom - sl_mult * cur_atr)
            risk = entry - sl
            if risk <= 0:
                continue
            tp = float(entry + rr * risk)
            direction = "long"
        else:
            # Confirmation: close below OB top (price rejected higher)
            if close[confirm_i] > ob.top:
                continue
            entry = float(close[confirm_i])
            sl = float(ob.top + sl_mult * cur_atr)
            risk = sl - entry
            if risk <= 0:
                continue
            tp = float(entry - rr * risk)
            direction = "short"
        signals.append(
            Signal(
                index=confirm_i,
                timestamp=candles.index[confirm_i],
                direction=direction,
                entry=entry,
                stop_loss=sl,
                take_profit=tp,
                setup_type="ob_mitigation",
                rationale=f"{ob.direction} OB@{ob.index} mitigated+confirmed@{confirm_i}",
                risk_atr=float(risk / cur_atr),
            )
        )

    signals.sort(key=lambda s: s.index)
    return signals


def signals_to_setups(signals: List[Signal]) -> List[Setup]:
    """Convert Signal objects to Setup objects for the labels module."""
    return [
        Setup(
            index=s.index,
            direction=s.direction,
            entry=s.entry,
            stop_loss=s.stop_loss,
            take_profit=s.take_profit,
        )
        for s in signals
    ]


def position_size(
    account_balance: float,
    risk_per_trade: float,
    entry: float,
    stop_loss: float,
    contract_value: float = 1.0,
) -> float:
    """Volume sized so a SL hit costs ``risk_per_trade * account_balance``.

    `contract_value` lets you adapt to instrument tick value (default 1 = forex-like).
    """
    risk_amount = account_balance * risk_per_trade
    risk_per_contract = abs(entry - stop_loss) * contract_value
    if risk_per_contract <= 0:
        return 0.0
    return float(risk_amount / risk_per_contract)
