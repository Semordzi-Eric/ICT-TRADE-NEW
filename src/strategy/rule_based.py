"""Rule-based ICT entry logic — the baseline strategy.

Two setup families:

1. **Liquidity Sweep + FVG Bounce.** Price sweeps a recent swing low (bullish)
   or high (bearish), forms an FVG in the opposite direction within a few bars,
   and we enter on FVG retest.

2. **Order Block Mitigation.** Price retraces into a fresh, unmitigated order
   block in the direction of the prevailing structure.

Both families share a common scoring mechanism so the *best* signal on a bar
always wins when multiple setups coincide.  A killzone filter (optional) limits
entries to London open and NY open windows where ICT setups have historically
the highest win rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time as dt_time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..detection.fvg import FVGZone
from ..detection.liquidity import LiquiditySweep, _atr
from ..detection.orderblock import OrderBlock
from ..detection.structure import StructureEvent
from ..features.labels import Setup


# ---------------------------------------------------------------------------
# Killzone definitions (UTC). Only signals whose bar timestamp falls within
# one of these windows are kept when killzone_only=True.
# ---------------------------------------------------------------------------
_DEFAULT_KILLZONE_WINDOWS: List[Tuple[dt_time, dt_time]] = [
    (dt_time(8, 0),  dt_time(10, 0)),   # London open
    (dt_time(13, 0), dt_time(15, 30)),  # NY open
]


def _in_killzone(
    timestamp: pd.Timestamp,
    windows: Optional[List[Tuple[dt_time, dt_time]]] = None,
) -> bool:
    """Return True if *timestamp* falls inside any killzone window (UTC)."""
    if windows is None:
        windows = _DEFAULT_KILLZONE_WINDOWS
    t = timestamp.time()
    return any(start <= t <= end for start, end in windows)


def _parse_killzone_windows(
    cfg_windows: Optional[Dict[str, List[str]]],
) -> List[Tuple[dt_time, dt_time]]:
    """Convert strategy_config killzone_windows dict to (time, time) pairs."""
    if not cfg_windows:
        return _DEFAULT_KILLZONE_WINDOWS
    result: List[Tuple[dt_time, dt_time]] = []
    for _, (start_s, end_s) in cfg_windows.items():
        result.append((
            dt_time.fromisoformat(start_s),
            dt_time.fromisoformat(end_s),
        ))
    return result


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
    score: float = 0.0        # higher = higher quality; used for ranking


def _score_signal(sig: "Signal", sweep_pierce_atr: float = 0.0, ob_strength: float = 0.0) -> float:
    """Heuristic signal quality score.

    Combines:
    * ``risk_atr`` — how cleanly defined the risk is vs ATR (smaller is tighter)
    * ``ob_strength`` — displacement magnitude of the order block (if applicable)
    * ``sweep_pierce_atr`` — how decisively liquidity was taken (deeper sweep = more conviction)

    Returns a positive float; higher = better.
    """
    # Prefer tight risk (risk_atr close to 1.0 is ideal) — penalise very wide stops
    tightness = 1.0 / max(sig.risk_atr, 0.5)
    return float(tightness + ob_strength * 0.5 + sweep_pierce_atr * 0.3)


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
    sweep_to_fvg_max_bars: int = 15,
) -> List[Signal]:
    """Walk the bars and generate rule-based signals.

    Args:
        candles: OHLC DataFrame.
        detections: dict from the detection layer.
        risk_config: dict with keys ``stop_loss_atr_multiplier``,
            ``take_profit_r_multiple``, and optionally ``killzone_only``,
            ``killzone_windows``.
        proximity_atr: how close (in ATR units) price must be to a zone to trigger.
        sweep_to_fvg_max_bars: a sweep is only paired with an FVG that forms within
            this window after it.
    """
    sl_mult = float(risk_config.get("stop_loss_atr_multiplier", 1.5))
    rr = float(risk_config.get("take_profit_r_multiple", 2.0))
    killzone_only = bool(risk_config.get("killzone_only", False))
    kz_windows = _parse_killzone_windows(risk_config.get("killzone_windows"))

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

    # Build sweep pierce_atr lookup for scoring
    sweep_pierce_by_idx: Dict[int, float] = {sw.index: sw.pierce_atr for sw in sweeps_sorted}

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
            # Bullish FVG retest: low taps zone, close confirms above bottom.
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
                sig = Signal(
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
                sig.score = _score_signal(sig, sweep_pierce_atr=sw.pierce_atr)
                signals.append(sig)
                break
            # Bearish FVG retest: high taps zone, close confirms below top.
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
                sig = Signal(
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
                sig.score = _score_signal(sig, sweep_pierce_atr=sw.pierce_atr)
                signals.append(sig)
                break

    # Setup 2: Order block mitigation with up to 8 bars of consolidation before confirmation.
    for ob in obs:
        if ob.mitigation_index is None:
            continue
        # Look for a valid confirmation bar within 8 bars after mitigation
        for confirm_i in range(ob.mitigation_index + 1, min(n, ob.mitigation_index + 8)):
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
            sig = Signal(
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
            sig.score = _score_signal(sig, ob_strength=ob.strength)
            signals.append(sig)
            break

    # --- Killzone filter: drop signals outside high-probability windows ----
    if killzone_only and signals:
        before = len(signals)
        signals = [s for s in signals if _in_killzone(s.timestamp, kz_windows)]
        dropped = before - len(signals)
        if dropped:
            # Use module-level logger lazily to avoid circular import issues.
            import logging
            logging.getLogger(__name__).debug(
                "Killzone filter removed %d/%d signals", dropped, before
            )

    # --- Deduplication: keep the highest-scored signal per bar index -------
    # Multiple setups can fire on the same bar; prefer the highest quality one
    # so the executor always receives the best available signal.
    signals.sort(key=lambda s: s.index)
    deduped: List[Signal] = []
    best_by_bar: Dict[int, Signal] = {}
    for s in signals:
        existing = best_by_bar.get(s.index)
        if existing is None or s.score > existing.score:
            best_by_bar[s.index] = s
    deduped = [best_by_bar[idx] for idx in sorted(best_by_bar)]
    return deduped


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
