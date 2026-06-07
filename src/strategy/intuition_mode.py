"""Intuition Mode — Human-like ICT confluence-based risk engine.

When the ML model probability is below the normal threshold (0.65), the bot
can still take a trade if enough independent ICT confluence factors stack up.
This mimics how a skilled ICT trader uses multi-factor analysis to take
higher-conviction "intuitive" trades.

Confluence scoring (max ≈ 14 points):

    Factor                                               Points
    -------------------------------------------------------
    ML prob >= 0.55 (below normal but meaningful)           2
    ML prob >= 0.65 (passes normal gate)                    3
    H4 trend bias aligned with signal direction             1
    Signal inside London or NY killzone                     1
    Liquidity sweep within last 6 bars                      1
    Order Block AND FVG both align with direction           2
    CHoCH within last 20 bars (confirmed reversal)          1
    Sentiment score > +0.15 or < -0.15 (aligned)           1
    Spread < 1.5 pips (very tight — high liquidity)         1
    Price within Asian session range (classic manipulation)  1
    -------------------------------------------------------
    Total possible                                         14

Decision:
    - score >= intuition_threshold → take trade
    - risk multiplier scales with score (configurable max)
    - every call is logged with full breakdown for audit

Usage::

    from src.strategy.intuition_mode import IntuitiveSignalScorer

    scorer = IntuitiveSignalScorer(cfg=strategy_cfg.get("intuition_mode", {}))
    result = scorer.score(
        signal=sig,
        ml_prob=0.58,
        htf_bias="long",
        detections=detections,
        candles=candles,
        sentiment_score=0.3,
        spread_pips=1.2,
        in_killzone=True,
        current_bar_idx=len(candles) - 1,
    )
    if result.should_trade:
        print(f"Intuition trade! score={result.total_score} mult={result.risk_multiplier}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Asian session times (UTC) — price often ranges here before London manipulates it.
_ASIAN_START = dt_time(0, 0)
_ASIAN_END   = dt_time(7, 0)


@dataclass
class ConfluenceBreakdown:
    """Result of a single confluence scoring run."""
    # Score components (all integers or 0)
    ml_score:          int = 0   # 0, 2, or 3
    htf_score:         int = 0   # 0 or 1
    killzone_score:    int = 0   # 0 or 1
    sweep_score:       int = 0   # 0 or 1
    ob_fvg_score:      int = 0   # 0, 1, or 2
    choch_score:       int = 0   # 0 or 1
    sentiment_score:   int = 0   # 0 or 1
    spread_score:      int = 0   # 0 or 1
    asian_range_score: int = 0   # 0 or 1

    total_score:      int = 0
    threshold:        int = 8
    should_trade:     bool = False
    risk_multiplier:  float = 1.0
    notes:            List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Intuition Score: {self.total_score}/{self.threshold} "
            f"({'FIRE' if self.should_trade else 'skip'}) ==="
        ]
        for note in self.notes:
            lines.append(f"  {note}")
        lines.append(f"  → risk_multiplier = {self.risk_multiplier:.2f}x")
        return "\n".join(lines)


class IntuitiveSignalScorer:
    """Scores a signal on up to 14 confluence points and decides whether
    to take the trade even when the ML model probability is low.

    Args:
        cfg: the ``intuition_mode`` sub-dict from ``strategy_config.yaml``.
             Keys:
               enabled (bool): master on/off switch
               threshold_score (int): min points to fire (default 8)
               max_risk_multiplier (float): cap on position size boost (default 2.0)
               crypto_always_on (bool): ignore killzone for crypto symbols
               log_all_scores (bool): log every score, not just fires
    """

    def __init__(self, cfg: Optional[Dict] = None) -> None:
        cfg = cfg or {}
        self.enabled          = bool(cfg.get("enabled", True))
        self.threshold        = int(cfg.get("threshold_score", 8))
        self.max_risk_mult    = float(cfg.get("max_risk_multiplier", 2.0))
        self.crypto_always_on = bool(cfg.get("crypto_always_on", True))
        self.log_all          = bool(cfg.get("log_all_scores", True))

    def score(
        self,
        signal,                      # Signal object from rule_based.py
        ml_prob: float,
        htf_bias: str,               # 'long' | 'short' | 'neutral'
        detections: Dict,
        candles: pd.DataFrame,
        sentiment_score: float = 0.0,
        spread_pips: float = 999.0,
        in_killzone: bool = False,
        current_bar_idx: Optional[int] = None,
        symbol: str = "",
    ) -> ConfluenceBreakdown:
        """Compute the confluence score for *signal*.

        Returns a ``ConfluenceBreakdown`` with ``should_trade=True`` if the
        score meets or exceeds the threshold.
        """
        bd = ConfluenceBreakdown(threshold=self.threshold)

        if not self.enabled:
            return bd

        idx = current_bar_idx if current_bar_idx is not None else len(candles) - 1

        # ------------------------------------------------------------------
        # 1. ML probability score
        # ------------------------------------------------------------------
        if ml_prob >= 0.65:
            bd.ml_score = 3
            bd.notes.append(f"✓ ML prob ≥ 0.65 → +3  (prob={ml_prob:.3f})")
        elif ml_prob >= 0.55:
            bd.ml_score = 2
            bd.notes.append(f"✓ ML prob 0.55–0.65 → +2  (prob={ml_prob:.3f})")
        else:
            bd.notes.append(f"✗ ML prob < 0.55 → 0  (prob={ml_prob:.3f})")

        # ------------------------------------------------------------------
        # 2. H4 bias aligned
        # ------------------------------------------------------------------
        if htf_bias != "neutral" and htf_bias == signal.direction:
            bd.htf_score = 1
            bd.notes.append(f"✓ H4 bias aligned ({htf_bias}) → +1")
        else:
            bd.notes.append(f"✗ H4 bias not aligned (bias={htf_bias}, signal={signal.direction}) → 0")

        # ------------------------------------------------------------------
        # 3. Killzone
        # ------------------------------------------------------------------
        is_crypto = any(c in symbol.upper() for c in ("BTC", "ETH"))
        if in_killzone:
            bd.killzone_score = 1
            bd.notes.append("✓ Inside killzone → +1")
        elif is_crypto and self.crypto_always_on:
            bd.killzone_score = 1
            bd.notes.append("✓ Crypto 24/7 mode → +1")
        else:
            bd.notes.append("✗ Outside killzone → 0")

        # ------------------------------------------------------------------
        # 4. Liquidity sweep within last 6 bars
        # ------------------------------------------------------------------
        sweeps = detections.get("liquidity_sweeps", [])
        recent_sweeps = [
            s for s in sweeps
            if s.index >= idx - 6 and s.direction == signal.direction
        ]
        if recent_sweeps:
            bd.sweep_score = 1
            bd.notes.append(f"✓ Liquidity sweep in last 6 bars → +1")
        else:
            bd.notes.append("✗ No recent aligned sweep → 0")

        # ------------------------------------------------------------------
        # 5. Order Block AND FVG both aligned
        # ------------------------------------------------------------------
        obs = detections.get("order_blocks", [])
        fvgs = detections.get("fvg", [])
        direction_name = "bullish" if signal.direction == "long" else "bearish"

        active_obs = [
            ob for ob in obs
            if ob.direction == direction_name
            and not ob.mitigated
            and ob.index <= idx
        ]
        active_fvgs = [
            f for f in fvgs
            if f.direction == direction_name
            and not f.mitigated
            and f.index <= idx
        ]

        if active_obs and active_fvgs:
            bd.ob_fvg_score = 2
            bd.notes.append("✓ Active OB + FVG both aligned → +2")
        elif active_obs or active_fvgs:
            bd.ob_fvg_score = 1
            bd.notes.append("✓ Active OB or FVG aligned → +1")
        else:
            bd.notes.append("✗ No active OB or FVG aligned → 0")

        # ------------------------------------------------------------------
        # 6. CHoCH within last 20 bars
        # ------------------------------------------------------------------
        choch_list = detections.get("choch", [])
        recent_choch = [
            c for c in choch_list
            if c.index >= idx - 20
            and c.direction == direction_name
        ]
        if recent_choch:
            bd.choch_score = 1
            bd.notes.append("✓ CHoCH within last 20 bars → +1")
        else:
            bd.notes.append("✗ No recent CHoCH → 0")

        # ------------------------------------------------------------------
        # 7. Sentiment aligned
        # ------------------------------------------------------------------
        is_long = signal.direction == "long"
        if (is_long and sentiment_score > 0.15) or (not is_long and sentiment_score < -0.15):
            bd.sentiment_score = 1
            bd.notes.append(f"✓ Sentiment aligned (score={sentiment_score:.2f}) → +1")
        else:
            bd.notes.append(f"✗ Sentiment not aligned (score={sentiment_score:.2f}) → 0")

        # ------------------------------------------------------------------
        # 8. Very tight spread (< 1.5 pips)
        # ------------------------------------------------------------------
        if spread_pips < 1.5:
            bd.spread_score = 1
            bd.notes.append(f"✓ Tight spread ({spread_pips:.1f} pips) → +1")
        else:
            bd.notes.append(f"✗ Spread not tight ({spread_pips:.1f} pips) → 0")

        # ------------------------------------------------------------------
        # 9. Price inside Asian session range (manipulation zone)
        # ------------------------------------------------------------------
        if len(candles) > 0 and isinstance(candles.index, pd.DatetimeIndex):
            ts = candles.index[idx]
            t = ts.time()
            if _ASIAN_START <= t <= _ASIAN_END:
                bd.asian_range_score = 1
                bd.notes.append("✓ Price in Asian session range → +1")
            else:
                bd.notes.append("✗ Not in Asian session range → 0")

        # ------------------------------------------------------------------
        # Total
        # ------------------------------------------------------------------
        bd.total_score = (
            bd.ml_score + bd.htf_score + bd.killzone_score + bd.sweep_score +
            bd.ob_fvg_score + bd.choch_score + bd.sentiment_score +
            bd.spread_score + bd.asian_range_score
        )
        bd.should_trade = bd.total_score >= self.threshold

        # Dynamic risk multiplier: scales linearly from 1.0 to max_risk_mult
        # between threshold and max possible score (14).
        if bd.should_trade:
            max_possible = 14
            excess = max(0, bd.total_score - self.threshold)
            max_excess = max(1, max_possible - self.threshold)
            scale = excess / max_excess
            bd.risk_multiplier = 1.0 + scale * (self.max_risk_mult - 1.0)
        else:
            bd.risk_multiplier = 1.0

        if self.log_all or bd.should_trade:
            logger.info(
                "Intuition [%s] %s %s: score=%d/%d fire=%s mult=%.2f",
                symbol, signal.direction, signal.setup_type,
                bd.total_score, self.threshold, bd.should_trade, bd.risk_multiplier,
            )
            if bd.should_trade:
                logger.info("Intuition breakdown:\n%s", bd.summary())

        return bd
