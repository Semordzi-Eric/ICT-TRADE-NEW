"""Tests for IntuitiveSignalScorer (Intuition Mode)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategy.intuition_mode import IntuitiveSignalScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 100, in_asian: bool = False) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame."""
    if in_asian:
        # 03:00 UTC — inside Asian session
        base = pd.Timestamp("2024-01-02 03:00", tz=None)
    else:
        # 09:00 UTC — London open
        base = pd.Timestamp("2024-01-02 09:00", tz=None)
    idx = pd.date_range(base, periods=n, freq="15min")
    close = 1.10 + np.random.randn(n) * 0.001
    return pd.DataFrame({
        "open":   close * 0.9999,
        "high":   close * 1.0005,
        "low":    close * 0.9995,
        "close":  close,
        "volume": np.ones(n) * 1000,
    }, index=idx)


@dataclass
class _DummySignal:
    direction: str = "long"
    setup_type: str = "sweep_fvg"
    timestamp: pd.Timestamp = None
    score: float = 1.0
    index: int = 99


@dataclass
class _DummySweep:
    index: int
    direction: str


@dataclass
class _DummyOB:
    index: int
    direction: str
    mitigated: bool = False


@dataclass
class _DummyFVG:
    index: int
    direction: str
    mitigated: bool = False


@dataclass
class _DummyCHoCH:
    index: int
    direction: str


def _make_detections(
    sweeps=True, ob=True, fvg=True, choch=True, direction="bullish", idx=99
) -> dict:
    return {
        "liquidity_sweeps": [_DummySweep(index=idx - 3, direction=direction)] if sweeps else [],
        "order_blocks":     [_DummyOB(index=idx - 10, direction=direction)] if ob else [],
        "fvg":              [_DummyFVG(index=idx - 5, direction=direction)] if fvg else [],
        "choch":            [_DummyCHoCH(index=idx - 15, direction=direction)] if choch else [],
        "bos":              [],
        "equal_levels":     [],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIntuitiveSignalScorer:
    def _scorer(self, threshold=8, enabled=True, max_mult=2.0):
        return IntuitiveSignalScorer(cfg={
            "enabled": enabled,
            "threshold_score": threshold,
            "max_risk_multiplier": max_mult,
            "crypto_always_on": True,
            "log_all_scores": False,
        })

    def test_disabled_returns_no_trade(self):
        scorer = self._scorer(enabled=False)
        sig = _DummySignal()
        candles = _make_candles()
        result = scorer.score(sig, ml_prob=0.70, htf_bias="long",
                              detections=_make_detections(),
                              candles=candles, sentiment_score=0.5,
                              spread_pips=1.0, in_killzone=True)
        assert result.should_trade is False

    def test_high_confluence_fires(self):
        scorer = self._scorer(threshold=8)
        sig = _DummySignal(direction="long")
        candles = _make_candles()
        # ML 0.60 → 2pts, htf aligned 1pt, killzone 1pt, sweep 1pt,
        # OB+FVG 2pts, choch 1pt, sentiment 1pt → total 9
        result = scorer.score(
            sig, ml_prob=0.60, htf_bias="long",
            detections=_make_detections(direction="bullish"),
            candles=candles, sentiment_score=0.5,
            spread_pips=2.0, in_killzone=True,
            current_bar_idx=99, symbol="EURUSD",
        )
        assert result.should_trade is True
        assert result.total_score >= 8

    def test_low_confluence_no_fire(self):
        scorer = self._scorer(threshold=8)
        sig = _DummySignal(direction="long")
        candles = _make_candles()
        # Only ML 0.45 (0pts) + killzone (1pt) → total ≤ 3
        result = scorer.score(
            sig, ml_prob=0.45, htf_bias="neutral",
            detections=_make_detections(sweeps=False, ob=False, fvg=False, choch=False),
            candles=candles, sentiment_score=0.0,
            spread_pips=5.0, in_killzone=True,
            current_bar_idx=99, symbol="EURUSD",
        )
        assert result.should_trade is False

    def test_risk_multiplier_scales_with_score(self):
        scorer = self._scorer(threshold=4, max_mult=2.0)
        sig = _DummySignal(direction="long")
        candles = _make_candles()
        # All factors on → high score → multiplier near max
        result = scorer.score(
            sig, ml_prob=0.70, htf_bias="long",
            detections=_make_detections(direction="bullish"),
            candles=candles, sentiment_score=0.5,
            spread_pips=1.0, in_killzone=True,
            current_bar_idx=99, symbol="EURUSD",
        )
        assert result.should_trade is True
        assert result.risk_multiplier > 1.0
        assert result.risk_multiplier <= 2.0

    def test_htf_counter_trend_reduces_score(self):
        scorer = self._scorer(threshold=8)
        sig = _DummySignal(direction="long")   # long signal
        candles = _make_candles()
        # htf_bias = "short" → htf_score = 0
        result_aligned    = scorer.score(sig, ml_prob=0.60, htf_bias="long",
                                         detections=_make_detections(direction="bullish"),
                                         candles=candles, sentiment_score=0.5,
                                         spread_pips=1.0, in_killzone=True,
                                         current_bar_idx=99, symbol="EURUSD")
        result_misaligned = scorer.score(sig, ml_prob=0.60, htf_bias="short",
                                         detections=_make_detections(direction="bullish"),
                                         candles=candles, sentiment_score=0.5,
                                         spread_pips=1.0, in_killzone=True,
                                         current_bar_idx=99, symbol="EURUSD")
        assert result_aligned.total_score > result_misaligned.total_score

    def test_crypto_always_on_adds_killzone_point(self):
        scorer = self._scorer(threshold=1)
        sig = _DummySignal(direction="long")
        candles = _make_candles()
        result = scorer.score(
            sig, ml_prob=0.0, htf_bias="neutral",
            detections=_make_detections(sweeps=False, ob=False, fvg=False, choch=False),
            candles=candles, sentiment_score=0.0,
            spread_pips=999.0, in_killzone=False,
            current_bar_idx=99, symbol="BTCUSD",
        )
        assert result.killzone_score == 1

    def test_asian_session_adds_point(self):
        scorer = self._scorer(threshold=1)
        sig = _DummySignal(direction="long")
        candles = _make_candles(in_asian=True)
        result = scorer.score(
            sig, ml_prob=0.0, htf_bias="neutral",
            detections=_make_detections(sweeps=False, ob=False, fvg=False, choch=False),
            candles=candles, sentiment_score=0.0,
            spread_pips=999.0, in_killzone=False,
            current_bar_idx=99, symbol="EURUSD",
        )
        assert result.asian_range_score == 1
