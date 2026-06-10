"""Tests for SentimentEngine."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.sentiment_engine import SentimentEngine, SentimentResult, _COUNTRY_CURRENCY_MAP


class TestSentimentResult:
    def test_score_clamped(self):
        r = SentimentResult("EURUSD", score=5.0, sources=[], events_today=0, is_blocked=False)
        assert r.score == 1.0
        r2 = SentimentResult("EURUSD", score=-5.0, sources=[], events_today=0, is_blocked=False)
        assert r2.score == -1.0

    def test_label_bullish(self):
        r = SentimentResult("EURUSD", score=0.5, sources=[], events_today=0, is_blocked=False)
        assert r.label == "bullish"

    def test_label_bearish(self):
        r = SentimentResult("EURUSD", score=-0.5, sources=[], events_today=0, is_blocked=False)
        assert r.label == "bearish"

    def test_label_neutral(self):
        r = SentimentResult("EURUSD", score=0.05, sources=[], events_today=0, is_blocked=False)
        assert r.label == "neutral"

    def test_to_dict_keys(self):
        r = SentimentResult("EURUSD", score=0.3, sources=["ff"], events_today=1, is_blocked=False)
        d = r.to_dict()
        assert "score" in d and "label" in d and "is_blocked" in d


class TestSentimentEngine:
    def _engine(self, tmp_path):
        return SentimentEngine(
            cache_dir=str(tmp_path),
            market_config_path="config/market_config.yaml",
        )

    def test_default_result_neutral(self, tmp_path):
        engine = self._engine(tmp_path)
        result = engine.get_sentiment("EURUSD")
        assert result.label == "neutral"
        assert result.score == 0.0

    def test_is_trade_blocked_no_events(self, tmp_path):
        engine = self._engine(tmp_path)
        blocked, msg = engine.is_trade_blocked(datetime.utcnow(), "EURUSD")
        assert blocked is False
        assert msg == "ok"

    def test_is_trade_blocked_with_high_impact(self, tmp_path):
        engine = self._engine(tmp_path)
        now = datetime.utcnow()
        # Inject a high-impact event 10 minutes from now for USD.
        event_dt = (now + timedelta(minutes=10)).replace(tzinfo=None)
        with engine._lock:
            engine._ff_events = [{
                "currency": "USD",
                "title": "FOMC Rate Decision",
                "impact": "High",
                "dt": event_dt.isoformat(),
            }]
        blocked, msg = engine.is_trade_blocked(now, "EURUSD")
        assert blocked is True
        assert "FOMC" in msg

    def test_is_trade_not_blocked_different_currency(self, tmp_path):
        engine = self._engine(tmp_path)
        now = datetime.utcnow()
        event_dt = (now + timedelta(minutes=5)).replace(tzinfo=None)
        with engine._lock:
            engine._ff_events = [{
                "currency": "JPY",        # not related to EURUSD
                "title": "BOJ Rate",
                "impact": "High",
                "dt": event_dt.isoformat(),
            }]
        blocked, _ = engine.is_trade_blocked(now, "EURUSD")
        assert blocked is False

    def test_pre_event_warning(self, tmp_path):
        engine = self._engine(tmp_path)
        now = datetime.utcnow()
        event_dt = (now + timedelta(minutes=45)).replace(tzinfo=None)
        with engine._lock:
            engine._ff_events = [{
                "currency": "GBP",
                "title": "BOE",
                "impact": "High",
                "dt": event_dt.isoformat(),
            }]
        assert engine.pre_event_warning(now, "GBPUSD") is True
        assert engine.pre_event_warning(now, "USDJPY") is False  # JPY not GBP

    def test_keyword_score_bullish(self, tmp_path):
        engine = self._engine(tmp_path)
        headlines = ["Gold surges on safe haven demand", "Gold rally continues"]
        score = engine._keyword_score(headlines, ["gold"])
        assert score > 0

    def test_keyword_score_bearish(self, tmp_path):
        engine = self._engine(tmp_path)
        headlines = ["Market crash fears grow", "Dollar falls on weak data"]
        score = engine._keyword_score(headlines, ["dollar"])
        assert score < 0

    def test_keyword_score_no_relevant_headlines(self, tmp_path):
        engine = self._engine(tmp_path)
        score = engine._keyword_score(["weather is nice", "sports results"], ["gold"])
        assert score == 0.0

    def test_country_currency_map_coverage(self):
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "NAS100"]:
            from src.utils.sentiment_engine import _SYMBOL_CURRENCIES
            assert sym in _SYMBOL_CURRENCIES
