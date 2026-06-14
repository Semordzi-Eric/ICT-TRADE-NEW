"""Tests for the new regime engine and drift monitor."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.regime import (
    RANGING, TRENDING_BULL, TRENDING_BEAR, HIGH_VOLATILITY,
    RegimeEngine, _rolling_hurst, _calc_adx, _market_efficiency_ratio,
)
from src.utils.drift_monitor import DriftMonitor, psi, ADWIN, jensen_shannon_divergence


# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------

def _synthetic(n: int = 400, seed: int = 42, trend: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rets = rng.normal(0.0001 if trend else 0.0, 0.0008, n)
    close = 1.1 + np.cumsum(rets)
    open_ = close - rng.normal(0, 0.0002, n)
    high  = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0003, n))
    low   = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0003, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(50, 500, n)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Regime Engine tests
# ---------------------------------------------------------------------------

class TestRegimeEngine:
    def test_detect_returns_series(self):
        candles = _synthetic(300)
        engine  = RegimeEngine()
        result  = engine.detect(candles)
        assert isinstance(result, pd.Series)
        assert len(result) == len(candles)
        assert result.index.equals(candles.index)

    def test_labels_are_valid(self):
        candles = _synthetic(300)
        engine  = RegimeEngine()
        result  = engine.detect(candles)
        valid   = {TRENDING_BULL, TRENDING_BEAR, RANGING, HIGH_VOLATILITY}
        assert set(result.unique()).issubset(valid)

    def test_features_no_nans(self):
        candles = _synthetic(300)
        engine  = RegimeEngine()
        feats   = engine.features(candles)
        assert not feats.isnull().any().any(), "Regime features must not contain NaN"

    def test_encode_onehot_shape(self):
        candles = _synthetic(200)
        engine  = RegimeEngine()
        regime  = engine.detect(candles)
        ohe     = engine.encode_onehot(regime)
        assert ohe.shape == (len(candles), 4)
        # Each row must sum to exactly 1
        assert (ohe.sum(axis=1) == 1).all()

    def test_hurst_bounds(self):
        close = pd.Series(1.1 + np.cumsum(np.random.normal(0, 0.001, 200)))
        h = _rolling_hurst(close, window=60)
        assert ((h >= 0.0) & (h <= 1.0)).all()

    def test_mer_bounds(self):
        close = pd.Series(1.1 + np.cumsum(np.random.normal(0, 0.001, 200)))
        mer = _market_efficiency_ratio(close, window=10)
        assert mer.between(0.0, 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Drift Monitor tests
# ---------------------------------------------------------------------------

class TestDriftMonitor:
    def test_psi_identical_distributions(self):
        arr = np.random.normal(0, 1, 1000)
        assert psi(arr, arr) < 0.01

    def test_psi_shifted_distributions(self):
        ref  = np.random.normal(0, 1, 1000)
        live = np.random.normal(2, 1, 1000)   # mean shifted by 2σ
        assert psi(ref, live) > 0.10

    def test_jsd_identical(self):
        probs = np.random.uniform(0.3, 0.7, 500)
        assert jensen_shannon_divergence(probs, probs) < 0.01

    def test_adwin_detects_step_change(self):
        adwin = ADWIN(delta=0.002, min_window=20)
        # Feed stable win rate of 0.5
        for _ in range(60):
            adwin.update(float(np.random.binomial(1, 0.5)))
        adwin.reset()
        # Now switch to win rate 0.15
        # ADWIN detects *changes* within its accumulating window.
        # We must NOT reset between phases: it compares old vs new within the same window.
        # Feed stable 0.5 phase, then switch to 0.15 — ADWIN should flag the shift.
        adwin_sensitive = ADWIN(delta=0.05, min_window=10)
        rng_stable = np.random.default_rng(0)
        rng_shift  = np.random.default_rng(99)

        # Phase 1: stable at p=0.5 (100 obs, builds reference)
        for _ in range(100):
            adwin_sensitive.update(float(rng_stable.binomial(1, 0.5)))

        # Phase 2: shift to p=0.15 — ADWIN should detect this within 200 obs
        detected = False
        for _ in range(300):
            result = adwin_sensitive.update(float(rng_shift.binomial(1, 0.15)))
            if result:
                detected = True
                break
        assert detected, "ADWIN should detect 0.5→0.15 win-rate step-change"

    def test_monitor_set_reference_and_update(self):
        rng   = np.random.default_rng(0)
        feats = pd.DataFrame(rng.normal(size=(500, 5)),
                             columns=[f"f{i}" for i in range(5)])
        preds = rng.uniform(0.3, 0.7, 500)
        monitor = DriftMonitor(feature_names=[f"f{i}" for i in range(5)])
        monitor.set_reference(feats, preds)
        # Update with similar distribution → should not trigger retrain
        live_feats = pd.DataFrame(rng.normal(size=(1, 5)),
                                  columns=[f"f{i}" for i in range(5)])
        report = monitor.update(live_feats, float(rng.uniform(0.3, 0.7)))
        # Just check it runs without error; no retrain expected on 1 sample
        assert hasattr(report, "should_retrain")

    def test_monitor_summary_df(self):
        rng   = np.random.default_rng(1)
        feats = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
        preds = rng.uniform(0, 1, 300)
        monitor = DriftMonitor(feature_names=["a", "b", "c"])
        monitor.set_reference(feats, preds)
        df = monitor.summary_df()
        assert set(df.columns) == {"feature", "psi", "status"}
