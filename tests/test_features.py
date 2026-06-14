"""Tests for the feature builder pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fvg import detect_fvg
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from src.detection.orderblock import detect_order_blocks
from src.detection.structure import detect_bos, detect_choch
from src.features.builder import FEATURE_COLUMNS, build_feature_pipeline


def _synthetic(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    rets = rng.normal(0, 0.0005, n)
    close = 1.1 + np.cumsum(rets)
    open_ = close - rng.normal(0, 0.0002, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0003, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0003, n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(50, 500, n)}, index=idx,
    )


def _detections(candles: pd.DataFrame) -> dict:
    return {
        "fvg": detect_fvg(candles, min_gap_atr=0.5),
        "order_blocks": detect_order_blocks(candles, min_move_atr=0.5),
        "liquidity_sweeps": detect_liquidity_sweeps(candles, threshold_atr=0.3),
        "equal_levels": detect_equal_highs_lows(candles),
        "bos": detect_bos(candles),
        "choch": detect_choch(candles),
    }


def test_feature_pipeline_shape():
    candles = _synthetic()
    feats = build_feature_pipeline(candles, _detections(candles), normalize=False)
    assert len(feats) == len(candles)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 47   # 47 features (48 - sess_London_NY dropped)


def test_feature_pipeline_no_nans_or_infs():
    candles = _synthetic()
    feats = build_feature_pipeline(candles, _detections(candles), normalize=True)
    assert not feats.isnull().any().any()
    assert not np.isinf(feats.values).any()


def test_feature_pipeline_normalization_keeps_one_hots():
    candles = _synthetic()
    feats = build_feature_pipeline(candles, _detections(candles), normalize=True)
    # Session dummies and FVG/OB active flags must remain binary (0/1)
    one_hot_cols = ["sess_London", "sess_NY", "sess_Asian",
                    "active_bull_fvg", "active_bear_fvg",
                    "active_bull_ob", "active_bear_ob",
                    "regime_TRENDING_BULL", "regime_TRENDING_BEAR",
                    "regime_RANGING", "regime_HIGH_VOLATILITY"]
    for col in one_hot_cols:
        if col not in feats.columns:
            continue
        unique = set(np.unique(feats[col].values))
        assert unique.issubset({0.0, 1.0}), f"{col} should remain one-hot, got {unique}"


def test_correlation_filter():
    candles = _synthetic(300)
    detections = _detections(candles)
    feats_no_filter = build_feature_pipeline(candles, detections, normalize=True)
    feats_filtered = build_feature_pipeline(candles, detections, normalize=True,
                                            drop_columns=["ret_1"])
    assert feats_filtered.shape[1] <= feats_no_filter.shape[1]
