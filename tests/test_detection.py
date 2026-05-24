"""Unit tests for the detection layer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.fvg import detect_fvg
from src.detection.liquidity import (
    detect_equal_highs_lows,
    detect_liquidity_sweeps,
    find_previous_session_highs_lows,
)
from src.detection.orderblock import detect_order_blocks
from src.detection.session import get_session, seconds_into_session
from src.detection.structure import detect_bos, detect_choch


def _synthetic_candles(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build deterministic synthetic OHLCV bars for testing."""
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


def test_detect_fvg_basic():
    candles = _synthetic_candles(300)
    # Inject a clear bullish FVG: high[i-2]=1.10, low[i]=1.12
    candles.iloc[100, candles.columns.get_loc("high")] = 1.10
    candles.iloc[100, candles.columns.get_loc("low")] = 1.095
    candles.iloc[101, candles.columns.get_loc("low")] = 1.105
    candles.iloc[101, candles.columns.get_loc("high")] = 1.115
    candles.iloc[102, candles.columns.get_loc("low")] = 1.12
    candles.iloc[102, candles.columns.get_loc("high")] = 1.13
    candles.iloc[102, candles.columns.get_loc("close")] = 1.125

    zones = detect_fvg(candles, min_gap_atr=0.5)
    assert isinstance(zones, list)
    bullish = [z for z in zones if z.direction == "bullish"]
    assert any(z.index == 102 for z in bullish)


def test_detect_liquidity_sweeps_returns_list():
    candles = _synthetic_candles(500)
    sweeps = detect_liquidity_sweeps(candles, lookback=50, threshold_atr=0.3)
    assert isinstance(sweeps, list)
    for s in sweeps:
        assert s.direction in ("bullish", "bearish")
        assert s.pierce_atr >= 0.3


def test_detect_equal_highs_lows():
    candles = _synthetic_candles(400)
    levels = detect_equal_highs_lows(candles, tolerance_atr=2.0, swing_lookback=3)
    assert isinstance(levels, list)
    for lv in levels:
        assert lv.kind in ("EQH", "EQL")
        assert lv.count >= 2


def test_detect_order_blocks():
    candles = _synthetic_candles(500)
    obs = detect_order_blocks(candles, min_move_atr=0.5)
    assert isinstance(obs, list)
    for ob in obs:
        assert ob.direction in ("bullish", "bearish")
        assert ob.top >= ob.bottom


def test_detect_bos_and_choch():
    candles = _synthetic_candles(500)
    bos = detect_bos(candles)
    choch = detect_choch(candles)
    assert isinstance(bos, list)
    assert isinstance(choch, list)


def test_session_classification():
    ts = pd.Timestamp("2024-01-01 09:00:00")
    label = get_session(ts)
    assert label in ("London", "Asian", "NY", "London_NY", "Asian_London", "OffHours")
    assert seconds_into_session(ts) >= 0


def test_previous_session_highs_lows_shape():
    candles = _synthetic_candles(800)
    sessions = {"london": ("08:00", "17:00"), "newyork": ("13:00", "22:00")}
    out = find_previous_session_highs_lows(candles, sessions)
    assert len(out) == len(candles)
    assert "london_prev_high" in out.columns
    assert "newyork_prev_low" in out.columns
