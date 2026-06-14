"""Tests for the ModelRegistry."""
import pickle
import tempfile
from pathlib import Path

import pytest

from src.models.registry import ModelRegistry


def _dummy_artifacts(auc_hint: float = 0.6):
    """Return a minimal artifacts dict (no real ML models)."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    meta = LogisticRegression()
    meta.fit([[0, 0, 0], [1, 1, 1]], [0, 1])
    return {
        "lightgbm": None,
        "xgboost": None,
        "meta": meta,
        "feature_columns": ["a", "b", "c"],
        "lstm_timesteps": 20,
        "lstm": None,
    }


def _metrics(gt: float, auc: float):
    return {"gt_score": gt, "avg_auc": auc, "data_start": "2021-01-01", "data_end": "2024-01-01"}


class TestModelRegistry:
    def test_no_champion_initially(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg.get_champion("EURUSD") is None
        assert not reg.has_champion("EURUSD")
        assert reg.champion_auc("EURUSD") is None

    def test_first_model_always_promoted(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        arts = _dummy_artifacts()
        promoted = reg.evaluate_and_promote("EURUSD", arts, _metrics(0.55, 0.58))
        assert promoted is True
        assert reg.has_champion("EURUSD")
        assert abs(reg.champion_auc("EURUSD") - 0.58) < 1e-3

    def test_better_model_promotes(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        arts = _dummy_artifacts()
        reg.evaluate_and_promote("GBPUSD", arts, _metrics(0.55, 0.58))
        promoted = reg.evaluate_and_promote("GBPUSD", arts, _metrics(0.70, 0.62))
        assert promoted is True
        assert abs(reg.champion_auc("GBPUSD") - 0.62) < 1e-3

    def test_worse_model_not_promoted(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        arts = _dummy_artifacts()
        reg.evaluate_and_promote("USDJPY", arts, _metrics(0.70, 0.62))
        promoted = reg.evaluate_and_promote("USDJPY", arts, _metrics(0.50, 0.55))
        assert promoted is False
        # AUC should still be the original champion's.
        assert abs(reg.champion_auc("USDJPY") - 0.62) < 1e-3

    def test_manual_approval_does_not_auto_promote(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path), require_manual_approval=True)
        arts = _dummy_artifacts()
        reg.evaluate_and_promote("XAUUSD", arts, _metrics(0.55, 0.58))  # first → promoted
        promoted = reg.evaluate_and_promote("XAUUSD", arts, _metrics(0.80, 0.70))
        assert promoted is False  # requires manual approval
        assert abs(reg.champion_auc("XAUUSD") - 0.58) < 1e-3  # old champion retained
        # Challenger file should exist (pkl when onnx not installed, json meta when onnx available).
        has_challenger = (
            (tmp_path / "XAUUSD" / "challenger_ensemble.pkl").exists()
            or (tmp_path / "XAUUSD" / "challenger_ensemble_meta.json").exists()
        )
        assert has_challenger, "Expected challenger artifact to be saved"

    def test_leaderboard_sorted(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        arts = _dummy_artifacts()
        reg.evaluate_and_promote("EURUSD", arts, _metrics(0.70, 0.65))
        reg.evaluate_and_promote("GBPUSD", arts, _metrics(0.50, 0.58))
        reg.evaluate_and_promote("USDJPY", arts, _metrics(0.90, 0.70))
        lb = reg.leaderboard()
        assert list(lb["symbol"]) == ["USDJPY", "EURUSD", "GBPUSD"]

    def test_list_champions_empty(self, tmp_path):
        reg = ModelRegistry(base_dir=str(tmp_path))
        assert reg.list_champions() == []
