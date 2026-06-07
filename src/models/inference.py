"""Inference: load trained ensemble and predict success probability."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .lstm_model import HAS_TF, make_sequences, predict_lstm

logger = logging.getLogger(__name__)


class EnsembleModel:
    """Wraps LightGBM + XGBoost + LSTM + meta-logistic for inference."""

    def __init__(self, ensemble_path: str, lstm_path: Optional[str] = None):
        with open(ensemble_path, "rb") as f:
            self.payload = pickle.load(f)
        self.lstm = None
        if lstm_path and Path(lstm_path).exists() and HAS_TF:
            from tensorflow.keras.models import load_model  # type: ignore
            self.lstm = load_model(lstm_path)
        self.feature_columns = self.payload["feature_columns"]
        self.lstm_timesteps = int(self.payload.get("lstm_timesteps", 20))

    @classmethod
    def from_dir(cls, dir_path: str) -> "EnsembleModel":
        """Load from a specific directory (legacy path — use from_registry instead)."""
        d = Path(dir_path)
        return cls(str(d / "ensemble.pkl"), str(d / "lstm.keras"))

    @classmethod
    def from_registry(cls, symbol: str, base_dir: str = "models_artifacts") -> Optional["EnsembleModel"]:
        """Load the champion model for *symbol* from the model registry.

        Returns ``None`` if no champion has been saved for this symbol yet.
        """
        sym_dir = Path(base_dir) / symbol.upper()
        pkl_path = sym_dir / "ensemble.pkl"
        if not pkl_path.exists():
            return None
        try:
            return cls(str(pkl_path), str(sym_dir / "lstm.keras"))
        except Exception:
            logger.warning("EnsembleModel.from_registry: failed to load %s", symbol)
            return None


    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict success probability for each row of `features`."""
        X = features[self.feature_columns].values

        # LightGBM
        lgb_model = self.payload["lightgbm"]
        p_lgb = (
            np.asarray(lgb_model.predict(X)) if lgb_model is not None
            else np.full(len(X), 0.5)
        )

        # XGBoost
        xgb_model = self.payload.get("xgboost")
        if xgb_model is not None:
            p_xgb = xgb_model.predict_proba(X)[:, 1]
        else:
            p_xgb = p_lgb.copy()

        # LSTM
        if self.lstm is not None and len(features) >= self.lstm_timesteps:
            dummy_y = pd.Series(np.zeros(len(features)), index=features.index)
            X_seq, _ = make_sequences(features[self.feature_columns], dummy_y, self.lstm_timesteps)
            p_lstm_full = predict_lstm(self.lstm, X_seq)
            pad = np.full(len(features) - len(p_lstm_full), 0.5)
            p_lstm = np.concatenate([pad, p_lstm_full])
        else:
            p_lstm = p_lgb.copy()

        meta_X = np.column_stack([p_lgb, p_xgb, p_lstm])
        meta = self.payload["meta"]
        p = meta.predict_proba(meta_X)[:, 1]
        return p


def predict(features: pd.DataFrame, model_ensemble: EnsembleModel) -> np.ndarray:
    """Convenience wrapper."""
    return model_ensemble.predict(features)


def should_trade(probability: float, threshold: float = 0.65) -> bool:
    return probability >= threshold
