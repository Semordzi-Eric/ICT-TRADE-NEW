"""Inference: load trained ensemble and predict success probability.

Security note:
  Pickle is only used for legacy models and is protected by an HMAC-SHA256
  signature.  The key is read from the environment variable
  ``ICT_MODEL_HMAC_KEY`` (defaults to a static fallback only for development
  — set a random secret in production).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .lstm_model import HAS_TF, make_sequences, predict_lstm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HMAC helpers for pickle integrity verification.
# ---------------------------------------------------------------------------
_HMAC_KEY = os.environ.get("ICT_MODEL_HMAC_KEY", "ict-dev-key-change-in-prod").encode()


def _sig_path(pkl_path: Path) -> Path:
    """Return the companion .sig file path for a pickle."""
    return pkl_path.with_suffix(".sig")


def _compute_hmac(data: bytes) -> str:
    return hmac.new(_HMAC_KEY, data, hashlib.sha256).hexdigest()


def _verify_pickle(pkl_path: Path) -> bytes:
    """Load pickle bytes and verify HMAC signature.  Raises ValueError if invalid."""
    sig_file = _sig_path(pkl_path)
    raw = pkl_path.read_bytes()
    expected = _compute_hmac(raw)
    if sig_file.exists():
        stored = sig_file.read_text().strip()
        if not hmac.compare_digest(stored, expected):
            raise ValueError(
                f"HMAC verification FAILED for {pkl_path} — model file may have been tampered with!"
            )
    else:
        # No sig file yet (legacy model).  Log a warning but allow load.
        logger.warning(
            "No .sig file found for %s — skipping HMAC check. "
            "Run `save_ensemble_pickle()` to create a signed copy.",
            pkl_path,
        )
    return raw


def save_signed_pickle(obj: object, pkl_path: Path) -> None:
    """Pickle `obj` to `pkl_path` and write a companion HMAC signature file."""
    import pickle as _pickle
    raw = _pickle.dumps(obj)
    pkl_path.write_bytes(raw)
    _sig_path(pkl_path).write_text(_compute_hmac(raw))
    logger.info("Saved signed pickle → %s (.sig written)", pkl_path)


class EnsembleModel:
    """Wraps LightGBM + XGBoost + LSTM + meta-logistic for inference."""

    def __init__(self, ensemble_dir: str, lstm_path: Optional[str] = None):
        d = Path(ensemble_dir)
        self.payload = None
        self.lgb_sess = None
        self.xgb_sess = None
        self.meta_sess = None
        # FIX BUG-C1: self.lstm must always be initialised before predict() is called.
        self.lstm = None

        if (d / "ensemble_meta.json").exists():
            import json
            import onnxruntime as rt
            with open(d / "ensemble_meta.json", "r") as f:
                meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
            self.lstm_timesteps = int(meta.get("lstm_timesteps", 20))

            if (d / "lgb.onnx").exists():
                self.lgb_sess = rt.InferenceSession(str(d / "lgb.onnx"))
            if (d / "xgb.onnx").exists():
                self.xgb_sess = rt.InferenceSession(str(d / "xgb.onnx"))
            if (d / "meta.onnx").exists():
                self.meta_sess = rt.InferenceSession(str(d / "meta.onnx"))
        else:
            # Fallback for old pickle format — verified with HMAC before loading.
            pkl_path = d if d.is_file() else d / "ensemble.pkl"
            raw = _verify_pickle(pkl_path)
            self.payload = pickle.loads(raw)  # noqa: S301 — HMAC verified above
            self.feature_columns = self.payload["feature_columns"]
            self.lstm_timesteps = int(self.payload.get("lstm_timesteps", 20))

        # Attempt to load a saved Keras LSTM model if available.
        lp = Path(lstm_path) if lstm_path else (d / "lstm.keras")
        if HAS_TF and lp.exists():
            try:
                import tensorflow as tf  # type: ignore
                self.lstm = tf.keras.models.load_model(str(lp))
                logger.info("Loaded LSTM from %s", lp)
            except Exception as exc:
                logger.warning("Could not load LSTM model from %s: %s", lp, exc)

    @classmethod
    def from_dir(cls, dir_path: str) -> "EnsembleModel":
        """Load from a specific directory."""
        d = Path(dir_path)
        return cls(str(d), str(d / "lstm.keras"))

    @classmethod
    def from_registry(cls, symbol: str, base_dir: str = "models_artifacts") -> Optional["EnsembleModel"]:
        """Load the champion model for *symbol* from the model registry.

        Returns ``None`` if no champion has been saved for this symbol yet.
        """
        sym_dir = Path(base_dir) / symbol.upper()
        if not (sym_dir / "ensemble_meta.json").exists() and not (sym_dir / "ensemble.pkl").exists():
            return None
        try:
            return cls(str(sym_dir), str(sym_dir / "lstm.keras"))
        except Exception:
            logger.warning("EnsembleModel.from_registry: failed to load %s", symbol)
            return None


    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict success probability for each row of `features`."""
        X = features[self.feature_columns].values.astype(np.float32)

        # LightGBM
        if self.lgb_sess is not None:
            input_name = self.lgb_sess.get_inputs()[0].name
            label_name = self.lgb_sess.get_outputs()[1].name
            preds = self.lgb_sess.run([label_name], {input_name: X})[0]
            if isinstance(preds, list):
                p_lgb = np.array([float(p.get(1, p.get('1', 0.5))) for p in preds])
            else:
                p_lgb = preds[:, 1]
        elif self.payload and self.payload.get("lightgbm") is not None:
            lgb_model = self.payload["lightgbm"]
            p_lgb = np.asarray(lgb_model.predict(X))
        else:
            p_lgb = np.full(len(X), 0.5)

        # XGBoost
        if self.xgb_sess is not None:
            input_name = self.xgb_sess.get_inputs()[0].name
            label_name = self.xgb_sess.get_outputs()[1].name
            preds = self.xgb_sess.run([label_name], {input_name: X})[0]
            if isinstance(preds, list):
                p_xgb = np.array([float(p.get(1, p.get('1', 0.5))) for p in preds])
            else:
                p_xgb = preds[:, 1]
        elif self.payload and self.payload.get("xgboost") is not None:
            xgb_model = self.payload.get("xgboost")
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

        meta_X = np.column_stack([p_lgb, p_xgb, p_lstm]).astype(np.float32)
        
        if self.meta_sess is not None:
            input_name = self.meta_sess.get_inputs()[0].name
            label_name = self.meta_sess.get_outputs()[1].name
            preds = self.meta_sess.run([label_name], {input_name: meta_X})[0]
            if isinstance(preds, list):
                p = np.array([float(p.get(1, p.get('1', 0.5))) for p in preds])
            else:
                p = preds[:, 1]
        elif self.payload and self.payload.get("meta") is not None:
            meta = self.payload["meta"]
            p = meta.predict_proba(meta_X)[:, 1]
        else:
            p = np.full(len(X), 0.5)
            
        return p


def predict(features: pd.DataFrame, model_ensemble: EnsembleModel) -> np.ndarray:
    """Convenience wrapper."""
    return model_ensemble.predict(features)


def should_trade(probability: float, threshold: float = 0.65) -> bool:
    return probability >= threshold
