"""LSTM sequence model: 20-bar window → P(success).

Tensorflow / Keras is heavy and optional. We import lazily and fail with a
clear error if it isn't installed.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    HAS_TF = True
except ImportError:  # pragma: no cover
    HAS_TF = False


def make_sequences(
    features: pd.DataFrame,
    labels: pd.Series,
    timesteps: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build supervised sequences (n, timesteps, n_features) aligned to labels.

    For label at row ``i``, we use rows ``[i-timesteps+1 ... i]``.
    """
    X = features.values
    y = labels.values
    n = len(features)
    if n < timesteps:
        return np.zeros((0, timesteps, X.shape[1])), np.zeros((0,))
    Xs, ys = [], []
    for i in range(timesteps - 1, n):
        Xs.append(X[i - timesteps + 1 : i + 1])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    units: int = 64,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 50,
    patience: int = 5,
):
    """Train a 2-layer LSTM with early stopping; returns the trained model."""
    if not HAS_TF:
        raise ImportError("tensorflow is not installed")
    if X_train.size == 0:
        raise ValueError("Empty training set")
    timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(timesteps, n_features)),
            Dropout(dropout),
            LSTM(units // 2),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    return model


def predict_lstm(model, X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0,))
    return model.predict(X, verbose=0).flatten()
