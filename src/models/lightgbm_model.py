"""LightGBM trainer with optional Bayesian hyperparameter optimization."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover
    HAS_LGB = False
    lgb = None

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    HAS_SKOPT = True
except ImportError:  # pragma: no cover
    HAS_SKOPT = False


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    bayesian_iterations: int = 0,
    sample_weight: Optional[np.ndarray] = None,
):
    """Train a LightGBM binary classifier; optionally tune hyperparams.

    Args:
        sample_weight: per-sample weights aligned to X_train/y_train.
                       Pass confidence scores so high-conviction setups
                       receive more gradient signal.

    Returns the trained booster (and best params if tuned).
    """
    if not HAS_LGB:
        raise ImportError("lightgbm is not installed")

    base_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
    }
    if params:
        base_params.update(params)

    if bayesian_iterations > 0 and HAS_SKOPT:
        best_params = _bayesian_tune(
            X_train, y_train, X_val, y_val, base_params,
            bayesian_iterations, sample_weight=sample_weight,
        )
        base_params.update(best_params)
        logger.info("Bayesian-tuned params: %s", best_params)

    train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        base_params,
        train_set,
        num_boost_round=base_params.get("n_estimators", 500),
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)],
    )
    return model


def _bayesian_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: Dict,
    n_iter: int,
    sample_weight=None,
) -> Dict:
    space = [
        Integer(15, 127, name="num_leaves"),
        Real(0.01, 0.2, prior="log-uniform", name="learning_rate"),
        Real(0.5, 1.0, name="feature_fraction"),
        Real(0.5, 1.0, name="bagging_fraction"),
        Integer(10, 200, name="min_data_in_leaf"),
    ]

    def objective(values):
        params = dict(base_params)
        params["num_leaves"]        = int(values[0])
        params["learning_rate"]     = float(values[1])
        params["feature_fraction"]  = float(values[2])
        params["bagging_fraction"]  = float(values[3])
        params["min_data_in_leaf"]  = int(values[4])
        train_set = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_set   = lgb.Dataset(X_val,   label=y_val, reference=train_set)
        m = lgb.train(
            params,
            train_set,
            num_boost_round=200,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )
        preds = m.predict(X_val)
        # Maximise ROC-AUC (negate for minimiser).
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_val, preds))
        except Exception:
            auc = 0.5
        return -auc   # gp_minimize minimises; negate to maximise AUC

    result = gp_minimize(objective, space, n_calls=n_iter, random_state=42)
    return {
        "num_leaves": int(result.x[0]),
        "learning_rate": float(result.x[1]),
        "feature_fraction": float(result.x[2]),
        "bagging_fraction": float(result.x[3]),
        "min_data_in_leaf": int(result.x[4]),
    }


def predict_lightgbm(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))
