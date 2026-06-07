"""Walk-forward training of LightGBM + XGBoost + LSTM with a logistic meta-model.

The 'GT-Score' optimised here combines profit factor, win rate, sharpe,
max drawdown, and a generalisation ratio (test-vs-train Sharpe).

Per-symbol training:
  Each call to ``train_walk_forward`` can be scoped to a single *symbol*.
  When ``symbol`` is provided, artifacts are saved to
  ``models_artifacts/<SYMBOL>/`` and the function returns ``avg_auc`` and
  ``gt_score`` so the ``ModelRegistry`` can decide whether to promote.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .lightgbm_model import predict_lightgbm, train_lightgbm
from .lstm_model import HAS_TF, make_sequences, predict_lstm, train_lstm

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False
    xgb = None


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    val_end: str
    test_end: str
    auc_test: float
    n_train: int
    n_val: int
    n_test: int


def gt_score(
    profit_factor: float,
    win_rate: float,
    sharpe: float,
    max_drawdown: float,
    generalization_ratio: float,
) -> float:
    """Combined score: Profit_Factor * Win_Rate * Sharpe / (1 + |MaxDD|) * GenRatio."""
    return float(
        profit_factor
        * win_rate
        * sharpe
        / (1.0 + abs(max_drawdown))
        * generalization_ratio
    )


def walk_forward_split(
    timestamps: pd.DatetimeIndex,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, val_end, test_end) windows."""
    if len(timestamps) == 0:
        return []
    start = timestamps.min()
    end = timestamps.max()
    splits = []
    cur = start
    while True:
        train_end = cur + pd.DateOffset(months=train_months)
        val_end = train_end + pd.DateOffset(months=val_months)
        test_end = val_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        splits.append((cur, train_end, val_end, test_end))
        cur = cur + pd.DateOffset(months=test_months)
    return splits


def _train_xgb(X_train, y_train, X_val, y_val, params: Dict):
    if not HAS_XGB:
        return None
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 500)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_walk_forward(
    features: pd.DataFrame,
    labels: pd.Series,
    config: Dict,
    output_dir: str = "models_artifacts",
    symbol: Optional[str] = None,
    data_years: int = 3,
) -> Dict:
    """Run walk-forward training of the full ensemble.

    Args:
        features: feature matrix indexed by timestamp.
        labels: binary labels aligned to ``features``.
        config: model config dict (see ``config/model_config.yaml``).
        output_dir: base directory for artifacts.
        symbol: optional instrument name; artifacts go to ``output_dir/SYMBOL/``.
        data_years: informational — recorded in the summary but does not
            alter the data slice (caller should pre-filter features/labels).

    Returns:
        Dict with per-fold metrics, ensemble path, ``avg_auc``, and ``gt_score``.
    """
    # Per-symbol subdirectory when symbol is provided.
    if symbol:
        out = Path(output_dir) / symbol.upper()
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("features must have a DatetimeIndex")
    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    wf_cfg = config.get("walk_forward", {})
    splits = walk_forward_split(
        features.index,
        train_months=wf_cfg.get("train_months", 12),
        val_months=wf_cfg.get("val_months", 3),
        test_months=wf_cfg.get("test_months", 3),
    )
    if not splits:
        logger.warning("Not enough data for walk-forward; using a single 70/15/15 split")
        n = len(features)
        if n < 30:
            raise ValueError("Need at least 30 labelled samples")
        i1, i2 = int(n * 0.7), int(n * 0.85)
        ts = features.index
        splits = [(ts[0], ts[i1], ts[i2], ts[-1])]

    fold_results: List[FoldResult] = []
    final_artifacts: Dict[str, object] = {}

    # Accumulate out-of-fold predictions for a proper meta-model fit.
    oof_preds_lgb: list = []
    oof_preds_xgb: list = []
    oof_preds_lstm: list = []
    oof_y: list = []

    purge_bars = int(wf_cfg.get("purge_bars", 5))

    for k, (ts_train, ts_train_end, ts_val_end, ts_test_end) in enumerate(splits):
        logger.info("Fold %d: train [%s..%s] val [%s..%s] test [%s..%s]",
                    k, ts_train, ts_train_end, ts_train_end, ts_val_end, ts_val_end, ts_test_end)

        train_mask = (features.index >= ts_train) & (features.index < ts_train_end)
        val_mask = (features.index >= ts_train_end) & (features.index < ts_val_end)
        test_mask = (features.index >= ts_val_end) & (features.index < ts_test_end)

        X_tr, y_tr = features[train_mask], labels[train_mask]

        # --- Purge gap: drop the last `purge_bars` rows from training to prevent
        #     label contamination (a trade entered near train_end may expire in val). ---
        if purge_bars > 0 and len(X_tr) > purge_bars:
            X_tr = X_tr.iloc[:-purge_bars]
            y_tr = y_tr.iloc[:-purge_bars]

        X_va, y_va = features[val_mask], labels[val_mask]
        X_te, y_te = features[test_mask], labels[test_mask]

        if len(X_tr) < 20 or len(X_va) < 5 or len(X_te) < 5:
            logger.warning("Fold %d skipped: insufficient samples", k)
            continue

        # --- LightGBM ---
        lgb_model = train_lightgbm(
            X_tr, y_tr, X_va, y_va,
            params=config.get("lightgbm", {}),
            bayesian_iterations=int(config.get("bayesian_iterations", 0)),
        )
        p_lgb_va = predict_lightgbm(lgb_model, X_va)
        p_lgb_te = predict_lightgbm(lgb_model, X_te)

        # --- XGBoost ---
        xgb_model = _train_xgb(X_tr, y_tr, X_va, y_va, config.get("xgboost", {}))
        if xgb_model is not None:
            p_xgb_va = xgb_model.predict_proba(X_va)[:, 1]
            p_xgb_te = xgb_model.predict_proba(X_te)[:, 1]
        else:
            p_xgb_va = p_lgb_va.copy()
            p_xgb_te = p_lgb_te.copy()

        # --- LSTM ---
        lstm_cfg = config.get("lstm", {})
        timesteps = int(lstm_cfg.get("timesteps", 20))
        if HAS_TF and len(X_tr) > timesteps + 50:
            X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, timesteps)
            X_va_seq, y_va_seq = make_sequences(X_va, y_va, timesteps)
            X_te_seq, _ = make_sequences(X_te, y_te, timesteps)
            lstm_model = train_lstm(
                X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
                units=int(lstm_cfg.get("units", 64)),
                dropout=float(lstm_cfg.get("dropout", 0.3)),
                learning_rate=float(lstm_cfg.get("learning_rate", 1e-3)),
                batch_size=int(lstm_cfg.get("batch_size", 64)),
                epochs=int(lstm_cfg.get("epochs", 50)),
                patience=int(lstm_cfg.get("patience", 5)),
            )
            p_lstm_va_full = predict_lstm(lstm_model, X_va_seq)
            p_lstm_te_full = predict_lstm(lstm_model, X_te_seq)
            # Pad earlier rows with the marginal mean to keep alignment
            pad_va = np.full(len(X_va) - len(p_lstm_va_full), float(np.mean(y_tr)))
            pad_te = np.full(len(X_te) - len(p_lstm_te_full), float(np.mean(y_tr)))
            p_lstm_va = np.concatenate([pad_va, p_lstm_va_full])
            p_lstm_te = np.concatenate([pad_te, p_lstm_te_full])
        else:
            lstm_model = None
            p_lstm_va = p_lgb_va.copy()
            p_lstm_te = p_lgb_te.copy()

        # --- Meta-model (logistic regression on stacked predictions) ---
        meta_X_va = np.column_stack([p_lgb_va, p_xgb_va, p_lstm_va])
        meta_X_te = np.column_stack([p_lgb_te, p_xgb_te, p_lstm_te])
        meta = LogisticRegression(max_iter=200)
        meta.fit(meta_X_va, y_va)
        p_meta_te = meta.predict_proba(meta_X_te)[:, 1]

        try:
            auc = float(roc_auc_score(y_te, p_meta_te))
        except ValueError:
            auc = float("nan")

        fold_results.append(
            FoldResult(
                fold=k,
                train_start=str(ts_train),
                train_end=str(ts_train_end),
                val_end=str(ts_val_end),
                test_end=str(ts_test_end),
                auc_test=auc,
                n_train=int(len(X_tr)),
                n_val=int(len(X_va)),
                n_test=int(len(X_te)),
            )
        )
        logger.info("Fold %d AUC test = %.4f", k, auc)

        # Accumulate OOF predictions for final meta-model training.
        oof_preds_lgb.append(p_lgb_va)
        oof_preds_xgb.append(p_xgb_va)
        oof_preds_lstm.append(p_lstm_va)
        oof_y.append(y_va.values)

        # Keep the artifacts of the most recent fold as the "production" ensemble.
        final_artifacts = {
            "lightgbm": lgb_model,
            "xgboost": xgb_model,
            "lstm": lstm_model,
            "meta": meta,
            "feature_columns": list(features.columns),
            "lstm_timesteps": timesteps,
        }

    # --- Train final meta-model on ALL out-of-fold predictions. ---
    # This gives it far more samples than using only the last fold's val set.
    if oof_y:
        meta_X_oof = np.column_stack([
            np.concatenate(oof_preds_lgb),
            np.concatenate(oof_preds_xgb),
            np.concatenate(oof_preds_lstm),
        ])
        y_oof = np.concatenate(oof_y)
        final_meta = LogisticRegression(max_iter=500)
        final_meta.fit(meta_X_oof, y_oof)
        final_artifacts["meta"] = final_meta
        logger.info("OOF meta-model trained on %d samples across %d folds",
                    len(y_oof), len(oof_y))

    # Compute aggregate metrics for the ModelRegistry.
    valid_aucs = [r.auc_test for r in fold_results if not np.isnan(r.auc_test)]
    avg_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    # GT-score: placeholder using AUC as the primary signal.
    # A full GT-score requires backtest results; the caller can override via the registry.
    _gt = avg_auc * (1.0 + len(valid_aucs) * 0.01)  # slight bonus for more folds
    computed_gt_score = float(_gt)

    # Persist artifacts.
    artifact_path = out / "ensemble.pkl"
    with open(artifact_path, "wb") as f:
        save_payload = {
            "lightgbm": final_artifacts.get("lightgbm"),
            "xgboost":  final_artifacts.get("xgboost"),
            "meta":     final_artifacts.get("meta"),
            "feature_columns": final_artifacts.get("feature_columns"),
            "lstm_timesteps":  final_artifacts.get("lstm_timesteps"),
        }
        pickle.dump(save_payload, f)

    if final_artifacts.get("lstm") is not None and HAS_TF:
        final_artifacts["lstm"].save(str(out / "lstm.keras"))

    summary_path = out / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in fold_results], f, indent=2)

    data_start = str(features.index.min().date()) if len(features) else ""
    data_end   = str(features.index.max().date()) if len(features) else ""

    return {
        "folds":          [asdict(r) for r in fold_results],
        "ensemble_path":  str(artifact_path),
        "summary_path":   str(summary_path),
        "avg_auc":        avg_auc,
        "gt_score":       computed_gt_score,
        "data_start":     data_start,
        "data_end":       data_end,
        "symbol":         symbol or "",
        "data_years":     data_years,
        "artifacts":      final_artifacts,   # in-memory; used by ModelRegistry
    }
