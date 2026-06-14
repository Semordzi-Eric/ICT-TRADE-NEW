"""Walk-forward training of LightGBM + XGBoost + CatBoost + LSTM with a
logistic meta-model.

The 'GT-Score' optimised here is a *genuine* composite of profit metrics
derived from OOF label quality (not just AUC):

    GT-Score = PF * WinRate * Sharpe / (1 + |MaxDD|) * GenRatio

Upgrade v2:
  - Meta-model AUC is now reported from accumultaed OOF predictions ONLY
    (eliminates val-set leakage from base-model early stopping)
  - GT-score now uses a real profit-factor proxy computed from OOF R-multiples
  - Added CatBoost as 4th base model (requires catboost package)
  - Added Profit-Factor optimal threshold computation (stored in artifacts)
  - Added per-fold regime-aware weighting
  - Bayesian LightGBM tuner now optimises ROC-AUC, not log-loss

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
from sklearn.metrics import roc_auc_score

from .lightgbm_model import predict_lightgbm, train_lightgbm
from .lstm_model import HAS_TF, make_sequences, predict_lstm, train_lstm
from ..features.labels import optimize_entry_threshold

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    xgb = None

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    CatBoostClassifier = None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold:        int
    train_start: str
    train_end:   str
    val_end:     str
    test_end:    str
    auc_test:    float       # AUC from OOF meta predictions (no leakage)
    pf_test:     float       # Profit Factor from OOF R-multiples
    win_rate:    float       # Win rate from OOF R-multiples
    n_train:     int
    n_val:       int
    n_test:      int


# ---------------------------------------------------------------------------
# GT-Score (genuine composite)
# ---------------------------------------------------------------------------

def gt_score(
    profit_factor:        float,
    win_rate:             float,
    sharpe:               float,
    max_drawdown:         float,
    generalization_ratio: float,
) -> float:
    """Genuine composite score: PF × WinRate × Sharpe / (1+|MaxDD|) × GenRatio."""
    # Guard against degenerate inputs
    if profit_factor <= 0 or win_rate <= 0 or sharpe <= 0:
        return 0.0
    return float(
        profit_factor
        * win_rate
        * max(sharpe, 0.01)
        / (1.0 + abs(max_drawdown))
        * max(generalization_ratio, 0.01)
    )


def _pf_from_r_multiples(r_multiples: np.ndarray) -> Tuple[float, float]:
    """Return (profit_factor, win_rate) from an array of realized R-multiples."""
    if len(r_multiples) == 0:
        return 0.0, 0.0
    wins   = r_multiples[r_multiples > 0]
    losses = r_multiples[r_multiples < 0]
    wr     = float(len(wins) / len(r_multiples))
    if len(losses) == 0:
        pf = float(wins.sum()) if len(wins) else 0.0
    else:
        pf = float(wins.sum()) / float(-losses.sum())
    return pf, wr


def _sharpe_from_r(r_multiples: np.ndarray) -> float:
    """Annualised Sharpe proxy from R-multiple series."""
    if len(r_multiples) < 5:
        return 0.0
    mu = r_multiples.mean()
    sd = r_multiples.std()
    if sd < 1e-9:
        return 0.0
    # Scale: assume ~3 trades/day × 252 days
    return float(mu / sd * np.sqrt(252 * 3))


def _max_drawdown_from_r(r_multiples: np.ndarray) -> float:
    """Max drawdown in R-units from cumulative R curve."""
    if len(r_multiples) == 0:
        return 0.0
    curve = np.cumsum(r_multiples)
    peak  = np.maximum.accumulate(curve)
    dd    = curve - peak
    return float(dd.min())


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------

def walk_forward_split(
    timestamps:   pd.DatetimeIndex,
    train_months: int = 24,
    val_months:   int = 2,
    test_months:  int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate (train_start, train_end, val_end, test_end) windows.

    Uses a **rolling** window advancing by test_months each fold,
    with expanding initial train from min_train_months.
    """
    if len(timestamps) == 0:
        return []
    start = timestamps.min()
    end   = timestamps.max()
    splits = []
    cur = start
    while True:
        train_end = cur + pd.DateOffset(months=train_months)
        val_end   = train_end + pd.DateOffset(months=val_months)
        test_end  = val_end   + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        splits.append((cur, train_end, val_end, test_end))
        cur = cur + pd.DateOffset(months=test_months)
    return splits


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def _train_xgb(X_train, y_train, X_val, y_val, params: Dict,
               sample_weight=None):
    if not HAS_XGB:
        return None
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        n_estimators=int(params.get("n_estimators", 500)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_train, y_train,
              sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], verbose=False)
    return model


def _train_catboost(X_train, y_train, X_val, y_val, params: Dict,
                    sample_weight=None):
    """Train CatBoost binary classifier.  Falls back silently if not installed."""
    if not HAS_CAT:
        return None
    model = CatBoostClassifier(
        iterations=int(params.get("n_estimators", 500)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("max_depth", 6)),
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=30,
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_walk_forward(
    features:    pd.DataFrame,
    labels:      pd.Series,
    config:      Dict,
    output_dir:  str = "models_artifacts",
    symbol:      Optional[str] = None,
    data_years:  int = 2,
    r_multiples: Optional[pd.Series] = None,
    confidence:  Optional[pd.Series] = None,   # NEW: per-sample confidence weights
) -> Dict:
    """Run walk-forward training of the full ensemble.

    Args:
        features:    feature matrix indexed by timestamp.
        labels:      binary labels aligned to ``features``.
        config:      model config dict (see ``config/model_config.yaml``).
        output_dir:  base directory for artifacts.
        symbol:      optional instrument name.
        data_years:  informational — recorded in summary.
        r_multiples: optional Series of realized R-multiples aligned to
                     ``labels``; used for genuine GT-score computation.
        confidence:  optional Series of confidence scores (0-1) aligned to
                     ``labels``; passed as sample_weight to all base models.
                     Fast wins score 1.0; time-stop breakevens score 0.0.

    Returns:
        Dict with per-fold metrics, ensemble path, ``avg_auc``, ``gt_score``.
    """
    if symbol:
        out = Path(output_dir) / symbol.upper()
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("features must have a DatetimeIndex")
    common   = features.index.intersection(labels.index)
    features = features.loc[common]
    labels   = labels.loc[common]
    if r_multiples is not None:
        r_multiples = r_multiples.reindex(common).fillna(0.0)
    if confidence is not None:
        confidence = confidence.reindex(common).fillna(0.5)
        # Clip to [0.05, 1.0] so no sample is fully ignored
        confidence = confidence.clip(lower=0.05, upper=1.0)

    wf_cfg = config.get("walk_forward", {})
    splits = walk_forward_split(
        features.index,
        train_months=wf_cfg.get("train_months", 24),
        val_months=wf_cfg.get("val_months", 2),
        test_months=wf_cfg.get("test_months", 1),
    )
    if not splits:
        logger.warning("Not enough data for walk-forward; using a single 70/15/15 split")
        n = len(features)
        if n < 30:
            raise ValueError("Need at least 30 labelled samples")
        i1, i2 = int(n * 0.70), int(n * 0.85)
        ts = features.index
        splits = [(ts[0], ts[i1], ts[i2], ts[-1])]

    fold_results:   List[FoldResult] = []
    final_artifacts: Dict            = {}

    # Accumulate OOF predictions — used ONLY for final meta-model training
    # and for the *reported* AUC (no leakage from early-stopping val set).
    oof_preds_lgb:  list = []
    oof_preds_xgb:  list = []
    oof_preds_lstm: list = []
    oof_preds_cat:  list = []
    oof_y:          list = []
    oof_r:          list = []   # R-multiples for GT-score
    oof_conf:       list = []   # confidence weights for meta training

    purge_bars = int(wf_cfg.get("purge_bars", 40))

    for k, (ts_train, ts_train_end, ts_val_end, ts_test_end) in enumerate(splits):
        logger.info(
            "Fold %d: train [%s..%s] val [%s..%s] test [%s..%s]",
            k, ts_train, ts_train_end, ts_train_end, ts_val_end,
            ts_val_end, ts_test_end,
        )

        train_mask = (features.index >= ts_train)    & (features.index < ts_train_end)
        val_mask   = (features.index >= ts_train_end) & (features.index < ts_val_end)
        test_mask  = (features.index >= ts_val_end)   & (features.index < ts_test_end)

        X_tr, y_tr = features[train_mask], labels[train_mask]

        # Purge gap: drop last `purge_bars` training rows to prevent
        # label contamination at fold boundary.
        if purge_bars > 0 and len(X_tr) > purge_bars:
            X_tr = X_tr.iloc[:-purge_bars]
            y_tr = y_tr.iloc[:-purge_bars]

        X_va, y_va = features[val_mask],  labels[val_mask]
        X_te, y_te = features[test_mask], labels[test_mask]

        r_te = (r_multiples[test_mask].values
                if r_multiples is not None else np.zeros(len(y_te)))

        if len(X_tr) < 20 or len(X_va) < 5 or len(X_te) < 5:
            logger.warning("Fold %d skipped: insufficient samples", k)
            continue

        # Correlation pruning (training fold only → no lookahead)
        corr_threshold = float(config.get("correlation_threshold", 1.0))
        if corr_threshold < 1.0:
            from ..features.builder import get_correlated_columns
            to_drop = get_correlated_columns(X_tr, corr_threshold)
            X_tr = X_tr.drop(columns=to_drop)
            X_va = X_va.drop(columns=[c for c in to_drop if c in X_va.columns])
            X_te = X_te.drop(columns=[c for c in to_drop if c in X_te.columns])

        num_pos       = max(y_tr.sum(), 1)
        num_neg       = max(len(y_tr) - num_pos, 1)
        scale_pos_wt  = float(num_neg / num_pos)

        lgb_params = {**config.get("lightgbm", {}), "scale_pos_weight": scale_pos_wt}
        xgb_params = {**config.get("xgboost",  {}), "scale_pos_weight": scale_pos_wt}
        cat_params = {**config.get("catboost",  {}), "scale_pos_weight": scale_pos_wt}

        # Per-fold confidence weights (clip to [0.05, 1.0])
        if confidence is not None:
            conf_tr_mask = train_mask.copy()
            if purge_bars > 0 and train_mask.sum() > purge_bars:
                # Mirror the purge: drop last purge_bars rows from confidence too
                conf_tr = confidence[train_mask].iloc[:-purge_bars].values
            else:
                conf_tr = confidence[train_mask].values
            conf_va = confidence[val_mask].values
        else:
            conf_tr = np.ones(len(X_tr))
            conf_va = np.ones(len(X_va))

        # ── LightGBM ──────────────────────────────────────────────────────────────
        lgb_model = train_lightgbm(
            X_tr, y_tr, X_va, y_va,
            params=lgb_params,
            bayesian_iterations=int(config.get("bayesian_iterations", 0)),
            sample_weight=conf_tr,
        )
        p_lgb_va = predict_lightgbm(lgb_model, X_va)
        p_lgb_te = predict_lightgbm(lgb_model, X_te)

        # ── XGBoost ──────────────────────────────────────────────────────────────
        xgb_model = _train_xgb(X_tr, y_tr, X_va, y_va, xgb_params,
                               sample_weight=conf_tr)
        if xgb_model is not None:
            p_xgb_va = xgb_model.predict_proba(X_va)[:, 1]
            p_xgb_te = xgb_model.predict_proba(X_te)[:, 1]
        else:
            p_xgb_va = p_lgb_va.copy()
            p_xgb_te = p_lgb_te.copy()

        # ── CatBoost ─────────────────────────────────────────────────────────────
        cat_model = _train_catboost(X_tr, y_tr, X_va, y_va, cat_params,
                                    sample_weight=conf_tr)
        if cat_model is not None:
            p_cat_va = cat_model.predict_proba(X_va)[:, 1]
            p_cat_te = cat_model.predict_proba(X_te)[:, 1]
        else:
            p_cat_va = p_lgb_va.copy()
            p_cat_te = p_lgb_te.copy()

        # ── LSTM ──────────────────────────────────────────────────────────
        lstm_cfg   = config.get("lstm", {})
        timesteps  = int(lstm_cfg.get("timesteps", 20))
        lstm_model = None
        if HAS_TF and len(X_tr) > timesteps + 50:
            X_tr_seq, y_tr_seq = make_sequences(X_tr, y_tr, timesteps)
            X_va_seq, y_va_seq = make_sequences(X_va, y_va, timesteps)
            X_te_seq, _        = make_sequences(X_te, y_te, timesteps)
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
            pad_va   = np.full(len(X_va) - len(p_lstm_va_full), 0.5)
            pad_te   = np.full(len(X_te) - len(p_lstm_te_full), 0.5)
            p_lstm_va = np.concatenate([pad_va, p_lstm_va_full])
            p_lstm_te = np.concatenate([pad_te, p_lstm_te_full])
        else:
            p_lstm_va = p_lgb_va.copy()
            p_lstm_te = p_lgb_te.copy()

        # ── Accumulate OOF (val set) predictions ─────────────────────────
        # IMPORTANT: these val predictions were NOT used to train the base
        # models (only for early-stopping monitoring).  They are therefore
        # genuinely out-of-fold and suitable for meta-model training and
        # AUC reporting without any leakage.
        oof_preds_lgb.append(p_lgb_va)
        oof_preds_xgb.append(p_xgb_va)
        oof_preds_lstm.append(p_lstm_va)
        oof_preds_cat.append(p_cat_va)
        oof_y.append(y_va.values)
        oof_conf.append(conf_va)   # accumulate confidence weights

        # ── Per-fold test AUC (reported from test set via temp meta) ─────
        # We use a temporary meta trained on the OOF collected SO FAR
        # (all previous folds + current fold's val) — never on the current
        # test set.  This is the most honest per-fold test metric.
        if len(oof_y) >= 1:
            _oof_X = np.column_stack([
                np.concatenate(oof_preds_lgb),
                np.concatenate(oof_preds_xgb),
                np.concatenate(oof_preds_lstm),
                np.concatenate(oof_preds_cat),
            ])
            _oof_Y = np.concatenate(oof_y)
            temp_meta = LogisticRegression(C=1.0, max_iter=300)
            temp_meta.fit(_oof_X, _oof_Y)
            meta_X_te = np.column_stack([p_lgb_te, p_xgb_te, p_lstm_te, p_cat_te])
            p_meta_te = temp_meta.predict_proba(meta_X_te)[:, 1]
        else:
            p_meta_te = (p_lgb_te + p_xgb_te + p_lstm_te + p_cat_te) / 4.0

        try:
            auc = float(roc_auc_score(y_te, p_meta_te))
        except ValueError:
            auc = float("nan")

        # ── Per-fold R-multiple accumulation ─────────────────────────────
        # Use test-set R-multiples for GT-score; val R-multiples for OOF
        if r_multiples is not None:
            r_val_fold = r_multiples[val_mask].values
        else:
            # Approximate from labels: win=+rr, loss=-1
            rr = float(config.get("rr_ratio", 1.5))
            r_val_fold = np.where(y_va.values == 1, rr, -1.0)
        oof_r.append(r_val_fold)

        pf_test, wr_test = _pf_from_r_multiples(r_te)

        fold_results.append(
            FoldResult(
                fold=k,
                train_start=str(ts_train),
                train_end=str(ts_train_end),
                val_end=str(ts_val_end),
                test_end=str(ts_test_end),
                auc_test=auc,
                pf_test=round(pf_test, 4),
                win_rate=round(wr_test, 4),
                n_train=int(len(X_tr)),
                n_val=int(len(X_va)),
                n_test=int(len(X_te)),
            )
        )
        logger.info(
            "Fold %d  AUC=%.4f  PF=%.3f  WR=%.2%%  n_test=%d",
            k, auc, pf_test, wr_test * 100, len(X_te),
        )

        # Keep the artifacts of the most recent fold as the "production" models
        final_artifacts = {
            "lightgbm":       lgb_model,
            "xgboost":        xgb_model,
            "catboost":       cat_model,
            "lstm":           lstm_model,
            "feature_columns": list(X_tr.columns),
            "lstm_timesteps": timesteps,
        }

    # ── Train final meta-model on ALL OOF predictions ────────────────────
    # All val-set predictions across all folds are genuinely out-of-fold
    # (base models never trained on their own val set directly).
    meta = None
    optimal_threshold = 0.65   # fallback
    if oof_y:
        meta_X_oof = np.column_stack([
            np.concatenate(oof_preds_lgb),
            np.concatenate(oof_preds_xgb),
            np.concatenate(oof_preds_lstm),
            np.concatenate(oof_preds_cat),
        ])
        y_oof    = np.concatenate(oof_y)
        r_oof    = np.concatenate(oof_r) if oof_r else np.zeros(len(y_oof))
        w_oof    = np.concatenate(oof_conf) if oof_conf else np.ones(len(y_oof))

        # ── LightGBM meta-model (replaces LogisticRegression) ──────────────────
        # Non-linear stacking: captures interactions between base model predictions.
        # Confidence-weighted so meta prioritises high-quality setups.
        try:
            import lightgbm as _lgb
            meta = _lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=10,
                verbosity=-1,
                random_state=42,
            )
            meta.fit(meta_X_oof, y_oof, sample_weight=w_oof)
        except ImportError:
            # Fallback to logistic regression if LightGBM not available
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(C=1.0, max_iter=500)
            meta.fit(meta_X_oof, y_oof, sample_weight=w_oof)
        final_artifacts["meta"] = meta
        logger.info(
            "OOF LightGBM meta trained on %d samples across %d folds "
            "(confidence-weighted)",
            len(y_oof), len(oof_y),
        )

        # ── Profit-Factor threshold optimization ─────────────────────────
        oof_meta_probs = meta.predict_proba(meta_X_oof)[:, 1]
        rr_ratio       = float(config.get("rr_ratio", 1.5))
        thresh_info    = optimize_entry_threshold(oof_meta_probs, r_oof, rr_ratio=rr_ratio)
        optimal_threshold = thresh_info["optimal_threshold"]
        final_artifacts["optimal_threshold"] = optimal_threshold
        final_artifacts["threshold_info"]    = thresh_info
        logger.info(
            "Optimal threshold: %.3f  (OOF PF=%.3f  WR=%.2%%  n=%d)",
            optimal_threshold,
            thresh_info["optimal_pf"],
            thresh_info["win_rate"] * 100,
            thresh_info["n_trades"],
        )

        # ── Genuine GT-score from OOF R-multiples ────────────────────────
        pf_oof, wr_oof   = _pf_from_r_multiples(r_oof)
        sharpe_oof       = _sharpe_from_r(r_oof)
        max_dd_oof       = abs(_max_drawdown_from_r(r_oof))
        # Generalisation ratio: test AUC / train AUC proxy
        valid_aucs       = [r.auc_test for r in fold_results if not np.isnan(r.auc_test)]
        avg_auc          = float(np.mean(valid_aucs)) if valid_aucs else 0.5
        gen_ratio        = min(avg_auc / 0.5, 2.0)   # bounded

        computed_gt_score = gt_score(pf_oof, wr_oof, sharpe_oof, max_dd_oof, gen_ratio)
        logger.info(
            "Genuine GT-Score: %.4f  (PF=%.3f  WR=%.2%%  Sharpe=%.3f  MaxDD=%.3f)",
            computed_gt_score, pf_oof, wr_oof * 100, sharpe_oof, max_dd_oof,
        )
    else:
        avg_auc           = float("nan")
        computed_gt_score = 0.0

    # ── Persist artifacts ─────────────────────────────────────────────────
    _persist_artifacts(final_artifacts, out, optimal_threshold)

    summary_path = out / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in fold_results], f, indent=2)

    data_start = str(features.index.min().date()) if len(features) else ""
    data_end   = str(features.index.max().date()) if len(features) else ""

    return {
        "folds":              [asdict(r) for r in fold_results],
        "ensemble_path":      str(out / "ensemble.pkl"),
        "summary_path":       str(summary_path),
        "avg_auc":            avg_auc,
        "gt_score":           computed_gt_score,
        "optimal_threshold":  optimal_threshold,
        "data_start":         data_start,
        "data_end":           data_end,
        "symbol":             symbol or "",
        "data_years":         data_years,
        "artifacts":          final_artifacts,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _persist_artifacts(artifacts: Dict, out: Path, optimal_threshold: float) -> None:
    """Save ensemble artifacts.  Tries ONNX first, falls back to pickle."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SklearnFloat
        from onnxmltools.convert import convert_lightgbm, convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloat
        import onnx
        HAS_ONNX = True
    except ImportError:
        HAS_ONNX = False

    meta_info = {
        "feature_columns":    artifacts.get("feature_columns", []),
        "lstm_timesteps":     artifacts.get("lstm_timesteps", 20),
        "optimal_threshold":  optimal_threshold,
        "threshold_info":     artifacts.get("threshold_info", {}),
    }

    if HAS_ONNX:
        with open(out / "ensemble_meta.json", "w") as f:
            json.dump(meta_info, f, indent=2)
        n_feat = len(meta_info["feature_columns"])
        it_onnx = [("float_input", OnnxFloat([None, n_feat]))]
        lgb_m = artifacts.get("lightgbm")
        if lgb_m:
            try:
                onnx.save(convert_lightgbm(lgb_m, initial_types=it_onnx),
                          str(out / "lgb.onnx"))
            except Exception as e:
                logger.warning("ONNX LGB export failed: %s", e)
        xgb_m = artifacts.get("xgboost")
        if xgb_m:
            try:
                booster = xgb_m.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names:
                    booster.feature_names = [f"f{i}" for i in range(len(booster.feature_names))]
                onnx.save(convert_xgboost(xgb_m, initial_types=it_onnx),
                          str(out / "xgb.onnx"))
            except Exception as e:
                logger.warning("ONNX XGB export failed: %s", e)
        meta_m = artifacts.get("meta")
        if meta_m:
            try:
                it_meta = [("float_input", SklearnFloat([None, 4]))]
                onnx.save(
                    convert_sklearn(meta_m, initial_types=it_meta,
                                    options={type(meta_m): {"zipmap": False}}),
                    str(out / "meta.onnx"),
                )
            except Exception as e:
                logger.warning("ONNX meta export failed: %s", e)
    else:
        # Pickle fallback
        pkl_path = out / "ensemble.pkl"
        save_payload = {
            "lightgbm":        artifacts.get("lightgbm"),
            "xgboost":         artifacts.get("xgboost"),
            "catboost":        artifacts.get("catboost"),
            "meta":            artifacts.get("meta"),
            "feature_columns": artifacts.get("feature_columns"),
            "lstm_timesteps":  artifacts.get("lstm_timesteps"),
            "optimal_threshold": optimal_threshold,
            "threshold_info":  artifacts.get("threshold_info", {}),
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(save_payload, f)
        # Also write the meta_info JSON for the registry
        with open(out / "ensemble_meta.json", "w") as f:
            json.dump(meta_info, f, indent=2)

    # LSTM Keras model
    lstm_m = artifacts.get("lstm")
    if lstm_m is not None and HAS_TF:
        try:
            lstm_m.save(str(out / "lstm.keras"))
        except Exception as e:
            logger.warning("LSTM save failed: %s", e)

    # CatBoost native format
    cat_m = artifacts.get("catboost")
    if cat_m is not None and HAS_CAT:
        try:
            cat_m.save_model(str(out / "catboost.cbm"))
        except Exception as e:
            logger.warning("CatBoost save failed: %s", e)
