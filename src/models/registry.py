"""Model Registry — per-symbol champion model management.

Each symbol gets its own subdirectory under ``models_artifacts/<SYMBOL>/``.
A ``registry.json`` file in that directory records the champion's metadata
(AUC, GT-score, training date, data window).

A new trained model is promoted to champion only if it beats the currently
saved champion on the composite GT-score.  Auto-promotion is the default;
set ``require_manual_approval=True`` in the registry to flag challengers for
human review instead.

Usage::

    from src.models.registry import ModelRegistry

    reg = ModelRegistry()
    model = reg.get_champion("EURUSD")          # returns EnsembleModel or None
    reg.evaluate_and_promote("EURUSD", new_artifacts, metrics)
    print(reg.leaderboard())                    # DataFrame of all champions
"""
from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"
_ENSEMBLE_FILE = "ensemble.pkl"
_LSTM_FILE     = "lstm.keras"
_SUMMARY_FILE  = "training_summary.json"


class ModelRegistry:
    """Manages per-symbol champion models on disk.

    Args:
        base_dir: root directory; per-symbol dirs are created under it.
        require_manual_approval: if True, challengers that beat the champion
            are saved as ``challenger.pkl`` and flagged but NOT auto-promoted.
    """

    def __init__(
        self,
        base_dir: str = "models_artifacts",
        require_manual_approval: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.require_manual_approval = require_manual_approval
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_champion(self, symbol: str):
        """Load and return the champion EnsembleModel for *symbol*, or None.

        Returns:
            An ``EnsembleModel`` instance loaded from disk, or ``None`` if no
            champion has been saved yet.
        """
        sym_dir = self._sym_dir(symbol)
        pkl_path = sym_dir / _ENSEMBLE_FILE
        if not pkl_path.exists():
            logger.info("ModelRegistry: no champion for %s yet", symbol)
            return None
        try:
            from .inference import EnsembleModel
            lstm_path = str(sym_dir / _LSTM_FILE)
            return EnsembleModel(str(pkl_path), lstm_path)
        except Exception:
            logger.exception("ModelRegistry: failed to load champion for %s", symbol)
            return None

    def evaluate_and_promote(
        self,
        symbol: str,
        artifacts: Dict,
        metrics: Dict,
        fold_results: Optional[List[Dict]] = None,
    ) -> bool:
        """Compare a newly trained model against the saved champion.

        Args:
            symbol: instrument name, e.g. ``"EURUSD"``.
            artifacts: dict from ``train_walk_forward`` — must contain keys
                ``lightgbm``, ``xgboost``, ``meta``, ``feature_columns``,
                ``lstm_timesteps``.  ``lstm`` (Keras model) is optional.
            metrics: dict with keys ``avg_auc``, ``gt_score``, ``data_start``,
                ``data_end``.
            fold_results: list of per-fold metric dicts (for the summary file).

        Returns:
            True if the challenger was promoted (or is the first model),
            False if the existing champion won.
        """
        sym_dir = self._sym_dir(symbol)
        sym_dir.mkdir(parents=True, exist_ok=True)

        registry = self._load_registry(symbol)
        challenger_score = float(metrics.get("gt_score", 0.0))
        challenger_auc   = float(metrics.get("avg_auc",  0.0))
        champion_score   = float(registry.get("gt_score", -9999.0))

        logger.info(
            "ModelRegistry [%s]: challenger GT=%.4f AUC=%.4f | champion GT=%.4f",
            symbol, challenger_score, challenger_auc, champion_score,
        )

        is_better = challenger_score > champion_score
        is_first  = champion_score == -9999.0

        if not is_first and not is_better:
            logger.info("ModelRegistry [%s]: champion retained (challenger did not improve)", symbol)
            return False

        if is_better and self.require_manual_approval and not is_first:
            # Save as challenger — do NOT overwrite champion.
            self._save_artifacts(artifacts, sym_dir, prefix="challenger_")
            registry["challenger"] = {
                "gt_score": challenger_score,
                "avg_auc":  challenger_auc,
                "date":     datetime.now(timezone.utc).isoformat(),
                **{k: metrics.get(k) for k in ("data_start", "data_end")},
            }
            self._save_registry(symbol, registry)
            logger.info(
                "ModelRegistry [%s]: challenger flagged for manual review "
                "(gt=%.4f > champion gt=%.4f)",
                symbol, challenger_score, champion_score,
            )
            return False

        # Promote challenger → champion.
        self._save_artifacts(artifacts, sym_dir, prefix="")
        if fold_results:
            summary_path = sym_dir / _SUMMARY_FILE
            summary_path.write_text(json.dumps(fold_results, indent=2), encoding="utf-8")

        registry.update({
            "symbol":      symbol,
            "gt_score":    challenger_score,
            "avg_auc":     challenger_auc,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "data_start":  metrics.get("data_start", ""),
            "data_end":    metrics.get("data_end", ""),
            "n_folds":     len(fold_results) if fold_results else 0,
        })
        registry.pop("challenger", None)  # clear pending challenger if any
        self._save_registry(symbol, registry)
        logger.info(
            "ModelRegistry [%s]: new champion promoted — GT=%.4f AUC=%.4f",
            symbol, challenger_score, challenger_auc,
        )
        return True

    def promote_challenger(self, symbol: str) -> bool:
        """Manually promote a pending challenger to champion.

        Only relevant when ``require_manual_approval=True`` and a challenger
        has been saved but not yet promoted.
        """
        sym_dir = self._sym_dir(symbol)
        challenger_pkl = sym_dir / "challenger_ensemble.pkl"
        if not challenger_pkl.exists():
            logger.warning("ModelRegistry [%s]: no challenger to promote", symbol)
            return False

        # Replace champion files with challenger files.
        for fname in [_ENSEMBLE_FILE, _LSTM_FILE, _SUMMARY_FILE]:
            challenger_file = sym_dir / f"challenger_{fname}"
            if challenger_file.exists():
                shutil.move(str(challenger_file), str(sym_dir / fname))

        registry = self._load_registry(symbol)
        challenger = registry.pop("challenger", {})
        registry.update(challenger)
        registry["promoted_at"] = datetime.now(timezone.utc).isoformat()
        self._save_registry(symbol, registry)
        logger.info("ModelRegistry [%s]: challenger manually promoted to champion", symbol)
        return True

    def list_champions(self) -> List[Dict]:
        """Return metadata for all symbols that have a saved champion."""
        results = []
        for sym_dir in sorted(self.base_dir.iterdir()):
            if not sym_dir.is_dir():
                continue
            pkl = sym_dir / _ENSEMBLE_FILE
            if not pkl.exists():
                continue
            registry = self._load_registry(sym_dir.name)
            results.append({
                "symbol":      sym_dir.name,
                "gt_score":    registry.get("gt_score", None),
                "avg_auc":     registry.get("avg_auc",  None),
                "promoted_at": registry.get("promoted_at", ""),
                "data_start":  registry.get("data_start", ""),
                "data_end":    registry.get("data_end", ""),
                "n_folds":     registry.get("n_folds", 0),
                "has_challenger": (sym_dir / "challenger_ensemble.pkl").exists(),
            })
        return results

    def leaderboard(self) -> pd.DataFrame:
        """Return a ranked DataFrame of all champions sorted by GT-score."""
        data = self.list_champions()
        if not data:
            return pd.DataFrame(columns=[
                "symbol", "gt_score", "avg_auc", "promoted_at",
                "data_start", "data_end", "n_folds",
            ])
        df = pd.DataFrame(data)
        df = df.sort_values("gt_score", ascending=False).reset_index(drop=True)
        return df

    def has_champion(self, symbol: str) -> bool:
        """Return True if a trained champion exists for *symbol*."""
        return (self._sym_dir(symbol) / _ENSEMBLE_FILE).exists()

    def champion_auc(self, symbol: str) -> Optional[float]:
        """Return the AUC of the saved champion, or None."""
        reg = self._load_registry(symbol)
        v = reg.get("avg_auc")
        return float(v) if v is not None else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sym_dir(self, symbol: str) -> Path:
        return self.base_dir / symbol.upper()

    def _load_registry(self, symbol: str) -> Dict:
        path = self._sym_dir(symbol) / _REGISTRY_FILE
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_registry(self, symbol: str, data: Dict) -> None:
        path = self._sym_dir(symbol) / _REGISTRY_FILE
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _save_artifacts(artifacts: Dict, sym_dir: Path, prefix: str = "") -> None:
        """Persist ensemble artifacts to *sym_dir* with optional *prefix*."""
        pkl_path = sym_dir / f"{prefix}{_ENSEMBLE_FILE}"
        save_payload = {
            "lightgbm":       artifacts.get("lightgbm"),
            "xgboost":        artifacts.get("xgboost"),
            "meta":           artifacts.get("meta"),
            "feature_columns": artifacts.get("feature_columns"),
            "lstm_timesteps": artifacts.get("lstm_timesteps"),
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(save_payload, f)

        lstm = artifacts.get("lstm")
        if lstm is not None:
            try:
                lstm.save(str(sym_dir / f"{prefix}{_LSTM_FILE}"))
            except Exception:
                logger.warning("ModelRegistry: could not save LSTM model for %s", sym_dir.name)
