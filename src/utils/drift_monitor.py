"""Drift monitoring: PSI, KS-Test, ADWIN, Jensen-Shannon Divergence.

Detects three types of drift:
    1. Feature drift     — distribution of input features has shifted
    2. Prediction drift  — model output distribution has shifted
    3. Performance drift — rolling win rate / AUC has degraded

Usage::

    from src.utils.drift_monitor import DriftMonitor

    monitor = DriftMonitor(feature_names=FEATURE_COLUMNS)
    monitor.set_reference(train_features_df, train_predictions)

    # At inference time:
    report = monitor.update(live_features, live_prediction, actual_outcome)
    if report.should_retrain:
        trigger_retrain()
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Summary of drift state at a point in time."""
    # Feature-level PSI scores  {feature_name: psi_value}
    feature_psi: Dict[str, float] = field(default_factory=dict)
    # Max PSI across all features
    max_feature_psi: float = 0.0
    # Prediction distribution JSD (0 = identical, 1 = maximally different)
    prediction_jsd: float = 0.0
    # Rolling win-rate change detected by ADWIN
    adwin_drift_detected: bool = False
    # Current rolling win rate (last N trades)
    rolling_win_rate: float = 0.5
    # Number of trades in rolling window
    n_trades: int = 0
    # Overall recommendation
    should_retrain: bool = False
    # Human-readable notes
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.10 → stable
    PSI 0.10-0.20 → slight shift (monitor)
    PSI > 0.20 → significant drift (retrain)
    """
    # Build bins on the expected distribution
    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(expected, percentiles)
    bins[0]  = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual,   bins=bins)

    exp_pct = exp_counts / (exp_counts.sum() + epsilon)
    act_pct = act_counts / (act_counts.sum() + epsilon)

    # Replace zeros to avoid log(0)
    exp_pct = np.where(exp_pct == 0, epsilon, exp_pct)
    act_pct = np.where(act_pct == 0, epsilon, act_pct)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 20) -> float:
    """JSD between two distributions (0 = identical, 1 = max different).

    Works on raw probability arrays (histogrammed internally).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ph, _ = np.histogram(p, bins=bins, density=True)
    qh, _ = np.histogram(q, bins=bins, density=True)
    ph = ph / (ph.sum() + 1e-10)
    qh = qh / (qh.sum() + 1e-10)
    m = 0.5 * (ph + qh)
    eps = 1e-10
    kl_pm = np.sum(np.where(ph > 0, ph * np.log((ph + eps) / (m + eps)), 0))
    kl_qm = np.sum(np.where(qh > 0, qh * np.log((qh + eps) / (m + eps)), 0))
    return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0))


# ---------------------------------------------------------------------------
# ADWIN (Adaptive Windowing)  — lightweight streaming drift detector
# ---------------------------------------------------------------------------

class ADWIN:
    """A simplified ADWIN change detector for streaming binary outcomes.

    ADWIN detects when the mean of a binary stream (e.g. win=1/loss=0)
    has shifted significantly by comparing sub-windows within an adaptive
    window.

    Args:
        delta: confidence parameter (lower = more sensitive). Default 0.002.
        min_window: minimum observations before detection can fire.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30) -> None:
        self.delta      = delta
        self.min_window = min_window
        self._window: deque = deque()
        self._total  = 0.0
        self._n      = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add one observation (0 or 1). Returns True if drift detected."""
        self._window.append(value)
        self._total += value
        self._n     += 1
        self.drift_detected = False

        if self._n < self.min_window:
            return False

        # Test all possible split points
        w = list(self._window)
        n_total = len(w)
        cum = np.cumsum(w)
        total_sum = cum[-1]

        for cut in range(self.min_window // 2, n_total - self.min_window // 2):
            n0, n1 = cut, n_total - cut
            s0 = cum[cut - 1]
            s1 = total_sum - s0
            mu0 = s0 / n0
            mu1 = s1 / n1

            # Hoeffding-style bound
            eps_cut = np.sqrt(
                (1.0 / (2 * n0) + 1.0 / (2 * n1)) *
                np.log(4 * n_total / self.delta)
            )
            if abs(mu0 - mu1) > eps_cut:
                # Drift detected — shrink window to the newer half
                for _ in range(cut):
                    dropped = self._window.popleft()
                    self._total -= dropped
                    self._n     -= 1
                self.drift_detected = True
                break

        return self.drift_detected

    def current_mean(self) -> float:
        if self._n == 0:
            return 0.5
        return self._total / self._n

    def reset(self) -> None:
        self._window.clear()
        self._total = 0.0
        self._n     = 0
        self.drift_detected = False


# ---------------------------------------------------------------------------
# Main Monitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """Unified drift monitor for feature, prediction, and performance drift.

    Args:
        feature_names:      list of feature column names to monitor.
        psi_warn_threshold: PSI > this → log warning (default 0.10).
        psi_alert_threshold:PSI > this → trigger retrain (default 0.20).
        jsd_threshold:      JSD on predictions > this → alert (default 0.07).
        adwin_delta:        ADWIN sensitivity (lower = more sensitive).
        rolling_window:     number of recent trades for win-rate tracking.
        min_ref_samples:    minimum reference samples before PSI is computed.
    """

    def __init__(
        self,
        feature_names:       Optional[Sequence[str]] = None,
        psi_warn_threshold:  float = 0.10,
        psi_alert_threshold: float = 0.20,
        jsd_threshold:       float = 0.07,
        adwin_delta:         float = 0.002,
        rolling_window:      int   = 50,
        min_ref_samples:     int   = 200,
    ) -> None:
        self.feature_names       = list(feature_names) if feature_names else []
        self.psi_warn            = psi_warn_threshold
        self.psi_alert           = psi_alert_threshold
        self.jsd_threshold       = jsd_threshold
        self.rolling_window      = rolling_window
        self.min_ref_samples     = min_ref_samples

        # Reference distributions (set once on training data)
        self._ref_features: Dict[str, np.ndarray] = {}
        self._ref_predictions: Optional[np.ndarray] = None

        # Live sliding windows
        self._live_features: Dict[str, deque] = {
            f: deque(maxlen=min_ref_samples) for f in self.feature_names
        }
        self._live_predictions: deque = deque(maxlen=min_ref_samples)
        self._outcome_window:   deque = deque(maxlen=rolling_window)

        # ADWIN for win-rate
        self._adwin = ADWIN(delta=adwin_delta, min_window=20)

        # Counters
        self._n_updates = 0

    # ------------------------------------------------------------------
    def set_reference(
        self,
        features:    pd.DataFrame,
        predictions: np.ndarray,
    ) -> None:
        """Set the reference distribution from training/validation data.

        Call once after training a new champion model.
        """
        for col in self.feature_names:
            if col in features.columns:
                self._ref_features[col] = features[col].dropna().values.copy()
        self._ref_predictions = np.asarray(predictions).flatten().copy()
        self._adwin.reset()
        logger.info("DriftMonitor: reference set on %d samples.", len(features))

    # ------------------------------------------------------------------
    def update(
        self,
        features:   pd.DataFrame,
        prediction: float,
        outcome:    Optional[float] = None,    # 1.0=win, 0.0=loss, None=unknown
    ) -> DriftReport:
        """Record one inference step and return the current drift report.

        Args:
            features:   single-row (or batch) feature DataFrame.
            prediction: model probability output (0-1).
            outcome:    realized trade outcome (1=win, 0=loss) if available.
        """
        self._n_updates += 1

        # Accumulate live windows
        for col in self.feature_names:
            if col in features.columns:
                vals = features[col].dropna().values
                for v in vals:
                    self._live_features[col].append(float(v))

        preds = np.atleast_1d(np.asarray(prediction).flatten())
        for p in preds:
            self._live_predictions.append(float(p))

        if outcome is not None:
            win = float(outcome > 0.5)
            self._outcome_window.append(win)
            self._adwin.update(win)

        return self._generate_report()

    # ------------------------------------------------------------------
    def _generate_report(self) -> DriftReport:
        report = DriftReport()
        n_live = len(self._live_predictions)

        if n_live < self.min_ref_samples // 4:
            report.notes.append(f"Insufficient live data ({n_live} samples); skipping drift check.")
            return report

        # --- Feature PSI ---
        if self._ref_features:
            for feat, ref_arr in self._ref_features.items():
                live_arr = np.array(list(self._live_features.get(feat, [])))
                if len(live_arr) < 30 or len(ref_arr) < 30:
                    continue
                score = psi(ref_arr, live_arr)
                report.feature_psi[feat] = round(score, 4)

            if report.feature_psi:
                report.max_feature_psi = max(report.feature_psi.values())
                high_psi = {f: v for f, v in report.feature_psi.items() if v > self.psi_warn}
                if high_psi:
                    report.notes.append(
                        f"PSI warning: {dict(list(high_psi.items())[:5])}"
                    )
                if report.max_feature_psi > self.psi_alert:
                    report.should_retrain = True
                    report.notes.append(
                        f"PSI ALERT: max_psi={report.max_feature_psi:.3f} > {self.psi_alert} → retrain recommended"
                    )

        # --- Prediction JSD ---
        if self._ref_predictions is not None and n_live >= 30:
            live_preds = np.array(list(self._live_predictions))
            report.prediction_jsd = jensen_shannon_divergence(
                self._ref_predictions, live_preds
            )
            if report.prediction_jsd > self.jsd_threshold:
                report.notes.append(
                    f"Prediction drift: JSD={report.prediction_jsd:.3f} > {self.jsd_threshold}"
                )
                report.should_retrain = True

        # --- ADWIN win-rate drift ---
        report.adwin_drift_detected = self._adwin.drift_detected
        report.rolling_win_rate     = self._adwin.current_mean()
        report.n_trades             = len(self._outcome_window)
        if self._adwin.drift_detected:
            report.notes.append(
                f"ADWIN: win-rate drift detected. "
                f"Current rolling WR = {report.rolling_win_rate:.2%} "
                f"(n={report.n_trades})"
            )
            report.should_retrain = True

        if report.should_retrain:
            logger.warning("DriftMonitor: RETRAIN TRIGGERED — %s", "; ".join(report.notes))
        elif report.notes:
            logger.info("DriftMonitor: %s", "; ".join(report.notes))

        return report

    # ------------------------------------------------------------------
    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame of latest PSI scores for all monitored features."""
        if not self._ref_features:
            return pd.DataFrame(columns=["feature", "psi", "status"])
        rows = []
        for feat, ref_arr in self._ref_features.items():
            live_arr = np.array(list(self._live_features.get(feat, [])))
            if len(live_arr) < 30:
                score, status = None, "insufficient_data"
            else:
                score = psi(ref_arr, live_arr)
                status = (
                    "alert"   if score > self.psi_alert else
                    "warning" if score > self.psi_warn  else
                    "stable"
                )
            rows.append({"feature": feat, "psi": score, "status": status})
        return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
