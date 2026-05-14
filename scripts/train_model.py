"""Train the ensemble model with walk-forward validation.

Usage::

    python -m scripts.train_model --symbol EURUSD --timeframe M5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.fvg import detect_fvg  # noqa: E402
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps  # noqa: E402
from src.detection.orderblock import detect_order_blocks  # noqa: E402
from src.detection.structure import detect_bos, detect_choch  # noqa: E402
from src.features.builder import build_feature_pipeline  # noqa: E402
from src.features.labels import create_labels  # noqa: E402
from src.models.train_ensemble import train_walk_forward  # noqa: E402
from src.strategy.rule_based import generate_signals, signals_to_setups  # noqa: E402
from src.utils.data_loader import load_data  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--output-dir", default="models_artifacts")
    parser.add_argument("--max-holding", type=int, default=24)
    args = parser.parse_args()

    setup_logging("INFO")
    cfg_dir = Path(args.config_dir)
    det_cfg = load_yaml(cfg_dir / "detection_config.yaml")["detection"]
    model_cfg = load_yaml(cfg_dir / "model_config.yaml")["model"]
    risk_cfg = load_yaml(cfg_dir / "risk_config.yaml")["risk"]

    candles = load_data(args.symbol, args.timeframe)
    if candles.empty:
        print("No data available")
        return
    print(f"Loaded {len(candles)} bars")

    detections = {
        "fvg": detect_fvg(candles, min_gap_atr=det_cfg["fvg_min_gap_atr"]),
        "order_blocks": detect_order_blocks(
            candles, min_move_atr=det_cfg["order_block_min_move_atr"],
            lookback=det_cfg["ob_lookback"],
        ),
        "liquidity_sweeps": detect_liquidity_sweeps(
            candles, lookback=det_cfg["liquidity_lookback"],
            threshold_atr=det_cfg.get("liquidity_sweep_atr_multiplier", 0.5),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "equal_levels": detect_equal_highs_lows(
            candles, swing_lookback=det_cfg["swing_lookback"],
        ),
        "bos": detect_bos(
            candles, confirmation_bars=det_cfg["bos_confirmation_bars"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "choch": detect_choch(candles, swing_lookback=det_cfg["choch_swing_lookback"]),
    }

    signals = generate_signals(candles, detections, risk_cfg)
    setups = signals_to_setups(signals)
    print(f"Generated {len(setups)} setups")

    if len(setups) < 30:
        print("Not enough setups to train — generate more data or relax detection thresholds")
        return

    labels_df = create_labels(setups, candles, max_holding_bars=args.max_holding)
    if labels_df.empty:
        print("No labels — exiting")
        return

    feats_full = build_feature_pipeline(candles, detections, normalize=True)
    feats = feats_full.iloc[labels_df["index"].values]
    feats.index = candles.index[labels_df["index"].values]
    y = pd.Series(labels_df["binary"].values, index=feats.index, name="target")

    print(f"Training on {len(feats)} samples; positive rate = {y.mean():.3f}")

    summary = train_walk_forward(feats, y, model_cfg, output_dir=args.output_dir)
    print("Training summary:")
    for fold in summary["folds"]:
        print(f"  fold {fold['fold']}: AUC={fold['auc_test']:.4f}  "
              f"n_test={fold['n_test']}")
    print("Saved ensemble to", summary["ensemble_path"])


if __name__ == "__main__":
    main()
