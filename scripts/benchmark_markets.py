"""Multi-market benchmarking script.

Runs a backtest on every symbol that has a trained champion model,
produces a ranked leaderboard, saves the results to JSON, and prints
a formatted table to the terminal.

Usage::

    # Benchmark all symbols with existing champion models
    python scripts/benchmark_markets.py

    # Also train missing symbols first (2-year window)
    python scripts/benchmark_markets.py --train --data-years 2

    # Save results to a custom path
    python scripts/benchmark_markets.py --output logs/bench.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on path.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.models.registry import ModelRegistry
from src.models.inference import EnsembleModel
from src.utils.data_loader import load_csv
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.detection.fvg import detect_fvg
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from src.detection.orderblock import detect_order_blocks
from src.detection.structure import detect_bos, detect_choch
from src.features.builder import build_feature_pipeline
from src.features.labels import label_signals
from src.strategy.rule_based import generate_signals
from src.models.train_ensemble import train_walk_forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_configs():
    root = Path(__file__).parent.parent / "config"
    with open(root / "strategy_config.yaml") as f:
        strategy_cfg = yaml.safe_load(f)["strategy"]
    with open(root / "risk_config.yaml") as f:
        risk_cfg = yaml.safe_load(f)["risk"]
    with open(root / "detection_config.yaml") as f:
        det_cfg = yaml.safe_load(f)["detection"]
    with open(root / "model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)["model"]
    mkt_path = root / "market_config.yaml"
    market_cfg = {}
    if mkt_path.exists():
        with open(mkt_path) as f:
            market_cfg = yaml.safe_load(f).get("markets", {})
    return strategy_cfg, risk_cfg, det_cfg, model_cfg, market_cfg


def _run_detections(candles, det_cfg):
    return {
        "fvg": detect_fvg(candles, min_gap_atr=det_cfg["fvg_min_gap_atr"]),
        "order_blocks": detect_order_blocks(
            candles, min_move_atr=det_cfg["order_block_min_move_atr"],
            lookback=det_cfg.get("ob_lookback", 100),
        ),
        "liquidity_sweeps": detect_liquidity_sweeps(
            candles, lookback=det_cfg.get("liquidity_lookback", 50),
            threshold_atr=det_cfg.get("liquidity_sweep_atr_multiplier", 0.5),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "equal_levels": detect_equal_highs_lows(
            candles, tolerance_atr=det_cfg.get("equal_hl_tolerance_atr", 0.1),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "bos": detect_bos(
            candles, confirmation_bars=det_cfg["bos_confirmation_bars"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "choch": detect_choch(candles, swing_lookback=det_cfg.get("choch_swing_lookback", 5)),
    }


def _train_symbol(symbol, strategy_cfg, risk_cfg, det_cfg, model_cfg, data_years, registry):
    tf = strategy_cfg.get("default_timeframe", "M15")
    data_path = Path("data") / f"{symbol}_{tf}.csv"
    if not data_path.exists():
        logger.warning("No data for %s — skipping training", symbol)
        return None

    candles = load_csv(str(data_path))
    if candles is None or len(candles) < 500:
        logger.warning("%s: insufficient bars — skipping", symbol)
        return None

    # Trim to data_years.
    cutoff = candles.index[-1] - pd.DateOffset(years=data_years)
    candles = candles[candles.index >= cutoff]
    logger.info("Training %s: %d bars (%.1f yr)", symbol, len(candles), data_years)

    detections = _run_detections(candles, det_cfg)
    signals = generate_signals(candles, detections, {**risk_cfg, **strategy_cfg})
    if not signals:
        logger.warning("%s: no signals generated — skipping", symbol)
        return None

    from src.features.labels import signals_to_setups, label_signals
    from src.strategy.rule_based import signals_to_setups as sts
    setups = sts(signals)
    labels = label_signals(candles, setups)
    feats = build_feature_pipeline(candles, detections, normalize=True)

    common = feats.index.intersection(labels.index)
    feats = feats.loc[common]
    labels = labels.loc[common]

    result = train_walk_forward(
        feats, labels, model_cfg,
        output_dir="models_artifacts",
        symbol=symbol,
        data_years=data_years,
    )
    promoted = registry.evaluate_and_promote(
        symbol,
        artifacts=result["artifacts"],
        metrics={
            "avg_auc":    result["avg_auc"],
            "gt_score":   result["gt_score"],
            "data_start": result["data_start"],
            "data_end":   result["data_end"],
        },
        fold_results=result["folds"],
    )
    logger.info(
        "%s: avg_auc=%.4f gt=%.4f promoted=%s",
        symbol, result["avg_auc"], result["gt_score"], promoted,
    )
    return result


def _benchmark_symbol(symbol, strategy_cfg, risk_cfg, det_cfg, registry):
    tf = strategy_cfg.get("default_timeframe", "M15")
    data_path = Path("data") / f"{symbol}_{tf}.csv"
    if not data_path.exists():
        return None

    candles = load_csv(str(data_path))
    if candles is None or len(candles) < 500:
        return None

    model = registry.get_champion(symbol)
    detections = _run_detections(candles, det_cfg)
    signals = generate_signals(candles, detections, {**risk_cfg, **strategy_cfg})
    if not signals:
        return None

    results = run_backtest(candles, signals, risk_cfg, model=model)
    metrics = compute_metrics(results)

    auc = registry.champion_auc(symbol)
    return {
        "symbol":      symbol,
        "n_trades":    int(metrics.get("trades", 0)),
        "win_rate":    round(float(metrics.get("win_rate", 0)), 4),
        "profit_factor": round(float(metrics.get("profit_factor", 0)), 4),
        "sharpe":      round(float(metrics.get("sharpe", 0)), 4),
        "max_dd":      round(float(metrics.get("max_drawdown", 0)), 4),
        "net_pnl":     round(float(metrics.get("net_pnl", 0)), 2),
        "champion_auc": round(auc, 4) if auc is not None else None,
        "drift_alert": auc is not None and auc < 0.55,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark all market champions")
    parser.add_argument("--train", action="store_true",
                        help="Train missing symbols before benchmarking")
    parser.add_argument("--train-all", action="store_true",
                        help="Re-train ALL symbols (overwrites champions if better)")
    parser.add_argument("--data-years", type=int, default=3,
                        help="Years of historical data to use for training (default: 3)")
    parser.add_argument("--output", type=str, default="",
                        help="Path to save JSON results (default: logs/market_benchmark_<date>.json)")
    parser.add_argument("--symbols", nargs="*",
                        help="Specific symbols to benchmark (default: all in strategy_config)")
    args = parser.parse_args()

    strategy_cfg, risk_cfg, det_cfg, model_cfg, market_cfg = _load_configs()
    registry = ModelRegistry()
    symbols = args.symbols or strategy_cfg.get("symbols", [])

    # --- Optional training pass ---
    if args.train or args.train_all:
        for sym in symbols:
            if args.train_all or not registry.has_champion(sym):
                logger.info("=== Training %s ===", sym)
                _train_symbol(sym, strategy_cfg, risk_cfg, det_cfg, model_cfg,
                              args.data_years, registry)

    # --- Benchmark pass ---
    logger.info("=== Benchmarking %d symbols ===", len(symbols))
    rows = []
    for sym in symbols:
        logger.info("Benchmarking %s ...", sym)
        r = _benchmark_symbol(sym, strategy_cfg, risk_cfg, det_cfg, registry)
        if r:
            rows.append(r)
        else:
            logger.warning("%s: no data or no signals — skipped", sym)

    if not rows:
        logger.error("No results — check that data files exist in data/")
        return

    # Sort by Sharpe descending.
    rows.sort(key=lambda r: r.get("sharpe", 0), reverse=True)

    # --- Print table ---
    df = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 90)
    print("  MARKET BENCHMARK LEADERBOARD")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # Drift alerts
    alerts = [r["symbol"] for r in rows if r.get("drift_alert")]
    if alerts:
        print(f"\n⚠  MODEL DRIFT ALERT: {', '.join(alerts)} — champion AUC < 0.55, consider retraining")
    print()

    # --- Save JSON ---
    out_path = args.output
    if not out_path:
        Path("logs").mkdir(exist_ok=True)
        date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
        out_path = f"logs/market_benchmark_{date_str}.json"
    Path(out_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
