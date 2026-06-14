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
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    from src.features.labels import create_labels
    from src.strategy.rule_based import signals_to_setups as sts
    setups = sts(signals)
    labels_df = create_labels(setups, candles)
    if labels_df.empty:
        logger.warning("%s: no labels generated — skipping", symbol)
        return None

    feats_full = build_feature_pipeline(candles, detections, normalize=True)
    feats = feats_full.iloc[labels_df["index"].values]
    feats.index = candles.index[labels_df["index"].values]

    import pandas as pd
    labels = pd.Series(labels_df["binary"].values, index=feats.index, name="target")
    # Extract confidence scores: fast wins → 1.0, time-stops → 0.0
    confidence = pd.Series(labels_df["confidence"].values, index=feats.index, name="confidence")

    result = train_walk_forward(
        feats, labels, model_cfg,
        output_dir="models_artifacts",
        symbol=symbol,
        data_years=data_years,
        confidence=confidence,
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
        "n_trades":    int(metrics.get("n_trades", 0)),
        "win_rate":    round(float(metrics.get("win_rate", 0)), 4),
        "profit_factor": round(float(metrics.get("profit_factor", 0)), 4),
        "sharpe":      round(float(metrics.get("sharpe", 0)), 4),
        "max_dd":      round(float(metrics.get("max_drawdown_pct", 0)), 4),
        "net_pnl":     round(float(metrics.get("net_pnl", 0)), 2),
        "champion_auc": round(auc, 4) if auc is not None else None,
        "drift_alert": auc is not None and auc < 0.55,
    }


# ---------------------------------------------------------------------------
# Parallel worker helpers (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _train_symbol_worker(args_tuple):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    sym, strategy_cfg, risk_cfg, det_cfg, model_cfg, data_years = args_tuple
    # Re-import registry inside worker process
    from src.models.registry import ModelRegistry as _Reg
    registry = _Reg()
    return sym, _train_symbol(sym, strategy_cfg, risk_cfg, det_cfg, model_cfg,
                              data_years, registry)


def _benchmark_symbol_worker(args_tuple):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    sym, strategy_cfg, risk_cfg, det_cfg = args_tuple
    from src.models.registry import ModelRegistry as _Reg
    registry = _Reg()
    return _benchmark_symbol(sym, strategy_cfg, risk_cfg, det_cfg, registry)


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
    parser.add_argument("--workers", type=int, default=0,
                        help="Max parallel workers (default: auto = min(CPU count, symbols))")
    args = parser.parse_args()

    strategy_cfg, risk_cfg, det_cfg, model_cfg, market_cfg = _load_configs()
    registry = ModelRegistry()
    symbols = args.symbols or strategy_cfg.get("symbols", [])

    # Determine worker count
    cpu_count = os.cpu_count() or 1
    max_workers = args.workers if args.workers > 0 else min(cpu_count, len(symbols))
    max_workers = max(1, max_workers)

    # --- Optional training pass (parallel) ---
    if args.train or args.train_all:
        to_train = [
            sym for sym in symbols
            if args.train_all or not registry.has_champion(sym)
        ]
        if to_train:
            logger.info(
                "=== Training %d symbol(s) in parallel (workers=%d) ===",
                len(to_train), min(max_workers, len(to_train)),
            )
            worker_args = [
                (sym, strategy_cfg, risk_cfg, det_cfg, model_cfg, args.data_years)
                for sym in to_train
            ]
            with ProcessPoolExecutor(max_workers=min(max_workers, len(to_train))) as pool:
                futures = {pool.submit(_train_symbol_worker, a): a[0] for a in worker_args}
                for future in as_completed(futures):
                    sym = futures[future]
                    try:
                        sym_out, result = future.result()
                        if result:
                            logger.info("=== Finished training %s ===", sym_out)
                        else:
                            logger.warning("=== Training skipped for %s ===", sym_out)
                    except Exception as exc:
                        logger.error("=== Training FAILED for %s: %s ===", sym, exc)

    # --- Benchmark pass (parallel) ---
    logger.info("=== Benchmarking %d symbols (workers=%d) ===",
                len(symbols), min(max_workers, len(symbols)))
    rows = []
    bench_args = [
        (sym, strategy_cfg, risk_cfg, det_cfg)
        for sym in symbols
    ]
    with ProcessPoolExecutor(max_workers=min(max_workers, len(symbols))) as pool:
        futures = {pool.submit(_benchmark_symbol_worker, a): a[0] for a in bench_args}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                r = future.result()
                if r:
                    rows.append(r)
                    logger.info("Benchmarked %s", sym)
                else:
                    logger.warning("%s: no data or no signals — skipped", sym)
            except Exception as exc:
                logger.error("Benchmark FAILED for %s: %s", sym, exc)

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
