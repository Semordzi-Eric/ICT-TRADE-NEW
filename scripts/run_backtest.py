"""Run a complete backtest end-to-end, single or multi-symbol.

Pipeline (per symbol):

    load data → run detections → generate rule-based signals → backtest → metrics

Usage::

    # Single symbol (legacy)
    python -m scripts.run_backtest --symbol EURUSD --timeframe M15 --plot

    # Multi-symbol (uses strategy.symbols from config by default)
    python -m scripts.run_backtest --multi --plot

    # Multi-symbol explicit list
    python -m scripts.run_backtest --symbols EURUSD GBPUSD USDJPY --timeframe M15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import run_backtest  # noqa: E402
from src.backtest.metrics import compute_metrics  # noqa: E402
from src.detection.fvg import detect_fvg  # noqa: E402
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps  # noqa: E402
from src.detection.orderblock import detect_order_blocks  # noqa: E402
from src.detection.structure import detect_bos, detect_choch  # noqa: E402
from src.strategy.rule_based import generate_signals  # noqa: E402
from src.utils.data_loader import load_data  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def backtest_one(
    symbol: str,
    timeframe: str,
    det_cfg: dict,
    risk_cfg: dict,
    strat_cfg: dict,
    starting_balance: float,
):
    """Run the full pipeline for a single symbol. Returns (result, metrics) or (None, None)."""
    candles = load_data(symbol, timeframe)
    if candles is None or candles.empty:
        print(f"  [{symbol}] no data — skipping")
        return None, None
    print(f"  [{symbol}] loaded {len(candles)} bars")

    detections = {
        "fvg": detect_fvg(candles, min_gap_atr=det_cfg["fvg_min_gap_atr"]),
        "order_blocks": detect_order_blocks(
            candles,
            min_move_atr=det_cfg["order_block_min_move_atr"],
            lookback=det_cfg["ob_lookback"],
        ),
        "liquidity_sweeps": detect_liquidity_sweeps(
            candles,
            lookback=det_cfg["liquidity_lookback"],
            threshold_atr=det_cfg.get("liquidity_sweep_atr_multiplier", 0.5),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "equal_levels": detect_equal_highs_lows(
            candles,
            tolerance_atr=det_cfg["equal_hl_tolerance_atr"],   # fixed: was erroneously * 100
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "bos": detect_bos(
            candles,
            confirmation_bars=det_cfg["bos_confirmation_bars"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "choch": detect_choch(
            candles, swing_lookback=det_cfg["choch_swing_lookback"]
        ),
    }
    # Merge strategy config into risk config so generate_signals sees
    # killzone_only, killzone_windows, and any other strategy-level knobs.
    combined_cfg = {**risk_cfg, **strat_cfg}
    signals = generate_signals(candles, detections, combined_cfg)
    print(f"  [{symbol}] {len(signals)} signals "
          f"(FVG {len(detections['fvg'])}, OB {len(detections['order_blocks'])}, "
          f"Sweeps {len(detections['liquidity_sweeps'])}, BOS {len(detections['bos'])}, "
          f"CHoCH {len(detections['choch'])})")
    if not signals:
        return None, None

    result = run_backtest(
        candles, signals,
        starting_balance=starting_balance,
        risk_per_trade=strat_cfg["risk_per_trade"],
    )
    metrics = compute_metrics(result)
    return result, metrics


def aggregate(per_symbol: Dict[str, dict]) -> dict:
    """Combine per-symbol metrics into a portfolio-level summary."""
    if not per_symbol:
        return {}
    total_trades = sum(m.get("n_trades", 0) for m in per_symbol.values())
    total_wins   = sum(int(m.get("n_trades", 0) * m.get("win_rate", 0))
                       for m in per_symbol.values())
    avg_sharpe   = sum(m.get("sharpe", 0) for m in per_symbol.values()) / len(per_symbol)
    avg_pf       = sum(m.get("profit_factor", 0) for m in per_symbol.values()) / len(per_symbol)
    total_net    = sum(m.get("net_pnl", 0) for m in per_symbol.values())
    worst_dd     = min(m.get("max_drawdown_pct", 0) for m in per_symbol.values())
    return {
        "symbols":           len(per_symbol),
        "total_trades":      total_trades,
        "blended_win_rate":  total_wins / total_trades if total_trades else 0.0,
        "avg_sharpe":        avg_sharpe,
        "avg_profit_factor": avg_pf,
        "summed_net_pnl":    total_net,
        "worst_max_dd_pct":  worst_dd,
    }


def estimate_daily_trades(per_symbol: Dict[str, dict], candles_by_symbol: Dict[str, "pd.DataFrame"]) -> float:
    """Roughly estimate daily trade frequency across the portfolio."""
    total_trades = sum(m.get("n_trades", 0) for m in per_symbol.values())
    days_seen = 0
    for sym, df in candles_by_symbol.items():
        if df is None or df.empty:
            continue
        # Use index span in days
        span = (df.index[-1] - df.index[0]).total_seconds() / 86400
        if span > days_seen:
            days_seen = span
    return total_trades / days_seen if days_seen else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICT bot backtest")
    parser.add_argument("--symbol", default=None,
                        help="Single-symbol mode")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Multi-symbol explicit list")
    parser.add_argument("--multi", action="store_true",
                        help="Use the symbol list from strategy_config.yaml")
    parser.add_argument("--timeframe", default=None,
                        help="Default: strategy.default_timeframe (M15)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Optional path to write metrics JSON")
    args = parser.parse_args()

    setup_logging("INFO")
    cfg_dir = Path(args.config_dir)
    det_cfg = load_yaml(cfg_dir / "detection_config.yaml")["detection"]
    risk_cfg = load_yaml(cfg_dir / "risk_config.yaml")["risk"]
    strat_cfg = load_yaml(cfg_dir / "strategy_config.yaml")["strategy"]

    # Resolve symbol list
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    elif args.multi:
        symbols = strat_cfg.get("symbols", ["EURUSD"])
    else:
        # Default: single EURUSD for back-compat
        symbols = ["EURUSD"]

    timeframe = args.timeframe or strat_cfg.get("default_timeframe", "M15")
    print(f"Backtesting {len(symbols)} symbol(s) on {timeframe}: {', '.join(symbols)}\n")

    per_symbol_metrics: Dict[str, dict] = {}
    per_symbol_candles: Dict[str, "pd.DataFrame"] = {}
    per_symbol_results = {}

    for sym in symbols:
        print(f"--- {sym} ---")
        # Re-load candles inside backtest_one, but also keep a reference for stats
        try:
            cand = load_data(sym, timeframe)
        except Exception as exc:
            print(f"  [{sym}] load error: {exc}")
            continue
        per_symbol_candles[sym] = cand

        result, metrics = backtest_one(
            sym, timeframe, det_cfg, risk_cfg, strat_cfg, args.starting_balance
        )
        if metrics is None:
            continue
        per_symbol_metrics[sym] = metrics
        per_symbol_results[sym] = result

        print(f"  trades={metrics.get('n_trades', 0)}  "
              f"win_rate={metrics.get('win_rate', 0):.2%}  "
              f"pf={metrics.get('profit_factor', 0):.2f}  "
              f"sharpe={metrics.get('sharpe', 0):.2f}  "
              f"net_pnl={metrics.get('net_pnl', 0):+,.2f}  "
              f"max_dd={metrics.get('max_drawdown_pct', 0):.2%}")

    if not per_symbol_metrics:
        print("\nNo successful backtests — exiting")
        return

    # ------ Portfolio summary -------------------------------------------------
    agg = aggregate(per_symbol_metrics)
    daily = estimate_daily_trades(per_symbol_metrics, per_symbol_candles)

    print("\n=== Portfolio Summary ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.4f}")
        else:
            print(f"  {k:25s} {v}")
    print(f"  {'estimated_trades_per_day':25s} {daily:.2f}")

    # ------ Output JSON -------------------------------------------------------
    if args.output:
        payload = {"per_symbol": per_symbol_metrics,
                   "aggregate":  agg,
                   "estimated_trades_per_day": daily}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nWrote metrics JSON -> {args.output}")

    # ------ Plotting ----------------------------------------------------------
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            n = len(per_symbol_results)
            fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
            for ax, (sym, res) in zip(axes[:, 0], per_symbol_results.items()):
                res.equity_curve.plot(ax=ax)
                ax.set_title(f"{sym} — {timeframe}")
                ax.set_ylabel("Equity")
                ax.grid(alpha=0.3)
            plt.tight_layout()
            out_png = Path("logs") / f"equity_multi_{timeframe}.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_png, dpi=120)
            print(f"Saved equity curves -> {out_png}")
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    main()
