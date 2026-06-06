"""Start the live trading loop.

Usage::

    python -m scripts.run_live --symbols EURUSD GBPUSD --timeframe M5

Reads MT5 credentials from environment variables when present:
``MT5_ACCOUNT``, ``MT5_PASSWORD``, ``MT5_SERVER``, ``MT5_PATH``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.live.executor import LiveExecutor  # noqa: E402
from src.live.mt5_client import MT5Client  # noqa: E402
from src.models.inference import EnsembleModel  # noqa: E402
from src.strategy.risk_manager import RiskManager  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list (default: strategy.symbols from config)")
    parser.add_argument("--timeframe", default=None,
                        help="Override timeframe (default: strategy.default_timeframe)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--ensemble-dir", default="models_artifacts")
    parser.add_argument("--account", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    parser.add_argument("--mt5-path", default=None)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--no-model", action="store_true",
                        help="Run rule-based only, skip the ML ensemble")
    args = parser.parse_args()

    setup_logging("INFO")
    cfg_dir = Path(args.config_dir)
    det_cfg = load_yaml(cfg_dir / "detection_config.yaml")["detection"]
    risk_cfg = load_yaml(cfg_dir / "risk_config.yaml")["risk"]
    strat_cfg = load_yaml(cfg_dir / "strategy_config.yaml")["strategy"]

    # Resolve symbols / timeframe (CLI overrides config)
    symbols = args.symbols or strat_cfg.get("symbols", ["EURUSD"])
    timeframe = args.timeframe or strat_cfg.get("default_timeframe", "M15")
    print(f"Trading {len(symbols)} symbols on {timeframe}: {', '.join(symbols)}")

    # MT5 connection
    client = MT5Client()
    account = args.account or os.environ.get("MT5_ACCOUNT")
    password = args.password or os.environ.get("MT5_PASSWORD")
    server = args.server or os.environ.get("MT5_SERVER")
    path = args.mt5_path or os.environ.get("MT5_PATH")
    ok = client.connect(
        account=int(account) if account else None,
        password=password, server=server, path=path,
    )
    if not ok:
        print("MT5 connection failed — aborting")
        return

    info = client.account_info()
    starting_balance = float(info["balance"]) if info else float(strat_cfg.get("account_starting_balance", 10_000.0))

    risk_mgr = RiskManager(strat_cfg, risk_cfg, starting_balance)

    ensemble = None
    if not args.no_model:
        try:
            ensemble = EnsembleModel.from_dir(args.ensemble_dir)
            print("Loaded ensemble from", args.ensemble_dir)
        except FileNotFoundError:
            print("No ensemble found — running rule-based only")

    executor = LiveExecutor(
        symbols=symbols,
        timeframe=timeframe,
        ensemble=ensemble,
        risk_manager=risk_mgr,
        mt5_client=client,
        detection_cfg=det_cfg,
        risk_cfg=risk_cfg,
        strategy_cfg=strat_cfg,
        prob_threshold=strat_cfg.get("min_model_probability", 0.65),
    )
    try:
        executor.run(poll_seconds=args.poll_seconds)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
