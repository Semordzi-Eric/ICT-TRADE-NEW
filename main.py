"""
ICT Trading Bot — main entry point.

Usage:
    python main.py --download-data --symbols EURUSD GBPUSD
    python main.py --backtest --symbol EURUSD --timeframe M15
    python main.py --train --symbol EURUSD --timeframe M15
    python main.py --live  --symbol EURUSD --timeframe M15
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def _run(script_name: str, forwarded: list[str]) -> None:
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    sys.argv = [str(script_path), *forwarded]
    runpy.run_path(str(script_path), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICT Trading Bot — entry point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--download-data", action="store_true", help="Download historical OHLCV data")
    grp.add_argument("--backtest",      action="store_true", help="Run a vectorized backtest")
    grp.add_argument("--train",         action="store_true", help="Train the ensemble model")
    grp.add_argument("--live",          action="store_true", help="Run the live executor (MT5)")

    # Capture remaining args verbatim for the dispatched script
    args, forwarded = parser.parse_known_args()

    if args.download_data:
        _run("download_data.py", forwarded)
    elif args.backtest:
        _run("run_backtest.py", forwarded)
    elif args.train:
        _run("train_model.py", forwarded)
    elif args.live:
        _run("run_live.py", forwarded)


if __name__ == "__main__":
    main()
