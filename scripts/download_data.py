"""Download historical OHLCV data for the bot's symbols.

Usage::

    # Default: pulls strategy.symbols from config on M15
    python -m scripts.download_data

    # Override
    python -m scripts.download_data --symbols EURUSD GBPUSD NAS100 SPX500 --timeframe M5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.data_loader import load_data  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OHLCV data")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override (default: strategy.symbols from config)")
    parser.add_argument("--timeframe", default=None,
                        help="Default: strategy.default_timeframe (M15)")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download, ignoring local cache")
    args = parser.parse_args()

    setup_logging("INFO")

    # Pull defaults from config when not overridden
    with open(Path(args.config_dir) / "strategy_config.yaml") as f:
        strat_cfg = yaml.safe_load(f)["strategy"]
    symbols = args.symbols or strat_cfg.get("symbols", ["EURUSD"])
    timeframe = args.timeframe or strat_cfg.get("default_timeframe", "M15")

    print(f"Downloading {len(symbols)} symbol(s) on {timeframe}: {', '.join(symbols)}\n")

    for sym in symbols:
        df = load_data(
            symbol=sym,
            timeframe=timeframe,
            count=args.count,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
        )
        print(f"{sym}: {len(df)} rows  range: "
              f"{df.index.min() if len(df) else 'n/a'} -> "
              f"{df.index.max() if len(df) else 'n/a'}")


if __name__ == "__main__":
    main()
