# ICT Trading Bot

A production-oriented Python framework that turns the **Inner Circle Trader (ICT)** discretionary
playbook into a quantitative system. It detects key ICT structures, builds a 35-feature
representation, ranks setups with a stacked ML ensemble, backtests vectorized over years of data,
and can execute live through MetaTrader 5 (via the Python bridge or via ONNX/MQL5).

> ⚠️ **Educational use only.** Trading FX/CFD/futures involves substantial risk. Past performance
> does not guarantee future results. Validate everything on your own data before risking capital.

---

## 1. What's inside

```
ict_trading_bot/
├── config/                      # YAML configuration (detection, strategy, model, risk)
├── src/
│   ├── detection/               # FVG, order blocks, liquidity sweeps, BOS/CHoCH, sessions
│   ├── features/                # 35-feature builder + triple-barrier labelling
│   ├── strategy/                # Rule-based signal generator + risk manager
│   ├── models/                  # LightGBM + XGBoost + LSTM stacked ensemble (walk-forward)
│   ├── backtest/                # Vectorized backtester + performance metrics
│   ├── live/                    # MT5 client, ONNX export, live executor
│   └── utils/                   # Data loaders, logging
├── scripts/                     # download_data / run_backtest / train_model / run_live
├── tests/                       # Pytest suites for detection & features
├── notebooks/                   # Exploratory notebook
├── main.py                      # CLI dispatcher
├── requirements.txt
└── README.md
```

---

## 2. Installation

```bash
# 1. Clone / unzip
cd ict_trading_bot

# 2. Create a virtual env
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Optional dependencies** — the system degrades gracefully if any of these are missing:

| Package              | Needed for                                   |
| -------------------- | -------------------------------------------- |
| `MetaTrader5`        | Live trading + MT5 historical data (Windows) |
| `tensorflow`         | LSTM component of the ensemble               |
| `lightgbm`           | LightGBM component                           |
| `xgboost`            | XGBoost component                            |
| `scikit-optimize`    | Bayesian hyperparameter tuning               |
| `onnx*` / `skl2onnx` | Exporting models to ONNX for MQL5            |
| `yfinance`           | Free data fallback when MT5 isn't available  |

If a component is unavailable, the framework prints a warning and skips it — you can still run
detections, rule-based signals, and backtests.

---

## 3. Configuration

All knobs live in `config/*.yaml`. Highlights:

- `detection_config.yaml` — ATR period, sweep thresholds, FVG/OB minimum sizes, session hours
- `strategy_config.yaml` — risk per trade (default **0.5%**), daily loss limit, weekly loss limit,
  correlation filter pairs
- `model_config.yaml` — LightGBM/XGBoost/LSTM hyperparameters, walk-forward windows
  (12mo train / 3mo val / 3mo test by default), ensemble blend weights `[0.5, 0.3, 0.2]`
- `risk_config.yaml` — SL ATR multiplier, TP R-multiple, max DD halt, position sizing model

---

## 4. Typical workflow

> **Default universe:** the framework now ships configured for **6 FX majors on M15**
> (`EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD`). Risk per trade is **0.35%** with up
> to **6 concurrent positions**, capping simultaneous risk at ~2.1%. Expected aggregate is
> **~3.5–4 trades/day** — enough for daily activity even though each pair only fires
> ~0.65 times/day on its own. Edit `config/strategy_config.yaml` to change the universe.

### 4.1 Download data

```bash
# Pulls all 6 majors on M15 by default
python main.py --download-data

# Or override
python main.py --download-data --symbols EURUSD GBPUSD --timeframe M15 --years 5
```

Data is cached as Parquet/CSV under `data/`. Source priority: MT5 → yfinance → user-provided CSV.

### 4.2 Backtest

```bash
# Multi-symbol portfolio backtest (recommended) — uses config universe
python main.py --backtest --multi --plot

# Single symbol (legacy)
python main.py --backtest --symbol EURUSD --timeframe M15 --plot

# Explicit list
python main.py --backtest --symbols EURUSD GBPUSD USDJPY --timeframe M15
```

The multi-symbol backtest prints per-symbol metrics and a portfolio summary including an
**estimated trades/day** figure so you can sanity-check signal frequency before going live.

### 4.3 Train the ensemble

```bash
python main.py --train --symbol EURUSD --timeframe M15
```

This runs **walk-forward training** over rolling windows. For each fold:

1. Detect ICT structures and generate rule-based setups
2. Triple-barrier labels (TP / SL / time-stop) → binary win/loss target
3. Build the 35-feature matrix at each setup index
4. Train LightGBM, XGBoost, LSTM in parallel
5. Train a Logistic Regression **meta-model** on out-of-fold predictions
6. Average the meta-model across folds → final stacked ensemble

Artifacts saved to `models_artifacts/`:

- `ensemble.pkl` — meta-model + per-fold base models
- `lstm.keras` — Keras LSTM weights
- `training_summary.json` — fold-by-fold metrics
- `feature_columns.json` — exact feature order (critical for ONNX inference)

### 4.4 Run live (MT5)

```bash
# Set credentials (or pass via CLI)
export MT5_ACCOUNT=12345678
export MT5_PASSWORD=********
export MT5_SERVER=ICMarkets-Demo
export MT5_PATH="C:/Program Files/MetaTrader 5/terminal64.exe"

# Default: trades the 6 majors from strategy_config.yaml on M15
python main.py --live

# Or override
python main.py --live --symbols EURUSD GBPUSD --timeframe M15
```

The `LiveExecutor` polls new bars, runs detection → features → ensemble inference → risk gates,
and submits orders via the MT5 Python API. If the ensemble isn't present, it falls back to
rule-based signals only.

---

## 5. Deploying to MQL5 via ONNX

The Python MT5 bridge is convenient but adds latency and a Python dependency on the trading PC.
For production EAs, export the LightGBM model to ONNX and load it directly inside an MQL5 EA:

```python
from src.live.onnx_export import export_lightgbm_to_onnx, verify_onnx
export_lightgbm_to_onnx(
    model_path="models_artifacts/lgbm_fold_last.pkl",
    out_path="models_artifacts/ict_model.onnx",
    n_features=35,
)
verify_onnx("models_artifacts/ict_model.onnx", n_features=35)
```

In MQL5:

```mql5
long handle = OnnxCreateFromBuffer(model_buffer, ONNX_DEFAULT);
matrix features(1, 35);            // populate with the SAME 35 features in the SAME order
matrix output(1, 2);
OnnxRun(handle, ONNX_DEFAULT, features, output);
double prob_win = output[0][1];    // class-1 probability
```

The exact feature order is in `models_artifacts/feature_columns.json` and in
`src/features/builder.py::FEATURE_COLUMNS`.

---

## 6. Risk management

`src/strategy/risk_manager.py` enforces, in order:

1. **Per-trade risk** — position size derived from stop distance and `risk_per_trade`
   (default **0.35%** so 6 concurrent positions cap total risk at ~2.1%)
2. **Daily / weekly loss limits** — auto-halt at 3% / 6% drawdown
3. **Consecutive-loss cooldown** — pause for 4h after 3 losers in a row
4. **Correlation filter** — refuses concurrent same-direction exposure to known
   correlated pairs (EURUSD+GBPUSD, AUDUSD+NZDUSD, etc. — see `strategy_config.yaml`)
5. **Max concurrent positions** — global cap of 6 (one per major)
6. **Hard max drawdown halt** — disables trading at 10% peak-to-trough until manually reset

These rules apply identically in backtest and live.

---

## 7. Tests

```bash
pytest -q
```

Covers detection determinism (FVG, sweeps, OBs, BOS/CHoCH, sessions) and feature pipeline
contract (35 columns, no NaNs/infs, correlation-filter behaviour).

---

## 8. Known limitations / next steps

- **Slippage model** is a flat 0.5 pip per side; replace with a per-symbol, time-of-day model
  for higher fidelity.
- **News filter** isn't included — wire in a high-impact news calendar before going live.
- **Multi-symbol portfolio backtests** run symbol-by-symbol; a true joint backtester with
  shared margin is out of scope.
- **Reinforcement learning agent** could replace the rule-based gate; the feature pipeline is
  already framed for it.

---

## 9. License

For personal / educational use. Do not redistribute trained models or live credentials.
