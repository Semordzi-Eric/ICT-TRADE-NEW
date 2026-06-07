# ICT Trading Bot — Institutional-Grade Algorithmic Trader

> **Full ICT (Inner Circle Trader) concept implementation** — Fair Value Gaps, Order Blocks, Liquidity Sweeps, Market Structure, Killzones, and a 3-model ML ensemble running 24/7 across 16 markets.

---

## 🚀 Web Dashboard

Everything you need is accessible from the browser — no CLI required.

```bash
python main.py --dashboard
# → Open http://localhost:5000
```

| Page | What You Can Do |
|---|---|
| **Market Overview** | Live status grid for all 16 symbols — model AUC, sentiment score, news blackout, spread |
| **Sentiment** | Multi-source sentiment scores (-1 to +1), upcoming high-impact events, trade clearance status |
| **Supported Markets** | ICT compatibility guide for every asset class |
| **Data** | Download 2–5yr historical data per symbol (MT5 preferred, yfinance fallback) |
| **Train** | Walk-forward ensemble training — LightGBM + XGBoost + LSTM, per-symbol or all 16 at once |
| **Backtest** | Simulate strategy with equity curve, trade table, and detection counts |
| **Benchmark** | Full market leaderboard ranked by Sharpe — drift alerts for stale models |
| **Live Trading** | Start/stop 24/7 executor with Intuition Mode toggle and real-time log |
| **Model Registry** | Champion leaderboard, promote challengers, trigger retraining |
| **Setup / MT5** | Connect to MetaTrader 5, view account balance and equity |

---

## 📊 Supported Markets (16 Symbols)

### ✅ Native — Works Out of the Box

| Market | Symbols | Why It Works |
|---|---|---|
| **FX Majors** | EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD | Native — built for this. Highest ICT liquidity pools & OBs |
| **FX Minors** | EURGBP, EURJPY, GBPJPY | Same pip/OHLCV structure — same detection engine |
| **Gold (XAUUSD)** | Gold vs USD | Very ICT-friendly — ICT himself trades gold heavily. High-liquidity sweeps, crystal-clear OBs |
| **Silver (XAGUSD)** | Silver vs USD | Same as gold, slightly noisier — ATR adapts automatically |

### ⚡ Works — Small Tweaks Applied

| Market | Symbols | What Was Changed |
|---|---|---|
| **Equity Indices (CFD)** | NAS100, SPX500, US30 | `pip_size=1.0`, `contract_value=1.0`, US equity session gate (13:30–20:00 UTC), spread relaxed to 10–20pts |
| **Crypto CFDs (via MT5)** | BTCUSD, ETHUSD | 24/7 session (no killzone blocking), spread relaxed to 50pts, 3yr training window |
| **Crude Oil / WTI** | USOIL, UKOIL | High ATR — detection auto-adapts. Add to `symbols` list, reduce `risk_per_trade` to 0.002 |

### Adding Any Symbol

```yaml
# config/strategy_config.yaml
symbols:
  - EURUSD
  - GBPUSD
  - XAUUSD    # ← Gold (already configured)
  - USOIL     # ← Crude oil (add manually)
```

```yaml
# config/risk_config.yaml
contract_value_per_symbol:
  USOIL: 1000.0   # 1 lot = 1000 barrels
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Web Dashboard (Flask)                       │
│  Data │ Train │ Backtest │ Benchmark │ Live │ Registry          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    Live Executor (24/7)                          │
│  ThreadPoolExecutor — all 16 symbols simultaneously             │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ MT5 Feed │  │Detection │  │ ML Infer │  │ Intuition Mode│  │
│  │ M15 bars │  │FVG/OB/   │  │LightGBM  │  │ 9-factor ICT  │  │
│  │          │  │Sweeps/BOS│  │+XGBoost  │  │ confluence    │  │
│  │          │  │/CHoCH    │  │+LSTM     │  │ scorer        │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ News Filter  │  │ Sentiment    │  │ Risk Manager        │  │
│  │ FF XML feed  │  │ Engine       │  │ Adaptive sizing     │  │
│  │ 30min blackout│ │ Multi-source │  │ Correlation filter  │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                     Model Registry                               │
│  models_artifacts/EURUSD/ensemble.pkl (champion forever)        │
│  models_artifacts/XAUUSD/ensemble.pkl                           │
│  models_artifacts/BTCUSD/ensemble.pkl   ...                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ICT Strategy

The bot implements the following ICT concepts on **M15** (default):

| Concept | Implementation |
|---|---|
| **Fair Value Gaps (FVG)** | 3-candle gap detection, ATR-normalised minimum gap |
| **Order Blocks (OB)** | Last bearish/bullish candle before impulsive move, tracks mitigation |
| **Liquidity Sweeps** | Equal highs/lows broken then reversed (stop-hunt detection) |
| **Break of Structure (BOS)** | Swing high/low broken — trend continuation |
| **Change of Character (CHoCH)** | Counter-swing break — trend reversal signal |
| **Killzones** | London Open (08–10 UTC) + NY Open (13–15:30 UTC) session gates |
| **Higher-Timeframe Bias** | H4 structure must align with M15 signal direction |
| **Intuition Mode** | 9-factor confluence scorer — fires below ML threshold when ≥8 ICT factors stack |

### Killzones (UTC)

| Session | Window | What Happens |
|---|---|---|
| **London Open** | 08:00–10:00 | OB sweeps from Asian range, main entry window |
| **NY Open** | 13:00–15:30 | Second major liquidity sweep, continuation trades |
| **Crypto** | 24/7 | No killzone restriction — Intuition Mode handles signal quality |
| **Indices** | 13:30–20:00 | US equity session gate only |

### Typical Trade Count

| Scenario | Trades/Day |
|---|---|
| 3 FX majors, strict ML (≥0.65) | 1–3 |
| 16 markets, Intuition Mode ON | 5–12 |
| Max (all symbols, low threshold) | 15–30 |

---

## 🤖 ML Ensemble

```
Features (60+)     →  LightGBM  ─┐
(FVG/OB/sweep/     →  XGBoost   ─┼→  Logistic Meta-Model  →  Probability [0,1]
 BOS/CHoCH/        →  LSTM      ─┘
 session/ATR/RSI)
```

- **Walk-forward cross-validation** — no lookahead bias
- **Per-symbol champion models** saved to `models_artifacts/<SYMBOL>/`
- **Auto-promotion** — new models only replace champions when GT-score is higher
- **Drift detection** — AUC < 0.55 triggers retrain alert in dashboard

### Intuition Mode Confluence Factors (max 14 pts)

| Factor | Points |
|---|---|
| ML probability ≥ 0.65 | 3 |
| ML probability ≥ 0.55 (below normal) | 2 |
| H4 bias aligned | 1 |
| Inside London/NY killzone | 1 |
| Liquidity sweep in last 6 bars | 1 |
| Active Order Block + FVG both aligned | 2 |
| CHoCH within last 20 bars | 1 |
| Sentiment score aligned (>±0.15) | 1 |
| Spread < 1.5 pips | 1 |
| Price inside Asian range | 1 |
| **Threshold to fire** | **≥ 8** |

---

## 🗞️ News & Sentiment Engine

| Source | Coverage | Refresh |
|---|---|---|
| Forex Factory XML | Weekly economic calendar (high-impact events) | Every 12h |
| Reuters / BBC RSS | Keyword-based headline polarity | Every 1h |
| CryptoPanic API | BTC/ETH news sentiment | Every 1h |
| Pre-event warning | 60min before high-impact event → 50% risk reduction | Live |
| Blackout window | 30min around event → trading paused | Live |

---

## ⚙️ Configuration

### `config/strategy_config.yaml`
- `symbols` — list of all 16 markets to trade
- `killzone_windows` — London + NY session times (UTC)
- `risk_per_trade` — base risk per trade (0.35% default)
- `intuition_mode.enabled` — enable/disable Intuition Mode
- `intuition_mode.threshold_score` — confluence points required (default: 8)

### `config/risk_config.yaml`
- `max_open_positions` — portfolio cap (default: 10)
- `max_spread_pips_override` — per-symbol spread limits
- `contract_value_per_symbol` — notional per lot per instrument

### `config/market_config.yaml`
- Per-symbol metadata: pip_size, category, session_type, active_hours
- Sentiment keywords per symbol
- Training data year range per symbol

---

## 📦 Installation

```bash
git clone <repo>
cd ICT-TRADE-NEW
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Launch dashboard
python main.py --dashboard
```

> **MetaTrader 5 required** for live trading and data download. The bot attaches to the already-open terminal — no credentials needed if you're logged in.

---

## 🧪 Testing

```bash
venv\Scripts\pytest tests/ -v
# Expected: 39 passed
```

| Test File | Coverage |
|---|---|
| `test_detection.py` | FVG, OB, sweeps, BOS, CHoCH, session classification |
| `test_features.py` | Feature pipeline, normalisation, correlation filter |
| `test_registry.py` | Model promotion, leaderboard, manual approval mode |
| `test_intuition.py` | All 9 confluence factors, risk multiplier scaling |
| `test_sentiment.py` | Score clamping, news blackout, pre-event warning |

---

## 📁 Project Structure

```
ICT-TRADE-NEW/
├── config/
│   ├── strategy_config.yaml      # Symbols, killzones, intuition mode
│   ├── risk_config.yaml          # Position sizing, spread limits
│   ├── market_config.yaml        # Per-symbol metadata (NEW)
│   ├── detection_config.yaml     # FVG/OB/sweep thresholds
│   └── model_config.yaml         # ML hyperparameters
├── src/
│   ├── detection/                # FVG, OB, Sweeps, BOS, CHoCH
│   ├── features/                 # Feature pipeline, labels
│   ├── models/
│   │   ├── train_ensemble.py     # Walk-forward trainer
│   │   ├── inference.py          # EnsembleModel + from_registry()
│   │   └── registry.py           # Per-symbol champion registry (NEW)
│   ├── strategy/
│   │   ├── rule_based.py         # Signal generation
│   │   ├── risk_manager.py       # Adaptive sizing, portfolio gates
│   │   └── intuition_mode.py     # ICT confluence scorer (NEW)
│   ├── utils/
│   │   ├── news_filter.py        # ForexFactory blackout filter
│   │   └── sentiment_engine.py   # Multi-source sentiment (NEW)
│   └── live/
│       ├── executor.py           # 24/7 parallel executor (v2)
│       └── mt5_client.py         # MetaTrader 5 interface
├── dashboard/
│   ├── app.py                    # Flask API (all operations)
│   ├── templates/index.html      # Full SPA dashboard
│   └── static/css/ + js/         # Premium dark UI
├── scripts/
│   └── benchmark_markets.py      # CLI market leaderboard
├── tests/                        # 39 unit tests
├── models_artifacts/             # Per-symbol champion models
│   └── EURUSD/ensemble.pkl
└── data/                         # Downloaded CSV files
```

---

## ⚠️ Risk Disclaimer

This software is for **educational and research purposes only**. Algorithmic trading involves substantial risk of loss. Always:
- Test thoroughly on a **demo account** before going live
- Never risk capital you cannot afford to lose
- Monitor the bot regularly — no algorithm is infallible
- Consult a qualified financial advisor before trading

---

*Built on ICT methodology — Fair Value Gaps, Order Blocks, Liquidity Sweeps, Market Structure*
