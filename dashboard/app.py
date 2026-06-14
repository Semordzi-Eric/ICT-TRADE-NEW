"""ICT Trading Bot — Dashboard API Server v2.

Fully self-contained web application. Every bot function is accessible
from the browser:
  • Data Download  — per-symbol progress, yfinance fallback, data-year range
  • Backtest       — multi-symbol, equity curve, trade table, detection counts
  • Train          — walk-forward, per-fold AUC, auto-promote via ModelRegistry
  • Model Registry — champion leaderboard, retrain button, drift alerts
  • Benchmark      — full market leaderboard
  • Live Trading   — start/stop, real-time log, Intuition Mode toggle
  • Market Overview— all 16 symbols, bias, spread, news status, sentiment
  • Sentiment      — live scores per symbol, upcoming news events
  • Setup / MT5    — connect, account info

Run with:
    python main.py --dashboard
or directly:
    python dashboard/app.py
"""
from __future__ import annotations

import collections
import hmac
import json
import logging
import os
import pickle
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics
from src.detection.fvg import detect_fvg
from src.detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from src.detection.orderblock import detect_order_blocks
from src.detection.structure import detect_bos, detect_choch
from src.strategy.rule_based import generate_signals
from src.utils.data_loader import load_data
from src.utils.logging_utils import setup_logging
from src.live.mt5_client import MT5Client
from src.models.registry import ModelRegistry
from src.utils.sentiment_engine import SentimentEngine

import yaml

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5000"}})

CONFIG_DIR  = ROOT / "config"
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models_artifacts"

# ── Shared singletons ─────────────────────────────────────────────────────────
_mt5: MT5Client = MT5Client()  # created but NOT connected until user explicitly connects
_mt5_lock = threading.Lock()

_registry = ModelRegistry(base_dir=str(MODELS_DIR))

_sentiment: Optional[SentimentEngine] = None
_sentiment_lock = threading.Lock()

_live_thread: Optional[threading.Thread] = None
_live_stop_event = threading.Event()
_live_log: collections.deque = collections.deque(maxlen=1000)
_live_intuition_enabled = True
_live_executor = None

# ── YAML config cache (avoid disk reads on every API request) ─────────────────
_yaml_cache: Dict[str, dict] = {}
_yaml_cache_lock = threading.Lock()

# ── API key authentication ────────────────────────────────────────────────────
# Set ICT_DASHBOARD_API_KEY in your environment to protect all /api/* endpoints.
# Omit or leave blank to disable auth (development mode only).
_API_KEY: str = os.environ.get("ICT_DASHBOARD_API_KEY", "")


def _get_mt5() -> MT5Client:
    """Return the MT5 client.  Does NOT auto-connect — the user must call
    /api/mt5/connect explicitly.  Auto-connecting without credentials would
    silently authenticate against whatever terminal is running."""
    return _mt5


def _get_sentiment() -> SentimentEngine:
    global _sentiment
    with _sentiment_lock:
        if _sentiment is None:
            mkt_cfg_path = str(CONFIG_DIR / "market_config.yaml")
            _sentiment = SentimentEngine(
                cache_dir=str(DATA_DIR),
                market_config_path=mkt_cfg_path,
            )
            _sentiment.start_background_refresh()
    return _sentiment


def _load_yaml(name: str) -> dict:
    """Load a YAML config file with in-process caching (avoids disk read per request)."""
    with _yaml_cache_lock:
        if name not in _yaml_cache:
            with open(CONFIG_DIR / name) as f:
                _yaml_cache[name] = yaml.safe_load(f)
        return _yaml_cache[name]


def _reload_yaml(name: str) -> dict:
    """Force a reload of a YAML config (call after config edits)."""
    with _yaml_cache_lock:
        _yaml_cache.pop(name, None)
    return _load_yaml(name)


def _check_api_key() -> Optional["Response"]:
    """Return a 401 Response if API key auth is enabled and the request key is wrong.
    Returns None if auth passes.
    """
    if not _API_KEY:
        return None  # auth disabled
    key = request.headers.get("X-API-Key") or request.args.get("api_key", "")
    if not hmac.compare_digest(key, _API_KEY):
        return jsonify({"error": "Unauthorized"}), 401
    return None



def _load_market_cfg() -> dict:
    p = CONFIG_DIR / "market_config.yaml"
    if p.exists():
        return _load_yaml("market_config.yaml").get("markets", {})
    return {}


def _detect(candles, det_cfg: dict) -> dict:
    return {
        "fvg": detect_fvg(candles, min_gap_atr=det_cfg["fvg_min_gap_atr"]),
        "order_blocks": detect_order_blocks(
            candles,
            min_move_atr=det_cfg["order_block_min_move_atr"],
            lookback=det_cfg.get("ob_lookback", 100),
        ),
        "liquidity_sweeps": detect_liquidity_sweeps(
            candles,
            lookback=det_cfg.get("liquidity_lookback", 50),
            threshold_atr=det_cfg.get("liquidity_sweep_atr_multiplier", 0.5),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "equal_levels": detect_equal_highs_lows(
            candles,
            tolerance_atr=det_cfg.get("equal_hl_tolerance_atr", 0.1),
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "bos": detect_bos(
            candles,
            confirmation_bars=det_cfg["bos_confirmation_bars"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "choch": detect_choch(candles, swing_lookback=det_cfg.get("choch_swing_lookback", 5)),
    }


def _sse(data) -> str:
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"data: {payload}\n\n"


def _stream_job(target_fn, *args, **kwargs):
    log_q: queue.Queue = queue.Queue()

    class _QH(logging.Handler):
        def emit(self, record):
            log_q.put({"type": "log", "msg": self.format(record)})

    handler = _QH()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    result_box: list = []
    error_box:  list = []

    def _run():
        try:
            result_box.append(target_fn(*args, **kwargs))
        except Exception as exc:
            error_box.append(str(exc))
        finally:
            log_q.put(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    while True:
        try:
            item = log_q.get(timeout=30)
        except queue.Empty:
            yield _sse({"type": "keepalive"})
            continue
        if item is None:
            break
        yield _sse(item)

    root_logger.removeHandler(handler)
    t.join()

    if error_box:
        yield _sse({"type": "error", "msg": error_box[0]})
    else:
        yield _sse({"type": "result", "data": result_box[0] if result_box else {}})
    yield _sse({"type": "done"})


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Config & Symbols ──────────────────────────────────────────────────────────

@app.route("/api/symbols")
def api_symbols():
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    market_cfg = _load_market_cfg()
    symbols = strat.get("symbols", [])
    result = []
    for sym in symbols:
        meta = market_cfg.get(sym, {})
        result.append({
            "symbol":   sym,
            "category": meta.get("category", "fx"),
            "session":  meta.get("session_type", "forex"),
            "pip_size": meta.get("pip_size", 0.0001),
        })
    return jsonify({"symbols": result})


@app.route("/api/config")
def api_config():
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    risk  = _load_yaml("risk_config.yaml")["risk"]
    return jsonify({"strategy": strat, "risk": risk})


# ── MT5 ───────────────────────────────────────────────────────────────────────

@app.route("/api/mt5/status")
def api_mt5_status():
    mt5 = _get_mt5()
    if not mt5.connected:
        return jsonify({"connected": False, "account": None})
    info = mt5.account_info() or {}
    return jsonify({
        "connected": True,
        "account": {
            "login":    info.get("login", "—"),
            "name":     info.get("name", "—"),
            "server":   info.get("server", "—"),
            "balance":  info.get("balance", 0.0),
            "equity":   info.get("equity", 0.0),
            "currency": info.get("currency", "USD"),
            "leverage": info.get("leverage", 0),
            "company":  info.get("company", "—"),
        },
    })


@app.route("/api/mt5/connect", methods=["POST"])
def api_mt5_connect():
    body = request.json or {}
    account  = body.get("account")  or None
    password = body.get("password") or None
    server   = body.get("server")   or None
    with _mt5_lock:
        if _mt5.connected:
            _mt5.disconnect()
        ok = _mt5.connect(
            account=int(account) if account else None,
            password=password, server=server,
        )
    if ok:
        info = _mt5.account_info() or {}
        return jsonify({
            "connected": True,
            "account": {
                "login":    info.get("login", "—"),
                "name":     info.get("name", "—"),
                "server":   info.get("server", "—"),
                "balance":  info.get("balance", 0.0),
                "equity":   info.get("equity", 0.0),
                "currency": info.get("currency", "USD"),
                "leverage": info.get("leverage", 0),
                "company":  info.get("company", "—"),
            },
        })
    return jsonify({"connected": False, "error": "MT5 not running or terminal is closed"}), 400


# ── Data ──────────────────────────────────────────────────────────────────────

@app.route("/api/data/download", methods=["POST"])
def api_data_download():
    body = request.json or {}
    strat     = _load_yaml("strategy_config.yaml")["strategy"]
    symbols   = body.get("symbols",   strat.get("symbols", ["EURUSD"]))
    timeframe = body.get("timeframe", strat.get("default_timeframe", "M15"))
    bars      = int(body.get("bars", 200_000))
    data_years = int(body.get("data_years", 5))

    def _run():
        import logging as _log
        import pandas as _pd
        from src.utils.data_loader import load_from_yfinance, load_from_mt5, load_csv

        # Only H4 and D1 have enough Yahoo Finance history to be a useful fallback.
        # All intraday timeframes (M1–H1) REQUIRE MT5 for meaningful history.
        YF_FALLBACK_TF  = {"H4": ("4h", "730d"), "D1": ("1d", "max")}
        is_intraday = timeframe.upper() not in YF_FALLBACK_TF

        mt5 = _get_mt5()

        # ── Gate: MT5 required for all intraday timeframes ───────────────────
        if is_intraday and not mt5.connected:
            msg = (
                f"MT5 is not connected — {timeframe} data requires MetaTrader 5. "
                "To fix: (1) Open MetaTrader 5. (2) Log into your broker account. "
                "(3) Enable 'Allow Algorithmic Trading' (robot icon in toolbar). "
                "(4) Come back here, open Setup > MT5, and click Connect."
            )
            _log.getLogger().error("MT5 not connected. %s data requires MetaTrader 5.", timeframe)
            return {sym: {"bars": 0, "source": "MT5 required", "error": msg} for sym in symbols}

        # ── Per-symbol download loop ─────────────────────────────────────────
        results = {}
        for symbol in symbols:
            source_label = "MT5" if mt5.connected else "yfinance"
            _log.getLogger().info(
                "Downloading %s %s (%d yr / up to %d bars) via %s...",
                symbol, timeframe, data_years, bars, source_label,
            )
            try:
                cache = DATA_DIR / f"{symbol}_{timeframe}.csv"
                df_existing = None
                if cache.exists():
                    try:
                        df_existing = load_csv(str(cache))
                    except Exception:
                        df_existing = None

                # Primary: MT5
                if mt5.connected:
                    df = load_from_mt5(symbol, timeframe, bars)
                    source = "MT5"

                # Fallback: yfinance for H4 / D1 only
                else:
                    interval, period = YF_FALLBACK_TF[timeframe.upper()]
                    df = load_from_yfinance(symbol, period=period, interval=interval)
                    source = "yfinance"
                    # Merge new bars into existing cache to accumulate history
                    if df_existing is not None and not df_existing.empty \
                            and df is not None and not df.empty:
                        new_only = df[df.index > df_existing.index[-1]]
                        if not new_only.empty:
                            df = _pd.concat([df_existing, new_only]).sort_index()
                            _log.getLogger().info(
                                "%s: appended %d new bars (total %d)",
                                symbol, len(new_only), len(df),
                            )
                        else:
                            df = df_existing
                            source = "cache (already up-to-date)"

                # Handle empty result
                if df is None or df.empty:
                    _log.getLogger().warning("%s: no data returned from %s", symbol, source)
                    if df_existing is not None and not df_existing.empty:
                        results[symbol] = {"bars": len(df_existing), "source": "cache (no new data)"}
                    else:
                        results[symbol] = {"bars": 0, "source": source,
                                           "error": f"No data returned for {symbol} {timeframe}."}
                    continue

                # Save to cache
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache, index_label="time")
                n = len(df)
                _log.getLogger().info("%s: saved %d bars from %s", symbol, n, source)
                results[symbol] = {"bars": n, "source": source}

            except Exception as e:
                _log.getLogger().error("%s download failed: %s", symbol, e)
                results[symbol] = {"bars": 0, "source": "error", "error": str(e)}
        return results

    return Response(_stream_job(_run), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/data/status")
def api_data_status():
    rows = []
    for csv in sorted(DATA_DIR.glob("*.csv")):
        parts = csv.stem.rsplit("_", 1)
        symbol = parts[0] if len(parts) == 2 else csv.stem
        tf = parts[1] if len(parts) == 2 else "?"
        try:
            with open(csv, "rb") as f:
                count = sum(1 for _ in f) - 1
        except Exception:
            count = -1
        stat = csv.stat()
        rows.append({
            "symbol": symbol, "timeframe": tf, "bars": count,
            "file": csv.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return jsonify({"datasets": rows})


# ── Backtest ──────────────────────────────────────────────────────────────────

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = request.json or {}
    symbols           = body.get("symbols", ["EURUSD"])
    timeframe         = body.get("timeframe", "M15")
    starting_balance  = float(body.get("starting_balance", 10_000))
    use_model         = bool(body.get("use_model", True))

    det_cfg  = _load_yaml("detection_config.yaml")["detection"]
    risk_cfg = _load_yaml("risk_config.yaml")["risk"]
    strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]

    def _run():
        results = {}
        for symbol in symbols:
            import logging as _log
            _log.getLogger().info("Backtesting %s %s", symbol, timeframe)
            candles = load_data(symbol, timeframe)
            if candles.empty:
                results[symbol] = {"error": "no data"}
                continue
            _log.getLogger().info("%s: %d bars loaded", symbol, len(candles))

            detections = _detect(candles, det_cfg)
            combined_cfg = {**risk_cfg, **strat_cfg}
            signals = generate_signals(candles, detections, combined_cfg)
            _log.getLogger().info("%s: %d signals", symbol, len(signals))

            if not signals:
                results[symbol] = {"error": "no signals"}
                continue

            # Load champion model if available
            model = _registry.get_champion(symbol) if use_model else None

            bt = run_backtest(candles, signals, starting_balance=starting_balance,
                              risk_per_trade=strat_cfg["risk_per_trade"])
            metrics = compute_metrics(bt)

            eq = bt.equity_curve
            step = max(1, len(eq) // 2000)
            eq_s = eq.iloc[::step]

            trades_df = bt.trades_df()
            trades = []
            if not trades_df.empty:
                trades = trades_df.to_dict(orient="records")
                for t in trades:
                    for k, v in t.items():
                        if hasattr(v, "isoformat"):
                            t[k] = str(v)

            auc = _registry.champion_auc(symbol)
            results[symbol] = {
                "metrics": {k: (round(float(v), 6) if isinstance(v, float) else v)
                            for k, v in metrics.items()},
                "equity_curve": {
                    "timestamps": [str(ts) for ts in eq_s.index],
                    "values": [round(float(v), 2) for v in eq_s.values],
                },
                "trades": trades,
                "detection_counts": {
                    "fvg": len(detections["fvg"]),
                    "order_blocks": len(detections["order_blocks"]),
                    "sweeps": len(detections["liquidity_sweeps"]),
                    "bos": len(detections["bos"]),
                    "choch": len(detections["choch"]),
                },
                "model_auc": auc,
                "model_used": model is not None,
            }
        return results

    return Response(_stream_job(_run), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Training ──────────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    body      = request.json or {}
    symbol    = body.get("symbol", "EURUSD")
    timeframe = body.get("timeframe", "M15")
    data_years = int(body.get("data_years", 3))
    auto_promote = bool(body.get("auto_promote", True))

    det_cfg   = _load_yaml("detection_config.yaml")["detection"]
    model_cfg = _load_yaml("model_config.yaml")["model"]
    risk_cfg  = _load_yaml("risk_config.yaml")["risk"]

    def _run():
        import logging as _log
        import pandas as pd
        from src.features.builder import build_feature_pipeline
        from src.features.labels import create_labels
        from src.models.train_ensemble import train_walk_forward
        from src.strategy.rule_based import signals_to_setups

        _log.getLogger().info("Loading data for %s %s", symbol, timeframe)
        candles = load_data(symbol, timeframe)
        if candles.empty:
            raise ValueError(f"No data for {symbol} {timeframe}")

        # Trim to data_years
        cutoff = candles.index[-1] - pd.DateOffset(years=data_years)
        candles = candles[candles.index >= cutoff]
        _log.getLogger().info("Training window: %d bars (%.1f yr)", len(candles), data_years)

        strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]
        detections = _detect(candles, det_cfg)
        combined_cfg = {**risk_cfg, **strat_cfg}
        signals = generate_signals(candles, detections, combined_cfg)
        setups = signals_to_setups(signals)
        _log.getLogger().info("Generated %d setups", len(setups))

        if len(setups) < 30:
            raise ValueError("Not enough setups — download more data")

        labels_df = create_labels(setups, candles)
        feats_full = build_feature_pipeline(candles, detections, normalize=True)
        feats = feats_full.iloc[labels_df["index"].values]
        feats.index = candles.index[labels_df["index"].values]
        y = pd.Series(labels_df["binary"].values, index=feats.index)

        _log.getLogger().info("Training on %d samples; positive_rate=%.3f",
                               len(feats), float(y.mean()))

        result = train_walk_forward(
            feats, y, model_cfg,
            output_dir=str(MODELS_DIR),
            symbol=symbol,
            data_years=data_years,
        )

        # Registry promotion
        promoted = False
        if auto_promote:
            promoted = _registry.evaluate_and_promote(
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
            _log.getLogger().info(
                "Registry: %s — avg_auc=%.4f gt=%.4f promoted=%s",
                symbol, result["avg_auc"], result["gt_score"], promoted,
            )

        return {
            "folds":        result["folds"],
            "avg_auc":      result["avg_auc"],
            "gt_score":     result["gt_score"],
            "data_start":   result["data_start"],
            "data_end":     result["data_end"],
            "promoted":     promoted,
            "ensemble_path": result["ensemble_path"],
        }

    return Response(_stream_job(_run), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/train/all", methods=["POST"])
def api_train_all():
    """Train all symbols sequentially (background SSE stream)."""
    body = request.json or {}
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    symbols    = body.get("symbols", strat.get("symbols", []))
    data_years = int(body.get("data_years", 3))

    det_cfg   = _load_yaml("detection_config.yaml")["detection"]
    model_cfg = _load_yaml("model_config.yaml")["model"]
    risk_cfg  = _load_yaml("risk_config.yaml")["risk"]

    def _run():
        import logging as _log
        import pandas as pd
        from src.features.builder import build_feature_pipeline
        from src.features.labels import create_labels
        from src.models.train_ensemble import train_walk_forward
        from src.strategy.rule_based import signals_to_setups

        summary = []
        strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]

        for symbol in symbols:
            _log.getLogger().info("=== Training %s ===", symbol)
            try:
                candles = load_data(symbol, strat_cfg.get("default_timeframe", "M15"))
                if candles.empty:
                    _log.getLogger().warning("%s: no data — skipped", symbol)
                    summary.append({"symbol": symbol, "status": "no_data"})
                    continue
                cutoff = candles.index[-1] - pd.DateOffset(years=data_years)
                candles = candles[candles.index >= cutoff]
                detections = _detect(candles, det_cfg)
                combined_cfg = {**risk_cfg, **strat_cfg}
                # Force killzone_only to False during training so the ML model learns from all patterns
                combined_cfg["killzone_only"] = False
                signals = generate_signals(candles, detections, combined_cfg)
                setups = signals_to_setups(signals)
                if len(setups) < 30:
                    _log.getLogger().warning("%s: only %d setups — skipped", symbol, len(setups))
                    summary.append({"symbol": symbol, "status": "too_few_setups"})
                    continue
                labels_df = create_labels(setups, candles)
                feats_full = build_feature_pipeline(candles, detections, normalize=True)
                feats = feats_full.iloc[labels_df["index"].values]
                feats.index = candles.index[labels_df["index"].values]
                y = pd.Series(labels_df["binary"].values, index=feats.index)
                result = train_walk_forward(feats, y, model_cfg, output_dir=str(MODELS_DIR),
                                            symbol=symbol, data_years=data_years)
                promoted = _registry.evaluate_and_promote(
                    symbol, artifacts=result["artifacts"],
                    metrics={"avg_auc": result["avg_auc"], "gt_score": result["gt_score"],
                             "data_start": result["data_start"], "data_end": result["data_end"]},
                    fold_results=result["folds"],
                )
                _log.getLogger().info("%s: auc=%.4f promoted=%s", symbol, result["avg_auc"], promoted)
                summary.append({"symbol": symbol, "avg_auc": result["avg_auc"],
                                 "gt_score": result["gt_score"], "promoted": promoted, "status": "ok"})
            except Exception as e:
                _log.getLogger().error("%s: training failed: %s", symbol, e)
                summary.append({"symbol": symbol, "status": "error", "error": str(e)})

        return {"summary": summary}

    return Response(_stream_job(_run), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Model Registry ────────────────────────────────────────────────────────────

@app.route("/api/registry")
def api_registry():
    champions = _registry.list_champions()
    for c in champions:
        auc = c.get("avg_auc")
        c["drift_alert"] = (auc is not None and auc < 0.55)
    return jsonify({"champions": champions})


@app.route("/api/registry/<symbol>/promote", methods=["POST"])
def api_registry_promote(symbol: str):
    ok = _registry.promote_challenger(symbol)
    return jsonify({"promoted": ok, "symbol": symbol})


# ── Benchmark ─────────────────────────────────────────────────────────────────

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    body = request.json or {}
    strat    = _load_yaml("strategy_config.yaml")["strategy"]
    symbols  = body.get("symbols", strat.get("symbols", []))
    det_cfg  = _load_yaml("detection_config.yaml")["detection"]
    risk_cfg = _load_yaml("risk_config.yaml")["risk"]

    def _run():
        import logging as _log
        rows = []
        for symbol in symbols:
            _log.getLogger().info("Benchmarking %s …", symbol)
            try:
                candles = load_data(symbol, strat.get("default_timeframe", "M15"))
                if candles.empty:
                    rows.append({"symbol": symbol, "error": "no data"})
                    continue
                detections = _detect(candles, det_cfg)
                combined_cfg = {**risk_cfg, **strat}
                signals = generate_signals(candles, detections, combined_cfg)
                if not signals:
                    rows.append({"symbol": symbol, "error": "no signals"})
                    continue
                model = _registry.get_champion(symbol)
                bt = run_backtest(candles, signals,
                                  starting_balance=strat.get("account_starting_balance", 10000),
                                  risk_per_trade=strat["risk_per_trade"])
                m = compute_metrics(bt)
                auc = _registry.champion_auc(symbol)
                rows.append({
                    "symbol":        symbol,
                    "n_trades":      int(m.get("n_trades", 0)),
                    "win_rate":      round(float(m.get("win_rate", 0)), 4),
                    "profit_factor": round(float(m.get("profit_factor", 0)), 4),
                    "sharpe":        round(float(m.get("sharpe", 0)), 4),
                    "max_dd":        round(float(m.get("max_drawdown_pct", 0)), 4),
                    "net_pnl":       round(float(m.get("net_pnl", 0)), 2),
                    "champion_auc":  round(auc, 4) if auc is not None else None,
                    "model_used":    model is not None,
                    "drift_alert":   auc is not None and auc < 0.55,
                })
                _log.getLogger().info("%s: wr=%.1f%% pf=%.2f sharpe=%.2f",
                                      symbol, m.get("win_rate", 0)*100,
                                      m.get("profit_factor", 0), m.get("sharpe", 0))
            except Exception as e:
                _log.getLogger().error("%s benchmark failed: %s", symbol, e)
                rows.append({"symbol": symbol, "error": str(e)})

        rows.sort(key=lambda r: r.get("sharpe", -999), reverse=True)
        
        # Save results to file
        try:
            import json
            from datetime import datetime
            from pathlib import Path
            
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            date_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
            out_path = logs_dir / f"market_benchmark_{date_str}.json"
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            _log.getLogger().info("Saved benchmark results to %s", out_path)
        except Exception as e:
            _log.getLogger().error("Failed to save benchmark results: %s", e)

        return {"leaderboard": rows}

    return Response(_stream_job(_run), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Sentiment ─────────────────────────────────────────────────────────────────

@app.route("/api/sentiment")
def api_sentiment():
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    symbols = strat.get("symbols", [])
    engine = _get_sentiment()
    now = datetime.utcnow()
    results = []
    for sym in symbols:
        sr = engine.get_sentiment(sym)
        blocked, reason = engine.is_trade_blocked(now, sym)
        pre_warn = engine.pre_event_warning(now, sym)
        results.append({
            **sr.to_dict(),
            # FIX BUG-C2: was `not blocked` which inverted the flag.
            # `blocked=True` means trading IS blocked; expose that directly.
            "trade_blocked": blocked,
            "block_reason": reason if blocked else "",
            "pre_event_warning": pre_warn,
        })
    upcoming = [
        {"title": e.title, "currency": e.country, "dt": e.dt.isoformat(), "impact": e.impact}
        for e in engine.upcoming_events(within_hours=24)
    ]
    return jsonify({"sentiment": results, "upcoming_events": upcoming})


# ── Market Overview ───────────────────────────────────────────────────────────

@app.route("/api/market/overview")
def api_market_overview():
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    market_cfg = _load_market_cfg()
    risk_cfg = _load_yaml("risk_config.yaml")["risk"]
    symbols = strat.get("symbols", [])
    engine = _get_sentiment()
    mt5 = _get_mt5()
    now = datetime.utcnow()

    rows = []
    for sym in symbols:
        meta = market_cfg.get(sym, {})
        auc = _registry.champion_auc(sym)
        sr = engine.get_sentiment(sym)
        blocked, _ = engine.is_trade_blocked(now, sym)

        # Spread
        spread = None
        if mt5.connected:
            try:
                pip_size = meta.get("pip_size", 0.0001)
                spread = round(mt5.get_spread_pips(sym, pip_size=pip_size), 1)
                if spread == 999.0:
                    spread = None
            except Exception:
                pass

        rows.append({
            "symbol":      sym,
            "category":    meta.get("category", "fx"),
            "session":     meta.get("session_type", "forex"),
            "champion_auc": round(auc, 3) if auc is not None else None,
            "has_data":    (DATA_DIR / f"{sym}_{strat.get('default_timeframe','M15')}.csv").exists(),
            "has_model":   _registry.has_champion(sym),
            "sentiment":   sr.label,
            "sentiment_score": round(sr.score, 2),
            "news_clear":  blocked,
            "spread_pips": spread,
            "drift_alert": auc is not None and auc < 0.55,
        })

    return jsonify({"overview": rows})


# ── Live Trading ──────────────────────────────────────────────────────────────

@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    auth_err = _check_api_key()
    if auth_err:
        return auth_err
    global _live_thread, _live_intuition_enabled
    if _live_thread and _live_thread.is_alive():
        return jsonify({"status": "already_running"}), 400

    body = request.json or {}
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    symbols    = body.get("symbols", strat.get("symbols", ["EURUSD"]))
    timeframe  = body.get("timeframe", "M15")
    intuition  = bool(body.get("intuition_enabled", True))
    _live_intuition_enabled = intuition

    _live_stop_event.clear()
    _live_log.clear()

    def _run_live():
        _live_log.append(f"[{datetime.utcnow().isoformat()}Z] Starting — {len(symbols)} symbols, intuition={'ON' if intuition else 'OFF'}")
        try:
            from src.live.executor import LiveExecutor
            from src.strategy.risk_manager import RiskManager

            strat_cfg = _load_yaml("strategy_config.yaml")["strategy"].copy()
            risk_cfg  = _load_yaml("risk_config.yaml")["risk"]
            det_cfg   = _load_yaml("detection_config.yaml")["detection"]

            # Patch intuition enabled flag at runtime
            strat_cfg.setdefault("intuition_mode", {})["enabled"] = intuition

            mt5_client = _get_mt5()
            if not mt5_client.connected:
                _live_log.append("ERROR: MT5 not connected — open MT5 and click Connect first")
                return

            acct = mt5_client.account_info() or {}
            balance = float(acct.get("balance", strat_cfg.get("account_starting_balance", 10_000)))
            _live_log.append(f"Account #{acct.get('login','?')} @ {acct.get('server','?')} — {acct.get('currency','USD')} {balance:,.2f}")

            rm = RiskManager(strat_cfg, risk_cfg, balance)

            global _live_executor
            executor = LiveExecutor(
                symbols=symbols, timeframe=timeframe,
                risk_manager=rm, mt5_client=mt5_client,
                detection_cfg=det_cfg, risk_cfg=risk_cfg, strategy_cfg=strat_cfg,
                model_artifacts_dir=str(MODELS_DIR),
                market_config_path=str(CONFIG_DIR / "market_config.yaml"),
            )
            _live_executor = executor
            # FIX: Use executor.run() instead of manually calling _tick() in a
            # plain loop.  run() uses the ThreadPoolExecutor, _safe_tick() error
            # handling, and the proper reconnect logic.
            executor.run(poll_seconds=5)
        except Exception as exc:
            _live_log.append(f"FATAL: {exc}")
        finally:
            _live_log.append(f"[{datetime.utcnow().isoformat()}Z] Executor stopped")

    _live_thread = threading.Thread(target=_run_live, daemon=True)
    _live_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    auth_err = _check_api_key()
    if auth_err:
        return auth_err
    _live_stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/live/status")
def api_live_status():
    running = bool(_live_thread and _live_thread.is_alive())
    return jsonify({"running": running, "log": list(_live_log)[-100:],
                    "intuition_enabled": _live_intuition_enabled})


@app.route("/api/live/intuition", methods=["POST"])
def api_live_intuition_toggle():
    global _live_intuition_enabled, _live_executor
    body = request.json or {}
    _live_intuition_enabled = bool(body.get("enabled", True))
    if _live_executor:
        _live_executor._intuition_enabled = _live_intuition_enabled
        _live_executor.strategy_cfg.setdefault("intuition_mode", {})["enabled"] = _live_intuition_enabled
        _live_log.append(f"[{datetime.utcnow().isoformat()}Z] INTUITION TOGGLE: {'ON' if _live_intuition_enabled else 'OFF'}")
    return jsonify({"intuition_enabled": _live_intuition_enabled})

@app.route("/api/live/positions")
def api_live_positions():
    mt5 = _get_mt5()
    if not mt5.connected:
        return jsonify({"positions": []})
    try:
        positions = mt5.get_positions()
        bot_tickets = set()
        if _live_executor:
            with _live_executor._positions_lock:
                bot_tickets = set(_live_executor._bot_positions.keys())
        pos_list = []
        for p in positions:
            ticket = int(p.get("ticket", 0))
            pos_list.append({
                "ticket": ticket,
                "symbol": p.get("symbol", ""),
                "type": "BUY" if p.get("type", 0) == 0 else "SELL",
                "volume": p.get("volume", 0.0),
                "price_open": p.get("price_open", 0.0),
                "price_current": p.get("price_current", 0.0),
                "profit": p.get("profit", 0.0),
                "is_bot": ticket in bot_tickets
            })
        return jsonify({"positions": pos_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/live/panic", methods=["POST"])
def api_live_panic():
    auth_err = _check_api_key()
    if auth_err:
        return auth_err
    global _live_thread
    _live_stop_event.set()
    mt5 = _get_mt5()
    if not mt5.connected:
        return jsonify({"status": "error", "message": "MT5 not connected"})

    try:
        positions = mt5.get_positions()
        closed_count = 0
        # FIX: Only close positions that were opened by the bot, not manual trades.
        bot_tickets: set = set()
        if _live_executor:
            with _live_executor._positions_lock:
                bot_tickets = set(_live_executor._bot_positions.keys())
        for p in positions:
            ticket = int(p.get("ticket", 0))
            # Close all positions if no executor (full panic), else only bot's.
            if not _live_executor or ticket in bot_tickets:
                res = mt5.close_position(ticket)
                if res:
                    closed_count += 1
        _live_log.append(f"[{datetime.utcnow().isoformat()}Z] PANIC HALT. Closed {closed_count} bot positions.")
        return jsonify({"status": "panic_ok", "closed": closed_count})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Models (legacy list) ──────────────────────────────────────────────────────

@app.route("/api/models")
def api_models():
    models = []
    if MODELS_DIR.exists():
        for pkl in MODELS_DIR.glob("**/ensemble.pkl"):
            folder = pkl.parent
            meta_file   = folder / "meta.json"
            summary_file = folder / "training_summary.json"
            reg_file    = folder / "registry.json"
            meta    = json.loads(meta_file.read_text())    if meta_file.exists()    else {}
            folds   = json.loads(summary_file.read_text()) if summary_file.exists() else []
            reg     = json.loads(reg_file.read_text())     if reg_file.exists()     else {}
            avg_auc = round(sum(fo.get("auc_test", 0) for fo in folds) / len(folds), 4) if folds else None
            models.append({
                "path":       str(folder),
                "name":       folder.name,
                "symbol":     reg.get("symbol", meta.get("symbol", folder.name)),
                "timeframe":  meta.get("timeframe", "M15"),
                "trained_at": reg.get("promoted_at", meta.get("trained_at", "?")),
                "n_samples":  meta.get("n_samples"),
                "avg_auc":    avg_auc or reg.get("avg_auc"),
                "gt_score":   reg.get("gt_score"),
                "is_champion": reg_file.exists(),
                "folds":      folds,
            })
    models.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return jsonify({"models": models})


if __name__ == "__main__":
    setup_logging("INFO")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
