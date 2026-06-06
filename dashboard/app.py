"""ICT Trading Bot — Dashboard API Server.

Run with:
    python main.py --dashboard
or directly:
    python dashboard/app.py
"""
from __future__ import annotations

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
from typing import Dict, Optional

# Ensure project root is on the path
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

import yaml

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models_artifacts"

# ── Shared MT5 client (connects once, reused by all routes) ──────────────────
# Calling mt5.initialize() with no arguments attaches to the currently open
# MetaTrader 5 terminal on the desktop — no credentials needed.
_mt5: MT5Client = MT5Client()
_mt5_lock = threading.Lock()


def _get_mt5() -> MT5Client:
    """Return the shared MT5 client, connecting if necessary."""
    with _mt5_lock:
        if not _mt5.connected:
            _mt5.connect()   # no-op if MT5 is not installed or terminal is closed
    return _mt5


# ── live executor state ──────────────────────────────────────────────────────
_live_thread: Optional[threading.Thread] = None
_live_stop_event = threading.Event()
_live_log: list = []


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def _detect(candles, det_cfg: dict) -> dict:
    return {
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
            tolerance_atr=det_cfg["equal_hl_tolerance_atr"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "bos": detect_bos(
            candles,
            confirmation_bars=det_cfg["bos_confirmation_bars"],
            swing_lookback=det_cfg["swing_lookback"],
        ),
        "choch": detect_choch(candles, swing_lookback=det_cfg["choch_swing_lookback"]),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _sse(data: dict | str) -> str:
    payload = json.dumps(data) if isinstance(data, dict) else data
    return f"data: {payload}\n\n"


def _stream_job(target_fn, *args, **kwargs):
    """Run target_fn in a thread, yield SSE events from a shared queue."""
    log_q: queue.Queue = queue.Queue()

    class _QueueHandler(logging.Handler):
        def emit(self, record):
            log_q.put({"type": "log", "msg": self.format(record)})

    handler = _QueueHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    result_box: list = []
    error_box: list = []

    def _run():
        try:
            result_box.append(target_fn(*args, **kwargs))
        except Exception as exc:
            error_box.append(str(exc))
        finally:
            log_q.put(None)  # sentinel

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


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/symbols")
def api_symbols():
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    return jsonify({"symbols": strat.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])})


# ── MT5 status & connect ─────────────────────────────────────────────────────

@app.route("/api/mt5/status")
def api_mt5_status():
    """Return MT5 connection state and account details."""
    mt5 = _get_mt5()
    if not mt5.connected:
        return jsonify({"connected": False, "account": None})
    info = mt5.account_info() or {}
    # Expose only the fields the UI needs — never expose passwords.
    account = {
        "login":   info.get("login", "—"),
        "name":    info.get("name", "—"),
        "server":  info.get("server", "—"),
        "balance": info.get("balance", 0.0),
        "equity":  info.get("equity", 0.0),
        "currency": info.get("currency", "USD"),
        "leverage": info.get("leverage", 0),
        "company": info.get("company", "—"),
    }
    return jsonify({"connected": True, "account": account})


@app.route("/api/mt5/connect", methods=["POST"])
def api_mt5_connect():
    """Attempt to connect (or re-connect) to the running MT5 terminal.

    Body (all optional — if omitted, attaches to the already-logged-in account):
        { "account": 12345, "password": "xxx", "server": "Broker-Live" }
    """
    body = request.json or {}
    account  = body.get("account")  or None
    password = body.get("password") or None
    server   = body.get("server")   or None
    with _mt5_lock:
        # Always re-initialize so a fresh connection is made.
        if _mt5.connected:
            _mt5.disconnect()
        ok = _mt5.connect(
            account=int(account) if account else None,
            password=password,
            server=server,
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


# ── Data download ─────────────────────────────────────────────────────────────

@app.route("/api/data/download", methods=["POST"])
def api_data_download():
    """Download historical data from MT5 (preferred) or yfinance.

    Body:
        { "symbols": ["EURUSD", ...], "timeframe": "M15", "bars": 200000 }

    Streams SSE events so the frontend can show per-symbol progress.
    """
    body = request.json or {}
    strat = _load_yaml("strategy_config.yaml")["strategy"]
    symbols   = body.get("symbols",   strat.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"]))
    timeframe = body.get("timeframe", strat.get("default_timeframe", "M15"))
    bars      = int(body.get("bars", 200_000))

    def _run_download():
        import logging as _log
        results = {}
        mt5 = _get_mt5()
        for symbol in symbols:
            _log.getLogger().info("Downloading %s %s (up to %d bars)…", symbol, timeframe, bars)
            try:
                if mt5.connected:
                    # Pull directly from the already-connected MT5 terminal —
                    # maximum history, no 60-day limit.
                    df = mt5.fetch_rates(symbol, timeframe, bars, from_pos=1)
                    source = "MT5"
                else:
                    # Fallback to yfinance
                    from src.utils.data_loader import load_from_yfinance
                    tf_map = {"M1":"1m","M5":"5m","M15":"15m","M30":"30m","H1":"1h","D1":"1d"}
                    interval = tf_map.get(timeframe.upper(), "15m")
                    period = "730d" if interval in ("1h", "1d") else "60d"
                    df = load_from_yfinance(symbol, period=period, interval=interval)
                    source = "yfinance"

                if df is None or df.empty:
                    _log.getLogger().warning("%s: no data returned", symbol)
                    results[symbol] = {"bars": 0, "source": source, "error": "no data"}
                    continue

                # Write to CSV cache
                cache = DATA_DIR / f"{symbol}_{timeframe}.csv"
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache, index_label="time")
                n = len(df)
                _log.getLogger().info("%s: saved %d bars from %s → %s", symbol, n, source, cache.name)
                results[symbol] = {"bars": n, "source": source}
            except Exception as e:
                _log.getLogger().error("%s download failed: %s", symbol, e)
                results[symbol] = {"bars": 0, "source": "error", "error": str(e)}
        return results

    def generate():
        yield from _stream_job(_run_download)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/data/status")
def api_data_status():
    rows = []
    for csv in sorted(DATA_DIR.glob("*.csv")):
        stem = csv.stem  # e.g. EURUSD_M15
        parts = stem.rsplit("_", 1)
        symbol = parts[0] if len(parts) == 2 else stem
        tf = parts[1] if len(parts) == 2 else "?"
        try:
            import pandas as pd
            df = pd.read_csv(csv, nrows=0)
            # count lines efficiently
            with open(csv, "rb") as f:
                count = sum(1 for _ in f) - 1  # minus header
        except Exception:
            count = -1
        rows.append({"symbol": symbol, "timeframe": tf, "bars": count, "file": csv.name})
    return jsonify({"datasets": rows})


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = request.json or {}
    symbols = body.get("symbols", ["EURUSD"])
    timeframe = body.get("timeframe", "M15")
    starting_balance = float(body.get("starting_balance", 10_000))

    det_cfg = _load_yaml("detection_config.yaml")["detection"]
    risk_cfg = _load_yaml("risk_config.yaml")["risk"]
    strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]

    def _run_all():
        results = {}
        for symbol in symbols:
            import logging as _log
            _log.getLogger().info("Loading data for %s %s", symbol, timeframe)
            candles = load_data(symbol, timeframe)
            if candles.empty:
                results[symbol] = {"error": "no data"}
                continue
            _log.getLogger().info("%s: %d bars", symbol, len(candles))

            detections = _detect(candles, det_cfg)
            _log.getLogger().info(
                "%s detections: FVG=%d OB=%d Sweeps=%d BOS=%d CHoCH=%d",
                symbol,
                len(detections["fvg"]),
                len(detections["order_blocks"]),
                len(detections["liquidity_sweeps"]),
                len(detections["bos"]),
                len(detections["choch"]),
            )

            combined_cfg = {**risk_cfg, **strat_cfg}
            signals = generate_signals(candles, detections, combined_cfg)
            _log.getLogger().info("%s: %d signals generated", symbol, len(signals))

            if not signals:
                results[symbol] = {"error": "no signals"}
                continue

            bt = run_backtest(
                candles, signals,
                starting_balance=starting_balance,
                risk_per_trade=strat_cfg["risk_per_trade"],
            )
            metrics = compute_metrics(bt)

            # Equity curve (downsample to ≤2000 points for transfer)
            eq = bt.equity_curve
            step = max(1, len(eq) // 2000)
            eq_sampled = eq.iloc[::step]

            trades_df = bt.trades_df()
            trades_list = []
            if not trades_df.empty:
                trades_list = trades_df.to_dict(orient="records")
                for t in trades_list:
                    for k, v in t.items():
                        if hasattr(v, "isoformat"):
                            t[k] = str(v)

            results[symbol] = {
                "metrics": {k: (round(float(v), 6) if isinstance(v, float) else v)
                            for k, v in metrics.items()},
                "equity_curve": {
                    "timestamps": [str(ts) for ts in eq_sampled.index],
                    "values": [round(float(v), 2) for v in eq_sampled.values],
                },
                "trades": trades_list,
                "detection_counts": {
                    "fvg": len(detections["fvg"]),
                    "order_blocks": len(detections["order_blocks"]),
                    "sweeps": len(detections["liquidity_sweeps"]),
                    "bos": len(detections["bos"]),
                    "choch": len(detections["choch"]),
                },
            }
        return results

    def generate():
        yield from _stream_job(_run_all)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/train", methods=["POST"])
def api_train():
    body = request.json or {}
    symbol = body.get("symbol", "EURUSD")
    timeframe = body.get("timeframe", "M15")
    base_dir = body.get("output_dir", str(MODELS_DIR))
    run_id = f"{symbol}_{timeframe}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    output_dir = str(Path(base_dir) / run_id)

    det_cfg = _load_yaml("detection_config.yaml")["detection"]
    model_cfg = _load_yaml("model_config.yaml")["model"]
    risk_cfg = _load_yaml("risk_config.yaml")["risk"]

    def _run_training():
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
        _log.getLogger().info("Loaded %d bars", len(candles))

        strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]
        detections = _detect(candles, det_cfg)
        combined_cfg = {**risk_cfg, **strat_cfg}
        signals = generate_signals(candles, detections, combined_cfg)
        setups = signals_to_setups(signals)
        _log.getLogger().info("Generated %d setups", len(setups))

        if len(setups) < 30:
            raise ValueError("Not enough setups — need at least 30")

        labels_df = create_labels(setups, candles)
        feats_full = build_feature_pipeline(candles, detections, normalize=True)
        feats = feats_full.iloc[labels_df["index"].values]
        feats.index = candles.index[labels_df["index"].values]
        y = pd.Series(labels_df["binary"].values, index=feats.index)

        _log.getLogger().info("Training on %d samples; positive rate=%.3f",
                               len(feats), float(y.mean()))

        summary = train_walk_forward(feats, y, model_cfg, output_dir=output_dir)

        # Tag the artifact with metadata
        meta_path = Path(output_dir) / "meta.json"
        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": int(len(feats)),
            "positive_rate": float(y.mean()),
            "folds": summary["folds"],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"folds": summary["folds"], "ensemble_path": summary["ensemble_path"]}

    def generate():
        yield from _stream_job(_run_training)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/models")
def api_models():
    models = []
    if MODELS_DIR.exists():
        for pkl in MODELS_DIR.glob("**/ensemble.pkl"):
            folder = pkl.parent
            meta_file = folder / "meta.json"
            summary_file = folder / "training_summary.json"
            meta = {}
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
            folds = []
            if summary_file.exists():
                with open(summary_file) as f:
                    folds = json.load(f)
            avg_auc = (
                round(sum(fo.get("auc_test", 0) for fo in folds) / len(folds), 4)
                if folds else None
            )
            models.append({
                "path": str(folder),
                "name": folder.name,
                "symbol": meta.get("symbol", "?"),
                "timeframe": meta.get("timeframe", "?"),
                "trained_at": meta.get("trained_at", "?"),
                "n_samples": meta.get("n_samples"),
                "positive_rate": meta.get("positive_rate"),
                "avg_auc": avg_auc,
                "folds": folds,
            })
    models.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return jsonify({"models": models})


@app.route("/api/live/start", methods=["POST"])
def api_live_start():
    global _live_thread
    if _live_thread and _live_thread.is_alive():
        return jsonify({"status": "already_running"}), 400

    body = request.json or {}
    symbols = body.get("symbols", ["EURUSD"])
    timeframe = body.get("timeframe", "M15")
    model_path = body.get("model_path")

    _live_stop_event.clear()
    _live_log.clear()

    def _run_live():
        import logging as _log
        _log.getLogger().info("Live executor starting: %s %s", symbols, timeframe)
        _live_log.append(f"[{datetime.utcnow().isoformat()}] Starting live on {symbols} {timeframe}")
        try:
            from src.live.executor import LiveExecutor
            from src.live.mt5_client import MT5Client
            from src.models.inference import EnsembleModel
            from src.strategy.risk_manager import RiskManager

            strat_cfg = _load_yaml("strategy_config.yaml")["strategy"]
            risk_cfg = _load_yaml("risk_config.yaml")["risk"]
            det_cfg = _load_yaml("detection_config.yaml")["detection"]

            ensemble = None
            if model_path and Path(model_path).exists():
                try:
                    ensemble = EnsembleModel.from_dir(model_path)
                    _live_log.append("Model loaded successfully")
                except Exception as e:
                    _live_log.append(f"Model load failed: {e}")

            # Reuse the shared MT5 client (already connected from Setup tab).
            # If not yet connected, attempt to connect to the open terminal.
            mt5_client = _get_mt5()
            if not mt5_client.connected:
                _live_log.append("MT5 not connected — open MetaTrader 5 on your desktop and click Connect in the Setup tab")
                return

            # Read actual account balance from MT5.
            acct = mt5_client.account_info() or {}
            balance = float(acct.get("balance", strat_cfg.get("account_starting_balance", 10_000)))
            _live_log.append(f"Connected: #{acct.get('login','?')} @ {acct.get('server','?')} — Balance: {acct.get('currency','USD')} {balance:,.2f}")

            rm = RiskManager(strat_cfg, risk_cfg, balance)
            executor = LiveExecutor(
                symbols=symbols, timeframe=timeframe,
                ensemble=ensemble, risk_manager=rm,
                mt5_client=mt5_client, detection_cfg=det_cfg,
                risk_cfg=risk_cfg, strategy_cfg=strat_cfg,
            )
            while not _live_stop_event.is_set():
                executor._check_closed_positions()
                for sym in symbols:
                    try:
                        executor._tick(sym)
                    except Exception as exc:
                        _live_log.append(f"Tick error {sym}: {exc}")
                _live_stop_event.wait(timeout=5)
            # Do NOT disconnect shared client here — other routes still use it.
        except Exception as exc:
            _live_log.append(f"Live executor error: {exc}")
        finally:
            _live_log.append("Live executor stopped")

    _live_thread = threading.Thread(target=_run_live, daemon=True)
    _live_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/live/stop", methods=["POST"])
def api_live_stop():
    _live_stop_event.set()
    return jsonify({"status": "stopping"})


@app.route("/api/live/status")
def api_live_status():
    running = bool(_live_thread and _live_thread.is_alive())
    return jsonify({
        "running": running,
        "log": _live_log[-50:],  # last 50 entries
    })


if __name__ == "__main__":
    setup_logging("INFO")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
