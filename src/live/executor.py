"""Live trading loop — v2 (24/7 multi-market, intuition mode, sentiment).

Key upgrades over v1:
  - Parallel symbol scanning via ThreadPoolExecutor (all markets simultaneously).
  - Per-category session gates:
      • FX / Metals  → London + NY killzones (configurable)
      • Equity Indices → US equity session 13:30–20:00 UTC
      • Crypto        → 24/7 (no session blocking)
  - Per-symbol champion models loaded from ModelRegistry.
  - Sentiment engine wired into every tick for news blackouts AND
    pre-event risk warnings.
  - Intuition Mode: takes trades below 0.65 ML threshold when ≥8 ICT
    confluence factors stack up.
  - Reconnect handling with exponential back-off (unchanged from v1).
  - Spread filter uses per-symbol overrides from risk_config.yaml.
  - Deal profit lookback uses risk_cfg.deal_profit_lookback_days.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional

import pandas as pd
import yaml

from ..backtest.metrics import compute_metrics  # noqa: F401
from ..detection.fvg import detect_fvg
from ..detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from ..detection.orderblock import detect_order_blocks
from ..detection.structure import detect_bos, detect_choch
from ..features.builder import build_feature_pipeline
from ..models.inference import EnsembleModel, should_trade
from ..models.registry import ModelRegistry
from ..strategy.risk_manager import RiskManager
from ..strategy.rule_based import _in_killzone, _parse_killzone_windows, generate_signals, position_size
from ..strategy.intuition_mode import IntuitiveSignalScorer
from ..utils.sentiment_engine import SentimentEngine
from .mt5_client import MT5Client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pip sizes for spread computation (extended to all 16 symbols).
# ---------------------------------------------------------------------------
_PIP_SIZES: Dict[str, float] = {
    "USDJPY": 0.01,  "EURJPY": 0.01,  "GBPJPY": 0.01,
    "NAS100": 1.0,   "SPX500": 0.1,   "US30":   1.0,
    "XAUUSD": 0.1,   "XAGUSD": 0.01,
    "BTCUSD": 1.0,   "ETHUSD": 1.0,
}
_DEFAULT_PIP_SIZE = 0.0001

# US equity session (UTC) for index CFDs.
_EQUITY_SESSION_START = dt_time(13, 30)
_EQUITY_SESSION_END   = dt_time(20, 0)

# Metal session (near-24/7, CME — approximate).
_METAL_SESSION_START = dt_time(23, 0)  # Sunday open
_METAL_SESSION_END   = dt_time(21, 0)  # Friday close (next day wrap)


def _pip_size(symbol: str) -> float:
    return _PIP_SIZES.get(symbol.upper(), _DEFAULT_PIP_SIZE)


def _load_market_config(path: str = "config/market_config.yaml") -> Dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("markets", {})
    except Exception:
        return {}


class LiveExecutor:
    """Glues all pieces together for 24/7 live trading across 16 markets."""

    def __init__(
        self,
        symbols: List[str],
        timeframe: str,
        risk_manager: RiskManager,
        mt5_client: MT5Client,
        detection_cfg: Dict,
        risk_cfg: Dict,
        strategy_cfg: Dict,
        ensemble: Optional[EnsembleModel] = None,   # legacy single-model (ignored if registry used)
        prob_threshold: float = 0.65,
        history_bars: int = 500,
        model_artifacts_dir: str = "models_artifacts",
        market_config_path: str = "config/market_config.yaml",
    ):
        self.symbols       = symbols
        self.timeframe     = timeframe
        self.risk_mgr      = risk_manager
        self.mt5           = mt5_client
        self.detection_cfg = detection_cfg
        self.risk_cfg      = risk_cfg
        self.strategy_cfg  = strategy_cfg
        self.prob_threshold = prob_threshold
        self.history_bars  = history_bars

        # Model registry (per-symbol champions).
        self._registry = ModelRegistry(base_dir=model_artifacts_dir)
        # Cache loaded models so we don't reload on every tick.
        self._models: Dict[str, Optional[EnsembleModel]] = {}

        # Sentiment engine — starts background daemon.
        self._sentiment = SentimentEngine(
            market_config_path=market_config_path,
        )
        self._sentiment.start_background_refresh()

        # Intuition mode scorer.
        intuition_cfg = strategy_cfg.get("intuition_mode", {})
        self._intuition = IntuitiveSignalScorer(cfg=intuition_cfg)
        self._intuition_enabled = bool(intuition_cfg.get("enabled", True))

        # Market config (per-symbol metadata).
        self._market_cfg = _load_market_config(market_config_path)

        # Per-symbol state.
        self._last_bar_time: Dict[str, pd.Timestamp] = {}
        self._bot_positions: Dict[int, dict] = {}

        # Contract values (from risk_cfg).
        self._contract_values: Dict[str, float] = risk_cfg.get("contract_value_per_symbol", {})
        # Per-symbol spread overrides.
        self._spread_overrides: Dict[str, float] = risk_cfg.get("max_spread_pips_override", {})
        self._default_spread: float = float(risk_cfg.get("max_spread_pips", 3.0))

        # Rule-based fallback probability.
        self._rule_based_prob = float(strategy_cfg.get("rule_based_prob", 0.55))

        # Reconnect settings.
        self._reconnect_retries = int(risk_cfg.get("reconnect_retries", 3))
        self._reconnect_delay   = float(risk_cfg.get("reconnect_delay_seconds", 30))

        # Deal profit lookback.
        self._deal_lookback_days = int(risk_cfg.get("deal_profit_lookback_days", 7))

        # HTF bias caches.
        self._htf_caches: Dict[str, object] = {}
        self._htf_enabled    = bool(strategy_cfg.get("htf_filter_enabled", True))
        self._htf_timeframe  = str(strategy_cfg.get("htf_timeframe", "H4"))
        self._htf_bars       = int(strategy_cfg.get("htf_bars", 200))

        # Killzone windows.
        self._kz_windows = _parse_killzone_windows(strategy_cfg.get("killzone_windows"))

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self, poll_seconds: int = 5, max_workers: int = 8) -> None:
        """Block forever, scanning all symbols in parallel every poll cycle."""
        logger.info(
            "LiveExecutor v2 starting: %d symbols, timeframe=%s, workers=%d",
            len(self.symbols), self.timeframe, max_workers,
        )
        try:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ict-tick") as pool:
                while True:
                    self._check_closed_positions()
                    futures = {
                        pool.submit(self._safe_tick, sym): sym
                        for sym in self.symbols
                    }
                    for fut in as_completed(futures):
                        sym = futures[fut]
                        try:
                            fut.result()
                        except Exception:
                            logger.exception("Tick raised for %s", sym)
                    time.sleep(poll_seconds)
        except KeyboardInterrupt:
            logger.info("LiveExecutor stopped by user")

    # ------------------------------------------------------------------ #
    #  Per-symbol tick                                                     #
    # ------------------------------------------------------------------ #

    def _safe_tick(self, symbol: str) -> None:
        try:
            self._tick(symbol)
        except Exception:
            logger.exception("Tick error for %s", symbol)

    def _tick(self, symbol: str) -> None:
        if not self._ensure_connected():
            return

        # --- Market-hours gate ---
        if not self._in_active_hours(symbol):
            return

        candles = self.mt5.fetch_rates(symbol, self.timeframe, self.history_bars, from_pos=1)
        if candles.empty:
            return
        latest_bar_time = candles.index[-1]
        if self._last_bar_time.get(symbol) == latest_bar_time:
            return  # no new closed bar
        self._last_bar_time[symbol] = latest_bar_time

        # --- Sentiment / news blackout gate ---
        now = datetime.utcnow()
        blocked, reason = self._sentiment.is_trade_blocked(now, symbol)
        if blocked:
            logger.info("News blackout [%s]: %s", symbol, reason)
            return

        # --- Pre-event warning: reduce risk ---
        pre_event = self._sentiment.pre_event_warning(now, symbol)

        # --- HTF bias ---
        htf_bias = self._get_htf_bias(symbol)

        # --- Detections & signals ---
        detections = self._run_detections(candles)
        signals = generate_signals(candles, detections, {**self.risk_cfg, **self.strategy_cfg})
        signals = [s for s in signals if s.index == len(candles) - 1]
        if not signals:
            return
        sig = max(signals, key=lambda s: s.score)

        # --- HTF bias filter ---
        if htf_bias != "neutral" and htf_bias != sig.direction:
            logger.debug("HTF conflict [%s]: H4=%s signal=%s — skip", symbol, htf_bias, sig.direction)
            return

        # --- Feature pipeline & ML inference ---
        feats = build_feature_pipeline(candles, detections, normalize=True)
        model = self._get_model(symbol)
        if model is not None:
            prob = float(model.predict(feats)[-1])
        else:
            prob = self._rule_based_prob

        logger.info("[%s] signal=%s setup=%s prob=%.3f", symbol, sig.direction, sig.setup_type, prob)

        # --- Normal ML gate ---
        passes_ml = should_trade(prob, self.prob_threshold)

        # --- Intuition Mode gate (if normal ML fails) ---
        sentiment_result = self._sentiment.get_sentiment(symbol)
        in_kz = _in_killzone(sig.timestamp, self._kz_windows)
        spread = self.mt5.get_spread_pips(symbol, pip_size=_pip_size(symbol))

        intuition_result = None
        if not passes_ml and self._intuition_enabled:
            intuition_result = self._intuition.score(
                signal=sig,
                ml_prob=prob,
                htf_bias=htf_bias,
                detections=detections,
                candles=candles,
                sentiment_score=sentiment_result.score,
                spread_pips=spread,
                in_killzone=in_kz,
                current_bar_idx=len(candles) - 1,
                symbol=symbol,
            )
            if not intuition_result.should_trade:
                logger.debug("[%s] Below ML threshold and intuition score %d/%d — skip",
                             symbol, intuition_result.total_score, intuition_result.threshold)
                return
            logger.info("[%s] INTUITION TRADE: score=%d risk_mult=%.2f",
                        symbol, intuition_result.total_score, intuition_result.risk_multiplier)

        elif not passes_ml:
            logger.debug("[%s] Below ML threshold (%.3f < %.2f) — skip",
                         symbol, prob, self.prob_threshold)
            return

        # --- Spread gate ---
        max_spread = self._spread_overrides.get(symbol.upper(), self._default_spread)
        if spread > max_spread:
            logger.info("[%s] Spread filter: %.1f pips > %.1f — skip", symbol, spread, max_spread)
            return

        # --- Adaptive position sizing ---
        risk_mult = self.risk_mgr.adaptive_risk_multiplier(now, in_killzone=in_kz)
        
        # Apply ML Sentiment Multiplier if it's a pure ML trade
        if passes_ml:
            max_sentiment_mult = float(self.strategy_cfg.get("intuition_mode", {}).get("max_risk_multiplier", 2.0))
            sentiment_score = sentiment_result.score
            if sig.direction == "long" and sentiment_score >= 0.8:
                risk_mult *= max_sentiment_mult
                logger.info("[%s] Extreme Bullish Sentiment! Doubling ML trade risk multiplier.", symbol)
            elif sig.direction == "short" and sentiment_score <= -0.8:
                risk_mult *= max_sentiment_mult
                logger.info("[%s] Extreme Bearish Sentiment! Doubling ML trade risk multiplier.", symbol)

        # Apply intuition risk multiplier on top (capped by config).
        if intuition_result is not None:
            risk_mult *= intuition_result.risk_multiplier
            
        # Apply pre-event warning reduction (50% risk).
        if pre_event:
            risk_mult *= 0.5
            logger.info("[%s] Pre-event warning: risk reduced to %.1f%%", symbol, risk_mult * 100)

        base_risk    = float(self.strategy_cfg.get("risk_per_trade", 0.0035))
        adjusted_risk = base_risk * risk_mult

        # --- Risk gates ---
        balance = self.risk_mgr.balance
        contract_value = self._contract_values.get(symbol, 100_000.0)
        vol = position_size(
            balance, risk_per_trade=adjusted_risk,
            entry=sig.entry, stop_loss=sig.stop_loss,
            contract_value=contract_value,
        )
        ok, reason = self.risk_mgr.can_trade(now, symbol, vol)
        if not ok:
            logger.info("[%s] Risk manager rejected: %s", symbol, reason)
            return

        # --- Place order ---
        action = "buy" if sig.direction == "long" else "sell"
        result = self.mt5.place_order(
            symbol=symbol, action=action, volume=vol,
            sl=sig.stop_loss, tp=sig.take_profit,
        )
        if result is not None:
            ticket = int(result.get("order", 0))
            self._bot_positions[ticket] = {
                "symbol": symbol,
                "entry":  sig.entry,
                "stop_loss": sig.stop_loss,
                "direction": sig.direction,
                "contract_value": contract_value,
                "intuition": intuition_result is not None,
            }
            self.risk_mgr.register_open(symbol)
            logger.info("[%s] Order placed ticket=%d action=%s vol=%.4f %s",
                        symbol, ticket, action, vol,
                        "(INTUITION)" if intuition_result else "")

    # ------------------------------------------------------------------ #
    #  Market-hours awareness                                              #
    # ------------------------------------------------------------------ #

    def _in_active_hours(self, symbol: str) -> bool:
        """Return True if this symbol should be checked right now."""
        mkt = self._market_cfg.get(symbol.upper(), {})
        session_type = mkt.get("session_type", "forex")
        now_t = datetime.utcnow().time()

        if session_type == "crypto":
            return True  # 24/7

        if session_type == "equity":
            ah = mkt.get("active_hours", ["13:30", "20:00"])
            try:
                start = dt_time.fromisoformat(ah[0])
                end   = dt_time.fromisoformat(ah[1])
                return start <= now_t <= end
            except Exception:
                return True

        if session_type == "metal":
            # Gold/Silver: near-24/7 — exclude only the brief daily CME closure (21–22 UTC).
            # We approximate: allow all hours except 21:00–22:00 UTC.
            excluded_start = dt_time(21, 0)
            excluded_end   = dt_time(22, 0)
            return not (excluded_start <= now_t <= excluded_end)

        # Default: forex — always allow (killzone filter handles session restriction).
        return True

    # ------------------------------------------------------------------ #
    #  Model loading                                                       #
    # ------------------------------------------------------------------ #

    def _get_model(self, symbol: str) -> Optional[EnsembleModel]:
        if symbol not in self._models:
            self._models[symbol] = self._registry.get_champion(symbol)
            if self._models[symbol] is None:
                logger.info("[%s] No champion model in registry — using rule-based fallback", symbol)
        return self._models[symbol]

    # ------------------------------------------------------------------ #
    #  Connection / reconnect                                              #
    # ------------------------------------------------------------------ #

    def _ensure_connected(self) -> bool:
        if self.mt5.connected:
            return True
        for attempt in range(1, self._reconnect_retries + 1):
            delay = self._reconnect_delay * (2 ** (attempt - 1))
            logger.warning("MT5 disconnected — attempt %d/%d in %.0fs",
                           attempt, self._reconnect_retries, delay)
            time.sleep(delay)
            if self.mt5.connect():
                logger.info("MT5 reconnected on attempt %d", attempt)
                return True
        logger.error("MT5 reconnect failed after %d attempts", self._reconnect_retries)
        return False

    # ------------------------------------------------------------------ #
    #  HTF bias                                                            #
    # ------------------------------------------------------------------ #

    def _get_htf_bias(self, symbol: str) -> str:
        if not self._htf_enabled:
            return "neutral"
        if symbol not in self._htf_caches:
            try:
                from ..utils.htf_bias import HTFBiasCache
                self._htf_caches[symbol] = HTFBiasCache(
                    self.mt5, symbol,
                    timeframe=self._htf_timeframe,
                    bars=self._htf_bars,
                )
            except Exception:
                return "neutral"
        try:
            return self._htf_caches[symbol].get_bias()
        except Exception:
            return "neutral"

    # ------------------------------------------------------------------ #
    #  Detections                                                          #
    # ------------------------------------------------------------------ #

    def _run_detections(self, candles: pd.DataFrame) -> dict:
        det_cfg = self.detection_cfg
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

    # ------------------------------------------------------------------ #
    #  Closed-position tracker                                             #
    # ------------------------------------------------------------------ #

    def _check_closed_positions(self) -> None:
        if not self._bot_positions:
            return
        try:
            open_positions = self.mt5.get_positions()
        except Exception:
            return
        current_tickets = {int(p.get("ticket", 0)) for p in open_positions}
        closed_tickets = [t for t in list(self._bot_positions.keys())
                          if t not in current_tickets]
        for ticket in closed_tickets:
            info = self._bot_positions.pop(ticket)
            profit = self.mt5.get_deal_profit(ticket, lookback_days=self._deal_lookback_days)
            if profit is None:
                profit = 0.0
            risk = abs(info["entry"] - info["stop_loss"])
            cv = info.get("contract_value", 100_000.0)
            r_mult = profit / (risk * cv) if risk > 0 and cv > 0 else 0.0
            self.risk_mgr.register_close(datetime.utcnow(), info["symbol"], profit, r_mult)
            logger.info(
                "Position closed ticket=%d pnl=%.2f r=%.2f %s",
                ticket, profit, r_mult,
                "(INTUITION)" if info.get("intuition") else "",
            )
