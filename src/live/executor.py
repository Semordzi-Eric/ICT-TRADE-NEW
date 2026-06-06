"""Live trading loop.

Polls the broker for a new bar, runs the full detection + features +
inference pipeline, and (if the signal and risk gates clear) places an order
via the MT5 client.

Improvements over the original:
- Reconnect handling with exponential back-off (3 retries by default).
- Spread filter: rejects entries when broker spread exceeds max_spread_pips.
- Higher-timeframe (H4) bias filter: skips M15 signals that conflict with
  the prevailing H4 trend direction.
- Adaptive position sizing: scales risk based on rolling win rate and whether
  the signal falls inside a London/NY killzone window.
- Rule-based-only fallback uses a configurable probability (not a hard 0.7)
  so the ML threshold is still meaningful without a trained model.
- Signals are pre-sorted by quality score; executor always takes the best one.
- Deal profit lookback now uses risk_cfg.deal_profit_lookback_days (default 7).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..backtest.metrics import compute_metrics  # noqa: F401  — re-exported for convenience
from ..detection.fvg import detect_fvg
from ..detection.liquidity import detect_equal_highs_lows, detect_liquidity_sweeps
from ..detection.orderblock import detect_order_blocks
from ..detection.structure import detect_bos, detect_choch
from ..features.builder import build_feature_pipeline
from ..models.inference import EnsembleModel, should_trade
from ..strategy.risk_manager import RiskManager
from ..strategy.rule_based import _in_killzone, _parse_killzone_windows, generate_signals, position_size
from .mt5_client import MT5Client

logger = logging.getLogger(__name__)

# Pip sizes for common instruments — used by the spread filter.
_PIP_SIZES: Dict[str, float] = {
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "NAS100": 1.0,  "SPX500": 0.1,  "XAUUSD": 0.1,
}
_DEFAULT_PIP_SIZE = 0.0001


def _pip_size(symbol: str) -> float:
    return _PIP_SIZES.get(symbol.upper(), _DEFAULT_PIP_SIZE)


class LiveExecutor:
    """Glues all pieces together for live trading."""

    def __init__(
        self,
        symbols: List[str],
        timeframe: str,
        ensemble: Optional[EnsembleModel],
        risk_manager: RiskManager,
        mt5_client: MT5Client,
        detection_cfg: Dict,
        risk_cfg: Dict,
        strategy_cfg: Dict,
        prob_threshold: float = 0.65,
        history_bars: int = 500,
    ):
        self.symbols = symbols
        self.timeframe = timeframe
        self.ensemble = ensemble
        self.risk_mgr = risk_manager
        self.mt5 = mt5_client
        self.detection_cfg = detection_cfg
        self.risk_cfg = risk_cfg
        self.strategy_cfg = strategy_cfg
        self.prob_threshold = prob_threshold
        self.history_bars = history_bars
        self._last_bar_time: Dict[str, pd.Timestamp] = {}
        # Track open positions placed by this bot: ticket -> trade metadata.
        self._bot_positions: Dict[int, dict] = {}
        # Per-symbol contract value (read from risk_cfg, default 100k for FX).
        self._contract_values: Dict[str, float] = risk_cfg.get(
            "contract_value_per_symbol", {}
        )
        # Rule-based fallback probability (used when ensemble is None).
        self._rule_based_prob = float(strategy_cfg.get("rule_based_prob", 0.55))
        # Reconnect settings
        self._reconnect_retries = int(risk_cfg.get("reconnect_retries", 3))
        self._reconnect_delay = float(risk_cfg.get("reconnect_delay_seconds", 30))
        # Spread filter
        self._max_spread_pips = float(risk_cfg.get("max_spread_pips", 3.0))
        # Deal profit lookback
        self._deal_lookback_days = int(risk_cfg.get("deal_profit_lookback_days", 7))
        # HTF bias caches (lazily initialised per symbol)
        self._htf_caches: Dict[str, object] = {}
        self._htf_enabled = bool(strategy_cfg.get("htf_filter_enabled", True))
        self._htf_timeframe = str(strategy_cfg.get("htf_timeframe", "H4"))
        self._htf_bars = int(strategy_cfg.get("htf_bars", 200))
        # Killzone windows for in_killzone checks inside executor
        self._kz_windows = _parse_killzone_windows(
            strategy_cfg.get("killzone_windows")
        )

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #
    def run(self, poll_seconds: int = 5) -> None:
        """Block forever, polling for new bars and acting on them."""
        logger.info("Live executor starting on %s [%s]", self.symbols, self.timeframe)
        try:
            while True:
                self._check_closed_positions()   # check before ticking each cycle
                for symbol in self.symbols:
                    try:
                        self._tick(symbol)
                    except Exception:  # noqa: BLE001
                        logger.exception("Tick failed for %s", symbol)
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            logger.info("Live executor stopped by user")

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #
    def _ensure_connected(self) -> bool:
        """Verify MT5 connection; attempt reconnect with exponential back-off.

        Returns True if connected (or reconnect succeeded), False otherwise.
        """
        if self.mt5.connected:
            return True
        for attempt in range(1, self._reconnect_retries + 1):
            delay = self._reconnect_delay * (2 ** (attempt - 1))
            logger.warning(
                "MT5 disconnected — reconnect attempt %d/%d in %.0fs",
                attempt, self._reconnect_retries, delay,
            )
            time.sleep(delay)
            ok = self.mt5.connect()
            if ok:
                logger.info("MT5 reconnected successfully on attempt %d", attempt)
                return True
        logger.error("MT5 reconnect failed after %d attempts — skipping tick", self._reconnect_retries)
        return False

    def _check_spread(self, symbol: str) -> bool:
        """Return True if spread is acceptable, False if too wide."""
        if self._max_spread_pips <= 0:
            return True  # filter disabled
        spread = self.mt5.get_spread_pips(symbol, pip_size=_pip_size(symbol))
        if spread > self._max_spread_pips:
            logger.info(
                "Spread filter: %s spread=%.1f pips > max=%.1f — skip",
                symbol, spread, self._max_spread_pips,
            )
            return False
        return True

    def _get_htf_bias(self, symbol: str) -> str:
        """Return H4 bias for *symbol*, using the per-symbol cache."""
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
                logger.exception("Failed to create HTFBiasCache for %s", symbol)
                return "neutral"
        try:
            return self._htf_caches[symbol].get_bias()
        except Exception:
            logger.exception("HTF bias lookup failed for %s", symbol)
            return "neutral"

    def _tick(self, symbol: str) -> None:
        # --- Connection gate (with reconnect) ---
        if not self._ensure_connected():
            return

        # BUG-1 FIX: fetch from_pos=1 so MT5 skips bar-0 (the active, unclosed
        # bar). Every candle in the returned frame is a fully closed bar.
        candles = self.mt5.fetch_rates(symbol, self.timeframe, self.history_bars, from_pos=1)
        if candles.empty:
            return
        latest_bar_time = candles.index[-1]
        if self._last_bar_time.get(symbol) == latest_bar_time:
            return  # no new closed bar yet
        self._last_bar_time[symbol] = latest_bar_time

        # --- Higher-timeframe bias gate ---
        htf_bias = self._get_htf_bias(symbol)

        # Run detections on the historical window (all bars are closed).
        detections = self._run_detections(candles)

        # Generate rule-based signals (killzone filter applied inside).
        # Pass the full strategy_cfg so killzone_only / killzone_windows are honoured.
        signals = generate_signals(candles, detections, {**self.risk_cfg, **self.strategy_cfg})
        signals = [s for s in signals if s.index == len(candles) - 1]
        if not signals:
            return

        # Signals are already scored & deduped; pick the best one on this bar.
        sig = max(signals, key=lambda s: s.score)

        # --- HTF bias filter: skip counter-trend signals ---
        if htf_bias != "neutral" and htf_bias != sig.direction:
            logger.info(
                "HTF bias conflict: H4=%s but signal=%s %s — skip",
                htf_bias, sig.direction, sig.setup_type,
            )
            return

        logger.info("Signal: %s %s @ %.5f  score=%.3f", sig.setup_type, sig.direction, sig.entry, sig.score)

        # BUG-5b FIX: pass the *full* feature matrix so the LSTM can use its
        # 20-bar lookback window. predict() handles slicing internally and
        # returns one probability per row; we take the last one.
        feats = build_feature_pipeline(candles, detections, normalize=True)
        if self.ensemble is not None:
            prob = float(self.ensemble.predict(feats)[-1])
        else:
            # No trained model: use configurable rule-based fallback probability.
            # This is intentionally below the default 0.65 threshold so not every
            # signal fires — the threshold gate still filters weaker setups.
            prob = self._rule_based_prob
            logger.debug("No ensemble — using rule_based_prob=%.2f", prob)

        logger.info("Model probability: %.3f", prob)
        if not should_trade(prob, self.prob_threshold):
            logger.info("Below threshold %.2f — skip", self.prob_threshold)
            return

        # --- Spread gate ---
        if not self._check_spread(symbol):
            return

        # --- Adaptive position sizing ---
        in_kz = _in_killzone(sig.timestamp, self._kz_windows)
        risk_mult = self.risk_mgr.adaptive_risk_multiplier(datetime.utcnow(), in_killzone=in_kz)
        base_risk = float(self.strategy_cfg.get("risk_per_trade", 0.0035))
        adjusted_risk = base_risk * risk_mult
        if risk_mult != 1.0:
            logger.info(
                "Adaptive sizing: mult=%.2f → risk_per_trade=%.4f (kz=%s)",
                risk_mult, adjusted_risk, in_kz,
            )

        # --- Risk gates ---
        balance = self.risk_mgr.balance
        contract_value = self._contract_values.get(symbol, 100_000.0)
        vol = position_size(
            balance,
            risk_per_trade=adjusted_risk,
            entry=sig.entry,
            stop_loss=sig.stop_loss,
            contract_value=contract_value,
        )
        ok, reason = self.risk_mgr.can_trade(datetime.utcnow(), symbol, vol)
        if not ok:
            logger.info("Risk manager rejected: %s", reason)
            return

        action = "buy" if sig.direction == "long" else "sell"
        result = self.mt5.place_order(
            symbol=symbol, action=action, volume=vol,
            sl=sig.stop_loss, tp=sig.take_profit,
        )
        if result is not None:
            ticket = int(result.get("order", 0))
            self._bot_positions[ticket] = {
                "symbol": symbol,
                "entry": sig.entry,
                "stop_loss": sig.stop_loss,
                "direction": sig.direction,
                "contract_value": contract_value,
            }
            self.risk_mgr.register_open(symbol)
            logger.info("Order placed ticket=%d: %s", ticket, result)

    def _run_detections(self, candles: pd.DataFrame) -> dict:
        """Run all detection algorithms on the current bar set."""
        det_cfg = self.detection_cfg
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
            "choch": detect_choch(
                candles, swing_lookback=det_cfg["choch_swing_lookback"]
            ),
        }

    # ------------------------------------------------------------------ #
    #  Closed-position tracker                                            #
    # ------------------------------------------------------------------ #
    def _check_closed_positions(self) -> None:
        """Detect fills that closed bot positions and call register_close().

        MT5 removes a position from positions_get() once it is fully closed.
        We compare our tracked set of tickets against current open positions.
        """
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
            self.risk_mgr.register_close(
                datetime.utcnow(), info["symbol"], profit, r_mult
            )
            logger.info("Position closed ticket=%d pnl=%.2f r=%.2f",
                        ticket, profit, r_mult)
