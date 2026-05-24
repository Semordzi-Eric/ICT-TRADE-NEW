"""Live trading loop.

Polls the broker for a new bar, runs the full detection + features +
inference pipeline, and (if the signal and risk gates clear) places an order
via the MT5 client.
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
from ..strategy.rule_based import generate_signals, position_size
from .mt5_client import MT5Client

logger = logging.getLogger(__name__)


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
        self.prob_threshold = prob_threshold
        self.history_bars = history_bars
        self._last_bar_time: Dict[str, pd.Timestamp] = {}
        # Track open positions placed by this bot: ticket -> trade metadata.
        self._bot_positions: Dict[int, dict] = {}
        # Per-symbol contract value (read from risk_cfg, default 100k for FX).
        self._contract_values: Dict[str, float] = risk_cfg.get(
            "contract_value_per_symbol", {}
        )

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

    # ---------- internals ----------
    def _tick(self, symbol: str) -> None:
        candles = self.mt5.fetch_rates(symbol, self.timeframe, self.history_bars)
        if candles.empty:
            return
        latest_bar_time = candles.index[-1]
        if self._last_bar_time.get(symbol) == latest_bar_time:
            return  # no new bar
        self._last_bar_time[symbol] = latest_bar_time

        # Run detections on the historical window (latest bar already closed).
        detections = self._run_detections(candles)

        # Generate the freshest rule-based signal at the last bar.
        signals = generate_signals(candles, detections, self.risk_cfg)
        signals = [s for s in signals if s.index == len(candles) - 1]
        if not signals:
            return
        sig = signals[0]
        logger.info("Signal: %s %s @ %s", sig.setup_type, sig.direction, sig.entry)

        # Build features and predict
        feats = build_feature_pipeline(candles, detections, normalize=True)
        if self.ensemble is not None:
            prob = float(self.ensemble.predict(feats.iloc[[-1]])[0])
        else:
            prob = 0.7  # rule-based-only mode
        logger.info("Model probability: %.3f", prob)
        if not should_trade(prob, self.prob_threshold):
            logger.info("Below threshold %.2f — skip", self.prob_threshold)
            return

        # Risk gates
        balance = self.risk_mgr.balance
        contract_value = self._contract_values.get(symbol, 100_000.0)
        vol = position_size(
            balance,
            risk_per_trade=self.risk_mgr.strategy_cfg["risk_per_trade"],
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

    # ---------- closed-position tracker ----------
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
            profit = self.mt5.get_deal_profit(ticket)
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
