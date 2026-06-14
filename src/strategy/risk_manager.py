"""Account-level risk management: loss limits, cooldowns, correlation."""
from __future__ import annotations

import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    pnl: float
    r_multiple: float


@dataclass
class RiskState:
    """Mutable state the RiskManager carries across trades."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    daily_trades: int = 0
    consecutive_losses: int = 0
    cooldown_until: Optional[datetime] = None
    open_symbols: Set[str] = field(default_factory=set)
    trade_history: List[TradeRecord] = field(default_factory=list)
    peak_balance: float = 0.0
    halted: bool = False


class RiskManager:
    """Enforces account-level rules. Call :meth:`can_trade` before every entry.

    Thread safety
    -------------
    ``can_trade`` and ``register_open`` are protected by ``_trade_gate`` (a
    ``threading.Lock``).  Callers **must** hold this lock for the full
    can_trade → place_order → register_open sequence to prevent the TOCTOU
    race where two threads both see ``open_positions < max`` and both place
    orders before either increments the counter.

    Usage in the executor::

        with risk_mgr.trade_gate:
            ok, reason = risk_mgr.can_trade(...)
            if ok:
                # place order here
                risk_mgr.register_open(symbol)
    """

    def __init__(self, strategy_cfg: dict, risk_cfg: dict, account_balance: float):
        self.strategy_cfg = strategy_cfg
        self.risk_cfg = risk_cfg
        self.balance = account_balance
        self.state = RiskState(peak_balance=account_balance)
        self._current_day: Optional[datetime] = None
        self._current_week: Optional[int] = None
        # Adaptive-sizing state: True when win rate is in the low zone.
        self._in_low_wr_regime: bool = False
        # FIX BUG-C3: lock that must be held for the entire can_trade →
        # register_open critical section to prevent the TOCTOU race.
        self.trade_gate = threading.Lock()

    # ---------- public API ----------
    def can_trade(
        self,
        now: datetime,
        symbol: str,
        proposed_volume: float,
        floating_pnl: float = 0.0,
    ) -> Tuple[bool, str]:
        """Check all gates. Returns ``(ok, reason)``.

        IMPORTANT: Callers must hold ``self.trade_gate`` while calling this
        method AND the subsequent ``register_open()`` to avoid the TOCTOU race
        on ``max_open_positions``.  See class docstring.
        """
        self._roll_periods(now)

        # FIX BUG-M6 (partial): halted due to *daily* loss limit is reset
        # at the start of a new trading day (handled in _roll_periods).
        # Max-DD halt is permanent until manual restart.
        if self.state.halted:
            return False, "account halted (max DD or daily loss — see logs)"

        # Cooldown
        if self.state.cooldown_until and now < self.state.cooldown_until:
            return False, f"in cooldown until {self.state.cooldown_until}"

        # Daily trade cap
        if self.state.daily_trades >= int(self.strategy_cfg["max_daily_trades"]):
            return False, "max daily trades reached"

        # Daily / weekly loss limits — measured against starting balance of period
        daily_limit = float(self.strategy_cfg["daily_loss_limit"]) * self.balance
        if self.state.daily_pnl + floating_pnl <= -daily_limit:
            self.state.halted = True
            return False, "daily loss limit hit (incl floating)"
        weekly_limit = float(self.strategy_cfg["weekly_loss_limit"]) * self.balance
        if self.state.weekly_pnl + floating_pnl <= -weekly_limit:
            return False, "weekly loss limit hit (incl floating)"

        # Drawdown halt
        max_dd = float(self.risk_cfg.get("max_drawdown_halt", 0.10))
        if self.state.peak_balance > 0:
            dd = (self.state.peak_balance - self.balance) / self.state.peak_balance
            if dd >= max_dd:
                self.state.halted = True
                return False, f"drawdown halt {dd:.2%}"

        # Correlation filter — entries may be [a, b] or [a, b, threshold].
        # We block any concurrent position in another listed symbol of the cluster.
        for cluster in self.strategy_cfg.get("correlation_filter_pairs", []):
            symbols_in_cluster = {s for s in cluster if isinstance(s, str)}
            if symbol in symbols_in_cluster and (self.state.open_symbols & symbols_in_cluster) - {symbol}:
                return False, f"correlated open position in {sorted(symbols_in_cluster)}"

        # Max positions
        max_open = int(self.risk_cfg.get("max_open_positions", 3))
        if len(self.state.open_symbols) >= max_open:
            return False, "max open positions"

        # Position-size cap
        max_contracts = float(self.risk_cfg.get("max_position_size_contracts", 1.0))
        if proposed_volume > max_contracts:
            return False, f"volume {proposed_volume} exceeds cap {max_contracts}"

        return True, "ok"

    def adaptive_risk_multiplier(
        self,
        now: datetime,
        in_killzone: bool = False,
    ) -> float:
        """Return a float multiplier to apply to the base risk_per_trade.

        Logic (all thresholds are configurable via strategy_cfg):

        * When the trailing win rate (last N trades) falls below
          ``adaptive_sizing_low_wr``, enter the *low-WR regime* and apply
          ``adaptive_sizing_low_mult`` (e.g. 0.7 = 70% of normal size).
        * Restore full size once WR recovers above ``adaptive_sizing_high_wr``
          (hysteresis prevents rapid switching).
        * If the signal was generated inside a killzone, multiply by
          ``adaptive_sizing_kz_mult`` (e.g. 1.1 = 10% larger).
        * Adaptive sizing is a no-op when ``adaptive_sizing`` is False in config.

        Args:
            now: current UTC datetime (used for regime log timestamps).
            in_killzone: True when the entry signal is inside London/NY open.

        Returns:
            A positive float; 1.0 means no change.
        """
        if not self.strategy_cfg.get("adaptive_sizing", False):
            return 1.0

        lookback = int(self.strategy_cfg.get("adaptive_sizing_lookback", 10))
        low_wr = float(self.strategy_cfg.get("adaptive_sizing_low_wr", 0.35))
        high_wr = float(self.strategy_cfg.get("adaptive_sizing_high_wr", 0.45))
        low_mult = float(self.strategy_cfg.get("adaptive_sizing_low_mult", 0.70))
        kz_mult = float(self.strategy_cfg.get("adaptive_sizing_kz_mult", 1.10))

        # Compute rolling win rate from trade history.
        history = self.state.trade_history[-lookback:] if self.state.trade_history else []
        if len(history) >= lookback:
            win_rate = sum(1 for t in history if t.pnl > 0) / len(history)
            if self._in_low_wr_regime:
                # Hysteresis: only exit regime when WR clearly recovers.
                if win_rate >= high_wr:
                    self._in_low_wr_regime = False
            else:
                if win_rate < low_wr:
                    self._in_low_wr_regime = True

        mult = low_mult if self._in_low_wr_regime else 1.0
        if in_killzone:
            mult *= kz_mult
        return float(mult)

    def compute_optimal_risk(
        self,
        ml_probability:      float,
        threshold:           float,
        current_atr:         float,
        reference_atr:       float,
        in_killzone:         bool = False,
        kelly_fraction:      float = 0.25,
    ) -> float:
        """Institutional-grade risk computation combining Kelly, vol-targeting, and confidence.

        Args:
            ml_probability:  model output probability (0-1).
            threshold:       optimal entry threshold (from PF optimizer).
            current_atr:     current bar ATR (used for vol scaling).
            reference_atr:   rolling 100-bar ATR (reference vol level).
            in_killzone:     True if entry is inside London/NY killzone.
            kelly_fraction:  fraction of full Kelly to bet (default 0.25 = quarter-Kelly).

        Returns:
            Risk fraction as a float (e.g. 0.0035 = 0.35% of account).
        """
        base_risk = float(self.strategy_cfg.get("risk_per_trade", 0.0035))

        # --- 1. Fractional Kelly ---
        # Win probability estimate from ML (calibrated probability)
        p_win   = float(np.clip(ml_probability, 0.01, 0.99))
        rr      = float(self.strategy_cfg.get("rr_ratio", 1.5))
        b       = rr          # odds: win rr for every 1 lost
        kelly   = (b * p_win - (1 - p_win)) / b
        kelly   = max(kelly, 0.0)  # never short the Kelly fraction
        kelly_risk = base_risk * min(kelly / 0.15, 2.0) * kelly_fraction
        # Normalise: assume optimal base_risk corresponds to Kelly ≈ 0.15
        # (i.e., a 42% win rate, 1.5 RR system)

        # --- 2. Volatility Targeting ---
        # Scale down when current vol > reference vol
        if reference_atr > 1e-9 and current_atr > 1e-9:
            vol_ratio    = reference_atr / current_atr   # < 1 when vol is elevated
            vol_adj      = float(np.clip(vol_ratio, 0.5, 1.5))
        else:
            vol_adj = 1.0

        # --- 3. Confidence Scaling ---
        # Scale by how far probability is above threshold (capped at +50%)
        if threshold > 0 and ml_probability > threshold:
            excess_prob  = (ml_probability - threshold) / max(1.0 - threshold, 0.01)
            conf_mult    = 1.0 + min(excess_prob, 1.0) * 0.5  # up to 1.5×
        else:
            conf_mult = 1.0

        # --- 4. Adaptive sizing multiplier (low win-rate regime) ---
        from datetime import datetime as _dt
        adapt_mult = self.adaptive_risk_multiplier(_dt.utcnow(), in_killzone)

        # --- Final: combine all factors ---
        optimal_risk = kelly_risk * vol_adj * conf_mult * adapt_mult

        # Hard bounds
        max_risk = float(self.strategy_cfg.get("max_risk_per_trade", 0.01))
        min_risk = float(self.strategy_cfg.get("min_risk_per_trade", 0.001))
        return float(np.clip(optimal_risk, min_risk, max_risk))

    def register_open(self, symbol: str) -> None:
        self.state.open_symbols.add(symbol)
        self.state.daily_trades += 1

    def register_close(
        self,
        now: datetime,
        symbol: str,
        pnl: float,
        r_multiple: float,
    ) -> None:
        """Record a closed trade and update counters."""
        self._roll_periods(now)
        self.state.open_symbols.discard(symbol)
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.balance += pnl
        self.state.peak_balance = max(self.state.peak_balance, self.balance)
        self.state.trade_history.append(
            TradeRecord(timestamp=now, symbol=symbol, pnl=pnl, r_multiple=r_multiple)
        )
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Trigger cooldown on streak
        n_losses = int(self.strategy_cfg.get("consecutive_loss_limit", 3))
        cd_hours = float(self.strategy_cfg.get("consecutive_loss_cooldown_hours", 4))
        if self.state.consecutive_losses >= n_losses:
            self.state.cooldown_until = now + timedelta(hours=cd_hours)
            self.state.consecutive_losses = 0

    # ---------- helpers ----------
    def _roll_periods(self, now: datetime) -> None:
        day = now.date()
        week = now.isocalendar()[1]
        if self._current_day != day:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self._current_day = day
            # FIX BUG-M6: Reset the daily-loss-triggered halt at the start of
            # each new trading day so the bot can resume after overnight reset.
            # Max-DD halts are intentionally permanent (require manual restart).
            max_dd = float(self.risk_cfg.get("max_drawdown_halt", 0.10))
            if self.state.halted and self.state.peak_balance > 0:
                current_dd = (self.state.peak_balance - self.balance) / self.state.peak_balance
                if current_dd < max_dd:
                    # Not in a max-DD state — halt was triggered by daily limit, clear it.
                    self.state.halted = False
        if self._current_week != week:
            self.state.weekly_pnl = 0.0
            self._current_week = week
