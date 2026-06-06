"""Account-level risk management: loss limits, cooldowns, correlation."""
from __future__ import annotations

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
    """Enforces account-level rules. Call :meth:`can_trade` before every entry."""

    def __init__(self, strategy_cfg: dict, risk_cfg: dict, account_balance: float):
        self.strategy_cfg = strategy_cfg
        self.risk_cfg = risk_cfg
        self.balance = account_balance
        self.state = RiskState(peak_balance=account_balance)
        self._current_day: Optional[datetime] = None
        self._current_week: Optional[int] = None
        # Adaptive-sizing state: True when win rate is in the low zone.
        self._in_low_wr_regime: bool = False

    # ---------- public API ----------
    def can_trade(
        self,
        now: datetime,
        symbol: str,
        proposed_volume: float,
    ) -> Tuple[bool, str]:
        """Check all gates. Returns ``(ok, reason)``."""
        self._roll_periods(now)

        if self.state.halted:
            return False, "account halted (max DD)"

        # Cooldown
        if self.state.cooldown_until and now < self.state.cooldown_until:
            return False, f"in cooldown until {self.state.cooldown_until}"

        # Daily trade cap
        if self.state.daily_trades >= int(self.strategy_cfg["max_daily_trades"]):
            return False, "max daily trades reached"

        # Daily / weekly loss limits — measured against starting balance of period
        daily_limit = float(self.strategy_cfg["daily_loss_limit"]) * self.balance
        if self.state.daily_pnl <= -daily_limit:
            return False, "daily loss limit hit"
        weekly_limit = float(self.strategy_cfg["weekly_loss_limit"]) * self.balance
        if self.state.weekly_pnl <= -weekly_limit:
            return False, "weekly loss limit hit"

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
        if self._current_week != week:
            self.state.weekly_pnl = 0.0
            self._current_week = week
