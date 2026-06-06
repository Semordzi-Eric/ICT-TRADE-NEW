"""Vectorised backtest engine.

Walks the bar series once and simulates every signal with realistic SL/TP,
slippage and commission. Output is a trade list and an equity curve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ..strategy.rule_based import Signal, position_size


@dataclass
class TradeResult:
    entry_index: int
    exit_index: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    volume: float
    pnl: float
    r_multiple: float
    bars_held: int
    exit_reason: str            # 'tp' | 'sl' | 'time'
    setup_type: str


@dataclass
class BacktestResult:
    trades: List[TradeResult]
    equity_curve: pd.Series
    starting_balance: float
    ending_balance: float

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])


def run_backtest(
    candles: pd.DataFrame,
    signals: List[Signal],
    starting_balance: float = 10_000.0,
    risk_per_trade: float = 0.005,
    slippage_pips: float = 0.5,
    pip_size: float = 0.0001,
    commission_per_lot: float = 5.0,
    contract_value: float = 100_000.0,
    max_holding_bars: int = 36,
    max_concurrent: int = 3,
    trail_activation_r: float = 0.5,
    trail_atr_mult: float = 1.0,
) -> BacktestResult:
    """Simulate execution of a list of signals.

    Args:
        candles: OHLCV DataFrame.
        signals: list of Signals from the strategy layer.
        starting_balance: account balance at t0.
        risk_per_trade: fraction of balance risked on each trade.
        slippage_pips: applied to both entry and exit, in the adverse direction.
        pip_size: price units per pip (0.0001 for typical FX, adapt for indices).
        commission_per_lot: round-turn cost in account currency.
        contract_value: notional per 1.0 volume (100k for FX).
        max_holding_bars: time stop.
        max_concurrent: cap simultaneous open trades.
    """
    high = candles["high"].values
    low = candles["low"].values
    close = candles["close"].values
    open_ = candles["open"].values
    n = len(candles)
    slippage = slippage_pips * pip_size
    balance = starting_balance
    equity = pd.Series(starting_balance, index=candles.index, dtype=float)

    from ..detection.liquidity import _atr as _compute_atr
    atr_vals = _compute_atr(candles, 14).ffill().fillna(0.0).values

    trades: List[TradeResult] = []
    open_trades: List[dict] = []

    # Build a per-bar queue so signals are never silently dropped when
    # max_concurrent is full — they stay eligible for the next bar.
    from collections import defaultdict
    sig_queue: dict = defaultdict(list)
    for s in signals:
        sig_queue[s.index].append(s)
    pending: List[Signal] = []  # spill-over from prior bars

    for i in range(n):
        # Update open trades — check exits first
        still_open = []
        for tr in open_trades:
            exited = False
            cur_atr = atr_vals[i] if np.isfinite(atr_vals[i]) and atr_vals[i] > 0 else 0.0

            # --- Snapshot SL *before* the trail update so that exits this bar
            # are evaluated against the level that was set at bar open, not the
            # level the trail may have just tightened to.  This prevents the
            # optimistic scenario where the trail fires first and causes a
            # premature stop on the tighter level within the same bar. ---
            sl_at_bar_open = tr["stop_loss"]

            if tr["direction"] == "long":
                if low[i] <= sl_at_bar_open:
                    exit_price = sl_at_bar_open - slippage
                    pnl = (exit_price - tr["entry_price"]) * tr["volume"] * contract_value - tr["commission"]
                    r = (exit_price - tr["entry_price"]) / abs(tr["entry_price"] - tr["initial_sl"])
                    trades.append(_finalise(tr, i, exit_price, pnl, r, "sl", candles))
                    balance += pnl
                    exited = True
                elif high[i] >= tr["take_profit"]:
                    exit_price = tr["take_profit"] - slippage
                    pnl = (exit_price - tr["entry_price"]) * tr["volume"] * contract_value - tr["commission"]
                    r = (exit_price - tr["entry_price"]) / abs(tr["entry_price"] - tr["initial_sl"])
                    trades.append(_finalise(tr, i, exit_price, pnl, r, "tp", candles))
                    balance += pnl
                    exited = True
            else:  # short
                if high[i] >= sl_at_bar_open:
                    exit_price = sl_at_bar_open + slippage
                    pnl = (tr["entry_price"] - exit_price) * tr["volume"] * contract_value - tr["commission"]
                    r = (tr["entry_price"] - exit_price) / abs(tr["entry_price"] - tr["initial_sl"])
                    trades.append(_finalise(tr, i, exit_price, pnl, r, "sl", candles))
                    balance += pnl
                    exited = True
                elif low[i] <= tr["take_profit"]:
                    exit_price = tr["take_profit"] + slippage
                    pnl = (tr["entry_price"] - exit_price) * tr["volume"] * contract_value - tr["commission"]
                    r = (tr["entry_price"] - exit_price) / abs(tr["entry_price"] - tr["initial_sl"])
                    trades.append(_finalise(tr, i, exit_price, pnl, r, "tp", candles))
                    balance += pnl
                    exited = True

            if not exited and i - tr["entry_index"] >= max_holding_bars:
                exit_price = close[i] - slippage if tr["direction"] == "long" else close[i] + slippage
                if tr["direction"] == "long":
                    pnl = (exit_price - tr["entry_price"]) * tr["volume"] * contract_value - tr["commission"]
                    r = (exit_price - tr["entry_price"]) / abs(tr["entry_price"] - tr["initial_sl"])
                else:
                    pnl = (tr["entry_price"] - exit_price) * tr["volume"] * contract_value - tr["commission"]
                    r = (tr["entry_price"] - exit_price) / abs(tr["entry_price"] - tr["initial_sl"])
                trades.append(_finalise(tr, i, exit_price, pnl, r, "time", candles))
                balance += pnl
                exited = True

            if not exited:
                # --- Update trailing stop *after* exit check (next-bar effect) ---
                if cur_atr > 0:
                    if tr["direction"] == "long":
                        unrealised_r = (close[i] - tr["entry_price"]) / abs(tr["entry_price"] - tr["initial_sl"])
                        if unrealised_r >= trail_activation_r:
                            new_trail_sl = close[i] - trail_atr_mult * cur_atr
                            tr["stop_loss"] = max(tr["stop_loss"], new_trail_sl)
                    else:
                        unrealised_r = (tr["entry_price"] - close[i]) / abs(tr["entry_price"] - tr["initial_sl"])
                        if unrealised_r >= trail_activation_r:
                            new_trail_sl = close[i] + trail_atr_mult * cur_atr
                            tr["stop_loss"] = min(tr["stop_loss"], new_trail_sl)
                still_open.append(tr)
        open_trades = still_open


        # --- Signal entry logic (P1 fix: use per-bar queue, no silent drops) ---
        # Collect signals that fired on the previous bar + any spill-over.
        pending.extend(sig_queue.get(i - 1, []))
        still_pending = []
        for sig in pending:
            if len(open_trades) >= max_concurrent:
                still_pending.append(sig)   # defer, not dropped
                continue
            # Execute at this bar's open (realistic; avoids same-bar lookahead).
            raw_entry = open_[i]
            entry_price = (
                raw_entry + slippage if sig.direction == "long"
                else raw_entry - slippage
            )
            vol = position_size(
                balance, risk_per_trade,
                entry_price, sig.stop_loss,
                contract_value=contract_value,
            )
            if vol > 0:
                commission = commission_per_lot * vol
                open_trades.append(
                    {
                        "entry_index": i,
                        "entry_price": entry_price,
                        "stop_loss": sig.stop_loss,
                        "initial_sl": sig.stop_loss,
                        "take_profit": sig.take_profit,
                        "direction": sig.direction,
                        "volume": vol,
                        "commission": commission,
                        "setup_type": sig.setup_type,
                    }
                )
        # Keep deferred signals only until next bar (stale after 1 bar).
        pending = []

        # --- Equity: closed balance + unrealized MTM (P2 fix) ---
        unrealised = 0.0
        for tr in open_trades:
            if tr["direction"] == "long":
                unrealised += (close[i] - tr["entry_price"]) * tr["volume"] * contract_value - tr["commission"]
            else:
                unrealised += (tr["entry_price"] - close[i]) * tr["volume"] * contract_value - tr["commission"]
        equity.iloc[i] = balance + unrealised

    # Force-close any still-open trades at the last bar
    last_i = n - 1
    for tr in open_trades:
        exit_price = close[last_i]
        if tr["direction"] == "long":
            pnl = (exit_price - tr["entry_price"]) * tr["volume"] * contract_value - tr["commission"]
            r = (exit_price - tr["entry_price"]) / abs(tr["entry_price"] - tr["stop_loss"])
        else:
            pnl = (tr["entry_price"] - exit_price) * tr["volume"] * contract_value - tr["commission"]
            r = (tr["entry_price"] - exit_price) / abs(tr["entry_price"] - tr["stop_loss"])
        trades.append(_finalise(tr, last_i, exit_price, pnl, r, "time", candles))
        balance += pnl
    equity.iloc[-1] = balance

    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        starting_balance=starting_balance,
        ending_balance=balance,
    )


def _finalise(tr: dict, exit_i: int, exit_price: float, pnl: float, r: float,
              reason: str, candles: pd.DataFrame) -> TradeResult:
    return TradeResult(
        entry_index=tr["entry_index"],
        exit_index=exit_i,
        entry_time=candles.index[tr["entry_index"]],
        exit_time=candles.index[exit_i],
        direction=tr["direction"],
        entry_price=tr["entry_price"],
        exit_price=float(exit_price),
        stop_loss=tr["stop_loss"],
        take_profit=tr["take_profit"],
        volume=tr["volume"],
        pnl=float(pnl),
        r_multiple=float(r),
        bars_held=exit_i - tr["entry_index"],
        exit_reason=reason,
        setup_type=tr["setup_type"],
    )
