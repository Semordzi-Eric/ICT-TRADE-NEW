"""Performance metrics: Sharpe, Sortino, Calmar, win rate, profit factor, etc."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .engine import BacktestResult


def compute_metrics(
    result: BacktestResult,
    bars_per_year: int = 252 * 78,   # 5-min bars × 78/day × 252 days
    risk_free_rate: float = 0.0,
    train_sharpe: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the full metrics suite from a BacktestResult.

    ``bars_per_year`` is used to annualise. Defaults to ~252 trading days
    × 78 five-minute bars per day. Adjust for your timeframe.
    """
    eq = result.equity_curve
    if len(eq) < 2:
        return _empty_metrics()
    rets = eq.pct_change().fillna(0).values
    trades_df = result.trades_df()

    sharpe = _sharpe(rets, bars_per_year, risk_free_rate)
    sortino = _sortino(rets, bars_per_year, risk_free_rate)
    max_dd, max_dd_pct = _max_drawdown(eq.values)
    calmar = _calmar(eq.values, bars_per_year, max_dd_pct)

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = float(len(wins) / len(trades_df))
        avg_win = float(wins["pnl"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) else 0.0
        gross_profit = float(wins["pnl"].sum())
        gross_loss = float(-losses["pnl"].sum())
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else float("inf") if gross_profit > 0 else 0.0
        )
        avg_r = float(trades_df["r_multiple"].mean())
        expectancy = float(trades_df["pnl"].mean())
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_r = expectancy = 0.0
        gross_profit = gross_loss = 0.0

    total_return = (
        float(result.ending_balance / result.starting_balance - 1.0)
        if result.starting_balance > 0 else 0.0
    )

    gen_ratio = 1.0
    if train_sharpe is not None and train_sharpe > 1e-6:
        gen_ratio = float(sharpe / train_sharpe)

    return {
        "total_return": total_return,
        "net_pnl": float(result.ending_balance - result.starting_balance),
        "ending_balance": float(result.ending_balance),
        "n_trades": int(len(trades_df)) if not trades_df.empty else 0,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "avg_r_multiple": avg_r,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "generalization_ratio": gen_ratio,
    }


# ---------- helpers ----------
def _sharpe(rets: np.ndarray, periods_per_year: int, rf: float) -> float:
    excess = rets - rf / periods_per_year
    sd = np.std(excess)
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def _sortino(rets: np.ndarray, periods_per_year: int, rf: float) -> float:
    excess = rets - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) <= 1e-12:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray):
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    dd_pct = np.where(peak > 0, dd / peak, 0.0)
    return float(dd.min()), float(dd_pct.min())


def _calmar(equity: np.ndarray, periods_per_year: int, max_dd_pct: float) -> float:
    if len(equity) < 2 or max_dd_pct == 0:
        return 0.0
    total_return = equity[-1] / equity[0] - 1.0
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    cagr = (1 + total_return) ** (1 / n_years) - 1
    return float(cagr / abs(max_dd_pct))


def _empty_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in [
        "total_return", "net_pnl", "ending_balance", "n_trades", "win_rate",
        "avg_win", "avg_loss", "expectancy", "avg_r_multiple",
        "gross_profit", "gross_loss", "profit_factor", "sharpe",
        "sortino", "calmar", "max_drawdown", "max_drawdown_pct",
        "generalization_ratio",
    ]}
