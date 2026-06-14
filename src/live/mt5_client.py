"""MetaTrader 5 client.

Wraps the ``MetaTrader5`` Python package. The package is Windows-only and
optional — every method degrades gracefully when MT5 is unavailable so the
rest of the system (backtest, training) keeps working on macOS / Linux.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5  # type: ignore
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None


TIMEFRAME_MAP_KEYS = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _tf(timeframe: str) -> int:
    if not HAS_MT5:
        raise RuntimeError("MetaTrader5 package not available")
    key = TIMEFRAME_MAP_KEYS.get(timeframe.upper())
    if key is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return getattr(mt5, key)


class MT5Client:
    """Thin wrapper around the MetaTrader5 package."""

    def __init__(self):
        if not HAS_MT5:
            logger.warning("MetaTrader5 package not installed — running in mock mode.")
        self.connected = False

    # ---------- connection ----------
    def connect(
        self,
        account: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Initialize MT5 and (optionally) log in to a specific account."""
        if not HAS_MT5:
            return False
        kwargs: Dict[str, Any] = {}
        if path:
            kwargs["path"] = path
        if not mt5.initialize(**kwargs):
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        if account is not None and password is not None and server is not None:
            if not mt5.login(int(account), password=password, server=server):
                logger.error("MT5 login failed: %s", mt5.last_error())
                mt5.shutdown()
                return False
        self.connected = True
        return True

    def disconnect(self) -> None:
        if HAS_MT5 and self.connected:
            mt5.shutdown()
            self.connected = False

    # ---------- data ----------
    def fetch_rates(
        self,
        symbol: str,
        timeframe: str,
        count: int,
        from_pos: int = 0,
    ) -> pd.DataFrame:
        """Fetch the latest ``count`` bars; returns an OHLCV DataFrame."""
        if not HAS_MT5 or not self.connected:
            raise RuntimeError("MT5 not connected")

        # Ensure symbol is selected in Market Watch so data fetch doesn't fail
        mt5.symbol_select(symbol, True)

        # Ensure we don't ask for more bars than the terminal's allowed maximum
        maxbars = mt5.terminal_info().maxbars
        if count > maxbars:
            logger.warning("Requested %d bars for %s but MT5 maxbars is %d. Capping request.", count, symbol, maxbars)
            count = maxbars

        rates = mt5.copy_rates_from_pos(symbol, _tf(timeframe), from_pos, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df = df.set_index("time")
        df = df.rename(columns={"tick_volume": "volume"})
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep]

    # ---------- trading ----------
    def place_order(
        self,
        symbol: str,
        action: str,                 # 'buy' or 'sell'
        volume: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        deviation: int = 20,
        magic: int = 234000,
        comment: str = "ict_bot",
    ) -> Optional[Dict[str, Any]]:
        """Submit a market order. Returns broker result dict or None on failure."""
        if not HAS_MT5 or not self.connected:
            raise RuntimeError("MT5 not connected")
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error("Symbol %s not found", symbol)
            return None
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        price = tick.ask if action.lower() == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if action.lower() == "buy" else mt5.ORDER_TYPE_SELL
        vol_step = getattr(symbol_info, "volume_step", 0.01)
        vol_min = getattr(symbol_info, "volume_min", 0.01)
        vol_max = getattr(symbol_info, "volume_max", 1000.0)
        safe_volume = round(float(volume) / vol_step) * vol_step
        safe_volume = max(vol_min, min(safe_volume, vol_max))
        # Handle float precision issues
        decimals = len(str(vol_step).split(".")[1]) if "." in str(vol_step) else 0
        safe_volume = round(safe_volume, decimals)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": safe_volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)
        result = mt5.order_send(request)
        if result is None:
            logger.error("order_send returned None: %s", mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("Order failed retcode=%s comment=%s", result.retcode, result.comment)
        return result._asdict() if hasattr(result, "_asdict") else dict(result)

    def modify_order(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Modify the SL and/or TP of an open position by ticket number.

        This enables live trailing-stop adjustments.  Returns the broker
        result dict on success, or None if the request fails / MT5 is
        unavailable.

        Args:
            ticket: the position ticket to modify.
            sl: new stop-loss price, or None to leave unchanged.
            tp: new take-profit price, or None to leave unchanged.
        """
        if not HAS_MT5 or not self.connected:
            logger.warning("modify_order called but MT5 is not connected.")
            return None
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning("modify_order: ticket %d not found in open positions", ticket)
            return None
        pos = positions[0]
        request: Dict[str, Any] = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol":   pos.symbol,
            "sl":       float(sl) if sl is not None else float(pos.sl),
            "tp":       float(tp) if tp is not None else float(pos.tp),
        }
        result = mt5.order_send(request)
        if result is None:
            logger.error("modify_order order_send returned None: %s", mt5.last_error())
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(
                "modify_order failed ticket=%d retcode=%s comment=%s",
                ticket, result.retcode, result.comment,
            )
        return result._asdict() if hasattr(result, "_asdict") else dict(result)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if not HAS_MT5 or not self.connected:
            return []
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        return [p._asdict() for p in positions]

    def close_position(self, ticket: int, deviation: int = 20) -> Optional[Dict[str, Any]]:
        if not HAS_MT5 or not self.connected:
            raise RuntimeError("MT5 not connected")
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return None
        pos = positions[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": int(ticket),
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": pos.magic,
            "comment": "ict_bot_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return result._asdict() if (result and hasattr(result, "_asdict")) else None

    def account_info(self) -> Optional[Dict[str, Any]]:
        if not HAS_MT5 or not self.connected:
            return None
        info = mt5.account_info()
        return info._asdict() if info else None

    def get_deal_profit(self, order_ticket: int, lookback_days: int = 7) -> Optional[float]:
        """Return the net profit of all deal rows matched by *order_ticket*.

        MT5 can produce multiple deal entries for a single order (partial fills,
        swap/commission rows, etc.).  We sum every deal whose ``order`` field
        matches to avoid under-counting PnL.  ``lookback_days`` defaults to 7
        so trades held over a weekend or on H1 are always captured — the
        previous hard-coded 24-hour window caused the risk manager to record
        phantom breakevens for longer-held positions.
        """
        if not HAS_MT5 or not self.connected:
            return None
        from datetime import datetime, timedelta
        date_from = datetime.utcnow() - timedelta(days=max(1, lookback_days))
        date_to = datetime.utcnow()
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return None
        total = sum(
            float(d.profit)
            for d in deals
            if int(d.order) == order_ticket
        )
        # Return None (not 0) when the ticket was genuinely not found so the
        # caller can distinguish "deal not yet settled" from "zero-profit deal".
        matched = any(int(d.order) == order_ticket for d in deals)
        return total if matched else None

    def get_spread_pips(
        self,
        symbol: str,
        pip_size: float = 0.0001,
    ) -> float:
        """Return the current bid/ask spread in pips for *symbol*.

        Returns a large sentinel (999.0) when MT5 is unavailable so callers
        can safely compare against a max-spread threshold and reject the trade.

        Args:
            symbol: instrument name, e.g. ``'EURUSD'``.
            pip_size: pip size in price units (0.0001 for 4-decimal FX,
                0.01 for JPY pairs, 1.0 for indices).
        """
        if not HAS_MT5 or not self.connected:
            return 999.0
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 999.0
        spread_price = abs(tick.ask - tick.bid)
        return float(spread_price / pip_size) if pip_size > 0 else 999.0
