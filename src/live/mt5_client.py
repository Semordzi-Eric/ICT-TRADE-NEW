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

    def get_deal_profit(self, order_ticket: int) -> Optional[float]:
        """Return the net profit of a closed deal matched by its order ticket.

        Scans the last 24 hours of deal history. Returns None if not found.
        """
        if not HAS_MT5 or not self.connected:
            return None
        from datetime import datetime, timedelta
        date_from = datetime.utcnow() - timedelta(days=1)
        date_to = datetime.utcnow()
        deals = mt5.history_deals_get(date_from, date_to)
        if deals is None:
            return None
        for deal in deals:
            if int(deal.order) == order_ticket:
                return float(deal.profit)
        return None
