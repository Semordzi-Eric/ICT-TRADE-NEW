"""Higher-Timeframe (H4) directional bias.

Runs Break-of-Structure + Change-of-Character detection on H4 candles and
exposes a simple ``get_bias()`` call that returns ``'long'``, ``'short'``, or
``'neutral'``.  The result is cached and only refreshed when a new H4 bar
closes, so it adds negligible overhead to the M15 live loop.

Usage::

    from src.utils.htf_bias import HTFBiasCache

    cache = HTFBiasCache(mt5_client, symbol="EURUSD", timeframe="H4", bars=200)
    bias = cache.get_bias()   # 'long' | 'short' | 'neutral'
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from ..detection.structure import detect_bos, detect_choch

logger = logging.getLogger(__name__)

# How many M15 bars elapse per H4 bar (4 * 4 = 16 … but we also poll at 5-sec
# intervals so we use a simple time-based guard instead).
_SECONDS_PER_H4_BAR = 4 * 3600


def get_htf_bias(
    candles_h4: pd.DataFrame,
    swing_lookback: int = 5,
    bos_confirmation_bars: int = 2,
) -> str:
    """Derive directional bias from H4 BOS / CHoCH events.

    Returns:
        ``'long'`` if the most-recent structural event is bullish.
        ``'short'`` if bearish.
        ``'neutral'`` if no events found.
    """
    if candles_h4 is None or candles_h4.empty:
        return "neutral"

    bos_list = detect_bos(
        candles_h4,
        confirmation_bars=bos_confirmation_bars,
        swing_lookback=swing_lookback,
    )
    choch_list = detect_choch(candles_h4, swing_lookback=swing_lookback)

    all_events = sorted(bos_list + choch_list, key=lambda e: e.index)
    if not all_events:
        return "neutral"

    last = all_events[-1]
    return "long" if last.direction == "bullish" else "short"


class HTFBiasCache:
    """Per-symbol H4 bias cache for the live trading loop.

    Fetches H4 data from MT5 once per H4 bar and caches the result.  Falls
    back gracefully when MT5 is unavailable or has insufficient data.

    Args:
        mt5_client: connected ``MT5Client`` instance.
        symbol: e.g. ``"EURUSD"``.
        timeframe: the higher timeframe string, typically ``"H4"``.
        bars: number of H4 bars to fetch (200 ≈ 33 trading days).
        swing_lookback: passed through to the structure detectors.
    """

    def __init__(
        self,
        mt5_client,
        symbol: str,
        timeframe: str = "H4",
        bars: int = 200,
        swing_lookback: int = 5,
    ):
        self._mt5 = mt5_client
        self.symbol = symbol
        self.timeframe = timeframe
        self.bars = bars
        self.swing_lookback = swing_lookback

        self._cached_bias: str = "neutral"
        self._last_fetch_ts: Optional[datetime] = None
        self._last_h4_bar_time: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bias(self) -> str:
        """Return the current H4 directional bias.

        Refreshes the cache when a new H4 bar has closed since the last call.
        Returns ``'neutral'`` on any error so the caller is never blocked.
        """
        try:
            self._maybe_refresh()
        except Exception:
            logger.exception("HTFBiasCache.get_bias failed for %s — returning neutral", self.symbol)
        return self._cached_bias

    def invalidate(self) -> None:
        """Force a cache refresh on the next ``get_bias()`` call."""
        self._last_h4_bar_time = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_refresh(self) -> None:
        """Refresh only when a new H4 bar has closed.

        BUG-M2 FIX: Previously the MT5 fetch was executed on *every* call,
        adding 16 unnecessary network round-trips per 5-second poll cycle.
        Now we gate the fetch behind a time check: we only query MT5 if at
        least ``_SECONDS_PER_H4_BAR`` seconds have elapsed since the last
        successful fetch.  The bar-time guard still short-circuits computation
        if the same H4 bar has not yet closed.
        """
        import time as _time
        now_ts = _time.monotonic()
        last_fetch_mono = getattr(self, "_last_fetch_mono", 0.0)
        # Only hit MT5 if enough time has passed for a new H4 bar to close.
        if now_ts - last_fetch_mono < _SECONDS_PER_H4_BAR:
            return  # cached bias still valid within this H4 bar window

        candles = self._mt5.fetch_rates(self.symbol, self.timeframe, self.bars, from_pos=1)
        self._last_fetch_mono = now_ts  # record fetch time even if empty
        if candles is None or candles.empty:
            logger.warning("HTFBiasCache: no H4 data for %s", self.symbol)
            return

        latest = candles.index[-1]
        if latest == self._last_h4_bar_time:
            return  # same H4 bar — no need to recalculate

        bias = get_htf_bias(candles, swing_lookback=self.swing_lookback)
        if bias != self._cached_bias:
            logger.info(
                "HTF bias changed for %s: %s → %s (bar %s)",
                self.symbol, self._cached_bias, bias, latest,
            )
        self._cached_bias = bias
        self._last_h4_bar_time = latest
