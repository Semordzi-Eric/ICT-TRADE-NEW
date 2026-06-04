"""Macroeconomic News Filter.

Fetches and caches high-impact news events from Forex Factory's public XML
feed and blocks trade entries within a configurable time window around them.

Usage::

    filter = NewsFilter(blackout_minutes=30)
    filter.refresh()                        # download / update cache
    ok, reason = filter.is_clear(now, "EURUSD")
    if not ok:
        print(f"Skip trade — {reason}")

Forex Factory XML:
  https://nfs.faireconomy.media/ff_calendar_thisweek.xml
  Updated each Sunday. Fields: country, date, time, title, impact.
"""
from __future__ import annotations

import logging
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# Currencies that each symbol affects (used for news relevance matching).
_SYMBOL_CURRENCIES: dict = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "USDCHF": ["USD", "CHF"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "NAS100":  ["USD"],
    "SPX500":  ["USD"],
    "XAUUSD":  ["USD"],
}

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
HIGH_IMPACT_VALUES = {"High", "high", "HIGH"}


class NewsEvent:
    """A single macroeconomic event parsed from the FF XML feed."""
    __slots__ = ("country", "title", "impact", "dt")

    def __init__(self, country: str, title: str, impact: str, dt: datetime):
        self.country = country.upper().strip()
        self.title = title.strip()
        self.impact = impact.strip()
        self.dt = dt

    def __repr__(self) -> str:
        return f"NewsEvent({self.dt.isoformat()} {self.country} '{self.title}' [{self.impact}])"


# Country-code → ISO currency
_COUNTRY_CURRENCY: dict = {
    "US": "USD",
    "USD": "USD",
    "EU": "EUR",
    "EUR": "EUR",
    "UK": "GBP",
    "GBP": "GBP",
    "JP": "JPY",
    "JPY": "JPY",
    "AU": "AUD",
    "AUD": "AUD",
    "NZ": "NZD",
    "NZD": "NZD",
    "CA": "CAD",
    "CAD": "CAD",
    "CH": "CHF",
    "CHF": "CHF",
}


def _country_to_currency(country: str) -> str:
    return _COUNTRY_CURRENCY.get(country.upper(), country.upper())


def _parse_feed(xml_text: str) -> List[NewsEvent]:
    """Parse Forex Factory XML into a list of NewsEvent objects."""
    events: List[NewsEvent] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse news XML: %s", exc)
        return events

    current_year = datetime.utcnow().year

    for item in root.findall("eventitem"):
        country_el = item.find("country")
        title_el   = item.find("title")
        impact_el  = item.find("impact")
        date_el    = item.find("date")
        time_el    = item.find("time")

        if any(el is None for el in [country_el, title_el, impact_el, date_el]):
            continue

        country = (country_el.text or "").strip()
        title   = (title_el.text or "").strip()
        impact  = (impact_el.text or "").strip()
        date_s  = (date_el.text or "").strip()
        time_s  = (time_el.text or "").strip() if time_el is not None else "00:00am"

        try:
            # FF date format: "Tuesday Jan 05" or "Thursday Jan 07"
            dt = datetime.strptime(f"{date_s} {current_year}", "%A %b %d %Y")
            # Time: "8:30am" / "12:00pm" / "All Day"
            if re.match(r"\d{1,2}:\d{2}[ap]m", time_s, re.IGNORECASE):
                t = datetime.strptime(time_s.lower(), "%I:%M%p")
                dt = dt.replace(hour=t.hour, minute=t.minute)
        except ValueError:
            continue

        events.append(NewsEvent(country=country, title=title, impact=impact, dt=dt))

    return events


class NewsFilter:
    """Downloads, caches, and exposes high-impact news events.

    Thread-safe: ``refresh()`` can be called from a background thread while
    ``is_clear()`` is called from the main trading loop.
    """

    def __init__(
        self,
        blackout_minutes: int = 30,
        cache_path: Optional[str] = None,
        auto_refresh_hours: float = 12.0,
    ):
        self.blackout_minutes = blackout_minutes
        self.cache_path = Path(cache_path) if cache_path else Path("data/news_cache.xml")
        self.auto_refresh_hours = auto_refresh_hours
        self._events: List[NewsEvent] = []
        self._lock = threading.Lock()
        self._last_refresh: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, force: bool = False) -> bool:
        """Download the latest Forex Factory feed and parse it.

        Args:
            force: If True, ignore the auto-refresh interval and re-download.

        Returns:
            True if refresh succeeded (network or cache), False on failure.
        """
        if not force and self._last_refresh is not None:
            age = (datetime.utcnow() - self._last_refresh).total_seconds() / 3600
            if age < self.auto_refresh_hours:
                return True  # still fresh

        xml_text = self._download() or self._load_cache()
        if xml_text is None:
            logger.warning("NewsFilter: no feed available — news filtering disabled")
            return False

        events = _parse_feed(xml_text)
        high_impact = [e for e in events if e.impact in HIGH_IMPACT_VALUES]

        with self._lock:
            self._events = high_impact
            self._last_refresh = datetime.utcnow()

        logger.info("NewsFilter: loaded %d high-impact events (%d total parsed)",
                    len(high_impact), len(events))
        return True

    def is_clear(self, now: datetime, symbol: str) -> Tuple[bool, str]:
        """Check whether trading is safe at `now` for `symbol`.

        Returns:
            (True, "ok") if safe, or (False, reason_string) if blocked.
        """
        with self._lock:
            events = list(self._events)

        if not events:
            return True, "ok"

        currencies = _SYMBOL_CURRENCIES.get(symbol.upper(), [symbol.upper()[:3], symbol.upper()[3:]])
        window = timedelta(minutes=self.blackout_minutes)

        for evt in events:
            evt_ccy = _country_to_currency(evt.country)
            if evt_ccy not in currencies:
                continue
            delta = abs((now - evt.dt).total_seconds())
            if delta <= window.total_seconds():
                direction = "before" if now < evt.dt else "after"
                mins = int(delta / 60)
                return False, (
                    f"news blackout: '{evt.title}' ({evt.country}) "
                    f"in {mins}min {direction} — {evt.dt.strftime('%H:%M UTC')}"
                )
        return True, "ok"

    def upcoming_events(self, within_hours: float = 24.0) -> List[NewsEvent]:
        """Return high-impact events occurring within the next N hours."""
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=within_hours)
        with self._lock:
            return [e for e in self._events if now <= e.dt <= cutoff]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _download(self) -> Optional[str]:
        try:
            import urllib.request
            req = urllib.request.Request(
                FEED_URL,
                headers={"User-Agent": "ICT-TradingBot/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(text, encoding="utf-8")
            logger.info("NewsFilter: downloaded feed -> %s", self.cache_path)
            return text
        except Exception as exc:
            logger.warning("NewsFilter: download failed: %s", exc)
            return None

    def _load_cache(self) -> Optional[str]:
        if self.cache_path.exists():
            logger.info("NewsFilter: loading from cache %s", self.cache_path)
            return self.cache_path.read_text(encoding="utf-8")
        return None
