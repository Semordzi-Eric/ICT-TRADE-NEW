"""Multi-source Market Sentiment Engine.

Aggregates signals from:
  1. Forex Factory XML (existing — weekly high-impact economic calendar)
  2. Investing.com economic calendar (monthly view, HTTP GET — no JS needed)
  3. CryptoPanic API (free tier) — BTC/ETH sentiment polarity
  4. Keyword sentiment scan over RSS headlines for metals & indices

The main class is ``SentimentEngine``. Call ``get_sentiment(symbol)`` to
retrieve a normalised score in [-1.0, +1.0] and a news-clear flag.

Thread-safe: all fetching runs in a background daemon thread and is never
called on the hot trading path.

Usage::

    from src.utils.sentiment_engine import SentimentEngine

    engine = SentimentEngine()
    engine.start_background_refresh()          # starts daemon thread

    result = engine.get_sentiment("XAUUSD")
    # result → {
    #     "score":        0.45,        # +1 = bullish, -1 = bearish
    #     "sources":      ["ff", "rss"],
    #     "events_today": 2,
    #     "is_blocked":   False,
    #     "label":        "mildly_bullish"
    # }
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .constants import COUNTRY_CURRENCY_MAP as _COUNTRY_CURRENCY_MAP
from .constants import SYMBOL_CURRENCIES as _SYMBOL_CURRENCIES

logger = logging.getLogger(__name__)

_FF_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?auth_token={token}&currencies={currency}&kind=news&public=true"
_RSS_URLS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.investing.com/rss/news_25.rss",   # Commodities RSS
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]

# Default CryptoPanic token (public read-only free tier — no auth needed for public posts).
_CRYPTOPANIC_PUBLIC_TOKEN = "public"


class SentimentResult:
    """Immutable sentiment snapshot for one symbol."""

    __slots__ = ("symbol", "score", "sources", "events_today",
                 "is_blocked", "blackout_reason", "label", "updated_at")

    def __init__(
        self,
        symbol: str,
        score: float,
        sources: List[str],
        events_today: int,
        is_blocked: bool,
        blackout_reason: str = "",
    ) -> None:
        self.symbol = symbol
        self.score = max(-1.0, min(1.0, score))
        self.sources = sources
        self.events_today = events_today
        self.is_blocked = is_blocked
        self.blackout_reason = blackout_reason
        self.updated_at = datetime.now(timezone.utc)

        if score >= 0.4:
            self.label = "bullish"
        elif score >= 0.15:
            self.label = "mildly_bullish"
        elif score <= -0.4:
            self.label = "bearish"
        elif score <= -0.15:
            self.label = "mildly_bearish"
        else:
            self.label = "neutral"

    def to_dict(self) -> Dict:
        return {
            "symbol":         self.symbol,
            "score":          round(self.score, 3),
            "sources":        self.sources,
            "events_today":   self.events_today,
            "is_blocked":     self.is_blocked,
            "blackout_reason": self.blackout_reason,
            "label":          self.label,
            "updated_at":     self.updated_at.isoformat(),
        }


class SentimentEngine:
    """Aggregates multi-source sentiment for all tracked symbols.

    Args:
        blackout_minutes: minutes around a high-impact event to block trading.
        pre_event_warning_minutes: minutes before event to reduce risk to 50%.
        refresh_interval_seconds: how often the background thread re-fetches.
        cache_dir: directory for persisting downloaded feeds.
        cryptopanic_token: CryptoPanic API token (default: public/free tier).
        market_config_path: path to market_config.yaml (for sentiment_keywords).
    """

    def __init__(
        self,
        blackout_minutes: int = 30,
        pre_event_warning_minutes: int = 60,
        refresh_interval_seconds: int = 3600,
        cache_dir: str = "data",
        cryptopanic_token: str = _CRYPTOPANIC_PUBLIC_TOKEN,
        market_config_path: str = "config/market_config.yaml",
    ) -> None:
        self.blackout_minutes = blackout_minutes
        self.pre_event_minutes = pre_event_warning_minutes
        self.refresh_interval = refresh_interval_seconds
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cp_token = cryptopanic_token

        # Per-symbol keyword overrides from market_config.yaml
        self._symbol_keywords: Dict[str, List[str]] = {}
        self._load_market_config(market_config_path)

        # Shared state (protected by lock)
        self._lock = threading.Lock()
        self._ff_events: List[Dict] = []
        self._rss_headlines: List[str] = []
        self._crypto_scores: Dict[str, float] = {}  # symbol → score
        self._cache: Dict[str, SentimentResult] = {}
        self._last_refresh: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment(self, symbol: str) -> SentimentResult:
        """Return the latest sentiment result for *symbol*.

        Returns a neutral result if no data has been fetched yet.
        """
        with self._lock:
            if symbol.upper() in self._cache:
                return self._cache[symbol.upper()]
        return SentimentResult(symbol, 0.0, [], 0, False)

    def is_trade_blocked(self, now: datetime, symbol: str) -> Tuple[bool, str]:
        """Check whether trading is blocked right now for *symbol*.

        Returns:
            (True, reason_string) if blocked; (False, "ok") if clear.
        """
        currencies = _SYMBOL_CURRENCIES.get(symbol.upper(),
                                            [symbol.upper()[:3], symbol.upper()[3:]])
        window = timedelta(minutes=self.blackout_minutes)
        with self._lock:
            events = list(self._ff_events)

        for evt in events:
            if evt.get("impact", "").lower() != "high":
                continue
            evt_ccy = evt.get("currency", "")
            if evt_ccy not in currencies:
                continue
            try:
                evt_dt = datetime.fromisoformat(evt["dt"])
            except (KeyError, ValueError):
                continue
            delta_secs = (now - evt_dt).total_seconds()
            if abs(delta_secs) <= window.total_seconds():
                mins = int(abs(delta_secs) / 60)
                direction = "before" if now < evt_dt else "after"
                return True, (
                    f"news blackout: '{evt.get('title', '')}' "
                    f"{mins}min {direction} — {evt_dt.strftime('%H:%M UTC')}"
                )
        return False, "ok"

    def upcoming_events(self, within_hours: float = 24.0) -> List[Dict]:
        """Return high-impact FF events occurring within the next N hours."""
        from datetime import timezone as _tz
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=within_hours)
        with self._lock:
            events = list(self._ff_events)
        result = []
        for e in events:
            if e.get("impact", "").lower() != "high":
                continue
            try:
                dt = datetime.fromisoformat(e["dt"])
                if now <= dt <= cutoff:
                    result.append({
                        "title":    e.get("title", ""),
                        "country":  e.get("currency", ""),
                        "dt":       e["dt"],
                        "impact":   e.get("impact", ""),
                    })
            except (KeyError, ValueError):
                continue
        return result

    def pre_event_warning(self, now: datetime, symbol: str) -> bool:
        """Return True if a high-impact event is coming within ``pre_event_minutes``."""
        currencies = _SYMBOL_CURRENCIES.get(symbol.upper(),
                                            [symbol.upper()[:3], symbol.upper()[3:]])
        with self._lock:
            events = list(self._ff_events)
        for evt in events:
            if evt.get("impact", "").lower() != "high":
                continue
            if evt.get("currency", "") not in currencies:
                continue
            try:
                evt_dt = datetime.fromisoformat(evt["dt"])
            except (KeyError, ValueError):
                continue
            mins_ahead = (evt_dt - now).total_seconds() / 60
            if 0 < mins_ahead <= self.pre_event_minutes:
                return True
        return False

    def refresh(self) -> None:
        """Fetch all sources synchronously. Called by background thread."""
        logger.info("SentimentEngine: refreshing all feeds")
        ff_events = self._fetch_ff_feed()
        rss_headlines = self._fetch_rss_headlines()
        crypto_scores = self._fetch_crypto_scores()

        with self._lock:
            self._ff_events = ff_events
            self._rss_headlines = rss_headlines
            self._crypto_scores = crypto_scores
            self._last_refresh = datetime.now(timezone.utc)
            # Re-compute cached results for all known symbols.
            new_cache: Dict[str, SentimentResult] = {}
            for sym in list(_SYMBOL_CURRENCIES.keys()):
                new_cache[sym] = self._compute_result(sym, ff_events, rss_headlines, crypto_scores)
            self._cache = new_cache

        logger.info("SentimentEngine: refresh complete — %d events, %d headlines",
                    len(ff_events), len(rss_headlines))

    def start_background_refresh(self) -> None:
        """Start a daemon thread that refreshes feeds every ``refresh_interval`` seconds."""
        def _loop():
            while True:
                try:
                    self.refresh()
                except Exception:
                    logger.exception("SentimentEngine: background refresh failed")
                time.sleep(self.refresh_interval)

        t = threading.Thread(target=_loop, daemon=True, name="SentimentEngine-refresh")
        t.start()
        logger.info("SentimentEngine: background refresh daemon started (interval=%ds)",
                    self.refresh_interval)

    # ------------------------------------------------------------------
    # Internal: score computation
    # ------------------------------------------------------------------

    def _compute_result(
        self,
        symbol: str,
        ff_events: List[Dict],
        headlines: List[str],
        crypto_scores: Dict[str, float],
    ) -> SentimentResult:
        sources: List[str] = []
        scores: List[float] = []

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # 1. Forex Factory event count & raw directional signal.
        currencies = _SYMBOL_CURRENCIES.get(symbol.upper(),
                                            [symbol.upper()[:3], symbol.upper()[3:]])
        events_today = 0
        is_blocked = False
        blackout_reason = ""
        window = timedelta(minutes=self.blackout_minutes)
        for evt in ff_events:
            if evt.get("currency", "") not in currencies:
                continue
            if evt.get("impact", "").lower() == "high":
                try:
                    evt_dt = datetime.fromisoformat(evt["dt"])
                except (KeyError, ValueError):
                    continue
                if evt_dt.date() == now.date():
                    events_today += 1
                if abs((now - evt_dt).total_seconds()) <= window.total_seconds():
                    is_blocked = True
                    blackout_reason = f"high-impact: {evt.get('title', '')}"
        if events_today > 0:
            sources.append("ff")

        # 2. Crypto-specific score.
        if symbol.upper() in ("BTCUSD", "ETHUSD"):
            ccy = "BTC" if "BTC" in symbol.upper() else "ETH"
            cs = crypto_scores.get(ccy)
            if cs is not None:
                scores.append(cs)
                sources.append("cryptopanic")

        # 3. RSS keyword scan using symbol-specific keywords.
        keywords = self._symbol_keywords.get(symbol.upper(), [])
        if not keywords:
            # Fallback generic keywords.
            keywords = [c.lower() for c in currencies]
        rss_score = self._keyword_score(headlines, keywords)
        if rss_score != 0.0:
            scores.append(rss_score)
            sources.append("rss")

        agg_score = float(sum(scores) / len(scores)) if scores else 0.0
        return SentimentResult(
            symbol=symbol,
            score=agg_score,
            sources=list(set(sources)),
            events_today=events_today,
            is_blocked=is_blocked,
            blackout_reason=blackout_reason,
        )

    @staticmethod
    def _keyword_score(headlines: List[str], keywords: List[str]) -> float:
        """Compute a polarity score in [-1, 1] from headlines filtered by keywords."""
        if not headlines or not keywords:
            return 0.0
        kw_set = {k.lower() for k in keywords}
        relevant = [h for h in headlines if any(k in h.lower() for k in kw_set)]
        if not relevant:
            return 0.0
        bull_count = sum(
            1 for h in relevant
            if any(w in h.lower() for w in _BULLISH_WORDS)
        )
        bear_count = sum(
            1 for h in relevant
            if any(w in h.lower() for w in _BEARISH_WORDS)
        )
        total = bull_count + bear_count
        if total == 0:
            return 0.0
        return (bull_count - bear_count) / total

    # ------------------------------------------------------------------
    # Internal: data fetching
    # ------------------------------------------------------------------

    def _fetch_ff_feed(self) -> List[Dict]:
        """Fetch and parse Forex Factory XML into a list of event dicts."""
        events: List[Dict] = []
        cache_file = self.cache_dir / "ff_calendar.xml"
        xml_text = self._download_text(_FF_FEED_URL, cache_file, ttl_hours=12)
        if not xml_text:
            return events
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return events
        now_utc = datetime.utcnow()
        current_year = now_utc.year
        for item in root.findall("eventitem"):
            def _t(tag):
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""
            country = _t("country").upper()
            currency = _COUNTRY_CURRENCY_MAP.get(country, country)
            impact  = _t("impact")
            title   = _t("title")
            date_s  = _t("date")
            time_s  = _t("time") or "00:00am"
            try:
                # BUG-M1 FIX: handle year-rollover weeks (Dec 28 – Jan 3).
                dt = datetime.strptime(f"{date_s} {current_year}", "%A %b %d %Y")
                if (now_utc - dt).days > 6:
                    dt = datetime.strptime(f"{date_s} {current_year + 1}", "%A %b %d %Y")
                if re.match(r"\d{1,2}:\d{2}[ap]m", time_s, re.IGNORECASE):
                    t = datetime.strptime(time_s.lower(), "%I:%M%p")
                    dt = dt.replace(hour=t.hour, minute=t.minute)
                events.append({
                    "currency": currency,
                    "title":    title,
                    "impact":   impact,
                    "dt":       dt.isoformat(),
                })
            except ValueError:
                continue
        return events

    def _fetch_rss_headlines(self) -> List[str]:
        """Fetch headlines from multiple RSS feeds."""
        headlines: List[str] = []
        for url in _RSS_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ICT-Bot/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                # Cheap extraction: grab all <title> tags.
                titles = re.findall(r"<title>([^<]{10,200})</title>", text)
                headlines.extend(titles)
            except Exception as exc:
                logger.debug("SentimentEngine: RSS fetch failed (%s): %s", url, exc)
        return headlines

    def _fetch_crypto_scores(self) -> Dict[str, float]:
        """Fetch CryptoPanic posts for BTC and ETH and compute polarity."""
        scores: Dict[str, float] = {}
        for ccy in ("BTC", "ETH"):
            try:
                url = f"https://cryptopanic.com/api/v1/posts/?currencies={ccy}&kind=news&public=true"
                req = urllib.request.Request(url, headers={"User-Agent": "ICT-Bot/2.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                headlines = [r.get("title", "") for r in results[:30]]
                score = self._keyword_score(headlines, [ccy.lower(), "crypto", "bitcoin" if ccy == "BTC" else "ethereum"])
                scores[ccy] = score
            except Exception as exc:
                logger.debug("SentimentEngine: CryptoPanic fetch failed for %s: %s", ccy, exc)
        return scores

    def _download_text(self, url: str, cache_file: Path, ttl_hours: float = 4.0) -> Optional[str]:
        """Download URL text with cache and TTL."""
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < ttl_hours:
                return cache_file.read_text(encoding="utf-8")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ICT-Bot/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            cache_file.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            logger.warning("SentimentEngine: download failed (%s): %s", url, exc)
            if cache_file.exists():
                return cache_file.read_text(encoding="utf-8")
        return None

    def _load_market_config(self, path: str) -> None:
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            markets = cfg.get("markets", {})
            for sym, meta in markets.items():
                kws = meta.get("sentiment_keywords", [])
                if kws:
                    self._symbol_keywords[sym.upper()] = [str(k).lower() for k in kws]
        except Exception:
            pass  # market_config is optional



# ---------------------------------------------------------------------------
# Keyword polarity dictionaries (kept here for sentiment scoring)
# ---------------------------------------------------------------------------
_BULLISH_WORDS = {
    "surge", "rally", "rise", "gain", "beat", "record", "strong", "growth",
    "bullish", "optimism", "recovery", "expansion", "hawkish", "rate hike",
    "safe haven", "demand", "inflow", "breakout", "buy",
}
_BEARISH_WORDS = {
    "fall", "drop", "decline", "miss", "weak", "loss", "recession", "crash",
    "bearish", "pessimism", "contraction", "dovish", "rate cut", "sell-off",
    "outflow", "concern", "risk off", "slowdown", "inflation fear", "sell",
}
