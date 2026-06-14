"""Shared constants used across the detection, news, and sentiment modules.

Centralises ``_SYMBOL_CURRENCIES`` and ``_COUNTRY_CURRENCY_MAP`` so that
the identical dictionaries that previously lived (duplicated) in both
``news_filter.py`` and ``sentiment_engine.py`` are maintained in one place.
"""
from __future__ import annotations

from typing import Dict, List

# Maps symbol → list of currency codes whose news events affect it.
SYMBOL_CURRENCIES: Dict[str, List[str]] = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"], "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"], "NZDUSD": ["NZD", "USD"],
    "USDCHF": ["USD", "CHF"],
    "EURGBP": ["EUR", "GBP"], "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "XAUUSD": ["USD"],         "XAGUSD": ["USD"],
    "NAS100": ["USD"],         "SPX500": ["USD"], "US30": ["USD"],
    "BTCUSD": ["BTC", "USD"],  "ETHUSD": ["ETH", "USD"],
}

# Maps country code / currency code → ISO currency code (Forex Factory uses
# country names like "US", "EU", or currency codes like "USD" directly).
COUNTRY_CURRENCY_MAP: Dict[str, str] = {
    "US": "USD", "USD": "USD",
    "EU": "EUR", "EUR": "EUR",
    "UK": "GBP", "GBP": "GBP",
    "JP": "JPY", "JPY": "JPY",
    "AU": "AUD", "AUD": "AUD",
    "NZ": "NZD", "NZD": "NZD",
    "CA": "CAD", "CAD": "CAD",
    "CH": "CHF", "CHF": "CHF",
}
