"""Trading session utilities (London / New York / Asian)."""
from __future__ import annotations

from datetime import datetime, time
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DEFAULT_SESSIONS: Dict[str, Tuple[str, str]] = {
    "Asian": ("00:00", "09:00"),
    "London": ("08:00", "17:00"),
    "NY": ("13:00", "22:00"),
}


def _parse_session(s: Tuple[str, str]) -> Tuple[time, time]:
    return time.fromisoformat(s[0]), time.fromisoformat(s[1])


def get_session(
    timestamp: pd.Timestamp,
    sessions: Dict[str, Tuple[str, str]] = None,
) -> str:
    """Return the dominant session label for a timestamp.

    Overlap windows resolve to ``'London_NY'``. Outside any session returns ``'OffHours'``.
    """
    if sessions is None:
        sessions = DEFAULT_SESSIONS
    t = timestamp.time() if hasattr(timestamp, "time") else timestamp
    active = []
    for name, (s, e) in sessions.items():
        st, et = _parse_session((s, e))
        if st <= t <= et:
            active.append(name)
    if not active:
        return "OffHours"
    if len(active) == 1:
        return active[0]
    # Prefer the canonical London_NY overlap label
    if {"London", "NY"} <= set(active):
        return "London_NY"
    return "_".join(active)


def seconds_into_session(
    timestamp: pd.Timestamp,
    sessions: Dict[str, Tuple[str, str]] = None,
) -> int:
    """Number of seconds elapsed since the start of the active session.

    Returns 0 if outside any session.
    """
    if sessions is None:
        sessions = DEFAULT_SESSIONS
    label = get_session(timestamp, sessions)
    if label == "OffHours":
        return 0
    primary = label.split("_")[0]
    if primary not in sessions:
        return 0
    start_t, _ = _parse_session(sessions[primary])
    cur = timestamp.time() if hasattr(timestamp, "time") else timestamp
    a = datetime.combine(datetime(2000, 1, 1).date(), cur)
    b = datetime.combine(datetime(2000, 1, 1).date(), start_t)
    delta = (a - b).total_seconds()
    return max(0, int(delta))


def session_volatility_index(
    candles: pd.DataFrame,
    session_name: str,
    period: int = 20,
    sessions: Dict[str, Tuple[str, str]] = None,
) -> pd.Series:
    """Rolling avg true-range during a given session, broadcast across all bars.

    Useful as a feature: 'How volatile is the London session lately?'
    """
    if sessions is None:
        sessions = DEFAULT_SESSIONS
    if session_name not in sessions:
        raise ValueError(f"Unknown session: {session_name}")
    start_t, end_t = _parse_session(sessions[session_name])
    in_sess = (candles.index.time >= start_t) & (candles.index.time <= end_t)
    tr = (candles["high"] - candles["low"]).where(in_sess)
    rolled = tr.rolling(period, min_periods=1).mean().ffill().bfill()
    return rolled.rename(f"{session_name}_vol_idx")


def add_session_features(
    candles: pd.DataFrame,
    sessions: Dict[str, Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Return one-hot session columns + seconds_into_session column."""
    if sessions is None:
        sessions = DEFAULT_SESSIONS
    labels = pd.Index([get_session(t, sessions) for t in candles.index])
    out = pd.DataFrame(index=candles.index)
    for name in list(sessions.keys()) + ["London_NY", "OffHours"]:
        out[f"sess_{name}"] = (labels == name).astype(int)
    out["seconds_into_session"] = [seconds_into_session(t, sessions) for t in candles.index]
    return out
