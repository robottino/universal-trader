"""MarketCalendar abstraction supporting 24/7 crypto and global stock exchanges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from universal_trader.core.models import ensure_utc

_TIMEFRAME_DELTAS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
    "1wk": timedelta(weeks=1),
}

_CALENDAR_ALIASES: dict[str, str] = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "XETRA": "XETR",
    "BORSA_ITALIANA": "XMIL",
    "MIL": "XMIL",
    "LSE": "XLON",
    "24/7": "24/7",
    "CRYPTO": "24/7",
}


def parse_timeframe(timeframe: str) -> timedelta:
    """Parse canonical timeframe string (e.g. '1h', '1d', '15m') to timedelta."""
    tf = timeframe.strip().lower()
    if tf in _TIMEFRAME_DELTAS:
        return _TIMEFRAME_DELTAS[tf]
    raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {sorted(_TIMEFRAME_DELTAS)}")


def timeframe_seconds(timeframe: str) -> float:
    return parse_timeframe(timeframe).total_seconds()


@lru_cache(maxsize=32)
def _get_exchange_calendar(code: str) -> xcals.ExchangeCalendar:
    return xcals.get_calendar(code)


class MarketCalendar:
    """Calendar abstraction for market hours, sessions/year, and future bar generation."""

    def __init__(self, calendar_code: str = "24/7") -> None:
        normalized = _CALENDAR_ALIASES.get(calendar_code.strip().upper(), calendar_code.strip())
        self.code = normalized
        self.is_24_7 = normalized == "24/7"
        self._xcal: xcals.ExchangeCalendar | None = (
            None if self.is_24_7 else _get_exchange_calendar(normalized)
        )

    @property
    def timezone_name(self) -> str:
        if self.is_24_7 or self._xcal is None:
            return "UTC"
        return str(self._xcal.tz)

    def to_exchange_tz(self, ts: datetime) -> datetime:
        """Convert UTC timestamp to the exchange's local timezone."""
        utc_dt = ensure_utc(ts)
        return utc_dt.astimezone(ZoneInfo(self.timezone_name))

    def is_market_open(self, ts: datetime) -> bool:
        """Return True if the exchange is open for trading at timestamp ts (UTC)."""
        if self.is_24_7 or self._xcal is None:
            return True
        pd_ts = pd.Timestamp(ensure_utc(ts))
        try:
            return bool(self._xcal.is_open_on_minute(pd_ts, ignore_breaks=False))
        except Exception:
            # Handle boundary timestamps gracefully
            day = pd_ts.tz_convert(self._xcal.tz).normalize().tz_localize(None)
            if not self._xcal.is_session(day):
                return False
            open_t = self._xcal.session_open(day)
            close_t = self._xcal.session_close(day)
            return bool(open_t <= pd_ts <= close_t)

    def next_session_open(self, ts: datetime) -> datetime:
        """Return the next UTC session opening timestamp at or after ts."""
        utc_dt = ensure_utc(ts)
        if self.is_24_7 or self._xcal is None:
            return utc_dt
        pd_ts = pd.Timestamp(utc_dt)
        next_open = self._xcal.next_open(pd_ts)
        return ensure_utc(next_open)

    def is_pre_holiday(self, ts: datetime) -> bool:
        """Return True if the next calendar weekday after ts is an exchange holiday."""
        if self.is_24_7 or self._xcal is None:
            return False
        local_day = pd.Timestamp(ensure_utc(ts)).tz_convert(self._xcal.tz).normalize().tz_localize(None)
        # Find the next weekday (Mon-Fri)
        next_day = local_day + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        try:
            return not bool(self._xcal.is_session(next_day))
        except Exception:
            return False

    def sessions_per_year(self) -> float:
        """Average trading sessions per year for annualization."""
        if self.is_24_7 or self._xcal is None:
            return 365.25
        # Estimate over 5 recent calendar years within exchange calendar bounds
        try:
            sessions = self._xcal.sessions_in_range("2020-01-01", "2024-12-31")
            return float(len(sessions) / 5.0)
        except Exception:
            return 252.0

    def typical_session_hours(self) -> float:
        """Typical trading hours per session."""
        if self.is_24_7 or self._xcal is None:
            return 24.0
        try:
            sessions = self._xcal.sessions_in_range("2024-01-01", "2024-12-31")
            if len(sessions) == 0:
                return 6.5
            durations = [
                (self._xcal.session_close(s) - self._xcal.session_open(s)).total_seconds() / 3600.0
                for s in sessions[:30]
            ]
            # Median handles half-days accurately
            durations.sort()
            return float(durations[len(durations) // 2])
        except Exception:
            return 6.5

    def bars_per_year(self, timeframe: str) -> float:
        """Calendar-aware annualization factor (bars per year) for Sharpe/Sortino/CAGR."""
        delta = parse_timeframe(timeframe)
        secs = delta.total_seconds()
        sessions = self.sessions_per_year()
        if self.is_24_7:
            return (365.25 * 86400.0) / secs
        if delta >= timedelta(days=1):
            days_per_bar = delta.total_seconds() / 86400.0
            if days_per_bar >= 7.0:
                return 52.0 / (days_per_bar / 7.0)
            return sessions / days_per_bar
        session_secs = self.typical_session_hours() * 3600.0
        bars_per_session = max(1.0, session_secs / secs)
        return sessions * bars_per_session

    def future_timestamps(
        self,
        last_ts: datetime,
        horizon: int,
        timeframe: str,
    ) -> pd.DatetimeIndex:
        """Generate `horizon` future UTC bar timestamps respecting the market calendar."""
        if horizon <= 0:
            return pd.DatetimeIndex([], tz=UTC)
        start_utc = ensure_utc(last_ts)
        delta = parse_timeframe(timeframe)

        if self.is_24_7 or self._xcal is None:
            stamps = [start_utc + delta * (i + 1) for i in range(horizon)]
            return pd.DatetimeIndex(stamps, tz=UTC)

        # Daily / weekly bars on an exchange calendar
        if delta >= timedelta(days=1):
            local_day = pd.Timestamp(start_utc).tz_convert(self._xcal.tz).normalize().tz_localize(None)
            end_bound = local_day + pd.Timedelta(days=max(horizon * 3, 30))
            try:
                future_sessions = self._xcal.sessions_in_range(
                    local_day + pd.Timedelta(days=1),
                    min(end_bound, self._xcal.last_session),
                )
                step_days = max(1, int(round(delta.days / 5.0)) * 5) if delta.days >= 7 else 1
                selected = future_sessions[::step_days][:horizon]
                stamps = [
                    ensure_utc(self._xcal.session_close(s))
                    for s in selected
                ]
                while len(stamps) < horizon:
                    prev = stamps[-1] if stamps else start_utc
                    nxt = prev + delta
                    while nxt.weekday() >= 5:
                        nxt += timedelta(days=1)
                    stamps.append(nxt)
                return pd.DatetimeIndex(stamps, tz=UTC)
            except Exception:
                pass

        # Intraday exchange bars: advance step by step, jumping to next session open when closed
        stamps = []
        cursor = start_utc
        max_guard = horizon * 500
        steps = 0
        while len(stamps) < horizon and steps < max_guard:
            steps += 1
            candidate = cursor + delta
            if self.is_market_open(candidate):
                stamps.append(candidate)
                cursor = candidate
            else:
                try:
                    nxt_open = self.next_session_open(candidate)
                    if nxt_open <= cursor:
                        cursor = candidate
                        continue
                    stamps.append(nxt_open)
                    cursor = nxt_open
                except Exception:
                    while candidate.weekday() >= 5:
                        candidate += timedelta(days=1)
                    stamps.append(candidate)
                    cursor = candidate
        return pd.DatetimeIndex(stamps[:horizon], tz=UTC)
