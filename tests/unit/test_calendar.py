"""Unit tests for MarketCalendar across 24/7 crypto and global stock exchanges."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from universal_trader.core.calendar import MarketCalendar, parse_timeframe


def test_crypto_24_7_calendar() -> None:
    cal = MarketCalendar("24/7")
    sat = datetime(2026, 1, 3, 15, 0, tzinfo=UTC)  # Saturday
    assert cal.is_market_open(sat) is True
    assert cal.is_pre_holiday(sat) is False
    assert cal.sessions_per_year() == pytest.approx(365.25)
    assert cal.bars_per_year("1h") == pytest.approx(365.25 * 24.0)
    assert cal.bars_per_year("1d") == pytest.approx(365.25)

    fut = cal.future_timestamps(sat, horizon=4, timeframe="1h")
    assert len(fut) == 4
    assert fut[0] == datetime(2026, 1, 3, 16, 0, tzinfo=UTC)
    assert fut[-1] == datetime(2026, 1, 3, 19, 0, tzinfo=UTC)


def test_nasdaq_and_eu_exchange_calendars() -> None:
    xnas = MarketCalendar("XNAS")
    # Saturday must be closed
    saturday = datetime(2025, 3, 1, 16, 0, tzinfo=UTC)
    assert xnas.is_market_open(saturday) is False
    # Regular trading hours on a Wednesday (16:00 UTC = 11:00 EST) must be open
    wed_open = datetime(2025, 3, 5, 16, 0, tzinfo=UTC)
    assert xnas.is_market_open(wed_open) is True
    # After hours (23:00 UTC = 18:00 EST) must be closed
    wed_closed = datetime(2025, 3, 5, 23, 0, tzinfo=UTC)
    assert xnas.is_market_open(wed_closed) is False

    # Sessions & bars per year should reflect ~252 trading days and ~6.5h/day
    assert 248.0 <= xnas.sessions_per_year() <= 255.0
    assert 248.0 <= xnas.bars_per_year("1d") <= 255.0
    assert 1500.0 <= xnas.bars_per_year("1h") <= 1700.0

    # Future daily timestamps must skip weekends
    fri_close = datetime(2025, 3, 7, 21, 0, tzinfo=UTC)
    fut_days = xnas.future_timestamps(fri_close, horizon=5, timeframe="1d")
    assert len(fut_days) == 5
    for ts in fut_days:
        assert ts.weekday() < 5  # Mon-Fri only

    # XETR & XMIL should instantiate cleanly too
    xetr = MarketCalendar("XETR")
    xmil = MarketCalendar("XMIL")
    assert xetr.timezone_name == "Europe/Berlin"
    assert xmil.timezone_name == "Europe/Rome"


def test_parse_timeframe_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        parse_timeframe("7m")
