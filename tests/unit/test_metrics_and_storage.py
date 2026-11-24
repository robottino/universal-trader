"""Unit tests for calendar-aware metrics, quantile calibration, SQLiteStateStore, and LocalPaperBroker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from universal_trader.brokers.local_paper import LocalPaperBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import FeeModelConfig
from universal_trader.core.models import (
    ApprovedOrder,
    AssetClass,
    FillResult,
    ForecastResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Quote,
    StepOutcome,
)
from universal_trader.engine.metrics import (
    compute_forecast_calibration,
    compute_performance_metrics,
)
from universal_trader.storage.sqlite_store import SQLiteStateStore


def test_metrics_calendar_aware_and_calibration() -> None:
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    idx = pd.date_range(t0, periods=60, freq="1D", tz=UTC)
    equity = 10000.0 * np.exp(np.linspace(0.0, 0.20, 60) + 0.01 * np.sin(np.arange(60)))
    bh = 10000.0 * np.exp(np.linspace(0.0, 0.10, 60))
    eq_df = pd.DataFrame({"equity": equity, "buy_and_hold_equity": bh}, index=idx)

    fills = [
        FillResult(order_id="1", instrument_id="NVDA", timestamp=idx[5], side=OrderSide.BUY, quantity=10, fill_price=100, notional=1000, fee=1.0, slippage=0.5, status=OrderStatus.FILLED),
        FillResult(order_id="2", instrument_id="NVDA", timestamp=idx[25], side=OrderSide.SELL, quantity=10, fill_price=115, notional=1150, fee=1.0, slippage=0.5, status=OrderStatus.FILLED),
    ]
    m_eq = compute_performance_metrics(equity_df=eq_df, fills=fills, calendar=MarketCalendar("XNAS"), timeframe="1d")
    m_crypto = compute_performance_metrics(equity_df=eq_df, fills=fills, calendar=MarketCalendar("24/7"), timeframe="1d")

    assert m_eq["total_roi"] > m_eq["buy_and_hold_roi"]
    assert m_eq["win_rate"] == pytest.approx(1.0)
    assert m_eq["total_fees"] == pytest.approx(2.0)
    # 24/7 annualization sqrt(365.25) > XNAS sqrt(~252) -> higher annualized Sharpe for same bar returns
    assert m_crypto["sharpe_ratio"] > m_eq["sharpe_ratio"]

    # Calibration check
    candles = pd.DataFrame({"close": np.linspace(100.0, 120.0, 60)}, index=idx)
    forecasts = {
        idx[10].to_pydatetime(): ForecastResult(
            instrument_id="NVDA",
            model_id="m",
            timestamp=idx[10].to_pydatetime(),
            horizon=3,
            quantiles={
                0.05: np.array([95.0, 95.0, 95.0]),
                0.10: np.array([98.0, 98.0, 98.0]),
                0.50: np.array([104.0, 104.5, 105.0]),
                0.90: np.array([112.0, 112.0, 112.0]),
                0.95: np.array([115.0, 115.0, 115.0]),
            },
        )
    }
    cal = compute_forecast_calibration(forecasts, candles)
    assert cal["evaluated_points"] == 3
    assert cal["mae"] is not None and cal["mean_pinball_loss"] is not None
    assert cal["interval_80_coverage"] == pytest.approx(1.0)


def test_sqlite_store_idempotency_hwm_and_local_paper_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    store = SQLiteStateStore(db_path)
    ts1 = datetime(2025, 3, 5, 16, 0, tzinfo=UTC)
    ts2 = ts1 + timedelta(hours=1)

    assert not store.has_processed("cfg1", "BTC/EUR", "1h", ts1)

    pf = PortfolioState(timestamp=ts1, quote_currency="EUR", cash=10000.0, equity=10500.0, high_water_mark=10500.0)
    fc = ForecastResult(
        instrument_id="BTC/EUR",
        model_id="mock",
        timestamp=ts1,
        horizon=1,
        quantiles={0.1: np.array([95.0]), 0.5: np.array([102.0]), 0.9: np.array([110.0])},
        future_timestamps=[ts2],
    )
    outcome = StepOutcome(
        config_name="cfg1",
        instrument_id="BTC/EUR",
        timeframe="1h",
        candle_timestamp=ts1,
        forecast=fc,
        portfolio_before=pf,
        portfolio_after=pf,
    )
    store.record_step(outcome)

    assert store.has_processed("cfg1", "BTC/EUR", "1h", ts1)
    assert store.get_high_water_mark("cfg1") == pytest.approx(10500.0)
    assert store.is_halted("cfg1") is False

    # Ex-post evaluation with realized price at ts2
    realized = pd.Series([104.0], index=pd.DatetimeIndex([ts2], tz=UTC))
    expost = store.evaluate_stored_forecasts("cfg1", realized_prices=realized)
    assert expost["evaluated_points"] == 1
    assert expost["mae"] == pytest.approx(2.0)

    # LocalPaperBroker persists cash & positions across fresh broker instances!
    btc = Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR")
    broker1 = LocalPaperBroker(
        store=store,
        account_key="paper_btc",
        quote_currency="EUR",
        initial_cash=5000.0,
        fee_model=FeeModelConfig(pct=0.001, fixed_min=0.0),
        slippage_bps=0.0,
        spread_bps=0.0,
    )
    q = Quote(instrument_id="BTC/EUR", timestamp=ts1, bid=50000.0, ask=50000.0, last=50000.0)
    broker1.set_quote(q)
    order = ApprovedOrder(
        instrument_id="BTC/EUR",
        timestamp=ts1,
        side=OrderSide.BUY,
        quantity=0.05,
        estimated_price=50000.0,
        estimated_notional=2500.0,
        estimated_fee=2.5,
        estimated_slippage=0.0,
    )
    fill = broker1.submit_order(order, btc)
    assert fill.status == OrderStatus.FILLED

    # Create a brand new LocalPaperBroker instance pointed at the same SQLite DB
    broker2 = LocalPaperBroker(store=store, account_key="paper_btc", quote_currency="EUR", initial_cash=5000.0)
    broker2.set_quote(q)
    pf2 = broker2.get_portfolio(btc)
    assert pf2.position_qty("BTC/EUR") == pytest.approx(0.05)
    assert pf2.cash == pytest.approx(5000.0 - 2500.0 - 2.5)
