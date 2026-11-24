"""Unit tests for SignalExtractor, all 6 PositionSizers, all 7 RiskGuards, SimulatedBroker, and DryRunBroker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from universal_trader import load_builtin_plugins
from universal_trader.brokers.dry_run import DryRunBroker
from universal_trader.brokers.simulated import SimulatedBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import FeeModelConfig, RiskConfig, StrategyConfig
from universal_trader.core.models import (
    ApprovedOrder,
    AssetClass,
    Candle,
    ForecastResult,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
    Quote,
)
from universal_trader.risk.manager import RiskManager
from universal_trader.strategies.signals import SignalExtractor
from universal_trader.strategies.sizing import build_strategy


def _make_forecast(
    spot: float = 100.0,
    median_end: float = 105.0,
    q10_end: float = 95.0,
    q90_end: float = 115.0,
    q05_end: float = 92.0,
    q01_end: float = 88.0,
    horizon: int = 5,
) -> ForecastResult:
    ts = datetime(2025, 3, 5, 16, 0, tzinfo=UTC)  # Wed US market open
    return ForecastResult(
        instrument_id="NVDA",
        model_id="test",
        timestamp=ts,
        horizon=horizon,
        quantiles={
            0.01: np.linspace(spot, q01_end, horizon),
            0.05: np.linspace(spot, q05_end, horizon),
            0.10: np.linspace(spot, q10_end, horizon),
            0.25: np.linspace(spot, 0.5 * (q10_end + median_end), horizon),
            0.50: np.linspace(spot, median_end, horizon),
            0.75: np.linspace(spot, 0.5 * (median_end + q90_end), horizon),
            0.90: np.linspace(spot, q90_end, horizon),
            0.95: np.linspace(spot, q90_end + 3.0, horizon),
            0.99: np.linspace(spot, q90_end + 8.0, horizon),
        },
    )


def _make_snapshot(inst: Instrument, price: float = 100.0, ts: datetime | None = None) -> MarketSnapshot:
    t = ts or datetime(2025, 3, 5, 16, 0, tzinfo=UTC)
    idx = pd.DatetimeIndex([t - timedelta(hours=1), t], tz=UTC)
    df = pd.DataFrame(
        {"open": [price, price], "high": [price, price], "low": [price, price], "close": [price, price], "volume": [1000.0, 1000.0]},
        index=idx,
    )
    return MarketSnapshot(
        instrument=inst,
        timestamp=t,
        candles=df,
        quote=Quote(instrument_id=inst.id, timestamp=t, bid=price - 0.05, ask=price + 0.05, last=price),
    )


def test_signal_extractor_and_all_six_sizers() -> None:
    load_builtin_plugins()
    inst = Instrument(id="NVDA", asset_class=AssetClass.EQUITY, base="NVDA", quote_currency="USD", calendar="XNAS", fractional=False, min_qty=1, qty_step=1)
    snap = _make_snapshot(inst, price=100.0)
    pf = PortfolioState(timestamp=snap.timestamp, quote_currency="USD", cash=10000.0, equity=10000.0, high_water_mark=10000.0)

    fc_bull = _make_forecast(spot=100.0, median_end=106.0, q10_end=98.0, q90_end=114.0, q05_end=96.0, q01_end=95.0)
    fc_tail_risk = _make_forecast(spot=100.0, median_end=106.0, q10_end=98.0, q90_end=114.0, q05_end=75.0, q01_end=65.0)

    for mode in ("median_last", "median_mean", "median_peak", "prob_positive", "quantile_asymmetry"):
        sig = SignalExtractor(signal_mode=mode, tail_quantile=0.05, theta_buy=0.01).extract(fc_bull, 100.0)
        assert sig.expected_return > 0
        assert sig.prob_positive > 0.5

    for stype in ("binary", "fractional", "proportional", "volatility_adjusted", "dynamic_kelly", "tail_risk_kelly"):
        strat = build_strategy(StrategyConfig(type=stype, theta_buy=0.01, theta_sell=-0.01, k_risk=0.5))
        intent = strat.evaluate(fc_bull, snap, pf)
        assert intent.side == OrderSide.BUY, f"Strategy {stype} should BUY on bullish forecast"
        assert 0.0 < intent.target_weight <= 1.0

    # TailRiskKelly must significantly shrink target weight when left tail (q05/q01) worsens!
    trk = build_strategy(StrategyConfig(type="tail_risk_kelly", theta_buy=0.01, k_risk=0.5, tail_risk_budget=0.03))
    w_clean = trk.evaluate(fc_bull, snap, pf).target_weight
    w_fat_tail = trk.evaluate(fc_tail_risk, snap, pf).target_weight
    assert w_fat_tail < w_clean

    # TrendQuantileRiskSizer: buys in uptrend & throttles exposure when Timestral tail risk rises
    idx_tr = pd.date_range(datetime(2025, 1, 1, tzinfo=UTC), periods=80, freq="1D", tz=UTC)
    up_px = np.linspace(80.0, 120.0, 80)
    up_df = pd.DataFrame(
        {"open": up_px, "high": up_px, "low": up_px, "close": up_px, "volume": np.full(80, 1000.0), "adj_close": up_px},
        index=idx_tr,
    )
    snap_up = MarketSnapshot(
        instrument=inst,
        timestamp=idx_tr[-1].to_pydatetime(),
        candles=up_df,
        quote=Quote(instrument_id=inst.id, timestamp=idx_tr[-1].to_pydatetime(), bid=119.9, ask=120.1, last=120.0),
    )
    tqr = build_strategy(
        StrategyConfig(
            type="trend_quantile_risk",
            trend_fast_window=10,
            trend_slow_window=30,
            target_vol=0.08,
            tail_risk_budget=0.04,
            k_risk=1.0,
        )
    )
    fc_up_calm = _make_forecast(spot=120.0, median_end=121.0, q10_end=115.0, q90_end=127.0, q05_end=114.0, q01_end=112.0)
    fc_up_risky = _make_forecast(spot=120.0, median_end=121.0, q10_end=100.0, q90_end=140.0, q05_end=90.0, q01_end=80.0)
    int_calm = tqr.evaluate(fc_up_calm, snap_up, pf)
    int_risky = tqr.evaluate(fc_up_risky, snap_up, pf)
    assert int_calm.side == OrderSide.BUY and int_risky.side == OrderSide.BUY
    assert int_risky.target_weight < int_calm.target_weight

    # EmaTrendStrategy: buys cleanly on confirmed uptrend and exits on downtrend
    ema_strat = build_strategy(
        StrategyConfig(
            type="ema_trend",
            trend_fast_window=10,
            trend_slow_window=30,
            trend_hysteresis_pct=0.015,
            target_weight_bull=0.98,
        )
    )
    ema_buy = ema_strat.evaluate(fc_up_calm, snap_up, pf)
    assert ema_buy.side == OrderSide.BUY and ema_buy.target_weight == pytest.approx(0.98)

    dn_px = np.linspace(120.0, 80.0, 80)
    dn_df = pd.DataFrame(
        {"open": dn_px, "high": dn_px, "low": dn_px, "close": dn_px, "volume": np.full(80, 1000.0), "adj_close": dn_px},
        index=idx_tr,
    )
    snap_dn = MarketSnapshot(
        instrument=inst,
        timestamp=idx_tr[-1].to_pydatetime(),
        candles=dn_df,
        quote=Quote(instrument_id=inst.id, timestamp=idx_tr[-1].to_pydatetime(), bid=79.9, ask=80.1, last=80.0),
    )
    pf_long = PortfolioState(
        timestamp=idx_tr[-1].to_pydatetime(),
        quote_currency="USD",
        cash=200.0,
        positions={"NVDA": Position(instrument_id="NVDA", quantity=100.0, avg_entry_price=100.0, market_price=80.0)},
        equity=8200.0,
        high_water_mark=10000.0,
    )
    ema_sell = build_strategy(
        StrategyConfig(type="ema_trend", trend_fast_window=10, trend_slow_window=30)
    ).evaluate(fc_up_calm, snap_dn, pf_long)
    assert ema_sell.side == OrderSide.SELL and ema_sell.target_weight == 0.0

    # Macro EMA bear regime filter suppresses BUY on a brief bounce below macro EMA
    crash_then_bounce = np.concatenate([np.full(80, 200.0), np.linspace(200.0, 90.0, 40), np.linspace(90.0, 115.0, 25)])
    idx_m = pd.date_range(datetime(2025, 1, 1, tzinfo=UTC), periods=len(crash_then_bounce), freq="1D", tz=UTC)
    df_m = pd.DataFrame(
        {"open": crash_then_bounce, "high": crash_then_bounce, "low": crash_then_bounce, "close": crash_then_bounce, "volume": np.full(len(crash_then_bounce), 1000.0), "adj_close": crash_then_bounce},
        index=idx_m,
    )
    snap_m = MarketSnapshot(
        instrument=inst,
        timestamp=idx_m[-1].to_pydatetime(),
        candles=df_m,
        quote=Quote(instrument_id=inst.id, timestamp=idx_m[-1].to_pydatetime(), bid=114.9, ask=115.1, last=115.0),
    )
    ema_macro_blocked = build_strategy(
        StrategyConfig(
            type="ema_trend",
            trend_fast_window=5,
            trend_slow_window=15,
            macro_ema_window=60,
            macro_hysteresis_pct=0.02,
            macro_bear_weight=0.0,
        )
    ).evaluate(fc_up_calm, snap_m, pf)
    assert ema_macro_blocked.side == OrderSide.HOLD and ema_macro_blocked.target_weight == 0.0


def test_all_seven_risk_guards() -> None:
    nvda = Instrument(
        id="NVDA",
        asset_class=AssetClass.EQUITY,
        base="NVDA",
        quote_currency="USD",
        calendar="XNAS",
        fractional=False,
        min_qty=1,
        qty_step=1,
        min_notional=10.0,
    )
    cal = MarketCalendar("XNAS")
    wed_open = datetime(2025, 3, 5, 16, 0, tzinfo=UTC)
    sat_closed = datetime(2025, 3, 1, 16, 0, tzinfo=UTC)

    fc = _make_forecast(spot=100.0, median_end=105.0)
    sig = SignalExtractor("median_last").extract(fc, 100.0)
    buy_intent = OrderIntent(instrument_id="NVDA", timestamp=wed_open, side=OrderSide.BUY, target_weight=1.0, signal=sig)
    hold_intent = OrderIntent(instrument_id="NVDA", timestamp=wed_open, side=OrderSide.HOLD, target_weight=0.5, signal=sig)
    pf = PortfolioState(timestamp=wed_open, quote_currency="USD", cash=1000.0, equity=1000.0, high_water_mark=1000.0)

    # 1. MarketHoursGuard blocks equity order on Saturday
    rm_hours = RiskManager(RiskConfig(enforce_market_hours=True, max_data_age_minutes=60), cal)
    snap_sat = _make_snapshot(nvda, price=100.0, ts=sat_closed)
    app, rej, _ = rm_hours.evaluate(instrument=nvda, snapshot=snap_sat, portfolio=pf, forecast=fc, intent=buy_intent, now=sat_closed)
    assert app is None and rej is not None and rej.guard_name == "MarketHoursGuard"

    # 2. DataStalenessGuard rejects old candle
    snap_old = _make_snapshot(nvda, price=100.0, ts=wed_open - timedelta(hours=3))
    app, rej, _ = rm_hours.evaluate(instrument=nvda, snapshot=snap_old, portfolio=pf, forecast=fc, intent=buy_intent, now=wed_open)
    assert app is None and rej is not None and rej.guard_name == "DataStalenessGuard"

    # 3. FeeAndSlippageGuard blocks trade when fixed_min fee exceeds expected gain
    rm_fee = RiskManager(
        RiskConfig(
            enforce_market_hours=True,
            max_exposure_pct=1.0,
            fee_model=FeeModelConfig(pct=0.0, fixed_min=40.0),  # $80 round-trip on ~$900 > 5% gain!
            slippage_bps=5.0,
        ),
        cal,
    )
    snap_open = _make_snapshot(nvda, price=100.0, ts=wed_open)
    app, rej, _ = rm_fee.evaluate(instrument=nvda, snapshot=snap_open, portfolio=pf, forecast=fc, intent=buy_intent, now=wed_open)
    assert app is None and rej is not None and rej.guard_name == "FeeAndSlippageGuard"

    # 4 & 5. ExposureLimitGuard + InstrumentConstraintGuard (whole shares + max_exposure_pct=0.5)
    rm_exp = RiskManager(
        RiskConfig(
            enforce_market_hours=True,
            max_exposure_pct=0.5,
            fee_model=FeeModelConfig(pct=0.0, fixed_min=1.0),
            slippage_bps=3.0,
        ),
        cal,
    )
    app, rej, _ = rm_exp.evaluate(instrument=nvda, snapshot=snap_open, portfolio=pf, forecast=fc, intent=buy_intent, now=wed_open)
    assert rej is None and app is not None
    # Equity=1000, max_exposure=50% => $500 max => ~4 whole shares at ~$100.08
    assert app.quantity == 4.0 and float(app.quantity).is_integer()

    # 6. DrawdownCircuitBreaker blocks BUY when drawdown >= max_drawdown_pct, but ALLOWS SELL!
    pf_dd = PortfolioState(
        timestamp=wed_open,
        quote_currency="USD",
        cash=500.0,
        positions={"NVDA": Position(instrument_id="NVDA", quantity=3.0, avg_entry_price=120.0, market_price=100.0)},
        equity=800.0,
        high_water_mark=1000.0,  # 20% drawdown!
        halted=False,
    )
    rm_dd = RiskManager(RiskConfig(enforce_market_hours=False, max_drawdown_pct=0.15, fee_model=FeeModelConfig(pct=0.0, fixed_min=1.0)), cal)
    app, rej, halted = rm_dd.evaluate(instrument=nvda, snapshot=snap_open, portfolio=pf_dd, forecast=fc, intent=buy_intent, now=wed_open)
    assert halted is True and app is None and rej is not None and rej.guard_name == "DrawdownCircuitBreaker"

    sell_intent = OrderIntent(instrument_id="NVDA", timestamp=wed_open, side=OrderSide.SELL, target_weight=0.0, signal=sig)
    app_sell, rej_sell, halted_sell = rm_dd.evaluate(instrument=nvda, snapshot=snap_open, portfolio=pf_dd, forecast=fc, intent=sell_intent, now=wed_open)
    assert halted_sell is True and rej_sell is None and app_sell is not None and app_sell.quantity == 3.0

    # 7. QuantileStopLossGuard promotes HOLD to protective SELL when trailing stop is breached
    rm_sl = RiskManager(
        RiskConfig(
            enforce_market_hours=False,
            max_drawdown_pct=0.50,
            enable_quantile_stop_loss=True,
            trailing_stop_pct=0.10,  # Entry 120 -> stop 108; spot is 100!
            fee_model=FeeModelConfig(pct=0.0, fixed_min=1.0),
        ),
        cal,
    )
    app_sl, rej_sl, _ = rm_sl.evaluate(instrument=nvda, snapshot=snap_open, portfolio=pf_dd, forecast=fc, intent=hold_intent, now=wed_open)
    assert rej_sl is None and app_sl is not None and app_sl.side == OrderSide.SELL and app_sl.is_stop_loss is True


def test_simulated_broker_next_open_and_dry_run_isolation() -> None:
    nvda = Instrument(id="NVDA", asset_class=AssetClass.EQUITY, base="NVDA", quote_currency="USD", calendar="XNAS", fractional=False, min_qty=1, qty_step=1, min_notional=10.0)
    broker = SimulatedBroker(
        initial_cash=1000.0,
        quote_currency="USD",
        fee_model=FeeModelConfig(pct=0.0, fixed_min=1.0),
        slippage_bps=10.0,
        spread_bps=0.0,
        execution_price="next_open",
    )
    t1 = datetime(2025, 3, 5, 21, 0, tzinfo=UTC)
    t2 = datetime(2025, 3, 6, 21, 0, tzinfo=UTC)
    broker.on_new_bar(Candle(timestamp=t1, open=99.0, high=101.0, low=98.0, close=100.0, volume=1000), nvda)

    order = ApprovedOrder(
        instrument_id="NVDA",
        timestamp=t1,
        side=OrderSide.BUY,
        quantity=5.0,
        estimated_price=100.0,
        estimated_notional=500.0,
        estimated_fee=1.0,
        estimated_slippage=0.5,
    )
    pending = broker.submit_order(order, nvda)
    assert pending.status == OrderStatus.PENDING
    assert broker.get_portfolio(nvda).position_qty("NVDA") == 0.0  # Not filled on bar t1!

    # On bar t2 open=104.0, queued order executes at 104.0 * (1 + 0.001) = 104.104
    fills = broker.on_new_bar(Candle(timestamp=t2, open=104.0, high=106.0, low=103.0, close=105.0, volume=1000), nvda)
    assert len(fills) == 1
    assert fills[0].status == OrderStatus.FILLED
    assert fills[0].fill_price == pytest.approx(104.0 * 1.001)
    assert fills[0].fee == pytest.approx(1.0)
    assert broker.get_portfolio(nvda).position_qty("NVDA") == 5.0

    # DryRunBroker must NEVER touch inner broker's submit_order
    class ExplosiveBroker(SimulatedBroker):
        def submit_order(self, order: ApprovedOrder, instrument: Instrument):  # type: ignore[override]
            raise AssertionError("DryRunBroker must never call underlying submit_order!")

    inner = ExplosiveBroker(initial_cash=1000.0, quote_currency="USD")
    inner.set_current_bar(Candle(timestamp=t1, open=100.0, high=100.0, low=100.0, close=100.0, volume=100))
    dry = DryRunBroker(inner)
    dry_fill = dry.submit_order(order, nvda)
    assert dry_fill.status == OrderStatus.DRY_RUN
