"""Integration tests: Backtest/Live parity, Walk-Forward + ForecastCache reuse, and CLI E2E."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from universal_trader import load_builtin_plugins
from universal_trader.brokers.simulated import SimulatedBroker
from universal_trader.cli.app import app
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import (
    AppConfig,
    BrokerConfig,
    DataConfig,
    FeaturesConfig,
    FeeModelConfig,
    ForecasterConfig,
    ForecasterParamsConfig,
    RiskConfig,
    StrategyConfig,
    WalkForwardConfig,
)
from universal_trader.core.models import AssetClass, Candle, Instrument
from universal_trader.data.features import FeaturePipeline
from universal_trader.engine.backtest import BacktestRunner
from universal_trader.engine.factory import build_risk_manager, build_strategy
from universal_trader.engine.forecast_cache import ForecastCache
from universal_trader.engine.runner import TradingStepRunner
from universal_trader.engine.walkforward import WalkForwardEngine
from universal_trader.forecasters.mock import MockForecaster


def _synthetic_trend_candles(n: int = 240, freq: str = "1h", seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=UTC), periods=n, freq=freq, tz=UTC)
    # Alternating bullish/bearish regimes to elicit real trades cleanly
    t = np.arange(n, dtype=np.float64)
    wave = 0.008 * np.sin(2.0 * np.pi * t / 48.0) + rng.normal(0.0, 0.003, size=n)
    close = 100.0 * np.exp(np.cumsum(wave))
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    vol = rng.uniform(1000.0, 5000.0, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "adj_close": close},
        index=idx,
    )


def test_backtest_and_single_step_runner_exact_parity() -> None:
    """BacktestRunner must match bar-by-bar TradingStepRunner bit-for-bit (zero backtest/live drift)."""
    load_builtin_plugins()
    candles = _synthetic_trend_candles(n=120, freq="1h")
    cfg = AppConfig(
        name="parity_test",
        instrument=Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR"),
        data=DataConfig(
            provider="local",
            timeframe="1h",
            lookback_bars=64,
            features=FeaturesConfig(past_only=["volume", "log_return_1"], known_future=["hour_sin", "hour_cos"]),
        ),
        forecaster=ForecasterConfig(
            type="mock",
            params=ForecasterParamsConfig(model_id="mock/parity", context_len=32, horizon=6),
        ),
        strategy=StrategyConfig(type="tail_risk_kelly", theta_buy=0.005, theta_sell=-0.005, k_risk=0.5),
        risk=RiskConfig(
            enforce_market_hours=False,
            max_exposure_pct=0.8,
            max_drawdown_pct=0.25,
            fee_model=FeeModelConfig(pct=0.001, fixed_min=0.0),
            slippage_bps=3.0,
        ),
        broker=BrokerConfig(type="simulated", initial_cash=10000.0, execution_price="next_open"),
    )
    cal = MarketCalendar(cfg.instrument.calendar)
    pipe = FeaturePipeline(cfg.data.features, cal, price_column=cfg.data.price_column)

    # 1. Run via BacktestRunner
    bt_fc = MockForecaster(model_id="mock/parity")
    bt_runner = BacktestRunner(config=cfg, forecaster=bt_fc, feature_pipeline=pipe, calendar=cal)
    bt_res = bt_runner.run(candles, initial_cash=10000.0, use_cache=False, min_warmup_bars=32)

    # 2. Manually loop TradingStepRunner bar-by-bar WITHOUT batch precomputation (just like live!)
    live_fc = MockForecaster(model_id="mock/parity")
    live_broker = SimulatedBroker(
        initial_cash=10000.0,
        quote_currency=cfg.instrument.quote_currency,
        fee_model=cfg.risk.fee_model,
        slippage_bps=cfg.risk.slippage_bps,
        spread_bps=cfg.risk.spread_bps,
        execution_price=cfg.broker.execution_price,
        max_drawdown_pct=cfg.risk.max_drawdown_pct,
    )
    live_runner = TradingStepRunner(
        config=cfg,
        feature_pipeline=pipe,
        forecaster=live_fc,
        strategy=build_strategy(cfg),
        risk_manager=build_risk_manager(cfg, cal),
        broker=live_broker,
    )
    live_equities: list[float] = []
    for idx in range(32, len(candles)):
        row = candles.iloc[idx]
        ts = candles.index[idx].to_pydatetime()
        bar = Candle(
            timestamp=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        live_broker.on_new_bar(bar, cfg.instrument)
        outcome = live_runner.run_step(
            candles=candles.iloc[: idx + 1],
            precomputed_forecast=None,
            now=ts,
            check_staleness=False,
            enforce_idempotency=False,
            persist=False,
            notify=False,
        )
        assert outcome.portfolio_after is not None
        live_equities.append(outcome.portfolio_after.equity)

    np.testing.assert_allclose(bt_res.equity_df["equity"].to_numpy(), np.array(live_equities), rtol=1e-12, atol=1e-12)
    assert len(bt_res.fills) == len(live_broker.fills)
    assert len(bt_res.fills) > 0  # Ensure actual trades occurred


def test_walkforward_e2e_and_disk_forecast_cache_reuse(tmp_path: Path) -> None:
    """Second walk-forward run must reuse disk ForecastCache with 0 new forecaster inference calls."""
    load_builtin_plugins()
    candles = _synthetic_trend_candles(n=220, freq="1h", seed=99)
    cfg = AppConfig(
        name="wf_cache_test",
        instrument=Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR"),
        data=DataConfig(
            provider="local",
            timeframe="1h",
            lookback_bars=64,
            features=FeaturesConfig(past_only=["volume", "log_return_1"], known_future=["hour_sin", "hour_cos"]),
        ),
        forecaster=ForecasterConfig(
            type="mock",
            params=ForecasterParamsConfig(model_id="mock/wf", context_len=24, horizon=6),
        ),
        strategy=StrategyConfig(type="tail_risk_kelly", theta_buy=0.005, theta_sell=-0.005),
        risk=RiskConfig(enforce_market_hours=False, fee_model=FeeModelConfig(pct=0.001, fixed_min=0.0)),
        broker=BrokerConfig(type="simulated", initial_cash=10000.0, execution_price="next_open"),
        walkforward=WalkForwardConfig(
            mode="rolling",
            train_bars=60,
            test_bars=30,
            step_bars=30,
            metric="sharpe_ratio",
            min_trades=1,
            param_grid={
                "strategy.type": ["tail_risk_kelly", "binary"],
                "strategy.theta_buy": [0.004, 0.01],
            },
        ),
    )
    cal = MarketCalendar(cfg.instrument.calendar)
    pipe = FeaturePipeline(cfg.data.features, cal)
    fc_cache = ForecastCache(tmp_path / "forecast_cache")

    # Run 1: computes forecasts once in batch across all folds & param combinations
    fc1 = MockForecaster(model_id="mock/wf")
    eng1 = WalkForwardEngine(config=cfg, forecaster=fc1, feature_pipeline=pipe, calendar=cal, forecast_cache=fc_cache)
    res1 = eng1.run(candles, use_cache=True)
    assert len(res1.folds) >= 3
    assert fc1.call_count == 1  # Computed once in batch, NOT per fold or per param!
    csv_out = res1.export_csv(tmp_path / "wf.csv")
    assert csv_out.exists()

    # Run 2: completely new Forecaster instance -> MUST hit disk cache with 0 calls!
    fc2 = MockForecaster(model_id="mock/wf")
    eng2 = WalkForwardEngine(config=cfg, forecaster=fc2, feature_pipeline=pipe, calendar=cal, forecast_cache=fc_cache)
    res2 = eng2.run(candles, use_cache=True)
    assert fc2.call_count == 0
    assert res2.oos_metrics["final_equity"] == pytest.approx(res1.oos_metrics["final_equity"])


def test_cli_end_to_end_doctor_predict_backtest_trade_idempotency_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full CLI test covering doctor, predict, backtest, walkforward, trade --dry-run, LocalPaper idempotency, report."""
    # Create fresh recent local OHLCV dataset ending near "now" so DataStalenessGuard passes
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=now_hour, periods=140, freq="1h", tz=UTC)
    t = np.arange(140, dtype=np.float64)
    # Upward recent momentum so MockForecaster triggers a BUY signal cleanly
    close = 100.0 * np.exp(0.004 * t + 0.01 * np.sin(t / 6.0))
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": np.full(140, 2000.0),
            "adj_close": close,
        }
    )
    data_file = tmp_path / "btc_local.parquet"
    df.to_parquet(data_file, index=False)

    db_file = tmp_path / "state" / "cli_trader.db"
    cfg_file = tmp_path / "cli_btc.yaml"
    cfg_file.write_text(
        f"""
name: "cli_btc_test"
mode: paper
instrument:
  id: "BTC/EUR"
  asset_class: crypto
  base: BTC
  quote_currency: EUR
  calendar: "24/7"
  fractional: true
  min_qty: 0.0001
  qty_step: 0.00001
  min_notional: 5.0
data:
  provider: local
  file_path: "{data_file}"
  timeframe: "1h"
  lookback_bars: 64
  use_cache: false
  features:
    past_only: ["volume", "log_return_1"]
    known_future: ["hour_sin", "hour_cos"]
forecaster:
  type: mock
  params:
    model_id: "mock/cli"
    context_len: 32
    horizon: 6
strategy:
  type: tail_risk_kelly
  theta_buy: 0.005
  theta_sell: -0.005
  k_risk: 0.5
risk:
  enforce_market_hours: false
  max_data_age_minutes: 180
  max_exposure_pct: 0.8
  max_drawdown_pct: 0.2
  fee_model: {{ pct: 0.001, fixed_min: 0.0 }}
  slippage_bps: 3.0
broker:
  type: local_paper
  initial_cash: 10000.0
storage:
  db_path: "{db_file}"
walkforward:
  mode: rolling
  train_bars: 40
  test_bars: 20
  step_bars: 20
  min_trades: 0
  param_grid:
    strategy.theta_buy: [0.005, 0.01]
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("UTRADER_ROOT", str(tmp_path))
    monkeypatch.setenv("UTRADER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("UTRADER_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("UTRADER_DATA_DIR", str(tmp_path / "data_cache"))
    monkeypatch.setenv("UTRADER_FORECAST_CACHE_DIR", str(tmp_path / "forecast_cache"))

    runner = CliRunner()

    # 1. doctor
    res_doc = runner.invoke(app, ["doctor", "--config", str(cfg_file)])
    assert res_doc.exit_code == 0, res_doc.output

    # 2. predict + fan chart
    chart_out = tmp_path / "reports" / "fan.png"
    res_pred = runner.invoke(app, ["predict", "--config", str(cfg_file), "--output", str(chart_out)])
    assert res_pred.exit_code == 0, res_pred.output
    assert chart_out.exists()

    # 3. backtest
    start_s = idx[40].strftime("%Y-%m-%d")
    end_s = idx[-1].strftime("%Y-%m-%d")
    res_bt = runner.invoke(
        app,
        ["backtest", "--config", str(cfg_file), "--start", start_s, "--end", end_s, "--output-dir", str(tmp_path / "reports")],
    )
    assert res_bt.exit_code == 0, res_bt.output
    assert (tmp_path / "reports" / "backtest_cli_btc_test.png").exists()

    # 4. walkforward
    res_wf = runner.invoke(
        app,
        ["walkforward", "--config", str(cfg_file), "--start", start_s, "--end", end_s, "--output-dir", str(tmp_path / "reports")],
    )
    assert res_wf.exit_code == 0, res_wf.output
    assert (tmp_path / "reports" / "walkforward_cli_btc_test.csv").exists()

    # 5. trade --dry-run (force so we test dry-run then real local paper separately)
    res_dry = runner.invoke(app, ["trade", "--config", str(cfg_file), "--dry-run", "--no-plot"])
    assert res_dry.exit_code == 0, res_dry.output
    assert "DRY-RUN" in res_dry.output

    # 6. Second trade on the SAME candle -> idempotent skip!
    res_dup = runner.invoke(app, ["trade", "--config", str(cfg_file), "--no-plot"])
    assert res_dup.exit_code == 0, res_dup.output
    assert "Idempotent skip" in res_dup.output

    # 7. report from SQLite
    res_rep = runner.invoke(app, ["report", "--db", str(db_file)])
    assert res_rep.exit_code == 0, res_rep.output
    assert "Operational Report" in res_rep.output
