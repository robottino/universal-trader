"""Unit tests for multi-asset portfolio config, allocator, Bitfinex adapters, and metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from universal_trader import load_builtin_plugins
from universal_trader.brokers.bitfinex import BitfinexBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import (
    AppConfig,
    DataConfig,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    load_config,
)
from universal_trader.core.models import (
    AssetClass,
    FillResult,
    Instrument,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Quote,
)
from universal_trader.data.bitfinex import BitfinexDataProvider
from universal_trader.engine.metrics import compute_round_trip_pnls
from universal_trader.engine.portfolio import PortfolioAllocator


def _make_instruments() -> list[Instrument]:
    return [
        Instrument(
            id="BTC/USD",
            asset_class=AssetClass.CRYPTO,
            base="BTC",
            quote_currency="USD",
            provider_symbols={
                "bitfinex": "tBTCUSD",
                "bitfinex_paper": "tTESTBTC:TESTUSD",
            },
        ),
        Instrument(
            id="ETH/USD",
            asset_class=AssetClass.CRYPTO,
            base="ETH",
            quote_currency="USD",
            provider_symbols={
                "bitfinex": "tETHUSD",
                "bitfinex_paper": "tTESTETH:TESTUSD",
            },
        ),
        Instrument(
            id="XAUT/USD",
            asset_class=AssetClass.CRYPTO,
            base="XAUT",
            quote_currency="USD",
            provider_symbols={
                "bitfinex": "tXAUT:USD",
                "bitfinex_paper": "tTESTXAUT:TESTUSD",
            },
        ),
    ]


def test_portfolio_config_validation_and_production_yaml() -> None:
    # Production Option 2 YAML must validate cleanly
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "bitfinex_portfolio_btc_eth_gold.yaml"
    cfg = load_config(cfg_path)
    assert cfg.is_portfolio is True
    assert [i.id for i in cfg.instruments] == ["BTC/USD", "ETH/USD", "XAUT/USD"]
    assert cfg.instrument.id == "BTC/USD"
    assert cfg.benchmark_instrument.id == "BTC/USD"
    assert cfg.portfolio.allocation_method == "inverse_volatility"
    assert cfg.portfolio.max_weight_per_asset == pytest.approx(0.60)
    assert cfg.risk.fee_model.pct == pytest.approx(0.0)
    assert cfg.strategy.trend_fast_window == 240
    assert cfg.strategy.trend_slow_window == 720
    assert cfg.strategy.macro_ema_window == 4800

    # Duplicate instrument IDs must fail
    insts = _make_instruments()
    with pytest.raises(ValueError, match="Duplicate instrument IDs"):
        AppConfig(name="dup_test", instrument=insts[0], instruments=[insts[0], insts[0]])

    # Mixed quote currencies must fail
    eur_btc = Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR")
    with pytest.raises(ValueError, match="share one quote_currency"):
        AppConfig(name="mixed_ccy", instrument=insts[0], instruments=[insts[0], eur_btc])

    # Unknown benchmark_instrument_id must fail
    with pytest.raises(ValueError, match="benchmark_instrument_id"):
        AppConfig(
            name="bad_bench",
            instrument=insts[0],
            instruments=insts,
            portfolio=PortfolioConfig(benchmark_instrument_id="SOL/USD"),
        )


def test_bitfinex_broker_symbols_monotonic_nonce_and_multi_portfolio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    insts = _make_instruments()
    paper_broker = BitfinexBroker(api_key="k", api_secret="s", paper=True)
    live_broker = BitfinexBroker(api_key="k", api_secret="s", paper=False)

    assert paper_broker._exchange_symbol(insts[0]) == "tTESTBTC:TESTUSD"
    assert paper_broker._exchange_symbol(insts[1]) == "tTESTETH:TESTUSD"
    assert paper_broker._exchange_symbol(insts[2]) == "tTESTXAUT:TESTUSD"

    assert live_broker._exchange_symbol(insts[0]) == "tBTCUSD"
    assert live_broker._exchange_symbol(insts[1]) == "tETHUSD"
    assert live_broker._exchange_symbol(insts[2]) == "tXAUT:USD"

    # Consecutive auth headers MUST have strictly increasing nonces
    n1 = int(live_broker._auth_headers("v2/auth/r/wallets", "{}")["bfx-nonce"])
    n2 = int(live_broker._auth_headers("v2/auth/r/wallets", "{}")["bfx-nonce"])
    n3 = int(live_broker._auth_headers("v2/auth/r/wallets", "{}")["bfx-nonce"])
    assert n1 < n2 < n3

    # Multi-asset wallet aggregation across BTC + ETH + XAUT + USD cash
    monkeypatch.setattr(
        paper_broker,
        "_fetch_exchange_wallets",
        lambda: {"TESTUSD": 2000.0, "TESTBTC": 0.1, "TESTETH": 2.0, "TESTXAUT": 1.0},
    )
    now = datetime(2026, 6, 1, tzinfo=UTC)
    quotes = {
        "BTC/USD": Quote(instrument_id="BTC/USD", timestamp=now, bid=59990, ask=60010, last=60000),
        "ETH/USD": Quote(instrument_id="ETH/USD", timestamp=now, bid=1999, ask=2001, last=2000),
        "XAUT/USD": Quote(instrument_id="XAUT/USD", timestamp=now, bid=3599, ask=3601, last=3600),
    }
    pf = paper_broker.get_multi_portfolio(insts, quotes=quotes)
    assert pf.cash == pytest.approx(2000.0)
    # 2000 + 0.1*60000 (6000) + 2*2000 (4000) + 1*3600 (3600) = 15600
    assert pf.equity == pytest.approx(15600.0)
    assert set(pf.positions.keys()) == {"BTC/USD", "ETH/USD", "XAUT/USD"}


def test_bitfinex_data_provider_candle_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    load_builtin_plugins()
    prov = BitfinexDataProvider()
    inst = _make_instruments()[2]  # XAUT/USD -> tXAUT:USD
    assert prov._public_symbol(inst) == "tXAUT:USD"

    # Bitfinex wire format: [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
    t0 = int(datetime(2026, 1, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
    t1 = int(datetime(2026, 1, 1, 1, 0, tzinfo=UTC).timestamp() * 1000)

    def _fake_req(tf_code: str, symbol: str, params: dict[str, int]) -> list[list[Any]]:
        assert tf_code == "1h"
        assert symbol == "tXAUT:USD"
        return [
            [t0, 3000.0, 3010.0, 3020.0, 2995.0, 12.5],
            [t1, 3010.0, 3005.0, 3015.0, 3001.0, 8.2],
        ]

    monkeypatch.setattr(prov, "_request_candles", _fake_req)
    df = prov.fetch_ohlcv(
        inst,
        "1h",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
    )
    assert len(df) == 2
    assert float(df["open"].iloc[0]) == pytest.approx(3000.0)
    assert float(df["high"].iloc[0]) == pytest.approx(3020.0)
    assert float(df["low"].iloc[0]) == pytest.approx(2995.0)
    assert float(df["close"].iloc[0]) == pytest.approx(3010.0)
    assert float(df["volume"].iloc[0]) == pytest.approx(12.5)


def test_portfolio_allocator_inv_vol_weighting_and_cap() -> None:
    insts = _make_instruments()
    cfg = AppConfig(
        name="alloc_unit",
        instrument=insts[0],
        instruments=insts,
        data=DataConfig(provider="local", timeframe="1h"),
        strategy=StrategyConfig(
            type="ema_trend",
            trend_fast_window=10,
            trend_slow_window=30,
            trend_hysteresis_pct=0.01,
            macro_ema_window=60,
            macro_hysteresis_pct=0.01,
        ),
        portfolio=PortfolioConfig(
            allocation_method="inverse_volatility",
            max_weight_per_asset=0.60,
            max_total_exposure=0.98,
            vol_lookback_bars=30,
            vol_min_bars=10,
            vol_floor=0.10,
            rebalance_band=0.12,
            warmup_bars=40,
        ),
        risk=RiskConfig(max_exposure_pct=0.98),
    )
    cal = MarketCalendar("24/7")
    allocator = PortfolioAllocator(cfg, cal)

    n = 120
    idx = pd.date_range(datetime(2026, 1, 1, tzinfo=UTC), periods=n, freq="1h", tz=UTC)
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)

    # All 3 deterministically uptrending with distinct volatilities (Gold << BTC < ETH) -> Gold hits 60% cap!
    def _df(drift: float, osc_amp: float) -> pd.DataFrame:
        rets = drift + osc_amp * alt
        close = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 100.0},
            index=idx,
        )

    candles = {
        "BTC/USD": _df(0.005, 0.015),
        "ETH/USD": _df(0.005, 0.025),
        "XAUT/USD": _df(0.003, 0.002),  # very low volatility gold uptrend
    }
    quotes = {
        k: Quote(
            instrument_id=k,
            timestamp=idx[-1].to_pydatetime(),
            bid=float(df["close"].iloc[-1]),
            ask=float(df["close"].iloc[-1]),
            last=float(df["close"].iloc[-1]),
        )
        for k, df in candles.items()
    }
    pf = PortfolioState(
        timestamp=idx[-1].to_pydatetime(),
        quote_currency="USD",
        cash=10000.0,
        positions={},
        equity=10000.0,
        high_water_mark=10000.0,
    )

    decision = allocator.evaluate(candles, pf, quotes)
    assert all(decision.active_signals.values())
    assert decision.realized_vols["XAUT/USD"] < decision.realized_vols["BTC/USD"]
    # Low-vol gold gets capped at 60%, and BTC gets more weight than higher-vol ETH
    assert decision.target_weights["XAUT/USD"] == pytest.approx(0.60)
    assert decision.target_weights["BTC/USD"] > decision.target_weights["ETH/USD"]
    assert sum(decision.target_weights.values()) <= 0.98 + 1e-9
    assert decision.rebalance_triggered is True


def test_compute_round_trip_pnls_tracks_multiple_instruments_independently() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    fills = [
        FillResult(
            order_id="1", instrument_id="BTC/USD", timestamp=ts, side=OrderSide.BUY,
            quantity=1.0, fill_price=100.0, notional=100.0, fee=0.0, status=OrderStatus.FILLED,
        ),
        FillResult(
            order_id="2", instrument_id="XAUT/USD", timestamp=ts, side=OrderSide.BUY,
            quantity=2.0, fill_price=50.0, notional=100.0, fee=0.0, status=OrderStatus.FILLED,
        ),
        FillResult(
            order_id="3", instrument_id="BTC/USD", timestamp=ts, side=OrderSide.SELL,
            quantity=1.0, fill_price=120.0, notional=120.0, fee=0.0, status=OrderStatus.FILLED,
        ),
        FillResult(
            order_id="4", instrument_id="XAUT/USD", timestamp=ts, side=OrderSide.SELL,
            quantity=2.0, fill_price=45.0, notional=90.0, fee=0.0, status=OrderStatus.FILLED,
        ),
    ]
    pnls = compute_round_trip_pnls(fills)
    assert pnls == pytest.approx([20.0, -10.0])


def test_telegram_notifier_event_filtering_html_escape_and_cli_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx
    from typer.testing import CliRunner

    from universal_trader.cli.app import app
    from universal_trader.core.models import PortfolioStepOutcome
    from universal_trader.reporting.telegram import TelegramNotifier

    insts = _make_instruments()
    notifier = TelegramNotifier(
        bot_token="123456:fake-token",
        chat_id="99999",
        enabled=True,
        notify_on="events",
        daily_summary_hour_utc=8,
    )

    # 1. Quiet HOLD at non-heartbeat hour -> suppressed
    quiet_14 = PortfolioStepOutcome(
        config_name="bitfinex_portfolio_btc_eth_gold",
        timeframe="1h",
        candle_timestamp=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
        active_signals={"BTC/USD": False, "ETH/USD": False, "XAUT/USD": True},
        realized_vols={"BTC/USD": 0.5, "ETH/USD": 0.7, "XAUT/USD": 0.15},
        target_weights={"BTC/USD": 0.0, "ETH/USD": 0.0, "XAUT/USD": 0.60},
        current_weights_before={"BTC/USD": 0.0, "ETH/USD": 0.0, "XAUT/USD": 0.60},
        current_weights_after={"BTC/USD": 0.0, "ETH/USD": 0.0, "XAUT/USD": 0.60},
        rebalance_triggered=False,
    )
    assert notifier.should_notify_portfolio_step(quiet_14) is False

    # 2. Quiet HOLD at daily heartbeat hour (08:00 UTC) -> sent with [DAILY] badge
    quiet_08 = quiet_14.model_copy(update={"candle_timestamp": datetime(2026, 6, 1, 8, 0, tzinfo=UTC)})
    assert notifier.should_notify_portfolio_step(quiet_08) is True
    msg_daily = notifier.format_portfolio_step_message(quiet_08, insts)
    assert "[DAILY]" in msg_daily
    assert "<b>bitfinex_portfolio_btc_eth_gold</b>" in msg_daily

    # 3. Rebalance / trade event at any hour -> sent immediately, HTML safely escaped
    err_outcome = quiet_14.model_copy(update={"error": "API <timeout> & retry_failed"})
    assert notifier.should_notify_portfolio_step(err_outcome) is True
    msg_err = notifier.format_portfolio_step_message(err_outcome, insts)
    assert "[ERROR]" in msg_err
    assert "&lt;timeout&gt; &amp; retry_failed" in msg_err

    # 4. CLI `utrader test-telegram` posts HTML message cleanly
    captured: dict[str, Any] = {}

    class _FakeResp:
        status_code = 200
        text = '{"ok": true}'

    def _fake_post(self: httpx.Client, url: str, **kwargs: Any) -> _FakeResp:
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _FakeResp()

    monkeypatch.setattr(httpx.Client, "post", _fake_post)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:fake-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "99999")

    runner = CliRunner()
    res = runner.invoke(app, ["test-telegram"])
    assert res.exit_code == 0, res.output
    assert "Telegram test sent successfully" in res.output
    assert captured["json"]["parse_mode"] == "HTML"
    assert captured["json"]["chat_id"] == "99999"
