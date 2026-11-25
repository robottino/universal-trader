"""Unit and integration tests for the standalone TelegramBotService microservice."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from universal_trader.brokers.dry_run import DryRunBroker
from universal_trader.brokers.local_paper import LocalPaperBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import (
    AppConfig,
    BrokerConfig,
    DataConfig,
    FeeModelConfig,
    ForecasterConfig,
    NotifierConfig,
    PortfolioConfig,
    RiskConfig,
    RuntimePaths,
    StorageConfig,
    StrategyConfig,
)
from universal_trader.core.models import AssetClass, Instrument
from universal_trader.data.local import CsvParquetProvider
from universal_trader.engine.portfolio import PortfolioAllocator, PortfolioStepRunner
from universal_trader.reporting.telegram import TelegramNotifier
from universal_trader.reporting.telegram_bot import TelegramBotService
from universal_trader.risk.manager import RiskManager
from universal_trader.storage.sqlite_store import SQLiteStateStore


def _setup_bot_env(tmp_path: Path) -> tuple[AppConfig, SQLiteStateStore, CsvParquetProvider, TelegramBotService]:
    now_hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = 120
    idx = pd.date_range(end=now_hour, periods=n, freq="1h", tz=UTC)
    t = np.arange(n, dtype=np.float64)
    for stem, base_p in [("BTC_USD", 60000.0), ("ETH_USD", 2000.0), ("XAUT_USD", 3600.0)]:
        close = base_p * np.exp(0.004 * t)
        df = pd.DataFrame(
            {
                "timestamp": idx,
                "open": close * 0.999,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(n, 1000.0),
                "adj_close": close,
            }
        )
        df.to_parquet(data_dir / f"{stem}.parquet", index=False)

    insts = [
        Instrument(id="BTC/USD", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="USD", min_qty=0.00004, qty_step=0.00001),
        Instrument(id="ETH/USD", asset_class=AssetClass.CRYPTO, base="ETH", quote_currency="USD", min_qty=0.0008, qty_step=0.0001),
        Instrument(id="XAUT/USD", asset_class=AssetClass.CRYPTO, base="XAUT", quote_currency="USD", min_qty=0.002, qty_step=0.0001),
    ]
    db_path = tmp_path / "state" / "bot_test.db"
    cfg = AppConfig(
        name="bot_portfolio_test",
        instrument=insts[0],
        instruments=insts,
        portfolio=PortfolioConfig(
            allocation_method="inverse_volatility",
            max_weight_per_asset=0.60,
            max_total_exposure=0.98,
            vol_lookback_bars=30,
            vol_min_bars=10,
            warmup_bars=40,
        ),
        data=DataConfig(provider="local", file_path=str(data_dir), timeframe="1h", lookback_bars=120, use_cache=False),
        forecaster=ForecasterConfig(type="noop"),
        strategy=StrategyConfig(
            type="ema_trend",
            trend_fast_window=10,
            trend_slow_window=30,
            trend_hysteresis_pct=0.01,
            macro_ema_window=60,
            macro_hysteresis_pct=0.01,
        ),
        risk=RiskConfig(
            enforce_market_hours=False,
            max_data_age_minutes=180,
            max_exposure_pct=0.98,
            max_drawdown_pct=0.40,
            fee_model=FeeModelConfig(pct=0.0, fixed_min=0.0),
            enable_fee_guard=False,
        ),
        broker=BrokerConfig(type="local_paper", initial_cash=10000.0),
        notifier=NotifierConfig(type="telegram", enabled=False, notify_on="events"),
        storage=StorageConfig(db_path=str(db_path)),
    )

    repo_root = Path(__file__).resolve().parents[2]
    paths = RuntimePaths.resolve(repo_root)
    store = SQLiteStateStore(db_path)
    provider = CsvParquetProvider(data_dir)
    paper_broker = LocalPaperBroker(
        store=store,
        account_key=cfg.name,
        quote_currency="USD",
        initial_cash=10000.0,
        data_provider=provider,
        timeframe="1h",
    )
    ro_broker = DryRunBroker(paper_broker)
    notifier = TelegramNotifier(
        bot_token="12345:test-token",
        chat_id="77777",
        enabled=False,
        template_path=repo_root / "configs" / "templates" / "telegram_portfolio.html.j2",
        daily_template_path=repo_root / "configs" / "templates" / "telegram_daily.html.j2",
    )
    bot = TelegramBotService(
        config=cfg,
        paths=paths,
        store=store,
        provider=provider,
        broker=ro_broker,
        notifier=notifier,
        bot_token="12345:test-token",
        allowed_chat_id="77777",
    )
    return cfg, store, provider, bot


def test_telegram_bot_auth_commands_and_halt_resume_integration(tmp_path: Path) -> None:
    cfg, store, provider, bot = _setup_bot_env(tmp_path)

    # 1. Unauthorized chat_id MUST be completely ignored (even /halt!)
    assert bot.handle_text(chat_id="999999", text="/halt") is None
    assert store.is_halted(cfg.name) is False
    assert bot.handle_text(chat_id="999999", text="/status") is None

    # 2. /help and /start@BotName
    help_reply = bot.handle_text(chat_id="77777", text="/help@MyBot")
    assert help_reply is not None
    assert "/status" in help_reply
    assert "/halt" in help_reply

    # 3. /status before any run
    status_reply = bot.handle_text(chat_id="77777", text="/status")
    assert status_reply is not None
    assert "ATTIVO" in status_reply
    assert "10,000.00 USD" in status_reply

    # 4. /signals evaluates live EMAs on all 3 assets
    sig_reply = bot.handle_text(chat_id="77777", text="/signals")
    assert sig_reply is not None
    assert "BTC" in sig_reply and "ETH" in sig_reply and "XAUT" in sig_reply

    # 5. /halt sets persistent halt in SQLite -> blocks BUY orders in PortfolioStepRunner!
    halt_reply = bot.handle_text(chat_id="77777", text="/halt")
    assert halt_reply is not None and "PAUSA" in halt_reply
    assert store.is_halted(cfg.name) is True

    cal = MarketCalendar("24/7")
    live_paper = LocalPaperBroker(
        store=store,
        account_key=cfg.name,
        quote_currency="USD",
        initial_cash=10000.0,
        data_provider=provider,
        timeframe="1h",
    )
    runner = PortfolioStepRunner(
        config=cfg,
        allocator=PortfolioAllocator(cfg, cal),
        risk_manager=RiskManager(cfg.risk, cal),
        broker=live_paper,
        store=store,
        notifier=None,
    )
    candles_by_id = {
        inst.id: provider.fetch_ohlcv(inst, "1h", limit=120)
        for inst in cfg.instruments
    }
    halted_outcome = runner.run_step(
        candles_by_id=candles_by_id,
        check_staleness=True,
        enforce_idempotency=False,
        persist=True,
        notify=False,
    )
    # Because /halt was active, zero fills occurred and DrawdownCircuitBreaker rejected BUYs
    assert len(halted_outcome.fills) == 0
    assert len(halted_outcome.rejected_orders) > 0
    assert halted_outcome.rejected_orders[0].guard_name == "DrawdownCircuitBreaker"

    # 6. /resume clears halt -> next step executes real BUY fills!
    resume_reply = bot.handle_text(chat_id="77777", text="/resume")
    assert resume_reply is not None and "RIATTIVATA" in resume_reply
    assert store.is_halted(cfg.name) is False

    active_outcome = runner.run_step(
        candles_by_id=candles_by_id,
        check_staleness=True,
        enforce_idempotency=False,
        persist=True,
        notify=False,
    )
    assert len(active_outcome.fills) > 0

    # 7. Now /fills and /daily reflect the executed trades!
    fills_reply = bot.handle_text(chat_id="77777", text="/fills")
    assert fills_reply is not None and "BUY" in fills_reply

    daily_reply = bot.handle_text(chat_id="77777", text="/daily")
    assert daily_reply is not None
    assert "Riepilogo Giornaliero" in daily_reply
    assert "Non allocato (Cash)" in daily_reply


def test_telegram_bot_poll_once_skips_stale_startup_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, store, _, bot = _setup_bot_env(tmp_path)
    sent_replies: list[str] = []
    now_epoch = int(time.time())

    class _FakeGetResp:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "ok": True,
                "result": [
                    # Stale /halt sent 2 hours ago while container was stopped -> MUST be skipped on initial drain!
                    {
                        "update_id": 100,
                        "message": {
                            "date": now_epoch - 7200,
                            "chat": {"id": 77777},
                            "text": "/halt",
                        },
                    },
                    # Fresh /help sent 5 seconds ago -> MUST be processed!
                    {
                        "update_id": 101,
                        "message": {
                            "date": now_epoch - 5,
                            "chat": {"id": 77777},
                            "text": "/help",
                        },
                    },
                ],
            }

    monkeypatch.setattr(httpx.Client, "get", lambda self, url, **kw: _FakeGetResp())
    monkeypatch.setattr(bot, "_send_reply", lambda text: sent_replies.append(text))

    processed = bot.poll_once(is_initial_drain=True)
    assert processed == 1
    assert store.is_halted("bot_portfolio_test") is False  # Stale /halt was safely skipped!
    assert bot._offset == 102
    assert len(sent_replies) == 1 and "Comandi disponibili" in sent_replies[0]
