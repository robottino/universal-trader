"""Dependency-injection factory wiring AppConfig into registered hexagonal adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from universal_trader import load_builtin_plugins
from universal_trader.brokers.dry_run import DryRunBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import AppConfig, RuntimePaths
from universal_trader.core.models import TradingMode
from universal_trader.core.protocols import Broker, DataProvider, Forecaster, Notifier, Strategy
from universal_trader.core.registry import registry
from universal_trader.data.cache import ParquetCacheStore
from universal_trader.data.features import FeaturePipeline
from universal_trader.data.yahoo import YahooDataProvider
from universal_trader.risk.manager import RiskManager
from universal_trader.storage.sqlite_store import SQLiteStateStore


def resolve_db_path(config: AppConfig, paths: RuntimePaths) -> Path:
    raw = Path(config.storage.db_path)
    if raw.is_absolute():
        return raw
    # If UTRADER_STATE_DIR is customized and config uses default "state/<name>", honor UTRADER_STATE_DIR
    if raw.parts and raw.parts[0] == "state":
        return paths.state_dir / Path(*raw.parts[1:])
    return paths.root / raw


def build_data_provider(config: AppConfig, paths: RuntimePaths) -> DataProvider:
    load_builtin_plugins()
    prov_cls: Any = registry.get("data_provider", config.data.provider)
    if config.data.provider.lower() in {"local", "csv_parquet"}:
        if not config.data.file_path:
            raise ValueError("data.file_path is required when data.provider is 'local'/'csv_parquet'")
        fp = Path(config.data.file_path)
        if not fp.is_absolute():
            fp = paths.root / fp
        provider: DataProvider = prov_cls(file_path=fp)
    else:
        provider = prov_cls()

    if config.data.use_cache and config.data.provider.lower() not in {"local", "csv_parquet"}:
        return ParquetCacheStore(provider=provider, cache_dir=paths.data_dir)
    return provider


def build_forecaster(config: AppConfig) -> Forecaster:
    load_builtin_plugins()
    fc_cls: Any = registry.get("forecaster", config.forecaster.type)
    params = config.forecaster.params.model_dump()
    return fc_cls(**params)  # type: ignore[no-any-return]


def build_strategy(config: AppConfig) -> Strategy:
    load_builtin_plugins()
    strat_cls: Any = registry.get("strategy", config.strategy.type)
    return strat_cls(config.strategy)  # type: ignore[no-any-return]


def build_notifier(config: AppConfig) -> Notifier:
    load_builtin_plugins()
    if not config.notifier.enabled or config.notifier.type.lower() == "noop":
        noop_cls: Any = registry.get("notifier", "noop")
        return noop_cls()  # type: ignore[no-any-return]
    notif_cls: Any = registry.get("notifier", config.notifier.type)
    return notif_cls(  # type: ignore[no-any-return]
        bot_token=config.notifier.bot_token,
        chat_id=config.notifier.chat_id,
        enabled=config.notifier.enabled,
        notify_on=config.notifier.notify_on,
        daily_summary_hour_utc=config.notifier.daily_summary_hour_utc,
    )


def build_broker(
    config: AppConfig,
    *,
    store: SQLiteStateStore | None = None,
    data_provider: DataProvider | None = None,
    dry_run_override: bool = False,
) -> Broker:
    load_builtin_plugins()
    btype = config.broker.type.lower()
    broker_cls: Any = registry.get("broker", btype)

    broker: Broker
    if btype == "local_paper":
        if store is None:
            raise ValueError("LocalPaperBroker requires a SQLiteStateStore")
        broker = broker_cls(
            store=store,
            account_key=config.name,
            quote_currency=config.instrument.quote_currency,
            initial_cash=config.broker.initial_cash,
            fee_model=config.risk.fee_model,
            slippage_bps=config.risk.slippage_bps,
            spread_bps=config.risk.spread_bps,
            data_provider=data_provider,
            timeframe=config.data.timeframe,
            price_column=config.data.price_column,
        )
    elif btype == "simulated":
        broker = broker_cls(
            initial_cash=config.broker.initial_cash,
            quote_currency=config.instrument.quote_currency,
            fee_model=config.risk.fee_model,
            slippage_bps=config.risk.slippage_bps,
            spread_bps=config.risk.spread_bps,
            execution_price=config.broker.execution_price,
            max_drawdown_pct=config.risk.max_drawdown_pct,
        )
    elif btype in {"bitfinex", "alpaca"}:
        broker = broker_cls(
            api_key=config.broker.api_key,
            api_secret=config.broker.api_secret,
            paper=config.broker.paper or (config.mode == TradingMode.PAPER),
        )
    elif btype == "ibkr":
        broker = broker_cls(
            host=config.broker.host,
            port=config.broker.port,
            client_id=config.broker.client_id,
            paper=config.broker.paper or (config.mode == TradingMode.PAPER),
        )
    else:
        broker = broker_cls(**config.broker.model_dump(exclude={"type"}))

    if dry_run_override or config.mode == TradingMode.DRY_RUN:
        return DryRunBroker(broker)
    return broker


def build_feature_pipeline(config: AppConfig, calendar: MarketCalendar) -> FeaturePipeline:
    return FeaturePipeline(
        features_config=config.data.features,
        calendar=calendar,
        price_column=config.data.price_column,
        macro_provider=YahooDataProvider(),
    )


def build_risk_manager(config: AppConfig, calendar: MarketCalendar) -> RiskManager:
    return RiskManager(config=config.risk, calendar=calendar)
