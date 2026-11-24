"""TradingStepRunner — the single shared execution pipeline for backtest, paper, dry-run, and live."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from universal_trader.brokers.local_paper import LocalPaperBroker
from universal_trader.core.config import AppConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import (
    FillResult,
    ForecastResult,
    MarketSnapshot,
    PortfolioState,
    Quote,
    StepOutcome,
    ensure_utc,
)
from universal_trader.core.protocols import Broker, Forecaster, Notifier, StateStore, Strategy
from universal_trader.data.features import FeaturePipeline
from universal_trader.reporting.plots import plot_fan_chart
from universal_trader.risk.manager import RiskManager

logger = get_logger(__name__)


class TradingStepRunner:
    """Executes one complete, atomic trading cycle identically in simulation and production."""

    def __init__(
        self,
        *,
        config: AppConfig,
        feature_pipeline: FeaturePipeline,
        forecaster: Forecaster,
        strategy: Strategy,
        risk_manager: RiskManager,
        broker: Broker,
        store: StateStore | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        self.config = config
        self.feature_pipeline = feature_pipeline
        self.forecaster = forecaster
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker
        self.store = store
        self.notifier = notifier

    def _quote_from_candles(self, candles: pd.DataFrame) -> Quote:
        col = self.config.data.price_column
        if col not in candles.columns:
            col = "close"
        px = float(candles[col].iloc[-1])
        ts = ensure_utc(candles.index[-1])
        half_spread = px * (self.config.risk.spread_bps / 20_000.0)
        return Quote(
            instrument_id=self.config.instrument.id,
            timestamp=ts,
            bid=max(1e-9, px - half_spread),
            ask=px + half_spread,
            last=px,
        )

    def _merge_persisted_risk_state(self, portfolio: PortfolioState) -> PortfolioState:
        if self.store is None:
            return portfolio
        persisted_hwm = self.store.get_high_water_mark(
            self.config.name,
            default=portfolio.high_water_mark,
        )
        persisted_halted = self.store.is_halted(self.config.name)
        hwm = max(portfolio.high_water_mark, persisted_hwm, portfolio.equity)
        halted = portfolio.halted or persisted_halted
        if hwm == portfolio.high_water_mark and halted == portfolio.halted:
            return portfolio
        return portfolio.model_copy(update={"high_water_mark": hwm, "halted": halted})

    def run_step(
        self,
        candles: pd.DataFrame,
        *,
        precomputed_forecast: ForecastResult | None = None,
        now: datetime | None = None,
        check_staleness: bool = True,
        enforce_idempotency: bool = True,
        force: bool = False,
        persist: bool = True,
        notify: bool = True,
        chart_output_path: Path | None = None,
    ) -> StepOutcome:
        """Run one atomic step: features -> forecast -> strategy -> risk -> broker -> audit -> notify."""
        if candles.empty:
            raise ValueError("TradingStepRunner.run_step received empty candles")

        instrument = self.config.instrument
        candle_ts = ensure_utc(candles.index[-1])
        eval_now = ensure_utc(now) if now is not None else datetime.now(UTC)

        # 1. Idempotency guard per (config_name, instrument_id, timeframe, candle_timestamp)
        if (
            enforce_idempotency
            and not force
            and self.store is not None
            and self.store.has_processed(
                self.config.name,
                instrument.id,
                self.config.data.timeframe,
                candle_ts,
            )
        ):
            logger.info(
                "Idempotent skip: %s (%s %s @ %s) already processed.",
                self.config.name,
                instrument.id,
                self.config.data.timeframe,
                candle_ts.isoformat(),
            )
            return StepOutcome(
                config_name=self.config.name,
                instrument_id=instrument.id,
                timeframe=self.config.data.timeframe,
                candle_timestamp=candle_ts,
                skipped_idempotent=True,
            )

        # 2. Seed LocalPaperBroker quote from candles if applicable, then query broker quote & portfolio
        inner_broker = getattr(self.broker, "_inner", self.broker)
        if isinstance(inner_broker, LocalPaperBroker):
            inner_broker.set_quote(self._quote_from_candles(candles))

        quote = self.broker.get_quote(instrument)
        portfolio_before = self._merge_persisted_risk_state(
            self.broker.get_portfolio(instrument, quote=quote)
        )
        snapshot = MarketSnapshot(
            instrument=instrument,
            timestamp=candle_ts,
            candles=candles,
            quote=quote,
        )

        # 3. Causal feature pipeline + probabilistic forecast (or precomputed batch forecast)
        if precomputed_forecast is not None:
            forecast = precomputed_forecast
        else:
            caps = self.forecaster.capabilities
            request = self.feature_pipeline.build_request(
                candles=candles,
                instrument=instrument,
                timeframe=self.config.data.timeframe,
                context_len=self.config.forecaster.params.context_len,
                horizon=self.config.forecaster.params.horizon,
                supports_nan=caps.supports_nan,
                supports_past_covariates=caps.supports_past_covariates,
                supports_known_future_covariates=caps.supports_known_future_covariates,
            )
            forecast = self.forecaster.predict(request)

        # 4. Strategy evaluation
        intent = self.strategy.evaluate(
            forecast=forecast,
            snapshot=snapshot,
            portfolio=portfolio_before,
        )

        # 5. Risk management chain
        approved_order, rejected_order, is_halted = self.risk_manager.evaluate(
            instrument=instrument,
            snapshot=snapshot,
            portfolio=portfolio_before,
            forecast=forecast,
            intent=intent,
            now=eval_now,
            check_staleness=check_staleness,
        )

        # 6. Order execution
        fill: FillResult | None = None
        if approved_order is not None:
            fill = self.broker.submit_order(approved_order, instrument)

        # 7. Refresh post-step portfolio and update HWM / halted flag
        portfolio_after = self.broker.get_portfolio(instrument, quote=quote)
        new_hwm = max(portfolio_before.high_water_mark, portfolio_after.high_water_mark, portfolio_after.equity)
        portfolio_after = portfolio_after.model_copy(
            update={
                "high_water_mark": new_hwm,
                "halted": portfolio_after.halted or is_halted,
            }
        )

        outcome = StepOutcome(
            config_name=self.config.name,
            instrument_id=instrument.id,
            timeframe=self.config.data.timeframe,
            candle_timestamp=candle_ts,
            skipped_idempotent=False,
            forecast=forecast,
            intent=intent,
            approved_order=approved_order,
            rejected_order=rejected_order,
            fill=fill,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
        )

        # 8. Persistence
        if persist and self.store is not None:
            self.store.record_step(outcome)

        # 9. Optional fan chart + notification
        saved_chart: Path | None = None
        if chart_output_path is not None:
            saved_chart = plot_fan_chart(
                candles=candles,
                forecast=forecast,
                instrument=instrument,
                output_path=chart_output_path,
                price_column=self.config.data.price_column,
            )

        if notify and self.notifier is not None:
            self.notifier.notify_step(outcome, instrument, chart_path=saved_chart)
            if is_halted and not portfolio_before.halted:
                self.notifier.notify_alert(
                    title="DRAWDOWN CIRCUIT BREAKER HALTED",
                    message=(
                        f"Portfolio drawdown reached {portfolio_after.drawdown:.2%} "
                        f"(limit {self.config.risk.max_drawdown_pct:.2%}). "
                        f"Equity: {portfolio_after.equity:,.2f} {portfolio_after.quote_currency}."
                    ),
                    instrument=instrument,
                )

        return outcome
