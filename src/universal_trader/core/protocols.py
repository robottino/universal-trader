"""Hexagonal Architecture Ports (typing.Protocol contracts)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from universal_trader.core.models import (
    ApprovedOrder,
    FillResult,
    ForecastRequest,
    ForecastResult,
    Instrument,
    MarketSnapshot,
    OrderIntent,
    PortfolioState,
    Quote,
    RejectedOrder,
    StepOutcome,
)


@dataclass(frozen=True)
class ForecasterCapabilities:
    """Declares native capabilities of a Forecaster plugin."""

    supports_nan: bool = True
    supports_past_covariates: bool = True
    supports_known_future_covariates: bool = True
    supports_categorical_covariates: bool = True
    max_context: int = 32768
    max_horizon: int = 1024


@runtime_checkable
class DataProvider(Protocol):
    """Port for market OHLCV data acquisition."""

    name: str

    def fetch_ohlcv(
        self,
        instrument: Instrument,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles indexed by UTC DatetimeIndex."""
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Port for probabilistic time-series forecasting models."""

    model_id: str
    capabilities: ForecasterCapabilities

    def predict(self, request: ForecastRequest) -> ForecastResult:
        """Run probabilistic forecast for a single request."""
        ...

    def predict_batch(self, requests: Sequence[ForecastRequest]) -> list[ForecastResult]:
        """Run batched probabilistic forecasts."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """Port for turning probabilistic forecasts into OrderIntents."""

    name: str

    def evaluate(
        self,
        forecast: ForecastResult,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> OrderIntent:
        """Produce a desired BUY/SELL/HOLD OrderIntent."""
        ...


@runtime_checkable
class RiskGuard(Protocol):
    """Composable guard inside RiskManager."""

    name: str

    def check(
        self,
        context: Any,
    ) -> RejectedOrder | None:
        """Inspect and optionally mutate risk context or reject the order."""
        ...


@runtime_checkable
class Broker(Protocol):
    """Port for order execution and portfolio/quote queries."""

    name: str

    def get_quote(self, instrument: Instrument) -> Quote:
        """Return latest bid/ask/last quote for instrument."""
        ...

    def get_portfolio(self, instrument: Instrument, quote: Quote | None = None) -> PortfolioState:
        """Return current portfolio state (cash, positions, equity)."""
        ...

    def submit_order(self, order: ApprovedOrder, instrument: Instrument) -> FillResult:
        """Execute an approved order and return fill details."""
        ...


@runtime_checkable
class StateStore(Protocol):
    """Port for persistent operational audit trail, HWM, and idempotency."""

    def has_processed(
        self,
        config_name: str,
        instrument_id: str,
        timeframe: str,
        candle_timestamp: datetime,
    ) -> bool:
        ...

    def get_high_water_mark(self, config_name: str, default: float = 0.0) -> float:
        ...

    def is_halted(self, config_name: str) -> bool:
        ...

    def record_step(self, outcome: StepOutcome) -> int:
        ...


@runtime_checkable
class Notifier(Protocol):
    """Port for operational reporting and risk/error alerting."""

    name: str

    def notify_step(
        self,
        outcome: StepOutcome,
        instrument: Instrument,
        chart_path: Path | None = None,
    ) -> None:
        ...

    def notify_alert(
        self,
        title: str,
        message: str,
        instrument: Instrument | None = None,
    ) -> None:
        ...
