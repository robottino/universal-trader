"""Forecaster plugins (Noop and Mock)."""

from universal_trader.forecasters.mock import MockForecaster
from universal_trader.forecasters.noop import NoopForecaster

__all__ = [
    "MockForecaster",
    "NoopForecaster",
]
