"""Forecaster plugins and hardware DeviceManager."""

from universal_trader.forecasters.device import DeviceInfo, DeviceManager
from universal_trader.forecasters.mock import MockForecaster
from universal_trader.forecasters.noop import NoopForecaster
from universal_trader.forecasters.timestral_adapter import TimestralForecaster

__all__ = [
    "DeviceInfo",
    "DeviceManager",
    "MockForecaster",
    "NoopForecaster",
    "TimestralForecaster",
]
