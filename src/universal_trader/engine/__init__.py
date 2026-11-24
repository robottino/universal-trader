"""Core trading execution engine: TradingStepRunner, BacktestRunner, WalkForwardEngine, metrics."""

from universal_trader.engine.backtest import BacktestResult, BacktestRunner
from universal_trader.engine.forecast_cache import ForecastCache
from universal_trader.engine.metrics import (
    compute_forecast_calibration,
    compute_performance_metrics,
)
from universal_trader.engine.runner import TradingStepRunner
from universal_trader.engine.walkforward import (
    WalkForwardEngine,
    WalkForwardFold,
    WalkForwardResult,
)

__all__ = [
    "BacktestResult",
    "BacktestRunner",
    "ForecastCache",
    "TradingStepRunner",
    "WalkForwardEngine",
    "WalkForwardFold",
    "WalkForwardResult",
    "compute_forecast_calibration",
    "compute_performance_metrics",
]
