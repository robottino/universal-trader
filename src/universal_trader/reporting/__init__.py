"""Reporting utilities: Matplotlib multi-quantile fan charts, equity curves, and Telegram notifier."""

from universal_trader.reporting.plots import plot_backtest_report, plot_fan_chart
from universal_trader.reporting.telegram import NoopNotifier, TelegramNotifier

__all__ = [
    "NoopNotifier",
    "TelegramNotifier",
    "plot_backtest_report",
    "plot_fan_chart",
]
