"""Universal Trader — modular, asset-agnostic probabilistic trading platform."""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_PLUGINS_LOADED = False

_BUILTIN_PLUGIN_MODULES: tuple[str, ...] = (
    "universal_trader.brokers.alpaca",
    "universal_trader.brokers.bitfinex",
    "universal_trader.brokers.ibkr",
    "universal_trader.brokers.local_paper",
    "universal_trader.brokers.simulated",
    "universal_trader.data.alpaca",
    "universal_trader.data.binance",
    "universal_trader.data.bitfinex",
    "universal_trader.data.local",
    "universal_trader.data.yahoo",
    "universal_trader.forecasters.mock",
    "universal_trader.forecasters.noop",
    "universal_trader.reporting.telegram",
    "universal_trader.strategies.sizing",
)


def load_builtin_plugins() -> None:
    """Import all built-in adapter modules so their registry decorators run once."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    for mod_name in _BUILTIN_PLUGIN_MODULES:
        importlib.import_module(mod_name)
    _PLUGINS_LOADED = True
