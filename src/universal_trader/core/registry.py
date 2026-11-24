"""Extensible PluginRegistry for dynamic instantiation of adapters from YAML config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class PluginRegistry:
    """Central plugin catalog mapping component names to classes or factory callables."""

    def __init__(self) -> None:
        self._data_providers: dict[str, Any] = {}
        self._forecasters: dict[str, Any] = {}
        self._strategies: dict[str, Any] = {}
        self._brokers: dict[str, Any] = {}
        self._notifiers: dict[str, Any] = {}

    def register(self, category: str, name: str, cls_or_factory: Any) -> Any:
        bucket = self._bucket(category)
        bucket[name.lower()] = cls_or_factory
        return cls_or_factory

    def get(self, category: str, name: str) -> Any:
        bucket = self._bucket(category)
        key = name.lower()
        if key not in bucket:
            available = sorted(bucket.keys())
            raise KeyError(
                f"Unknown {category} plugin '{name}'. Registered {category} plugins: {available}"
            )
        return bucket[key]

    def list_plugins(self, category: str | None = None) -> dict[str, list[str]]:
        if category is not None:
            return {category: sorted(self._bucket(category).keys())}
        return {
            "data_provider": sorted(self._data_providers.keys()),
            "forecaster": sorted(self._forecasters.keys()),
            "strategy": sorted(self._strategies.keys()),
            "broker": sorted(self._brokers.keys()),
            "notifier": sorted(self._notifiers.keys()),
        }

    def _bucket(self, category: str) -> dict[str, Any]:
        mapping = {
            "data_provider": self._data_providers,
            "forecaster": self._forecasters,
            "strategy": self._strategies,
            "broker": self._brokers,
            "notifier": self._notifiers,
        }
        if category not in mapping:
            raise ValueError(f"Unknown plugin category '{category}'")
        return mapping[category]


registry = PluginRegistry()


def register_data_provider(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        registry.register("data_provider", name, cls)
        return cls

    return decorator


def register_forecaster(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        registry.register("forecaster", name, cls)
        return cls

    return decorator


def register_strategy(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        registry.register("strategy", name, cls)
        return cls

    return decorator


def register_broker(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        registry.register("broker", name, cls)
        return cls

    return decorator


def register_notifier(name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        registry.register("notifier", name, cls)
        return cls

    return decorator
