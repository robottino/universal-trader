"""Unit tests for core domain models, YAML config validation, secret redaction, and locking."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from universal_trader.core.config import load_config
from universal_trader.core.locking import ConcurrencyLockError, ExecutionLock
from universal_trader.core.logging import SecretRedactingFilter, redact_text, register_secret
from universal_trader.core.models import AssetClass, ForecastResult, Instrument


def test_instrument_fractional_and_whole_lot_validation() -> None:
    btc = Instrument(
        id="BTC/EUR",
        asset_class=AssetClass.CRYPTO,
        base="BTC",
        quote_currency="EUR",
        calendar="24/7",
        fractional=True,
        min_qty=0.0001,
        qty_step=0.00001,
        min_notional=5.0,
        provider_symbols={"binance": "BTCEUR"},
    )
    assert btc.symbol_for("binance") == "BTCEUR"
    assert btc.symbol_for("unknown") == "BTC/EUR"
    assert btc.round_qty(0.1234567) == pytest.approx(0.12345)

    nvda = Instrument(
        id="NVDA",
        asset_class=AssetClass.EQUITY,
        base="NVDA",
        quote_currency="USD",
        calendar="XNAS",
        fractional=False,
        min_qty=1,
        qty_step=1,
        min_notional=10.0,
    )
    assert nvda.round_qty(3.99) == 3.0
    assert nvda.round_qty(0.8) == 0.0

    # Non-fractional equity with fractional qty_step must fail validation immediately
    with pytest.raises(ValueError, match="integer qty_step"):
        Instrument(
            id="NVDA",
            asset_class=AssetClass.EQUITY,
            base="NVDA",
            quote_currency="USD",
            calendar="XNAS",
            fractional=False,
            min_qty=1,
            qty_step=0.5,
        )


def test_forecast_result_quantiles_interpolation_and_serialization() -> None:
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    fc = ForecastResult(
        instrument_id="BTC/EUR",
        model_id="mock/linear",
        timestamp=ts,
        horizon=3,
        quantiles={
            0.10: np.array([90.0, 88.0, 85.0]),
            0.50: np.array([100.0, 102.0, 105.0]),
            0.90: np.array([110.0, 115.0, 125.0]),
        },
    )
    np.testing.assert_allclose(fc.median, [100.0, 102.0, 105.0])
    # Interpolated q=0.30 is halfway between q=0.10 and q=0.50
    np.testing.assert_allclose(fc.quantile(0.30), [95.0, 95.0, 95.0])
    # Expected return relative to 100
    assert fc.expected_return(100.0, mode="last") == pytest.approx(0.05)
    assert fc.expected_return(100.0, mode="mean") == pytest.approx((102.3333333 - 100.0) / 100.0, rel=1e-4)
    # Probability above 105 (median) should be 0.50
    assert fc.prob_above(105.0, step=-1) == pytest.approx(0.50)
    assert fc.prob_above(85.0, step=-1) == pytest.approx(0.90)

    # Round-trip JSON serialization
    restored = ForecastResult.from_serializable_dict(fc.to_serializable_dict())
    assert restored.instrument_id == fc.instrument_id
    assert restored.timestamp == fc.timestamp
    np.testing.assert_allclose(restored.median, fc.median)


def test_config_yaml_loading_env_interpolation_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_API_KEY", "super-secret-api-key-999")
    cfg_file = tmp_path / "valid.yaml"
    cfg_file.write_text(
        """
name: "test_cfg"
mode: paper
instrument:
  id: "NVDA"
  asset_class: equity
  base: NVDA
  quote_currency: USD
  calendar: "XNAS"
  fractional: false
  min_qty: 1
  qty_step: 1
data:
  provider: yahoo
  timeframe: "1d"
  price_column: adj_close
forecaster:
  type: mock
  params:
    model_id: "mock/linear"
    context_len: 128
    horizon: 5
    max_horizon: 32
broker:
  type: alpaca
  api_key: "${TEST_API_KEY}"
  api_secret: "${UNSET_SECRET:-fallback_val}"
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.broker.api_key == "super-secret-api-key-999"
    assert cfg.broker.api_secret == "fallback_val"
    # Secret must automatically be registered for redaction
    assert "super-secret-api-key-999" not in redact_text("leaked super-secret-api-key-999 here")

    # Invalid horizon > max_horizon must fail fast
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        """
name: "bad_cfg"
instrument:
  id: "BTC/EUR"
  asset_class: crypto
  base: BTC
  quote_currency: EUR
forecaster:
  type: mock
  params:
    horizon: 2048
    max_horizon: 1024
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exceeds max_horizon"):
        load_config(bad_cfg)


def test_secret_redacting_log_filter() -> None:
    register_secret("MY_TOP_SECRET_VALUE_12345")
    filt = SecretRedactingFilter()
    rec = logging.LogRecord(
        name="universal_trader.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Sending key=%s and url=https://api.telegram.org/bot123456:ABCdefGHIjklMNOpqrSTUvwx/sendMessage",
        args=("MY_TOP_SECRET_VALUE_12345",),
        exc_info=None,
    )
    filt.filter(rec)
    assert "MY_TOP_SECRET_VALUE_12345" not in str(rec.msg)
    assert "ABCdefGHIjklMNOpqrSTUvwx" not in str(rec.msg)
    assert "***REDACTED***" in str(rec.msg)


def test_execution_lock_prevents_concurrent_access(tmp_path: Path) -> None:
    lock_file = tmp_path / "trader.lock"
    with ExecutionLock(lock_file), pytest.raises(ConcurrencyLockError), ExecutionLock(lock_file):
        pass
    # Re-acquirable after release
    with ExecutionLock(lock_file):
        assert lock_file.exists()
