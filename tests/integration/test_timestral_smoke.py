"""Real Mistral Timestral-5M smoke tests (CPU + auto/MPS controlled fallback + covariates + NaN)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import FeaturesConfig
from universal_trader.core.models import AssetClass, Instrument
from universal_trader.data.features import FeaturePipeline
from universal_trader.forecasters.timestral_adapter import TimestralForecaster
from universal_trader.reporting.plots import plot_fan_chart


@pytest.mark.slow
def test_timestral_5m_inference_with_covariates_nan_and_mps_fallback(tmp_path: Path) -> None:
    """Smoke test real mistralai/Timestral-5M on CPU and auto (MPS/CUDA/CPU) with NaN + covariates."""
    rng = np.random.default_rng(42)
    idx = pd.date_range(datetime(2025, 1, 1, tzinfo=UTC), periods=160, freq="1h", tz=UTC)
    close = 50000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.005, size=160)))
    close[20] = np.nan  # Native NaN test!
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.uniform(10.0, 500.0, size=160),
            "adj_close": close,
        },
        index=idx,
    )
    btc = Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR")
    pipe = FeaturePipeline(
        FeaturesConfig(
            past_only=["volume", "log_return_1", "rolling_obi_24"],
            known_future=["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        ),
        MarketCalendar("24/7"),
    )
    req = pipe.build_request(df, btc, "1h", context_len=128, horizon=12, supports_nan=True)

    # 1. CPU inference
    fc_cpu = TimestralForecaster(model_id="mistralai/Timestral-5M", device="cpu", batch_size=4)
    res_cpu = fc_cpu.predict(req)
    assert len(res_cpu.quantile_levels) == 21
    assert res_cpu.median.shape == (12,)
    assert np.all(np.isfinite(res_cpu.median))
    # Monotonic quantiles check
    assert np.all(res_cpu.quantile(0.05) <= res_cpu.median + 1e-5)
    assert np.all(res_cpu.median <= res_cpu.quantile(0.95) + 1e-5)

    # 2. Auto device (tests CUDA or Apple Silicon MPS + controlled fallback if MPS hits float64)
    fc_auto = TimestralForecaster(model_id="mistralai/Timestral-5M", device="auto", batch_size=4)
    res_auto = fc_auto.predict(req)
    assert len(res_auto.quantile_levels) == 21
    assert np.all(np.isfinite(res_auto.median))

    # 3. Fan chart generation from real Timestral forecast
    png = plot_fan_chart(df.ffill(), res_cpu, btc, tmp_path / "timestral_5m_fan.png")
    assert png.exists() and png.stat().st_size > 5000
