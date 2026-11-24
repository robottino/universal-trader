"""Unit tests verifying strict feature causality (zero lookahead bias) & incremental Parquet cache."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import FeaturesConfig
from universal_trader.core.models import AssetClass, Instrument
from universal_trader.data.cache import ParquetCacheStore
from universal_trader.data.features import FeaturePipeline


def _make_ohlcv(n: int = 200, start: datetime | None = None, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = start or datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    idx = pd.date_range(t0, periods=n, freq=freq, tz=UTC)
    returns = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
    vol = rng.uniform(500.0, 5000.0, size=n)
    adj_close = close * 0.98
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "adj_close": adj_close},
        index=idx,
    )


def test_zero_lookahead_bias_bit_for_bit_causality() -> None:
    """Perturbing prices, volume, and macro series AFTER time t must leave ALL features <= t bit-for-bit identical."""
    candles_orig = _make_ohlcv(n=180, freq="1h", seed=7)
    candles_perturbed = candles_orig.copy()

    t_idx = 110
    # Violently perturb everything strictly AFTER t_idx
    candles_perturbed.iloc[t_idx + 1 :, candles_perturbed.columns.get_loc("close")] *= 10.0
    candles_perturbed.iloc[t_idx + 1 :, candles_perturbed.columns.get_loc("open")] *= 9.5
    candles_perturbed.iloc[t_idx + 1 :, candles_perturbed.columns.get_loc("high")] *= 12.0
    candles_perturbed.iloc[t_idx + 1 :, candles_perturbed.columns.get_loc("low")] *= 0.1
    candles_perturbed.iloc[t_idx + 1 :, candles_perturbed.columns.get_loc("volume")] *= 100.0

    # Daily macro series aligned backward via merge_asof
    macro_idx = pd.date_range(datetime(2024, 12, 28, tzinfo=UTC), periods=20, freq="1D", tz=UTC)
    vix_orig = pd.Series(np.linspace(15.0, 25.0, len(macro_idx)), index=macro_idx, name="vix")
    vix_perturbed = vix_orig.copy()
    cut_ts = candles_orig.index[t_idx]
    vix_perturbed.loc[vix_perturbed.index > cut_ts] = 999.0

    fcfg = FeaturesConfig(
        past_only=["volume", "log_return_1", "realized_vol_24", "rolling_obi_24", "rsi_14", "vix"],
        known_future=["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    )
    cal = MarketCalendar("24/7")

    pipe_orig = FeaturePipeline(fcfg, cal, price_column="close", macro_series={"vix": vix_orig})
    pipe_pert = FeaturePipeline(fcfg, cal, price_column="close", macro_series={"vix": vix_perturbed})

    feat_orig = pipe_orig.compute_past_frame(candles_orig).iloc[: t_idx + 1]
    feat_pert = pipe_pert.compute_past_frame(candles_perturbed).iloc[: t_idx + 1]

    for col in feat_orig.columns:
        np.testing.assert_array_equal(
            feat_orig[col].to_numpy(),
            feat_pert[col].to_numpy(),
            err_msg=f"Lookahead bias detected in past_only feature '{col}'!",
        )


def test_features_crypto_and_equity_nan_handling() -> None:
    # Crypto 1h with native NaN preserved for Timestral
    btc_df = _make_ohlcv(n=100, freq="1h", seed=1)
    btc = Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR", calendar="24/7")
    pipe_btc = FeaturePipeline(
        FeaturesConfig(
            past_only=["volume", "log_return_1", "rolling_obi_24"],
            known_future=["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
        ),
        MarketCalendar("24/7"),
        price_column="close",
    )
    req_nan = pipe_btc.build_request(btc_df, btc, "1h", context_len=64, horizon=12, supports_nan=True)
    assert req_nan.target.shape == (64,)
    assert len(req_nan.future_timestamps) == 12
    assert req_nan.known_future_covariates["hour_sin"].shape == (76,)
    assert req_nan.past_covariates["rolling_obi_24"].shape == (64,)

    # Equity 1d using adj_close and XETR calendar (e.g. VWCE.DE)
    eq_df = _make_ohlcv(n=80, freq="B", seed=2)
    eq_df.iloc[50, eq_df.columns.get_loc("adj_close")] = np.nan
    vwce = Instrument(
        id="VWCE.DE",
        asset_class=AssetClass.ETF,
        base="VWCE",
        quote_currency="EUR",
        calendar="XETR",
        fractional=False,
        min_qty=1,
        qty_step=1,
    )
    pipe_eq = FeaturePipeline(
        FeaturesConfig(
            past_only=["volume", "log_return_1"],
            known_future=["dow_sin", "dow_cos", "is_pre_holiday"],
        ),
        MarketCalendar("XETR"),
        price_column="adj_close",
    )
    req_eq_nan = pipe_eq.build_request(eq_df, vwce, "1d", context_len=60, horizon=5, supports_nan=True)
    assert np.isnan(req_eq_nan.target).any()  # Native NaN preserved for Timestral

    req_eq_imp = pipe_eq.build_request(eq_df, vwce, "1d", context_len=60, horizon=5, supports_nan=False)
    assert not np.isnan(req_eq_imp.target).any()  # Causal ffill applied when model doesn't support NaN


def test_parquet_cache_store_incremental_updates(tmp_path: Path) -> None:
    full_df = _make_ohlcv(n=100, start=datetime(2025, 1, 1, tzinfo=UTC), freq="1h")
    calls: list[tuple[datetime | None, datetime | None]] = []

    class CountingProvider:
        name = "mock_src"

        def fetch_ohlcv(
            self,
            instrument: Instrument,
            timeframe: str,
            start: datetime | None = None,
            end: datetime | None = None,
            limit: int | None = None,
        ) -> pd.DataFrame:
            calls.append((start, end))
            sub = full_df
            if start is not None:
                sub = sub[sub.index >= pd.Timestamp(start)]
            if end is not None:
                sub = sub[sub.index <= pd.Timestamp(end)]
            return sub

    inst = Instrument(id="BTC/EUR", asset_class=AssetClass.CRYPTO, base="BTC", quote_currency="EUR")
    store = ParquetCacheStore(CountingProvider(), cache_dir=tmp_path)

    # First call: fetches [20 .. 60]
    t20 = full_df.index[20].to_pydatetime()
    t60 = full_df.index[60].to_pydatetime()
    df1 = store.fetch_ohlcv(inst, "1h", start=t20, end=t60)
    assert len(df1) == 41
    assert len(calls) == 1

    # Second call on same sub-window [25 .. 55]: pure cache hit (0 new provider calls!)
    df2 = store.fetch_ohlcv(inst, "1h", start=full_df.index[25].to_pydatetime(), end=full_df.index[55].to_pydatetime())
    assert len(df2) == 31
    assert len(calls) == 1

    # Third call extending forward to [25 .. 90]: fetches ONLY incremental tail!
    df3 = store.fetch_ohlcv(inst, "1h", start=full_df.index[25].to_pydatetime(), end=full_df.index[90].to_pydatetime())
    assert len(df3) == 66
    assert len(calls) == 2
    assert calls[1][0] is not None and calls[1][0] >= t60 - timedelta(hours=2)
