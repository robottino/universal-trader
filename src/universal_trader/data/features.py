"""Strictly causal FeaturePipeline separating past_only and known_future covariates."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import FeaturesConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import ForecastRequest, Instrument, ensure_utc
from universal_trader.data.yahoo import MACRO_TICKER_MAP, YahooDataProvider

logger = get_logger(__name__)

_LOG_RET_RE = re.compile(r"^log_return_(\d+)$")
_RVOL_RE = re.compile(r"^realized_vol_(\d+)$")
_OBI_RE = re.compile(r"^rolling_obi_(\d+)$")
_RSI_RE = re.compile(r"^rsi_(\d+)$")
_SMA_RATIO_RE = re.compile(r"^sma_ratio_(\d+)$")


def _causal_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.where(avg_loss > 0, 100.0).where(~avg_gain.isna(), np.nan)


def _causal_rolling_obi(df: pd.DataFrame, window: int) -> pd.Series:
    """Order-book / candle pressure proxy normalized strictly with a causal rolling window."""
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    raw_pressure = ((df["close"] - df["open"]) / bar_range).fillna(0.0) * df["volume"]
    roll_mean = raw_pressure.rolling(window=window, min_periods=2).mean()
    roll_std = raw_pressure.rolling(window=window, min_periods=2).std(ddof=0).replace(0.0, np.nan)
    return (raw_pressure - roll_mean) / roll_std


class FeaturePipeline:
    """Builds strictly causal past_only and known_future covariates without lookahead bias."""

    def __init__(
        self,
        features_config: FeaturesConfig,
        calendar: MarketCalendar,
        price_column: str = "close",
        macro_series: dict[str, pd.Series] | None = None,
        macro_provider: YahooDataProvider | None = None,
    ) -> None:
        self.config = features_config
        self.calendar = calendar
        self.price_column = price_column
        self._macro_series: dict[str, pd.Series] = dict(macro_series or {})
        self._macro_provider = macro_provider

    def preload_macro_series(self, candles: pd.DataFrame, timeframe: str) -> None:
        """Fetch any requested macro past_only covariates not already supplied."""
        needed = [
            f.lower()
            for f in self.config.past_only
            if f.lower() in MACRO_TICKER_MAP and f.lower() not in self._macro_series
        ]
        if not needed:
            return
        provider = self._macro_provider or YahooDataProvider()
        start_ts = ensure_utc(candles.index.min())
        end_ts = ensure_utc(candles.index.max())
        for name in needed:
            self._macro_series[name] = provider.fetch_macro_series(
                macro_name=name,
                timeframe=timeframe,
                start=start_ts,
                end=end_ts,
            )

    def _compute_past_feature(self, name: str, df: pd.DataFrame, price: pd.Series) -> pd.Series:
        key = name.strip().lower()
        if key == "volume":
            return df["volume"].astype(np.float64)
        if key in {"open", "high", "low", "close", "adj_close"}:
            return df[key].astype(np.float64)
        if m := _LOG_RET_RE.match(key):
            lag = int(m.group(1))
            return np.log(price / price.shift(lag))
        if m := _RVOL_RE.match(key):
            win = int(m.group(1))
            lr = np.log(price / price.shift(1))
            return lr.rolling(window=win, min_periods=2).std(ddof=0)
        if m := _OBI_RE.match(key):
            win = int(m.group(1))
            return _causal_rolling_obi(df, window=win)
        if m := _RSI_RE.match(key):
            win = int(m.group(1))
            return _causal_rsi(price, window=win)
        if m := _SMA_RATIO_RE.match(key):
            win = int(m.group(1))
            sma = price.rolling(window=win, min_periods=1).mean()
            return (price / sma) - 1.0
        if key in self._macro_series:
            return self._align_macro_backward(self._macro_series[key], df.index)
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce").astype(np.float64)
        raise ValueError(
            f"Unknown past_only feature '{name}'. Provide it in candles/macro_series or use a built-in feature."
        )

    @staticmethod
    def _align_macro_backward(series: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
        """Align external macro series to target_index strictly backward (timestamp <= t)."""
        macro_df = series.dropna().sort_index().to_frame(name="val").reset_index()
        macro_df.columns = ["ts", "val"]
        macro_df["ts"] = pd.to_datetime(macro_df["ts"], utc=True)

        left = pd.DataFrame({"ts": pd.to_datetime(target_index, utc=True)})
        aligned = pd.merge_asof(
            left.sort_values("ts"),
            macro_df.sort_values("ts"),
            on="ts",
            direction="backward",
        )
        return pd.Series(aligned["val"].to_numpy(dtype=np.float64), index=target_index)

    def _compute_known_feature(self, name: str, full_index: pd.DatetimeIndex) -> np.ndarray:
        key = name.strip().lower()
        if key == "hour_sin":
            hours = full_index.hour.to_numpy(dtype=np.float64) + full_index.minute.to_numpy(dtype=np.float64) / 60.0
            return np.asarray(np.sin(2.0 * np.pi * hours / 24.0), dtype=np.float64)
        if key == "hour_cos":
            hours = full_index.hour.to_numpy(dtype=np.float64) + full_index.minute.to_numpy(dtype=np.float64) / 60.0
            return np.asarray(np.cos(2.0 * np.pi * hours / 24.0), dtype=np.float64)
        if key == "dow_sin":
            dow = full_index.day_of_week.to_numpy(dtype=np.float64)
            return np.asarray(np.sin(2.0 * np.pi * dow / 7.0), dtype=np.float64)
        if key == "dow_cos":
            dow = full_index.day_of_week.to_numpy(dtype=np.float64)
            return np.asarray(np.cos(2.0 * np.pi * dow / 7.0), dtype=np.float64)
        if key == "month_sin":
            month = full_index.month.to_numpy(dtype=np.float64) - 1.0
            return np.asarray(np.sin(2.0 * np.pi * month / 12.0), dtype=np.float64)
        if key == "month_cos":
            month = full_index.month.to_numpy(dtype=np.float64) - 1.0
            return np.asarray(np.cos(2.0 * np.pi * month / 12.0), dtype=np.float64)
        if key == "is_market_open":
            return np.array(
                [1.0 if self.calendar.is_market_open(ts.to_pydatetime()) else 0.0 for ts in full_index],
                dtype=np.float64,
            )
        if key == "is_pre_holiday":
            return np.array(
                [1.0 if self.calendar.is_pre_holiday(ts.to_pydatetime()) else 0.0 for ts in full_index],
                dtype=np.float64,
            )
        raise ValueError(f"Unknown known_future feature '{name}'")

    def _compute_categorical_feature(self, name: str, index: pd.DatetimeIndex) -> np.ndarray:
        key = name.strip().lower()
        if key == "day_of_week":
            return np.asarray(index.day_of_week.to_numpy(dtype=np.float64), dtype=np.float64)
        if key == "hour_of_day":
            return np.asarray(index.hour.to_numpy(dtype=np.float64), dtype=np.float64)
        if key == "month":
            return np.asarray(index.month.to_numpy(dtype=np.float64), dtype=np.float64)
        raise ValueError(f"Unknown categorical feature '{name}'")

    def compute_past_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured past_only covariates on a causal candle DataFrame."""
        price_col = self.price_column if self.price_column in candles.columns else "close"
        price = candles[price_col].astype(np.float64)
        cols: dict[str, Any] = {}
        for feat in self.config.past_only:
            cols[feat] = self._compute_past_feature(feat, candles, price)
        return pd.DataFrame(cols, index=candles.index)

    def build_request(
        self,
        candles: pd.DataFrame,
        instrument: Instrument,
        timeframe: str,
        context_len: int,
        horizon: int,
        *,
        supports_nan: bool = True,
        supports_past_covariates: bool = True,
        supports_known_future_covariates: bool = True,
    ) -> ForecastRequest:
        """Build a strictly causal ForecastRequest from history ending at candles.index[-1]."""
        if candles.empty:
            raise ValueError("Cannot build ForecastRequest from empty candles")

        # Keep enough warm-up rows to compute rolling features causally, then slice last context_len
        warmup = max(context_len + 128, context_len)
        window_df = candles.iloc[-warmup:] if len(candles) > warmup else candles

        price_col = self.price_column if self.price_column in window_df.columns else "close"
        price_series = window_df[price_col].astype(np.float64)

        past_df = self.compute_past_frame(window_df) if (self.config.past_only and supports_past_covariates) else pd.DataFrame(index=window_df.index)

        # Slice strictly to the final context_len rows
        ctx_index = window_df.index[-context_len:]
        target_series = price_series.loc[ctx_index]
        if not supports_nan:
            target_series = target_series.ffill()
        target_arr = target_series.to_numpy(dtype=np.float64)

        past_covs: dict[str, np.ndarray] = {}
        if supports_past_covariates and not past_df.empty:
            sliced_past = past_df.loc[ctx_index]
            if not supports_nan:
                # Strictly causal imputation only (ffill, NEVER bfill); remaining leading NaN -> 0.0
                sliced_past = sliced_past.ffill().fillna(0.0)
            for col in sliced_past.columns:
                past_covs[col] = sliced_past[col].to_numpy(dtype=np.float64)

        last_ts = ensure_utc(ctx_index[-1])
        future_idx = self.calendar.future_timestamps(last_ts, horizon=horizon, timeframe=timeframe)
        full_idx = ctx_index.append(future_idx)

        known_covs: dict[str, np.ndarray] = {}
        if supports_known_future_covariates:
            for feat in self.config.known_future:
                known_covs[feat] = self._compute_known_feature(feat, full_idx)

        cat_past: dict[str, np.ndarray] = {}
        for feat in self.config.categorical_past:
            cat_past[feat] = self._compute_categorical_feature(feat, ctx_index)

        cat_known: dict[str, np.ndarray] = {}
        if supports_known_future_covariates:
            for feat in self.config.categorical_known:
                cat_known[feat] = self._compute_categorical_feature(feat, full_idx)

        return ForecastRequest(
            instrument_id=instrument.id,
            timestamp=last_ts,
            target=target_arr,
            horizon=horizon,
            past_covariates=past_covs,
            known_future_covariates=known_covs,
            categorical_past_covariates=cat_past,
            categorical_known_covariates=cat_known,
            future_timestamps=[ensure_utc(ts) for ts in future_idx],
        )
