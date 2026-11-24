"""Unified BacktestRunner iterating TradingStepRunner bar-by-bar with SimulatedBroker."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from universal_trader.brokers.simulated import SimulatedBroker
from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import AppConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import (
    Candle,
    FillResult,
    ForecastRequest,
    ForecastResult,
    StepOutcome,
    ensure_utc,
)
from universal_trader.core.protocols import Forecaster
from universal_trader.data.features import FeaturePipeline
from universal_trader.engine.factory import build_risk_manager, build_strategy
from universal_trader.engine.forecast_cache import ForecastCache
from universal_trader.engine.metrics import (
    compute_forecast_calibration,
    compute_performance_metrics,
)
from universal_trader.engine.runner import TradingStepRunner

logger = get_logger(__name__)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class BacktestResult:
    """Complete backtest outputs: equity curve, fills, step outcomes, forecasts, and metrics."""

    config_name: str
    instrument_id: str
    timeframe: str
    equity_df: pd.DataFrame
    fills: list[FillResult]
    outcomes: list[StepOutcome]
    forecasts: dict[datetime, ForecastResult]
    metrics: dict[str, Any]
    calibration: dict[str, Any]


class BacktestRunner:
    """Historical simulation harness using the EXACT same TradingStepRunner as live trading."""

    def __init__(
        self,
        *,
        config: AppConfig,
        forecaster: Forecaster,
        feature_pipeline: FeaturePipeline,
        calendar: MarketCalendar,
        forecast_cache: ForecastCache | None = None,
    ) -> None:
        self.config = config
        self.forecaster = forecaster
        self.feature_pipeline = feature_pipeline
        self.calendar = calendar
        self.forecast_cache = forecast_cache

    def precompute_forecasts(
        self,
        candles: pd.DataFrame,
        eval_indices: list[int],
        *,
        use_cache: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[datetime, ForecastResult]:
        """Batch-compute strictly causal forecasts for all eval_indices with optional disk cache."""
        cache_key: str | None = None
        cached: dict[datetime, ForecastResult] = {}
        if use_cache and self.forecast_cache is not None:
            cache_key = self.forecast_cache.build_key(self.config, model_id=self.forecaster.model_id)
            cached = self.forecast_cache.load(cache_key)

        missing_indices: list[int] = []
        missing_requests: list[ForecastRequest] = []
        caps = self.forecaster.capabilities
        total_eval = len(eval_indices)

        if progress_callback is not None:
            progress_callback("features", 0, total_eval)

        for pos, idx in enumerate(eval_indices, start=1):
            ts = ensure_utc(candles.index[idx])
            if ts not in cached:
                # Strictly causal slice: ONLY bars <= idx are visible!
                causal_slice = candles.iloc[: idx + 1]
                req = self.feature_pipeline.build_request(
                    candles=causal_slice,
                    instrument=self.config.instrument,
                    timeframe=self.config.data.timeframe,
                    context_len=self.config.forecaster.params.context_len,
                    horizon=self.config.forecaster.params.horizon,
                    supports_nan=caps.supports_nan,
                    supports_past_covariates=caps.supports_past_covariates,
                    supports_known_future_covariates=caps.supports_known_future_covariates,
                )
                missing_indices.append(idx)
                missing_requests.append(req)
            if progress_callback is not None and (pos % 25 == 0 or pos == total_eval):
                progress_callback("features", pos, total_eval)

        if missing_requests:
            logger.info(
                "Computing %d forecasts in batch (cache hits: %d) using '%s'",
                len(missing_requests),
                total_eval - len(missing_requests),
                self.forecaster.model_id,
            )
            predict_fn: Any = self.forecaster.predict_batch
            sig = inspect.signature(predict_fn)
            if progress_callback is not None and "progress_callback" in sig.parameters:
                new_results = predict_fn(
                    missing_requests,
                    progress_callback=lambda done, tot: progress_callback("forecast", done, tot),
                )
            else:
                if progress_callback is not None:
                    progress_callback("forecast", 0, len(missing_requests))
                new_results = self.forecaster.predict_batch(missing_requests)
                if progress_callback is not None:
                    progress_callback("forecast", len(missing_requests), len(missing_requests))
            for idx, res in zip(missing_indices, new_results, strict=True):
                cached[ensure_utc(candles.index[idx])] = res
            if cache_key is not None and self.forecast_cache is not None and self.config.forecaster.type != "noop":
                self.forecast_cache.save(cache_key, cached)
        else:
            logger.info(
                "All %d forecasts loaded from disk cache (0 inference calls).",
                total_eval,
            )

        return {ensure_utc(candles.index[i]): cached[ensure_utc(candles.index[i])] for i in eval_indices}

    def run(
        self,
        candles: pd.DataFrame,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        initial_cash: float | None = None,
        precomputed_forecasts: dict[datetime, ForecastResult] | None = None,
        use_cache: bool = True,
        min_warmup_bars: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BacktestResult:
        """Execute bar-by-bar backtest over [start, end] using TradingStepRunner + SimulatedBroker."""
        if candles.empty:
            raise ValueError("Cannot run backtest on empty candles DataFrame")

        warmup = min_warmup_bars if min_warmup_bars is not None else min(
            self.config.forecaster.params.context_len,
            max(16, len(candles) // 4),
        )
        start_ts = pd.Timestamp(ensure_utc(start)) if start is not None else candles.index[min(warmup, len(candles) - 1)]
        end_ts = pd.Timestamp(ensure_utc(end)) if end is not None else candles.index[-1]

        eval_indices = [
            i
            for i, ts in enumerate(candles.index)
            if i >= warmup and start_ts <= ts <= end_ts
        ]
        if not eval_indices:
            raise ValueError(
                f"No evaluable bars in [{start_ts}, {end_ts}] after {warmup} warmup bars "
                f"(available: {candles.index.min()} .. {candles.index.max()}, rows={len(candles)})"
            )

        if precomputed_forecasts is not None:
            forecasts = precomputed_forecasts
        elif self.config.forecaster.type == "noop":
            # Fast-path for pure rule-based strategies: avoid building 50k heavy FeaturePipeline slices
            price_col_fc = self.config.data.price_column if self.config.data.price_column in candles.columns else "close"
            prices_arr = candles[price_col_fc].to_numpy(dtype=float)
            horizon_fc = self.config.forecaster.params.horizon
            forecasts = {}
            total_eval = len(eval_indices)
            if progress_callback is not None:
                progress_callback("features", 0, total_eval)
            for pos, i in enumerate(eval_indices, start=1):
                ts_i = ensure_utc(candles.index[i])
                spot_i = float(prices_arr[i])
                flat = np.full(horizon_fc, spot_i, dtype=np.float64)
                forecasts[ts_i] = ForecastResult(
                    instrument_id=self.config.instrument.id,
                    model_id=self.forecaster.model_id,
                    timestamp=ts_i,
                    horizon=horizon_fc,
                    quantiles={0.05: flat, 0.10: flat, 0.50: flat, 0.90: flat, 0.95: flat},
                    metadata={"noop": True},
                )
                if progress_callback is not None and (pos % 500 == 0 or pos == total_eval):
                    progress_callback("features", pos, total_eval)
        else:
            forecasts = self.precompute_forecasts(
                candles,
                eval_indices,
                use_cache=use_cache,
                progress_callback=progress_callback,
            )

        cash0 = float(initial_cash if initial_cash is not None else self.config.broker.initial_cash)
        broker = SimulatedBroker(
            initial_cash=cash0,
            quote_currency=self.config.instrument.quote_currency,
            fee_model=self.config.risk.fee_model,
            slippage_bps=self.config.risk.slippage_bps,
            spread_bps=self.config.risk.spread_bps,
            execution_price=self.config.broker.execution_price,
            max_drawdown_pct=self.config.risk.max_drawdown_pct,
        )
        strategy = build_strategy(self.config)
        risk_manager = build_risk_manager(self.config, self.calendar)
        runner = TradingStepRunner(
            config=self.config,
            feature_pipeline=self.feature_pipeline,
            forecaster=self.forecaster,
            strategy=strategy,
            risk_manager=risk_manager,
            broker=broker,
            store=None,
            notifier=None,
        )

        price_col = self.config.data.price_column if self.config.data.price_column in candles.columns else "close"
        bh_start_price = float(candles[price_col].iloc[eval_indices[0]])

        equity_records: list[dict[str, Any]] = []
        outcomes: list[StepOutcome] = []
        total_bars = len(eval_indices)
        if progress_callback is not None:
            progress_callback("simulate", 0, total_bars)

        for step_num, idx in enumerate(eval_indices, start=1):
            row = candles.iloc[idx]
            ts = ensure_utc(candles.index[idx])
            bar = Candle(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row[price_col]),
                volume=float(row["volume"]),
                adj_close=float(row["adj_close"]) if "adj_close" in row else float(row["close"]),
            )
            # Execute any pending next_open orders on the open of bar `t`
            broker.on_new_bar(bar, self.config.instrument)

            # Run the EXACT shared TradingStepRunner on causal slice candles[:idx+1]
            outcome = runner.run_step(
                candles=candles.iloc[: idx + 1],
                precomputed_forecast=forecasts[ts],
                now=ts,
                check_staleness=False,
                enforce_idempotency=False,
                persist=False,
                notify=False,
            )
            outcomes.append(outcome)

            pf = outcome.portfolio_after or broker.get_portfolio(self.config.instrument)
            bh_equity = cash0 * (bar.close / bh_start_price) if bh_start_price > 0 else cash0
            equity_records.append(
                {
                    "timestamp": ts,
                    "price": bar.close,
                    "cash": pf.cash,
                    "position_qty": pf.position_qty(self.config.instrument.id),
                    "equity": pf.equity,
                    "high_water_mark": pf.high_water_mark,
                    "drawdown": pf.drawdown,
                    "buy_and_hold_equity": bh_equity,
                    "halted": pf.halted,
                }
            )
            if progress_callback is not None and (step_num % 25 == 0 or step_num == total_bars):
                progress_callback("simulate", step_num, total_bars)

        equity_df = pd.DataFrame(equity_records).set_index("timestamp")
        equity_df.index = pd.to_datetime(equity_df.index, utc=True)

        metrics = compute_performance_metrics(
            equity_df=equity_df,
            fills=broker.fills,
            calendar=self.calendar,
            timeframe=self.config.data.timeframe,
        )
        calibration = compute_forecast_calibration(
            forecasts=forecasts,
            candles=candles,
            price_column=price_col,
        )

        return BacktestResult(
            config_name=self.config.name,
            instrument_id=self.config.instrument.id,
            timeframe=self.config.data.timeframe,
            equity_df=equity_df,
            fills=list(broker.fills),
            outcomes=outcomes,
            forecasts=forecasts,
            metrics=metrics,
            calibration=calibration,
        )
