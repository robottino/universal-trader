"""Walk-Forward out-of-sample optimization engine with single-load model & disk forecast caching."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from universal_trader.core.calendar import MarketCalendar
from universal_trader.core.config import AppConfig
from universal_trader.core.logging import get_logger
from universal_trader.core.models import FillResult, ensure_utc
from universal_trader.core.protocols import Forecaster
from universal_trader.data.features import FeaturePipeline
from universal_trader.engine.backtest import BacktestRunner, ProgressCallback
from universal_trader.engine.forecast_cache import ForecastCache
from universal_trader.engine.metrics import (
    compute_forecast_calibration,
    compute_performance_metrics,
)

logger = get_logger(__name__)


@dataclass
class WalkForwardFold:
    """In-sample selection and out-of-sample evaluation for one walk-forward split."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any]
    train_score: float
    train_trades: int
    test_roi: float
    test_bh_roi: float
    test_sharpe: float
    test_max_drawdown: float
    test_trades: int


@dataclass
class WalkForwardResult:
    """Aggregated out-of-sample walk-forward results across all folds."""

    config_name: str
    instrument_id: str
    timeframe: str
    folds: list[WalkForwardFold]
    oos_equity_df: pd.DataFrame
    oos_fills: list[FillResult]
    oos_metrics: dict[str, Any]
    calibration: dict[str, Any]

    def to_folds_dataframe(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            row: dict[str, Any] = {
                "fold": f.fold_index,
                "train_start": f.train_start.isoformat(),
                "train_end": f.train_end.isoformat(),
                "test_start": f.test_start.isoformat(),
                "test_end": f.test_end.isoformat(),
                "train_score": f.train_score,
                "train_trades": f.train_trades,
                "test_roi": f.test_roi,
                "test_bh_roi": f.test_bh_roi,
                "test_sharpe": f.test_sharpe,
                "test_max_drawdown": f.test_max_drawdown,
                "test_trades": f.test_trades,
            }
            for pk, pv in f.best_params.items():
                row[f"param_{pk}"] = pv
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_folds_dataframe().to_csv(path, index=False)
        return path


def _apply_param_overrides(base_config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    """Return a validated copy of AppConfig with dotted param overrides (e.g. 'strategy.theta_buy')."""
    data = base_config.model_dump(mode="python")
    for dotted_key, val in overrides.items():
        parts = dotted_key.split(".")
        cursor = data
        for seg in parts[:-1]:
            if seg not in cursor or not isinstance(cursor[seg], dict):
                cursor[seg] = {}
            cursor = cursor[seg]
        cursor[parts[-1]] = val
    return AppConfig.model_validate(data)


def _expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = sorted(param_grid.keys())
    values_lists = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values_lists)]


class WalkForwardEngine:
    """Rolling/expanding walk-forward optimization engine with strict OOS evaluation."""

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

    def run(
        self,
        candles: pd.DataFrame,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        use_cache: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> WalkForwardResult:
        wf = self.config.walkforward
        warmup = min(self.config.forecaster.params.context_len, max(16, len(candles) // 5))

        start_ts = pd.Timestamp(ensure_utc(start)) if start is not None else candles.index[min(warmup, len(candles) - 1)]
        end_ts = pd.Timestamp(ensure_utc(end)) if end is not None else candles.index[-1]

        eval_indices = [
            i
            for i, ts in enumerate(candles.index)
            if i >= warmup and start_ts <= ts <= end_ts
        ]
        if len(eval_indices) < wf.train_bars + wf.test_bars:
            raise ValueError(
                f"Not enough bars ({len(eval_indices)}) for walk-forward "
                f"(need at least train_bars={wf.train_bars} + test_bars={wf.test_bars})"
            )

        # Precompute ALL forecasts across the full evaluation span ONCE (cached on disk!)
        base_runner = BacktestRunner(
            config=self.config,
            forecaster=self.forecaster,
            feature_pipeline=self.feature_pipeline,
            calendar=self.calendar,
            forecast_cache=self.forecast_cache,
        )
        all_forecasts = base_runner.precompute_forecasts(
            candles,
            eval_indices,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )

        # Build rolling or expanding folds
        step = wf.step_bars or wf.test_bars
        splits: list[tuple[list[int], list[int]]] = []
        cursor = 0
        while cursor + wf.train_bars + wf.test_bars <= len(eval_indices):
            train_start_pos = 0 if wf.mode == "expanding" else cursor
            train_end_pos = cursor + wf.train_bars
            test_end_pos = min(len(eval_indices), train_end_pos + wf.test_bars)
            train_idx = eval_indices[train_start_pos:train_end_pos]
            test_idx = eval_indices[train_end_pos:test_end_pos]
            if len(test_idx) >= max(2, wf.test_bars // 2):
                splits.append((train_idx, test_idx))
            cursor += step

        if not splits:
            raise ValueError("Could not generate any walk-forward folds from provided window.")

        candidates = _expand_param_grid(wf.param_grid)
        logger.info(
            "Running %s walk-forward: %d folds × %d parameter combinations",
            wf.mode,
            len(splits),
            len(candidates),
        )

        folds: list[WalkForwardFold] = []
        oos_equity_segments: list[pd.DataFrame] = []
        oos_fills: list[FillResult] = []
        running_capital = float(wf.initial_cash)
        bh_anchor_price: float | None = None
        price_col = self.config.data.price_column if self.config.data.price_column in candles.columns else "close"

        if progress_callback is not None:
            progress_callback("walkforward", 0, len(splits))

        for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
            train_start = ensure_utc(candles.index[train_idx[0]])
            train_end = ensure_utc(candles.index[train_idx[-1]])
            test_start = ensure_utc(candles.index[test_idx[0]])
            test_end = ensure_utc(candles.index[test_idx[-1]])

            best_params: dict[str, Any] = candidates[0]
            best_score = float("-inf")
            best_trades = 0
            fallback_params = candidates[0]
            fallback_score = float("-inf")
            fallback_trades = 0

            for params in candidates:
                cand_cfg = _apply_param_overrides(self.config, params)
                bt_runner = BacktestRunner(
                    config=cand_cfg,
                    forecaster=self.forecaster,
                    feature_pipeline=self.feature_pipeline,
                    calendar=self.calendar,
                    forecast_cache=None,
                )
                res = bt_runner.run(
                    candles,
                    start=train_start,
                    end=train_end,
                    initial_cash=wf.initial_cash,
                    precomputed_forecasts=all_forecasts,
                    use_cache=False,
                    min_warmup_bars=warmup,
                )
                score = float(res.metrics.get(wf.metric, 0.0))
                trades = int(res.metrics.get("num_trades", 0))

                if score > fallback_score:
                    fallback_score = score
                    fallback_params = params
                    fallback_trades = trades

                if trades >= wf.min_trades and score > best_score:
                    best_score = score
                    best_params = params
                    best_trades = trades

            if best_score == float("-inf"):
                best_params = fallback_params
                best_score = fallback_score
                best_trades = fallback_trades

            # Strict Out-Of-Sample evaluation on unseen test fold using running_capital
            winner_cfg = _apply_param_overrides(self.config, best_params)
            oos_runner = BacktestRunner(
                config=winner_cfg,
                forecaster=self.forecaster,
                feature_pipeline=self.feature_pipeline,
                calendar=self.calendar,
                forecast_cache=None,
            )
            oos_res = oos_runner.run(
                candles,
                start=test_start,
                end=test_end,
                initial_cash=running_capital,
                precomputed_forecasts=all_forecasts,
                use_cache=False,
                min_warmup_bars=warmup,
            )
            seg_df = oos_res.equity_df.copy()
            if bh_anchor_price is None:
                bh_anchor_price = float(seg_df["price"].iloc[0])
            seg_df["buy_and_hold_equity"] = wf.initial_cash * (seg_df["price"] / bh_anchor_price)
            oos_equity_segments.append(seg_df)
            oos_fills.extend(oos_res.fills)
            running_capital = float(seg_df["equity"].iloc[-1])
            if progress_callback is not None:
                progress_callback("walkforward", fold_num, len(splits))

            folds.append(
                WalkForwardFold(
                    fold_index=fold_num,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    best_params=best_params,
                    train_score=best_score,
                    train_trades=best_trades,
                    test_roi=float(oos_res.metrics["total_roi"]),
                    test_bh_roi=float(oos_res.metrics["buy_and_hold_roi"]),
                    test_sharpe=float(oos_res.metrics["sharpe_ratio"]),
                    test_max_drawdown=float(oos_res.metrics["max_drawdown"]),
                    test_trades=int(oos_res.metrics["num_trades"]),
                )
            )

        oos_equity_df = pd.concat(oos_equity_segments)
        oos_equity_df = oos_equity_df[~oos_equity_df.index.duplicated(keep="last")].sort_index()
        # Recompute global OOS HWM & drawdown across concatenated segments
        oos_equity_df["high_water_mark"] = oos_equity_df["equity"].cummax()
        oos_equity_df["drawdown"] = (
            1.0 - (oos_equity_df["equity"] / oos_equity_df["high_water_mark"].replace(0.0, 1.0))
        ).clip(lower=0.0)

        oos_metrics = compute_performance_metrics(
            equity_df=oos_equity_df,
            fills=oos_fills,
            calendar=self.calendar,
            timeframe=self.config.data.timeframe,
        )
        oos_forecasts = {
            ensure_utc(ts): all_forecasts[ensure_utc(ts)]
            for ts in oos_equity_df.index
            if ensure_utc(ts) in all_forecasts
        }
        calibration = compute_forecast_calibration(
            forecasts=oos_forecasts,
            candles=candles,
            price_column=price_col,
        )

        return WalkForwardResult(
            config_name=self.config.name,
            instrument_id=self.config.instrument.id,
            timeframe=self.config.data.timeframe,
            folds=folds,
            oos_equity_df=oos_equity_df,
            oos_fills=oos_fills,
            oos_metrics=oos_metrics,
            calibration=calibration,
        )
