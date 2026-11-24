"""Typer CLI (`utrader`): doctor, predict, backtest, walkforward, trade, report."""

from __future__ import annotations

import os
import platform
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from universal_trader import __version__, load_builtin_plugins
from universal_trader.core.calendar import MarketCalendar, parse_timeframe
from universal_trader.core.config import RuntimePaths, load_config
from universal_trader.core.locking import ConcurrencyLockError, ExecutionLock
from universal_trader.core.logging import configure_logging, get_logger
from universal_trader.core.models import ensure_utc
from universal_trader.core.registry import registry
from universal_trader.engine.backtest import BacktestRunner
from universal_trader.engine.factory import (
    build_broker,
    build_data_provider,
    build_feature_pipeline,
    build_forecaster,
    build_notifier,
    build_risk_manager,
    build_strategy,
    resolve_db_path,
)
from universal_trader.engine.forecast_cache import ForecastCache
from universal_trader.engine.portfolio import (
    PortfolioAllocator,
    PortfolioBacktestRunner,
    PortfolioStepRunner,
)
from universal_trader.engine.runner import TradingStepRunner
from universal_trader.engine.walkforward import WalkForwardEngine
from universal_trader.forecasters.device import DeviceManager
from universal_trader.reporting.plots import (
    plot_backtest_report,
    plot_fan_chart,
    plot_portfolio_backtest_report,
)
from universal_trader.storage.sqlite_store import SQLiteStateStore

app = typer.Typer(
    name="utrader",
    help="Universal Trader — probabilistic forecasting, backtesting, walk-forward, and automated trading.",
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)


_STAGE_LABELS: dict[str, str] = {
    "features": "[cyan]Preparazione feature causali[/cyan]",
    "forecast": "[yellow]Inferenza previsioni[/yellow]",
    "simulate": "[green]Simulazione barra-per-barra[/green]",
    "walkforward": "[magenta]Ottimizzazione fold Walk-Forward[/magenta]",
}


def _create_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=32),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def _parse_date(date_str: str, *, end_of_day: bool = False) -> datetime:
    dt = ensure_utc(date_str)
    if end_of_day and len(date_str.strip()) <= 10:
        dt = dt + timedelta(hours=23, minutes=59, seconds=59)
    return dt


@app.command()
def doctor(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="Optional YAML config to validate")] = None,
    check_model: Annotated[bool, typer.Option("--check-model", help="Also test Timestral package import")] = False,
    log_level: Annotated[str, typer.Option("--log-level", help="Logging verbosity")] = "WARNING",
) -> None:
    """Inspect environment, hardware (CPU/CUDA/MPS), plugins, storage writability, and configs."""
    configure_logging(level=log_level)
    load_builtin_plugins()
    paths = RuntimePaths.resolve()
    dm = DeviceManager(requested_device="auto")
    dinfo = dm.resolve()

    table = Table(title=f"Universal Trader Doctor v{__version__}", show_lines=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    table.add_row("Python", "[green]OK[/green]", f"{sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    table.add_row("PyTorch", "[green]OK[/green]", dinfo.torch_version)
    table.add_row(
        "Compute Device",
        "[green]OK[/green]",
        (
            f"selected={dinfo.device} ({dinfo.accelerator_name}) | dtype={dinfo.dtype_name} | "
            f"cuda={dinfo.cuda_available}, mps={dinfo.mps_available}, cpus={dinfo.cpu_count}"
        ),
    )

    # Timestral import check
    try:
        import timestral

        ts_loc = getattr(timestral, "__file__", "installed")
        table.add_row("Timestral Package", "[green]OK[/green]", f"importable ({ts_loc})")
    except Exception as exc:
        table.add_row("Timestral Package", "[red]FAIL[/red]", str(exc))
        console.print(table)
        raise typer.Exit(code=1) from exc

    # Directory & SQLite write check
    dir_errors: list[str] = []
    for label, dpath in [
        ("data_cache", paths.data_dir),
        ("forecast_cache", paths.forecast_cache_dir),
        ("state", paths.state_dir),
        ("reports", paths.reports_dir),
        ("hf_cache", dinfo.hf_cache_dir),
    ]:
        try:
            dpath.mkdir(parents=True, exist_ok=True)
            probe = dpath / ".utrader_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            dir_errors.append(f"{label} ({dpath}): {exc}")

    if dir_errors:
        table.add_row("Storage & Cache", "[red]FAIL[/red]", "; ".join(dir_errors))
        console.print(table)
        raise typer.Exit(code=1)
    table.add_row(
        "Storage & Cache",
        "[green]OK[/green]",
        f"root={paths.root} | hf_cache={dinfo.hf_cache_dir}",
    )

    # SQLite WAL probe
    try:
        probe_db = paths.state_dir / ".doctor_probe.db"
        SQLiteStateStore(probe_db)
        for suf in ("", "-wal", "-shm"):
            Path(f"{probe_db}{suf}").unlink(missing_ok=True)
        table.add_row("SQLite WAL", "[green]OK[/green]", "Read/Write + WAL verified")
    except Exception as exc:
        table.add_row("SQLite WAL", "[red]FAIL[/red]", str(exc))
        console.print(table)
        raise typer.Exit(code=1) from exc

    # Registered plugins
    plugins = registry.list_plugins()
    plugin_summary = " | ".join(f"{k}: {', '.join(v)}" for k, v in plugins.items())
    table.add_row("Plugins", "[green]OK[/green]", plugin_summary)

    # Optional config validation (or validate all YAML files in configs/ if present)
    cfg_files = [config] if config is not None else sorted((paths.root / "configs").glob("*.yaml"))
    if cfg_files:
        bad_cfgs: list[str] = []
        ok_cfgs: list[str] = []
        for cf in cfg_files:
            try:
                parsed = load_config(cf)
                if parsed.is_portfolio:
                    basket_txt = "+".join(i.base for i in parsed.instruments)
                    ok_cfgs.append(f"{cf.name} (Portfolio: {basket_txt})")
                else:
                    ok_cfgs.append(f"{cf.name} ({parsed.instrument.id})")
            except Exception as exc:
                bad_cfgs.append(f"{cf.name}: {exc}")
        if bad_cfgs:
            table.add_row("Configs", "[red]FAIL[/red]", "; ".join(bad_cfgs))
            console.print(table)
            raise typer.Exit(code=1)
        table.add_row("Configs", "[green]OK[/green]", ", ".join(ok_cfgs))

    if check_model:
        try:
            from universal_trader.forecasters.timestral_adapter import TimestralForecaster

            fc = TimestralForecaster(model_id="mistralai/Timestral-5M", device=dinfo.device)
            fc._ensure_loaded()
            table.add_row("Timestral-5M Load", "[green]OK[/green]", f"loaded on {fc.effective_device}")
        except Exception as exc:
            table.add_row("Timestral-5M Load", "[red]FAIL[/red]", str(exc))
            console.print(table)
            raise typer.Exit(code=1) from exc

    console.print(table)


@app.command()
def predict(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output PNG path for fan chart")] = None,
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip generating fan chart PNG")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Fetch latest data, run probabilistic forecast, print quantile summary, and generate fan chart."""
    configure_logging(level=log_level)
    cfg = load_config(config)
    paths = RuntimePaths.resolve()
    calendar = MarketCalendar(cfg.instrument.calendar)
    provider = build_data_provider(cfg, paths)
    forecaster = build_forecaster(cfg)
    pipeline = build_feature_pipeline(cfg, calendar)

    candles = provider.fetch_ohlcv(
        instrument=cfg.instrument,
        timeframe=cfg.data.timeframe,
        limit=cfg.data.lookback_bars,
    )
    pipeline.preload_macro_series(candles, cfg.data.timeframe)

    caps = forecaster.capabilities
    req = pipeline.build_request(
        candles=candles,
        instrument=cfg.instrument,
        timeframe=cfg.data.timeframe,
        context_len=cfg.forecaster.params.context_len,
        horizon=cfg.forecaster.params.horizon,
        supports_nan=caps.supports_nan,
        supports_past_covariates=caps.supports_past_covariates,
        supports_known_future_covariates=caps.supports_known_future_covariates,
    )
    forecast = forecaster.predict(req)

    price_col = cfg.data.price_column if cfg.data.price_column in candles.columns else "close"
    spot = float(candles[price_col].iloc[-1])

    table = Table(
        title=f"Forecast: {cfg.instrument.id} ({cfg.data.timeframe}) | Model: {forecast.model_id}",
        show_lines=False,
    )
    table.add_column("Quantile", justify="right", style="cyan")
    table.add_column(f"Terminal Price ({cfg.instrument.quote_currency})", justify="right")
    table.add_column("Delta vs Spot", justify="right")

    for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        val = float(forecast.quantile(q)[-1])
        ret = (val - spot) / spot
        style = "green" if ret >= 0 else "red"
        table.add_row(f"q{int(round(q * 100)):02d}", f"{val:,.4f}", f"[{style}]{ret:+.2%}[/{style}]")

    console.print(f"[bold]Last Candle UTC:[/bold] {forecast.timestamp.isoformat()} | [bold]Spot:[/bold] {spot:,.4f} {cfg.instrument.quote_currency}")
    console.print(table)

    if not no_plot:
        out_path = output or (paths.reports_dir / f"forecast_{cfg.name}_{forecast.timestamp.strftime('%Y%m%d_%H%M')}.png")
        saved = plot_fan_chart(
            candles=candles,
            forecast=forecast,
            instrument=cfg.instrument,
            output_path=out_path,
            price_column=price_col,
        )
        console.print(f"[green]Fan chart saved:[/green] {saved}")


@app.command()
def backtest(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config")],
    start: Annotated[str, typer.Option("--start", help="Start date YYYY-MM-DD (UTC)")],
    end: Annotated[str, typer.Option("--end", help="End date YYYY-MM-DD (UTC)")],
    initial_cash: Annotated[float | None, typer.Option("--initial-cash", help="Override starting cash")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o", help="Directory for charts and CSVs")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable disk forecast cache")] = False,
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip chart generation")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Run unified bar-by-bar historical backtest with calendar-aware metrics & calibration."""
    configure_logging(level=log_level)
    cfg = load_config(config)
    paths = RuntimePaths.resolve()
    calendar = MarketCalendar(cfg.instrument.calendar)
    provider = build_data_provider(cfg, paths)
    forecaster = build_forecaster(cfg)
    pipeline = build_feature_pipeline(cfg, calendar)

    start_dt = _parse_date(start, end_of_day=False)
    end_dt = _parse_date(end, end_of_day=True)
    delta = parse_timeframe(cfg.data.timeframe)
    pad_factor = 1.25 if calendar.is_24_7 else 2.4
    ema_slow_need = max(
        [
            cfg.strategy.trend_slow_window,
            cfg.strategy.macro_ema_window or 0,
            cfg.portfolio.warmup_bars or 0,
            *[sw for _, sw in cfg.strategy.ema_pairs],
        ]
    )
    warmup_bars_need = max(
        cfg.data.lookback_bars,
        cfg.forecaster.params.context_len + 64,
        ema_slow_need + 64,
    )
    warmup_start = start_dt - timedelta(
        seconds=delta.total_seconds() * warmup_bars_need * pad_factor
    )

    if cfg.is_portfolio:
        candles_by_id = {}
        with console.status(
            f"[cyan]Scaricamento candele multi-asset ({', '.join(i.id for i in cfg.instruments)})...[/cyan]"
        ):
            for inst in cfg.instruments:
                candles_by_id[inst.id] = provider.fetch_ohlcv(
                    instrument=inst,
                    timeframe=cfg.data.timeframe,
                    start=warmup_start,
                    end=end_dt,
                )

        pf_runner = PortfolioBacktestRunner(config=cfg, calendar=calendar)
        with _create_progress() as progress:
            pf_stage_tasks: dict[str, TaskID] = {}

            def _on_pf_progress(stage: str, completed: int, total: int) -> None:
                if stage not in pf_stage_tasks:
                    label = _STAGE_LABELS.get(stage, stage)
                    pf_stage_tasks[stage] = progress.add_task(label, total=max(total, 1))
                progress.update(pf_stage_tasks[stage], completed=completed, total=max(total, 1))

            pf_result = pf_runner.run(
                candles_by_id,
                start=start_dt,
                end=end_dt,
                initial_cash=initial_cash,
                progress_callback=_on_pf_progress,
            )

        m = pf_result.metrics
        qccy = cfg.instruments[0].quote_currency
        basket_str = " + ".join(i.base for i in cfg.instruments)
        table = Table(title=f"Portfolio Backtest — {cfg.name} ({basket_str} {cfg.data.timeframe})")
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", justify="right")

        table.add_row("Initial Equity", f"{m['initial_equity']:,.2f} {qccy}")
        table.add_row("Final Equity", f"{m['final_equity']:,.2f} {qccy}")
        table.add_row("Total ROI", f"{m['total_roi']:+.2%}")
        table.add_row(f"Benchmark B&H ({pf_result.benchmark_id})", f"{m['buy_and_hold_roi']:+.2%}")
        table.add_row("Outperformance vs Benchmark", f"{m['outperformance']:+.2%}")
        table.add_row("CAGR", f"{m['cagr']:+.2%}")
        table.add_row("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        table.add_row("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
        table.add_row("Calmar Ratio", f"{m['calmar_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{m['max_drawdown']:.2%}")
        table.add_row("Win Rate", f"{m['win_rate']:.1%} ({m['num_round_trips']} closed)")
        table.add_row("Profit Factor", f"{m['profit_factor']:.2f}")
        table.add_row("Trades / Turnover", f"{m['num_trades']} / {m['turnover']:.2f}x")
        table.add_row(
            "Total Costs (Fees + Slip)",
            f"{m['total_costs']:,.2f} ({m['total_fees']:,.2f} + {m['total_slippage']:,.2f})",
        )
        for inst_id, bh_roi in pf_result.asset_bh_rois.items():
            table.add_row(f"Buy & Hold ({inst_id})", f"{bh_roi:+.2%}")
        if pf_result.yearly_returns:
            yr_summary = " | ".join(f"{yr}: {ret:+.1%}" for yr, ret in sorted(pf_result.yearly_returns.items()))
            table.add_row("Yearly Returns", yr_summary)
        console.print(table)

        out_dir = output_dir or paths.reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        eq_csv = out_dir / f"backtest_{cfg.name}_equity.csv"
        pf_result.equity_df.to_csv(eq_csv)
        console.print(f"[green]Portfolio Equity CSV saved:[/green] {eq_csv}")

        if not no_plot:
            chart_path = out_dir / f"backtest_{cfg.name}.png"
            plot_portfolio_backtest_report(
                equity_df=pf_result.equity_df,
                metrics=pf_result.metrics,
                instruments=pf_result.instruments,
                benchmark_id=pf_result.benchmark_id,
                config_name=cfg.name,
                output_path=chart_path,
            )
            console.print(f"[green]Portfolio 3-panel chart saved:[/green] {chart_path}")
        return

    with console.status("[cyan]Scaricamento candele e serie macro...[/cyan]"):
        candles = provider.fetch_ohlcv(
            instrument=cfg.instrument,
            timeframe=cfg.data.timeframe,
            start=warmup_start,
            end=end_dt,
        )
        pipeline.preload_macro_series(candles, cfg.data.timeframe)

    fc_cache = None if no_cache else ForecastCache(paths.forecast_cache_dir)
    runner = BacktestRunner(
        config=cfg,
        forecaster=forecaster,
        feature_pipeline=pipeline,
        calendar=calendar,
        forecast_cache=fc_cache,
    )
    with _create_progress() as progress:
        stage_tasks: dict[str, TaskID] = {}

        def _on_progress(stage: str, completed: int, total: int) -> None:
            if stage not in stage_tasks:
                label = _STAGE_LABELS.get(stage, stage)
                stage_tasks[stage] = progress.add_task(label, total=max(total, 1))
            progress.update(stage_tasks[stage], completed=completed, total=max(total, 1))

        result = runner.run(
            candles,
            start=start_dt,
            end=end_dt,
            initial_cash=initial_cash,
            use_cache=not no_cache,
            progress_callback=_on_progress,
        )

    m = result.metrics
    c = result.calibration
    table = Table(title=f"Backtest Report — {cfg.name} ({cfg.instrument.id} {cfg.data.timeframe})")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    table.add_row("Initial Equity", f"{m['initial_equity']:,.2f} {cfg.instrument.quote_currency}")
    table.add_row("Final Equity", f"{m['final_equity']:,.2f} {cfg.instrument.quote_currency}")
    table.add_row("Total ROI", f"{m['total_roi']:+.2%}")
    table.add_row("Buy & Hold ROI", f"{m['buy_and_hold_roi']:+.2%}")
    table.add_row("Outperformance", f"{m['outperformance']:+.2%}")
    table.add_row("CAGR", f"{m['cagr']:+.2%}")
    table.add_row("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
    table.add_row("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
    table.add_row("Calmar Ratio", f"{m['calmar_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{m['max_drawdown']:.2%}")
    table.add_row("Win Rate", f"{m['win_rate']:.1%} ({m['num_round_trips']} closed)")
    table.add_row("Profit Factor", f"{m['profit_factor']:.2f}")
    table.add_row("Trades / Turnover", f"{m['num_trades']} / {m['turnover']:.2f}x")
    table.add_row("Total Costs (Fees + Slip)", f"{m['total_costs']:,.2f} ({m['total_fees']:,.2f} + {m['total_slippage']:,.2f})")
    if c.get("evaluated_points", 0) > 0:
        table.add_row("Forecast MAE / RMSE", f"{c['mae']:.4f} / {c['rmse']:.4f}")
        table.add_row("Mean Pinball Loss", f"{c['mean_pinball_loss']:.4f}")
        if c.get("interval_80_coverage") is not None:
            table.add_row("80% / 90% Coverage", f"{c['interval_80_coverage']:.1%} / {c['interval_90_coverage']:.1%}")
    console.print(table)

    out_dir = output_dir or paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_csv = out_dir / f"backtest_{cfg.name}_equity.csv"
    result.equity_df.to_csv(eq_csv)
    console.print(f"[green]Equity CSV saved:[/green] {eq_csv}")

    if not no_plot:
        chart_path = out_dir / f"backtest_{cfg.name}.png"
        plot_backtest_report(
            equity_df=result.equity_df,
            fills=result.fills,
            metrics=result.metrics,
            instrument=cfg.instrument,
            output_path=chart_path,
        )
        console.print(f"[green]Backtest chart saved:[/green] {chart_path}")


@app.command()
def walkforward(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config")],
    start: Annotated[str, typer.Option("--start", help="Start date YYYY-MM-DD (UTC)")],
    end: Annotated[str, typer.Option("--end", help="End date YYYY-MM-DD (UTC)")],
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o", help="Directory for CSV & chart")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable disk forecast cache")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Run rolling/expanding Walk-Forward optimization and aggregate Out-Of-Sample validation."""
    configure_logging(level=log_level)
    cfg = load_config(config)
    paths = RuntimePaths.resolve()
    calendar = MarketCalendar(cfg.instrument.calendar)
    provider = build_data_provider(cfg, paths)
    forecaster = build_forecaster(cfg)
    pipeline = build_feature_pipeline(cfg, calendar)

    start_dt = _parse_date(start, end_of_day=False)
    end_dt = _parse_date(end, end_of_day=True)
    delta = parse_timeframe(cfg.data.timeframe)
    pad_factor = 1.25 if calendar.is_24_7 else 2.4
    ema_slow_need = max(
        [
            cfg.strategy.trend_slow_window,
            cfg.strategy.macro_ema_window or 0,
            cfg.portfolio.warmup_bars or 0,
            *[sw for _, sw in cfg.strategy.ema_pairs],
        ]
    )
    warmup_bars_need = max(
        cfg.data.lookback_bars,
        cfg.forecaster.params.context_len + 64,
        ema_slow_need + 64,
    )
    warmup_start = start_dt - timedelta(
        seconds=delta.total_seconds() * warmup_bars_need * pad_factor
    )

    if cfg.is_portfolio:
        console.print(
            "[yellow]Walk-forward grid optimization currently targets single-asset configs. "
            "Use 'utrader backtest' for multi-asset portfolio validation.[/yellow]"
        )
        raise typer.Exit(code=2)

    with console.status("[cyan]Scaricamento candele e serie macro...[/cyan]"):
        candles = provider.fetch_ohlcv(
            instrument=cfg.instrument,
            timeframe=cfg.data.timeframe,
            start=warmup_start,
            end=end_dt,
        )
        pipeline.preload_macro_series(candles, cfg.data.timeframe)

    fc_cache = None if no_cache else ForecastCache(paths.forecast_cache_dir)
    engine = WalkForwardEngine(
        config=cfg,
        forecaster=forecaster,
        feature_pipeline=pipeline,
        calendar=calendar,
        forecast_cache=fc_cache,
    )
    with _create_progress() as progress:
        stage_tasks: dict[str, TaskID] = {}

        def _on_wf_progress(stage: str, completed: int, total: int) -> None:
            if stage not in stage_tasks:
                label = _STAGE_LABELS.get(stage, stage)
                stage_tasks[stage] = progress.add_task(label, total=max(total, 1))
            progress.update(stage_tasks[stage], completed=completed, total=max(total, 1))

        wf_res = engine.run(
            candles,
            start=start_dt,
            end=end_dt,
            use_cache=not no_cache,
            progress_callback=_on_wf_progress,
        )

    folds_table = Table(title=f"Walk-Forward OOS Folds — {cfg.name} ({cfg.instrument.id})")
    folds_table.add_column("Fold", justify="right")
    folds_table.add_column("OOS Window (UTC)")
    folds_table.add_column("Selected Params")
    folds_table.add_column("OOS ROI", justify="right")
    folds_table.add_column("B&H ROI", justify="right")
    folds_table.add_column("OOS Sharpe", justify="right")
    folds_table.add_column("MaxDD", justify="right")
    folds_table.add_column("Trades", justify="right")

    for f in wf_res.folds:
        params_str = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in f.best_params.items()) or "default"
        folds_table.add_row(
            str(f.fold_index),
            f"{f.test_start.strftime('%Y-%m-%d')} → {f.test_end.strftime('%Y-%m-%d')}",
            params_str,
            f"{f.test_roi:+.2%}",
            f"{f.test_bh_roi:+.2%}",
            f"{f.test_sharpe:.2f}",
            f"{f.test_max_drawdown:.2%}",
            str(f.test_trades),
        )
    console.print(folds_table)

    m = wf_res.oos_metrics
    console.print(
        f"[bold green]Aggregate OOS:[/bold green] ROI={m['total_roi']:+.2%} "
        f"(B&H={m['buy_and_hold_roi']:+.2%}) | Sharpe={m['sharpe_ratio']:.2f} | "
        f"Calmar={m['calmar_ratio']:.2f} | MaxDD={m['max_drawdown']:.2%} | Trades={m['num_trades']}"
    )

    out_dir = output_dir or paths.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = wf_res.export_csv(out_dir / f"walkforward_{cfg.name}.csv")
    chart_path = plot_backtest_report(
        equity_df=wf_res.oos_equity_df,
        fills=wf_res.oos_fills,
        metrics=wf_res.oos_metrics,
        instrument=cfg.instrument,
        output_path=out_dir / f"walkforward_{cfg.name}.png",
    )
    console.print(f"[green]Walk-forward CSV saved:[/green] {csv_path}")
    console.print(f"[green]Walk-forward OOS chart saved:[/green] {chart_path}")


@app.command()
def trade(
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to YAML config")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Enforce DryRunBroker (zero broker writes)")] = False,
    force: Annotated[bool, typer.Option("--force", help="Bypass candle idempotency check")] = False,
    no_plot: Annotated[bool, typer.Option("--no-plot", help="Skip saving fan chart")] = False,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Execute one atomic trading cycle (suitable for manual runs or cron/container scheduling)."""
    configure_logging(level=log_level)
    cfg = load_config(config)
    paths = RuntimePaths.resolve()
    db_path = resolve_db_path(cfg, paths)
    lock_path = paths.state_dir / f"{cfg.name}.lock"
    notifier = build_notifier(cfg)

    try:
        with ExecutionLock(lock_path):
            store = SQLiteStateStore(db_path)
            calendar = MarketCalendar(cfg.instrument.calendar)
            provider = build_data_provider(cfg, paths)
            risk_manager = build_risk_manager(cfg, calendar)
            broker = build_broker(
                cfg,
                store=store,
                data_provider=provider,
                dry_run_override=dry_run,
            )

            if cfg.is_portfolio:
                candles_by_id = {
                    inst.id: provider.fetch_ohlcv(
                        instrument=inst,
                        timeframe=cfg.data.timeframe,
                        limit=cfg.data.lookback_bars,
                    )
                    for inst in cfg.instruments
                }
                allocator = PortfolioAllocator(cfg, calendar)
                pf_step_runner = PortfolioStepRunner(
                    config=cfg,
                    allocator=allocator,
                    risk_manager=risk_manager,
                    broker=broker,
                    store=store,
                    notifier=notifier,
                )
                pf_outcome = pf_step_runner.run_step(
                    candles_by_id=candles_by_id,
                    now=datetime.now(UTC),
                    check_staleness=True,
                    enforce_idempotency=True,
                    force=force,
                    persist=True,
                    notify=True,
                )
                if pf_outcome.skipped_idempotent:
                    console.print(
                        f"[yellow]Idempotent skip:[/yellow] candle {pf_outcome.candle_timestamp.isoformat()} "
                        f"already processed for portfolio '{cfg.name}'."
                    )
                    return

                mode_label = "DRY-RUN" if dry_run or cfg.mode.value == "dry_run" else cfg.mode.value.upper()
                basket_str = " + ".join(i.base for i in cfg.instruments)
                console.print(
                    f"[bold]{cfg.name}[/bold] [{mode_label}] | Portfolio ({basket_str}) @ "
                    f"{pf_outcome.candle_timestamp.isoformat()}"
                )
                alloc_table = Table(show_lines=False)
                alloc_table.add_column("Asset", style="bold cyan")
                alloc_table.add_column("Signal")
                alloc_table.add_column("Vol (ann.)", justify="right")
                alloc_table.add_column("Target", justify="right")
                alloc_table.add_column("Before -> After", justify="right")
                alloc_table.add_column("Action")
                for inst in cfg.instruments:
                    sig_txt = "[green]BULL[/green]" if pf_outcome.active_signals.get(inst.id) else "[dim]FLAT[/dim]"
                    vol_txt = f"{pf_outcome.realized_vols.get(inst.id, 0.0):.1%}"
                    tgt_txt = f"{pf_outcome.target_weights.get(inst.id, 0.0):.1%}"
                    w_b = pf_outcome.current_weights_before.get(inst.id, 0.0)
                    w_a = pf_outcome.current_weights_after.get(inst.id, 0.0)
                    sub = pf_outcome.outcomes.get(inst.id)
                    if sub and sub.fill is not None:
                        act_txt = (
                            f"[green]{sub.fill.side.value} {sub.fill.quantity:.6g} "
                            f"@ {sub.fill.fill_price:,.2f} ({sub.fill.status.value})[/green]"
                        )
                    elif sub and sub.rejected_order is not None:
                        act_txt = f"[yellow]REJECTED ({sub.rejected_order.guard_name}): {sub.rejected_order.reason}[/yellow]"
                    else:
                        act_txt = "HOLD"
                    alloc_table.add_row(inst.id, sig_txt, vol_txt, tgt_txt, f"{w_b:.1%} -> {w_a:.1%}", act_txt)
                console.print(alloc_table)
                if pf_outcome.portfolio_after is not None:
                    pf = pf_outcome.portfolio_after
                    console.print(
                        f"  Portfolio: Equity={pf.equity:,.2f} {pf.quote_currency} | "
                        f"Cash={pf.cash:,.2f} | DD={pf.drawdown:.2%} | Halted={pf.halted}"
                    )
                return

            forecaster = build_forecaster(cfg)
            strategy = build_strategy(cfg)
            pipeline = build_feature_pipeline(cfg, calendar)

            candles = provider.fetch_ohlcv(
                instrument=cfg.instrument,
                timeframe=cfg.data.timeframe,
                limit=cfg.data.lookback_bars,
            )
            pipeline.preload_macro_series(candles, cfg.data.timeframe)

            runner = TradingStepRunner(
                config=cfg,
                feature_pipeline=pipeline,
                forecaster=forecaster,
                strategy=strategy,
                risk_manager=risk_manager,
                broker=broker,
                store=store,
                notifier=notifier,
            )
            chart_path = (
                None
                if no_plot
                else (paths.reports_dir / f"step_{cfg.name}_{ensure_utc(candles.index[-1]).strftime('%Y%m%d_%H%M')}.png")
            )
            outcome = runner.run_step(
                candles=candles,
                now=datetime.now(UTC),
                check_staleness=True,
                enforce_idempotency=True,
                force=force,
                persist=True,
                notify=True,
                chart_output_path=chart_path,
            )
    except ConcurrencyLockError as exc:
        console.print(f"[yellow]Skipped:[/yellow] {exc}")
        raise typer.Exit(code=2) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        logger.exception("Unhandled error during trade cycle for '%s'", cfg.name)
        notifier.notify_alert(
            title=f"TRADE ERROR ({cfg.name})",
            message=f"{type(exc).__name__}: {exc}",
            instrument=None if cfg.is_portfolio else cfg.instrument,
        )
        console.print(f"[red]Trade error ({type(exc).__name__}):[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if outcome.skipped_idempotent:
        console.print(
            f"[yellow]Idempotent skip:[/yellow] candle {outcome.candle_timestamp.isoformat()} "
            f"already processed for '{cfg.name}'."
        )
        return

    mode_label = "DRY-RUN" if dry_run or cfg.mode.value == "dry_run" else cfg.mode.value.upper()
    console.print(
        f"[bold]{cfg.name}[/bold] [{mode_label}] | {cfg.instrument.id} @ "
        f"{outcome.candle_timestamp.isoformat()}"
    )
    if outcome.intent is not None:
        console.print(
            f"  Signal: {outcome.intent.side.value} | "
            f"E[r]={outcome.intent.signal.expected_return:+.2%} | "
            f"Reason: {outcome.intent.reason}"
        )
    if outcome.fill is not None:
        console.print(
            f"  [green]Fill:[/green] {outcome.fill.side.value} {outcome.fill.quantity:.6g} "
            f"@ {outcome.fill.fill_price:,.4f} {cfg.instrument.quote_currency} "
            f"(status={outcome.fill.status.value}, fee={outcome.fill.fee:.4f})"
        )
    elif outcome.rejected_order is not None:
        console.print(
            f"  [yellow]Risk Rejected ({outcome.rejected_order.guard_name}):[/yellow] "
            f"{outcome.rejected_order.reason}"
        )
    if outcome.portfolio_after is not None:
        pf = outcome.portfolio_after
        console.print(
            f"  Portfolio: Equity={pf.equity:,.2f} {pf.quote_currency} | "
            f"Cash={pf.cash:,.2f} | DD={pf.drawdown:.2%} | Halted={pf.halted}"
        )


@app.command()
def report(
    db: Annotated[Path, typer.Option("--db", help="Path to SQLite state database")] = Path("state/trader.db"),
    config_name: Annotated[str | None, typer.Option("--config-name", help="Filter by config name")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", "-o", help="Optional CSV export dir")] = None,
) -> None:
    """Generate PnL, drawdown, fills, and ex-post forecast calibration report from SQLite."""
    if not db.exists():
        console.print(f"[red]Database not found:[/red] {db}")
        raise typer.Exit(code=1)

    store = SQLiteStateStore(db)
    eq_df = store.get_equity_history(config_name=config_name)
    fills_df = store.get_fills(config_name=config_name)
    eval_metrics = store.evaluate_stored_forecasts(config_name=config_name)

    if eq_df.empty:
        console.print("[yellow]No equity snapshots recorded yet.[/yellow]")
        return

    table = Table(title=f"Operational Report — {db}")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")

    eq_start = float(eq_df["equity"].iloc[0])
    eq_end = float(eq_df["equity"].iloc[-1])
    pnl = eq_end - eq_start
    roi = (eq_end / eq_start - 1.0) if eq_start > 0 else 0.0
    max_dd = float(eq_df["drawdown"].max())
    currency = str(eq_df["quote_currency"].iloc[-1])

    table.add_row("Snapshots", str(len(eq_df)))
    table.add_row("Starting Equity", f"{eq_start:,.2f} {currency}")
    table.add_row("Current Equity", f"{eq_end:,.2f} {currency}")
    table.add_row("Realized + Unrealized PnL", f"{pnl:+,.2f} {currency} ({roi:+.2%})")
    table.add_row("Peak Drawdown", f"{max_dd:.2%}")
    table.add_row("Recorded Fills", str(len(fills_df)))
    if not fills_df.empty:
        table.add_row("Total Fees Paid", f"{float(fills_df['fee'].sum()):,.4f} {currency}")
    if eval_metrics.get("evaluated_points", 0) > 0:
        table.add_row("Ex-post Forecast Points", str(eval_metrics["evaluated_points"]))
        table.add_row("Forecast MAE / RMSE", f"{eval_metrics['mae']:.4f} / {eval_metrics['rmse']:.4f}")
        table.add_row("Mean Pinball Loss", f"{eval_metrics['mean_pinball_loss']:.4f}")

    console.print(table)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        eq_df.to_csv(output_dir / "report_equity.csv", index=False)
        if not fills_df.empty:
            fills_df.to_csv(output_dir / "report_fills.csv", index=False)
        console.print(f"[green]CSV reports exported to:[/green] {output_dir}")


@app.command("test-telegram")
def test_telegram(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Optional YAML config (defaults to TELEGRAM_* from .env)"),
    ] = None,
    log_level: Annotated[str, typer.Option("--log-level")] = "WARNING",
) -> None:
    """Send a test message via Telegram Bot API to verify token, chat_id, and HTML formatting."""
    configure_logging(level=log_level)
    from universal_trader.reporting.telegram import TelegramNotifier

    yaml_disabled = False
    if config is not None:
        cfg = load_config(config)
        bot_token = cfg.notifier.bot_token
        chat_id = cfg.notifier.chat_id
        notify_on = cfg.notifier.notify_on
        daily_hour = cfg.notifier.daily_summary_hour_utc
        label = cfg.name
        yaml_disabled = not cfg.notifier.enabled
    else:
        paths = RuntimePaths.resolve()
        dotenv_path = paths.root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        notify_on = "events"
        daily_hour = 8
        label = "universal_trader"

    if not bot_token or not chat_id:
        console.print(
            "[red]FAIL:[/red] TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti "
            "(verifica il file .env o la sezione notifier della YAML)."
        )
        raise typer.Exit(code=1)

    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        enabled=True,
        notify_on=notify_on,
        daily_summary_hour_utc=daily_hour,
    )
    ok, detail = notifier.send_test_message(label=label)
    if not ok:
        console.print(f"[red]Telegram test failed:[/red] {detail}")
        raise typer.Exit(code=1)

    console.print(f"[green]Telegram test sent successfully[/green] (label={label}, notify_on={notify_on}).")
    if yaml_disabled:
        console.print(
            "[yellow]Nota:[/yellow] nella YAML `notifier.enabled` è ancora `false` — "
            "impostalo a `true` per ricevere le notifiche da `utrader trade`."
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
