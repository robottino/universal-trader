"""Declarative YAML configuration schemas (Pydantic v2) with ${ENV_VAR} interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

from universal_trader.core.calendar import parse_timeframe
from universal_trader.core.logging import register_env_secrets, register_secret
from universal_trader.core.models import AssetClass, Instrument, TradingMode

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def find_project_root(start: Path | None = None) -> Path:
    """Resolve repository/project root dynamically without hardcoded paths."""
    if env_root := os.environ.get("UTRADER_ROOT"):
        return Path(env_root).expanduser().resolve()
    cur = (start or Path.cwd()).resolve()
    for candidate in (cur, *cur.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cur


class RuntimePaths(BaseModel):
    """Portable runtime directories resolved relative to project root or env vars."""

    model_config = ConfigDict(frozen=True)

    root: Path
    data_dir: Path
    forecast_cache_dir: Path
    state_dir: Path
    reports_dir: Path
    hf_cache_dir: Path

    @classmethod
    def resolve(cls, base_dir: Path | None = None) -> RuntimePaths:
        root = find_project_root(base_dir)

        def _dir(env_key: str, default_rel: str) -> Path:
            raw = os.environ.get(env_key)
            if raw:
                p = Path(raw).expanduser()
                return p if p.is_absolute() else (root / p)
            return root / default_rel

        return cls(
            root=root,
            data_dir=_dir("UTRADER_DATA_DIR", "data_cache"),
            forecast_cache_dir=_dir("UTRADER_FORECAST_CACHE_DIR", "forecast_cache"),
            state_dir=_dir("UTRADER_STATE_DIR", "state"),
            reports_dir=_dir("UTRADER_REPORTS_DIR", "reports"),
            hf_cache_dir=_dir("HF_HOME", ".hf_cache"),
        )


def _interpolate_env(obj: Any, strict: bool = False) -> Any:
    """Recursively expand ${ENV_VAR} and ${ENV_VAR:-default} in parsed YAML."""
    if isinstance(obj, dict):
        return {k: _interpolate_env(v, strict=strict) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v, strict=strict) for v in obj]
    if isinstance(obj, str):

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default_val = match.group(2)
            val = os.environ.get(var_name)
            if val is not None and val != "":
                return val
            if default_val is not None:
                return default_val
            if strict:
                raise ValueError(f"Required environment variable '${{{var_name}}}' is not set")
            return ""

        return _ENV_PATTERN.sub(_replace, obj)
    return obj


class FeaturesConfig(BaseModel):
    past_only: list[str] = Field(default_factory=list)
    known_future: list[str] = Field(default_factory=list)
    categorical_past: list[str] = Field(default_factory=list)
    categorical_known: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    provider: str = "yahoo"
    timeframe: str = "1d"
    lookback_bars: int = Field(default=512, ge=16)
    price_column: str = "close"
    use_cache: bool = True
    macro_provider: str = "yahoo"
    file_path: str | None = None
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    @model_validator(mode="after")
    def _validate_data_config(self) -> DataConfig:
        parse_timeframe(self.timeframe)
        if self.price_column not in {"close", "adj_close", "open", "high", "low"}:
            raise ValueError(f"Unsupported price_column '{self.price_column}'")
        return self


class ForecasterParamsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_id: str = "mistralai/Timestral-28M"
    context_len: int = Field(default=512, ge=8)
    horizon: int = Field(default=24, ge=1)
    max_horizon: int = Field(default=1024, ge=1)
    max_context: int = Field(default=32768, ge=8)
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    symmetric_averaging: bool = True
    clip_nonnegative: bool = True
    local_files_only: bool = False

    @model_validator(mode="after")
    def _validate_bounds(self) -> ForecasterParamsConfig:
        if self.horizon > self.max_horizon:
            raise ValueError(
                f"forecaster.params.horizon ({self.horizon}) exceeds max_horizon ({self.max_horizon})"
            )
        if self.context_len > self.max_context:
            raise ValueError(
                f"forecaster.params.context_len ({self.context_len}) exceeds max_context ({self.max_context})"
            )
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError(f"Invalid device '{self.device}'. Must be auto|cuda|mps|cpu")
        return self


class ForecasterConfig(BaseModel):
    type: str = "timestral"
    params: ForecasterParamsConfig = Field(default_factory=ForecasterParamsConfig)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "tail_risk_kelly"
    signal: str = "median_last"
    theta_buy: float = 0.01
    theta_sell: float = -0.01
    k_risk: float = Field(default=0.5, gt=0.0)
    fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    proportional_scale: float = Field(default=0.05, gt=0.0)
    target_vol: float = Field(default=0.05, gt=0.0)
    tail_quantile: float = Field(default=0.05, gt=0.0, lt=0.5)
    tail_risk_budget: float = Field(default=0.03, gt=0.0)
    prob_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    trend_fast_window: int = Field(default=20, ge=2)
    trend_slow_window: int = Field(default=60, ge=5)
    trend_hysteresis_pct: float = Field(default=0.015, ge=0.0, lt=0.25)
    macro_ema_window: int | None = Field(default=None, ge=10)
    macro_hysteresis_pct: float = Field(default=0.02, ge=0.0, lt=0.25)
    macro_bear_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    ema_pairs: list[tuple[int, int]] = Field(default_factory=list)
    target_weight_bull: float = Field(default=0.98, gt=0.0, le=1.0)
    rebalance_band: float = Field(default=0.15, ge=0.0, le=0.5)


class FeeModelConfig(BaseModel):
    pct: float = Field(default=0.001, ge=0.0, lt=0.5)
    fixed_min: float = Field(default=0.0, ge=0.0)


class RiskConfig(BaseModel):
    enforce_market_hours: bool = False
    max_data_age_minutes: int = Field(default=90, ge=1)
    max_exposure_pct: float = Field(default=1.0, gt=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, gt=0.0, le=1.0)
    cash_reserve_pct: float = Field(default=0.005, ge=0.0, lt=0.5)
    fee_model: FeeModelConfig = Field(default_factory=FeeModelConfig)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    spread_bps: float = Field(default=2.0, ge=0.0)
    enable_fee_guard: bool = True
    enable_quantile_stop_loss: bool = False
    stop_loss_quantile: float = Field(default=0.05, gt=0.0, lt=0.5)
    trailing_stop_pct: float | None = Field(default=None, gt=0.0, lt=1.0)


class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "local_paper"
    paper: bool = True
    initial_cash: float = Field(default=10000.0, gt=0.0)
    execution_price: str = "next_open"
    api_key: str = ""
    api_secret: str = ""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1

    @model_validator(mode="after")
    def _validate_execution_price(self) -> BrokerConfig:
        if self.execution_price not in {"next_open", "close"}:
            raise ValueError("broker.execution_price must be 'next_open' or 'close'")
        return self


class NotifierConfig(BaseModel):
    type: str = "noop"
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    notify_on: str = "all"  # all | events
    daily_summary_hour_utc: int | None = Field(default=8, ge=0, le=23)
    template_path: str = "configs/templates/telegram_portfolio.html.j2"
    daily_template_path: str = "configs/templates/telegram_daily.html.j2"

    @model_validator(mode="after")
    def _validate_notifier(self) -> NotifierConfig:
        if self.notify_on not in {"all", "events"}:
            raise ValueError(
                f"Unsupported notifier.notify_on '{self.notify_on}'. Must be 'all' or 'events'"
            )
        return self


class StorageConfig(BaseModel):
    db_path: str = "state/trader.db"


class WalkForwardConfig(BaseModel):
    mode: str = "rolling"  # rolling | expanding
    train_bars: int = Field(default=360, ge=20)
    test_bars: int = Field(default=120, ge=10)
    step_bars: int | None = None
    metric: str = "sharpe_ratio"  # sharpe_ratio | calmar_ratio | total_roi | sortino_ratio
    min_trades: int = Field(default=2, ge=0)
    initial_cash: float = Field(default=10000.0, gt=0.0)
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_wf(self) -> WalkForwardConfig:
        if self.mode not in {"rolling", "expanding"}:
            raise ValueError("walkforward.mode must be 'rolling' or 'expanding'")
        if self.metric not in {"sharpe_ratio", "calmar_ratio", "total_roi", "sortino_ratio"}:
            raise ValueError(f"Unsupported walkforward metric '{self.metric}'")
        return self


class PortfolioConfig(BaseModel):
    """Multi-asset portfolio allocation & rebalancing parameters."""

    allocation_method: str = "inverse_volatility"  # inverse_volatility | equal_weight
    max_weight_per_asset: float = Field(default=0.60, gt=0.0, le=1.0)
    max_total_exposure: float = Field(default=0.98, gt=0.0, le=1.0)
    vol_lookback_bars: int = Field(default=720, ge=2)
    vol_min_bars: int = Field(default=24, ge=2)
    vol_floor: float = Field(default=0.10, gt=0.0)
    rebalance_band: float = Field(default=0.12, ge=0.0, le=1.0)
    entry_min_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    exit_weight_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    min_trade_notional: float = Field(default=25.0, ge=0.0)
    warmup_bars: int | None = Field(default=None, ge=0)
    benchmark_instrument_id: str | None = None

    @model_validator(mode="after")
    def _validate_portfolio(self) -> PortfolioConfig:
        if self.allocation_method not in {"inverse_volatility", "equal_weight"}:
            raise ValueError(
                f"Unsupported portfolio.allocation_method '{self.allocation_method}'. "
                "Must be 'inverse_volatility' or 'equal_weight'"
            )
        if self.vol_min_bars > self.vol_lookback_bars:
            raise ValueError(
                f"portfolio.vol_min_bars ({self.vol_min_bars}) exceeds vol_lookback_bars ({self.vol_lookback_bars})"
            )
        return self


class AppConfig(BaseModel):
    """Top-level validated configuration for a single-asset or multi-asset Universal Trader pipeline."""

    name: str = Field(..., min_length=1)
    mode: TradingMode = TradingMode.PAPER
    instrument: Instrument
    instruments: list[Instrument] = Field(default_factory=list)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    forecaster: ForecasterConfig = Field(default_factory=ForecasterConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    notifier: NotifierConfig = Field(default_factory=NotifierConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    walkforward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_instruments_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        has_single = out.get("instrument") is not None
        raw_multi = out.get("instruments")
        if not has_single and isinstance(raw_multi, list) and len(raw_multi) > 0:
            out["instrument"] = raw_multi[0]
        elif has_single and not (isinstance(raw_multi, list) and len(raw_multi) > 0):
            out["instruments"] = [out["instrument"]]
        return out

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> AppConfig:
        if not self.instruments:
            self.instruments = [self.instrument]
        # Ensure unique instrument IDs
        ids = [inst.id for inst in self.instruments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate instrument IDs in portfolio config: {ids}")
        # Multi-asset portfolio requires a common quote currency for cash/equity accounting
        quote_currencies = {inst.quote_currency.upper() for inst in self.instruments}
        if len(quote_currencies) > 1:
            raise ValueError(
                f"All portfolio instruments must share one quote_currency, got {sorted(quote_currencies)}"
            )
        if (
            self.portfolio.benchmark_instrument_id is not None
            and self.portfolio.benchmark_instrument_id not in ids
        ):
            raise ValueError(
                f"portfolio.benchmark_instrument_id '{self.portfolio.benchmark_instrument_id}' "
                f"must match one of {ids}"
            )
        if self.forecaster.type == "timestral" and self.forecaster.params.context_len > 32768:
            raise ValueError("Timestral max context length is 32768")
        for inst in self.instruments:
            if inst.asset_class in {AssetClass.EQUITY, AssetClass.ETF} and self.data.provider == "binance":
                raise ValueError("BinanceDataProvider does not support equities/ETFs")
        return self

    @property
    def is_portfolio(self) -> bool:
        """True when managing a multi-asset basket (>1 instruments)."""
        return len(self.instruments) > 1

    @property
    def benchmark_instrument(self) -> Instrument:
        """Primary benchmark instrument for portfolio comparisons."""
        if self.portfolio.benchmark_instrument_id is not None:
            for inst in self.instruments:
                if inst.id == self.portfolio.benchmark_instrument_id:
                    return inst
        return self.instruments[0]


def load_config(
    path: str | Path,
    *,
    env_file: str | Path | None = None,
    strict_env: bool = False,
) -> AppConfig:
    """Load, interpolate environment variables, register secrets, and validate YAML config."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    # Load .env without overriding already-set process environment variables
    root = find_project_root(cfg_path.parent)
    dotenv_path = Path(env_file) if env_file is not None else (root / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    register_env_secrets()

    raw_text = cfg_path.read_text(encoding="utf-8")
    raw_data = yaml.safe_load(raw_text)
    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file '{cfg_path}' must contain a YAML mapping")

    interpolated = _interpolate_env(raw_data, strict=strict_env)
    config = AppConfig.model_validate(interpolated)

    # Register resolved secrets for log redaction
    register_secret(config.broker.api_key)
    register_secret(config.broker.api_secret)
    register_secret(config.notifier.bot_token)

    return config
