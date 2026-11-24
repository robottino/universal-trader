"""Immutable Pydantic v2 domain models for Universal Trader."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def ensure_utc(dt: datetime | pd.Timestamp | str) -> datetime:
    """Convert a datetime, pandas Timestamp, or ISO string to timezone-aware UTC datetime."""
    if isinstance(dt, str):
        parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        dt = parsed
    elif isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class AssetClass(StrEnum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"


class TradingMode(StrEnum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderStatus(StrEnum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    DRY_RUN = "DRY_RUN"


class Instrument(BaseModel):
    """Canonical representation of any tradable financial instrument."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    asset_class: AssetClass
    base: str = Field(..., min_length=1)
    quote_currency: str = Field(..., min_length=1)
    calendar: str = Field(default="24/7")
    fractional: bool = True
    min_qty: float = Field(default=0.0001, gt=0)
    qty_step: float = Field(default=0.00001, gt=0)
    min_notional: float = Field(default=5.0, ge=0)
    provider_symbols: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fractional_constraints(self) -> Instrument:
        if not self.fractional:
            if not float(self.min_qty).is_integer() or self.min_qty < 1:
                raise ValueError(
                    f"Non-fractional instrument '{self.id}' requires integer min_qty >= 1, got {self.min_qty}"
                )
            if not float(self.qty_step).is_integer() or self.qty_step < 1:
                raise ValueError(
                    f"Non-fractional instrument '{self.id}' requires integer qty_step >= 1, got {self.qty_step}"
                )
        return self

    def symbol_for(self, provider: str) -> str:
        """Return the provider-specific symbol, falling back to canonical instrument id."""
        return self.provider_symbols.get(provider, self.id)

    def round_qty(self, qty: float) -> float:
        """Floor quantity to the nearest allowed qty_step (whole units if non-fractional)."""
        if qty <= 0 or not math.isfinite(qty):
            return 0.0
        if not self.fractional:
            step_int = max(1, int(round(self.qty_step)))
            return float((int(math.floor(qty)) // step_int) * step_int)
        d_qty = Decimal(str(qty))
        d_step = Decimal(str(self.qty_step))
        units = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN)
        return float(units * d_step)


class Candle(BaseModel):
    """Single OHLCV bar normalized in UTC."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(default=0.0, ge=0.0)
    adj_close: float | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class Quote(BaseModel):
    """Point-in-time market quote for an instrument."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    timestamp: datetime
    bid: float = Field(..., gt=0)
    ask: float = Field(..., gt=0)
    last: float = Field(..., gt=0)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)


class MarketSnapshot(BaseModel):
    """Observed market state at a decision step."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    instrument: Instrument
    timestamp: datetime
    candles: pd.DataFrame
    quote: Quote

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class ForecastRequest(BaseModel):
    """Model-agnostic input contract for any Forecaster plugin."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    instrument_id: str
    timestamp: datetime
    target: np.ndarray
    horizon: int = Field(..., gt=0)
    past_covariates: dict[str, np.ndarray] = Field(default_factory=dict)
    known_future_covariates: dict[str, np.ndarray] = Field(default_factory=dict)
    categorical_past_covariates: dict[str, np.ndarray] = Field(default_factory=dict)
    categorical_known_covariates: dict[str, np.ndarray] = Field(default_factory=dict)
    future_timestamps: list[datetime] = Field(default_factory=list)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)

    @model_validator(mode="after")
    def _validate_shapes(self) -> ForecastRequest:
        t_len = int(self.target.shape[-1])
        if t_len < 1:
            raise ValueError("ForecastRequest.target must have length >= 1")
        for name, arr in self.past_covariates.items():
            if int(arr.shape[-1]) != t_len:
                raise ValueError(
                    f"past_covariate '{name}' length {arr.shape[-1]} != target length {t_len}"
                )
        for name, arr in self.known_future_covariates.items():
            expected = t_len + self.horizon
            if int(arr.shape[-1]) != expected:
                raise ValueError(
                    f"known_future_covariate '{name}' length {arr.shape[-1]} != {expected}"
                )
        for name, arr in self.categorical_past_covariates.items():
            if int(arr.shape[-1]) != t_len:
                raise ValueError(
                    f"categorical_past_covariate '{name}' length {arr.shape[-1]} != {t_len}"
                )
        for name, arr in self.categorical_known_covariates.items():
            expected = t_len + self.horizon
            if int(arr.shape[-1]) != expected:
                raise ValueError(
                    f"categorical_known_covariate '{name}' length {arr.shape[-1]} != {expected}"
                )
        return self


class ForecastResult(BaseModel):
    """Standardized probabilistic quantile forecast output."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    instrument_id: str
    model_id: str
    timestamp: datetime
    horizon: int = Field(..., gt=0)
    quantiles: dict[float, np.ndarray]
    future_timestamps: list[datetime] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)

    @field_validator("quantiles", mode="before")
    @classmethod
    def _normalize_quantiles(cls, v: Any) -> dict[float, np.ndarray]:
        if not isinstance(v, dict) or not v:
            raise ValueError("ForecastResult.quantiles must be a non-empty mapping")
        out: dict[float, np.ndarray] = {}
        for k, arr in v.items():
            q = round(float(k), 6)
            if not (0.0 < q < 1.0):
                raise ValueError(f"Quantile level must be in (0, 1), got {q}")
            out[q] = np.asarray(arr, dtype=np.float64).reshape(-1)
        return dict(sorted(out.items()))

    @model_validator(mode="after")
    def _validate_lengths(self) -> ForecastResult:
        for q, arr in self.quantiles.items():
            if len(arr) != self.horizon:
                raise ValueError(
                    f"Quantile {q} length {len(arr)} does not match horizon {self.horizon}"
                )
        return self

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        return tuple(sorted(self.quantiles.keys()))

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    def quantile(self, q: float) -> np.ndarray:
        """Return forecast trajectory for quantile q, using linear interpolation if absent."""
        q_norm = round(float(q), 6)
        if q_norm in self.quantiles:
            return self.quantiles[q_norm]
        levels = self.quantile_levels
        if q_norm <= levels[0]:
            return self.quantiles[levels[0]]
        if q_norm >= levels[-1]:
            return self.quantiles[levels[-1]]
        # Find bracketing quantiles
        for i in range(len(levels) - 1):
            lo, hi = levels[i], levels[i + 1]
            if lo <= q_norm <= hi:
                weight = (q_norm - lo) / (hi - lo)
                return (1.0 - weight) * self.quantiles[lo] + weight * self.quantiles[hi]
        return self.quantiles[levels[-1]]

    def iqr(self) -> np.ndarray:
        """Interquartile range q75 - q25 along the horizon."""
        return np.asarray(self.quantile(0.75) - self.quantile(0.25), dtype=np.float64)

    def spread(self, low: float = 0.10, high: float = 0.90) -> np.ndarray:
        """Quantile band width between high and low levels."""
        return np.asarray(self.quantile(high) - self.quantile(low), dtype=np.float64)

    def expected_return(self, price: float, mode: str = "last") -> float:
        """Compute expected return relative to current price using median forecast."""
        if price <= 0:
            raise ValueError(f"Current price must be > 0, got {price}")
        med = self.median
        if mode == "last":
            target_price = float(med[-1])
        elif mode == "mean":
            target_price = float(np.mean(med))
        elif mode == "peak":
            # Signed maximum excursion from current price
            rel = (med - price) / price
            idx = int(np.argmax(np.abs(rel)))
            return float(rel[idx])
        else:
            raise ValueError(f"Unknown expected_return mode: {mode}")
        return (target_price - price) / price

    def downside_tail(self, q: float = 0.05) -> np.ndarray:
        """Lower quantile trajectory (default q=0.05)."""
        return self.quantile(q)

    def prob_above(self, threshold: float, step: int = -1) -> float:
        """Estimate P(Y_step > threshold) by inverting the sorted quantile curve."""
        levels = np.array(self.quantile_levels, dtype=np.float64)
        values = np.array([float(self.quantiles[q][step]) for q in levels], dtype=np.float64)
        # Ensure monotonicity for CDF inversion
        values = np.maximum.accumulate(values)
        if threshold <= values[0]:
            return float(1.0 - levels[0])
        if threshold >= values[-1]:
            return float(1.0 - levels[-1])
        cdf_val = float(np.interp(threshold, values, levels))
        return float(np.clip(1.0 - cdf_val, 0.0, 1.0))

    def to_serializable_dict(self) -> dict[str, Any]:
        """Serialize ForecastResult to a JSON-compatible dictionary."""
        return {
            "instrument_id": self.instrument_id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
            "horizon": self.horizon,
            "quantiles": {str(q): arr.tolist() for q, arr in self.quantiles.items()},
            "future_timestamps": [ts.isoformat() for ts in self.future_timestamps],
            "metadata": self.metadata,
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict[str, Any]) -> ForecastResult:
        """Deserialize ForecastResult from a JSON-compatible dictionary."""
        return cls(
            instrument_id=payload["instrument_id"],
            model_id=payload["model_id"],
            timestamp=ensure_utc(payload["timestamp"]),
            horizon=int(payload["horizon"]),
            quantiles={float(k): np.asarray(v, dtype=np.float64) for k, v in payload["quantiles"].items()},
            future_timestamps=[ensure_utc(ts) for ts in payload.get("future_timestamps", [])],
            metadata=payload.get("metadata", {}),
        )


class Position(BaseModel):
    """Open holding in a single instrument."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    quantity: float = Field(default=0.0, ge=0.0)
    avg_entry_price: float = Field(default=0.0, ge=0.0)
    market_price: float = Field(default=0.0, ge=0.0)

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity <= 0 or self.avg_entry_price <= 0:
            return 0.0
        return (self.market_price - self.avg_entry_price) * self.quantity


class PortfolioState(BaseModel):
    """Point-in-time snapshot of cash, positions, equity, and drawdown state."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    quote_currency: str
    cash: float = Field(..., ge=0.0)
    positions: dict[str, Position] = Field(default_factory=dict)
    equity: float = Field(..., ge=0.0)
    high_water_mark: float = Field(..., ge=0.0)
    halted: bool = False

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)

    @property
    def drawdown(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.equity / self.high_water_mark))

    def position_qty(self, instrument_id: str) -> float:
        pos = self.positions.get(instrument_id)
        return pos.quantity if pos is not None else 0.0

    def position_value(self, instrument_id: str) -> float:
        pos = self.positions.get(instrument_id)
        return pos.market_value if pos is not None else 0.0


class SignalMetrics(BaseModel):
    """Summary statistics extracted from a probabilistic forecast distribution."""

    model_config = ConfigDict(frozen=True)

    signal_mode: str
    signal_value: float
    expected_return: float
    prob_positive: float
    quantile_asymmetry: float
    uncertainty_spread: float
    downside_tail_return: float
    current_price: float
    target_price_median: float
    raw_metrics: dict[str, float] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    """Desired trading action emitted by a Strategy before RiskManager checks."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    timestamp: datetime
    side: OrderSide
    target_weight: float = Field(..., ge=0.0, le=1.0)
    target_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    signal: SignalMetrics
    reason: str = ""
    is_stop_loss: bool = False

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class ApprovedOrder(BaseModel):
    """Validated, sized order approved by RiskManager for broker submission."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    timestamp: datetime
    side: OrderSide
    quantity: float = Field(..., gt=0.0)
    estimated_price: float = Field(..., gt=0.0)
    estimated_notional: float = Field(..., gt=0.0)
    estimated_fee: float = Field(..., ge=0.0)
    estimated_slippage: float = Field(..., ge=0.0)
    reason: str = ""
    is_stop_loss: bool = False
    client_order_id: str | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class RejectedOrder(BaseModel):
    """Order rejected by a RiskGuard with explicit diagnostic reason."""

    model_config = ConfigDict(frozen=True)

    instrument_id: str
    timestamp: datetime
    side: OrderSide
    guard_name: str
    reason: str
    halted: bool = False

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class FillResult(BaseModel):
    """Execution result returned by a Broker."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    instrument_id: str
    timestamp: datetime
    side: OrderSide
    quantity: float = Field(..., ge=0.0)
    fill_price: float = Field(..., gt=0.0)
    notional: float = Field(..., ge=0.0)
    fee: float = Field(..., ge=0.0)
    slippage: float = Field(default=0.0, ge=0.0)
    status: OrderStatus
    raw_response: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)


class StepOutcome(BaseModel):
    """Full audit record of one atomic TradingStepRunner cycle."""

    model_config = ConfigDict(frozen=True)

    config_name: str
    instrument_id: str
    timeframe: str
    candle_timestamp: datetime
    skipped_idempotent: bool = False
    forecast: ForecastResult | None = None
    intent: OrderIntent | None = None
    approved_order: ApprovedOrder | None = None
    rejected_order: RejectedOrder | None = None
    fill: FillResult | None = None
    portfolio_before: PortfolioState | None = None
    portfolio_after: PortfolioState | None = None
    error: str | None = None

    @field_validator("candle_timestamp", mode="before")
    @classmethod
    def _utc_timestamp(cls, v: Any) -> datetime:
        return ensure_utc(v)
