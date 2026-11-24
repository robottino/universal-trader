"""Probabilistic SignalExtractor operating on full quantile distributions."""

from __future__ import annotations

import numpy as np

from universal_trader.core.models import ForecastResult, SignalMetrics


class SignalExtractor:
    """Extracts rich distributional metrics from a ForecastResult relative to current price."""

    SUPPORTED_MODES = frozenset(
        {
            "median_last",
            "median_mean",
            "median_peak",
            "prob_positive",
            "quantile_asymmetry",
        }
    )

    def __init__(
        self,
        signal_mode: str = "median_last",
        tail_quantile: float = 0.05,
        theta_buy: float = 0.0,
    ) -> None:
        mode = signal_mode.strip().lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported signal mode '{signal_mode}'. Supported: {sorted(self.SUPPORTED_MODES)}"
            )
        self.signal_mode = mode
        self.tail_quantile = tail_quantile
        self.theta_buy = theta_buy

    def extract(self, forecast: ForecastResult, current_price: float) -> SignalMetrics:
        if current_price <= 0:
            raise ValueError(f"current_price must be > 0, got {current_price}")

        med = forecast.median
        q10 = forecast.quantile(0.10)
        q90 = forecast.quantile(0.90)
        q_tail = forecast.downside_tail(self.tail_quantile)

        median_last_ret = float((med[-1] - current_price) / current_price)
        median_mean_ret = float((float(np.mean(med)) - current_price) / current_price)
        rel_path = (med - current_price) / current_price
        peak_idx = int(np.argmax(np.abs(rel_path)))
        median_peak_ret = float(rel_path[peak_idx])

        # Implied probability that terminal price beats current_price * (1 + max(0, theta_buy))
        prob_threshold_price = current_price * (1.0 + max(0.0, self.theta_buy))
        prob_positive = forecast.prob_above(prob_threshold_price, step=-1)
        prob_above_spot = forecast.prob_above(current_price, step=-1)

        upside = max(0.0, float(q90[-1]) - current_price)
        downside = max(1e-9, current_price - float(q10[-1]))
        quantile_asymmetry = float(upside / downside)

        uncertainty_spread = float(max(0.0, float(q90[-1]) - float(q10[-1])) / current_price)
        downside_tail_return = float((float(q_tail[-1]) - current_price) / current_price)

        # Select primary decision signal value based on configured mode
        if self.signal_mode == "median_last":
            expected_return = median_last_ret
            signal_value = median_last_ret
        elif self.signal_mode == "median_mean":
            expected_return = median_mean_ret
            signal_value = median_mean_ret
        elif self.signal_mode == "median_peak":
            expected_return = median_peak_ret
            signal_value = median_peak_ret
        elif self.signal_mode == "prob_positive":
            expected_return = median_last_ret
            signal_value = prob_above_spot - 0.5
        else:  # quantile_asymmetry
            expected_return = median_last_ret
            signal_value = float(np.log(max(quantile_asymmetry, 1e-6)))

        # Estimate sigma from q90 - q10 (for normal distribution, q90 - q10 ≈ 2.5631 * sigma)
        sigma_est = max(1e-4, uncertainty_spread / 2.563103131)

        # Estimate left-tail CVaR from all quantiles <= tail_quantile
        tail_qs = [q for q in forecast.quantile_levels if q <= self.tail_quantile + 1e-9]
        if not tail_qs:
            tail_qs = [forecast.quantile_levels[0]]
        cvar_price = float(np.mean([float(forecast.quantiles[q][-1]) for q in tail_qs]))
        cvar_return = (cvar_price - current_price) / current_price

        return SignalMetrics(
            signal_mode=self.signal_mode,
            signal_value=signal_value,
            expected_return=expected_return,
            prob_positive=prob_positive,
            quantile_asymmetry=quantile_asymmetry,
            uncertainty_spread=uncertainty_spread,
            downside_tail_return=downside_tail_return,
            current_price=current_price,
            target_price_median=float(med[-1]),
            raw_metrics={
                "median_last_ret": median_last_ret,
                "median_mean_ret": median_mean_ret,
                "median_peak_ret": median_peak_ret,
                "prob_above_spot": prob_above_spot,
                "sigma_est": sigma_est,
                "cvar_return": cvar_return,
                "q10_last": float(q10[-1]),
                "q50_last": float(med[-1]),
                "q90_last": float(q90[-1]),
                "q_tail_last": float(q_tail[-1]),
            },
        )
