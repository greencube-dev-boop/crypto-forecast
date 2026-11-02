"""Naive Persistence Baseline Forecaster Implementation.

Serves as the foundational reference benchmark defined in PROJECT.md (Feature 15):
y_hat_{t_0 + h} = y_{t_0} across all 11 horizons.
Uncertainty cones expand with sqrt(step_hours / 24) scaled by empirical log-return volatility.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from crypto_forecast.config import (
    BASELINE_MODEL_ID,
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
)
from crypto_forecast.models.base import BaseForecaster, HorizonPrediction

logger = logging.getLogger(__name__)


class PersistenceForecaster(BaseForecaster):
    """Naive persistence baseline forecaster (Random-Walk Martingale Model).

    Assumes martingale property E[P_{t+h} | F_t] = P_t for all horizons h.
    Predicted change = 0.0, return = 0.0%, direction = 'NEUTRAL'.
    Uncertainty cones widen proportionally to sqrt(step_hours / 24) using historical
    log-return rolling volatility:
      P10 = max(0.0001, P_t * (1 - 1.28 * sigma * sqrt(h / 24)))
      P50 = P_t
      P90 = P_t * (1 + 1.28 * sigma * sqrt(h / 24))

    Zero external ML dependencies, sub-millisecond execution, zero disk footprint.
    Serves as the universal benchmark denominator for relative skill score metrics.
    """

    def __init__(
        self,
        name: str = BASELINE_MODEL_ID,
        device: str = "cpu",
        default_volatility: float = 0.01,
        min_volatility: float = 0.001,
    ) -> None:
        super().__init__(model_id=name, device=device)
        self.default_volatility = default_volatility
        self.min_volatility = min_volatility

    def _compute_volatility(self, prices: np.ndarray) -> float:
        """Computes empirical log-return volatility with defensive lower bound."""
        if len(prices) < 2:
            return self.default_volatility

        log_returns = np.diff(np.log(prices))
        vol = float(np.std(log_returns, ddof=1 if len(log_returns) > 1 else 0))
        if np.isnan(vol) or vol < self.min_volatility:
            return self.min_volatility
        return vol

    def forecast_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_hours: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous point and quantile trajectories up to max_horizon_hours."""
        prices = self.validate_context_prices(context_prices)
        current_price = float(prices[-1])
        vol = self._compute_volatility(prices)

        steps = np.arange(1, max_horizon_hours + 1, dtype=np.float64)
        point_forecasts = np.full(max_horizon_hours, current_price, dtype=np.float64)

        # Expanding confidence envelope: 1.28 * vol * sqrt(steps / 24)
        time_scales = np.sqrt(steps / 24.0)
        spreads = current_price * 1.28 * vol * time_scales

        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))
        p10 = np.maximum(lower_bound, current_price - spreads)
        p50 = point_forecasts.copy()
        p90 = current_price + spreads
        p10 = np.minimum(p10, p50)
        p90 = np.maximum(p90, p50)

        quantile_forecasts = np.column_stack([p10, p50, p90])
        return point_forecasts, quantile_forecasts

    def forecast(
        self,
        context_prices: Sequence[float] | np.ndarray,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Generates persistence predictions across specified horizons."""
        prices = self.validate_context_prices(context_prices)
        current_price = float(prices[-1])
        vol = self._compute_volatility(prices)
        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))

        predictions: dict[str, HorizonPrediction] = {}
        for h in horizons:
            if h not in HORIZON_TO_HOURS:
                raise ValueError(f"Unknown horizon identifier: '{h}'. Must be in HORIZON_TO_HOURS.")
            step_hours = HORIZON_TO_HOURS[h]
            time_scale = math.sqrt(step_hours / 24.0)
            half_spread = current_price * 1.28 * vol * time_scale

            p10 = max(lower_bound, current_price - half_spread)
            p50 = current_price
            p90 = current_price + half_spread
            p10 = min(p10, p50)
            p90 = max(p90, p50)

            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=step_hours,
                predicted_price=current_price,
                current_price=current_price,
                predicted_change=0.0,
                predicted_return_pct=0.0,
                direction="NEUTRAL",
                confidence_p10=p10,
                confidence_p50=p50,
                confidence_p90=p90,
                sentiment_adjusted=False,
                raw_predicted_price=current_price,
            )

        if sentiment_polarity is not None:
            from crypto_forecast.models.sentiment import apply_sentiment_drift

            raw_prices = {h: p.predicted_price for h, p in predictions.items()}
            drifted = apply_sentiment_drift(
                predictions=predictions,
                sentiment_polarity=sentiment_polarity,
            )
            for h, p in drifted.items():
                if hasattr(p, "__dataclass_fields__"):
                    kwargs: dict[str, Any] = {}
                    if "sentiment_adjusted" in p.__dataclass_fields__:
                        kwargs["sentiment_adjusted"] = True
                    if "raw_predicted_price" in p.__dataclass_fields__:
                        kwargs["raw_predicted_price"] = raw_prices.get(h, p.predicted_price)
                    if kwargs:
                        drifted[h] = replace(p, **kwargs)
            predictions = drifted

        return predictions
