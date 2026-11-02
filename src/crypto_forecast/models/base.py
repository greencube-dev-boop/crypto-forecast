"""Multi-Horizon Forecasting Engine - Base Abstractions & Prediction Schemas.

Defines:
1. HorizonPrediction: Immutable/validated dataclass capturing multi-horizon forecasts,
   quantiles (P10, P50, P90), returns, directional tags, and sentiment provenance.
2. BaseForecaster: Abstract base class enforcing trajectory generation, discrete
   horizon extraction, input validation, and sentiment conviction drift integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
)

DIRECTION_THRESHOLD: Final[float] = 1e-6


# ==============================================================================
# 1. HorizonPrediction Dataclass
# ==============================================================================

@dataclass
class HorizonPrediction:
    """Standardized multi-horizon price prediction contract.

    Attributes:
        horizon: Horizon identifier string (e.g. '1h', '24h', '7d', '30d').
        step_hours: Forecast step distance in integer hours.
        predicted_price: Point forecast price at horizon step.
        current_price: Cutoff context price at time t_0.
        predicted_change: Absolute price difference (predicted_price - current_price).
        predicted_return_pct: Percentage return relative to current_price.
        direction: Directional conviction ('BULLISH', 'BEARISH', or 'NEUTRAL').
        confidence_p10: 10th percentile lower confidence bound.
        confidence_p50: 50th percentile median forecast.
        confidence_p90: 90th percentile upper confidence bound.
        sentiment_adjusted: True if sentiment drift has been applied.
        raw_predicted_price: Pre-adjustment predicted price if sentiment_adjusted=True.
    """

    horizon: str
    step_hours: int
    predicted_price: float
    current_price: float
    predicted_change: float
    predicted_return_pct: float
    direction: str  # 'BULLISH', 'BEARISH', or 'NEUTRAL'
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float
    sentiment_adjusted: bool = False
    raw_predicted_price: float | None = None

    def __post_init__(self) -> None:
        """Validates contract invariants upon initialization."""
        if self.current_price <= 0.0:
            raise ValueError(f"current_price must be strictly positive, got {self.current_price}")
        if self.predicted_price <= 0.0:
            raise ValueError(f"predicted_price must be strictly positive, got {self.predicted_price}")
        if self.confidence_p10 <= 0.0:
            raise ValueError(f"confidence_p10 must be strictly positive, got {self.confidence_p10}")
        if not (self.confidence_p10 <= self.confidence_p50 <= self.confidence_p90):
            raise ValueError(
                f"Quantile monotonicity violated: p10 ({self.confidence_p10}) <= "
                f"p50 ({self.confidence_p50}) <= p90 ({self.confidence_p90})"
            )
        if self.direction not in ("BULLISH", "BEARISH", "NEUTRAL"):
            raise ValueError(
                f"Invalid direction '{self.direction}', must be 'BULLISH', 'BEARISH', or 'NEUTRAL'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serializes prediction to standard dictionary format."""
        return {
            "horizon": self.horizon,
            "step_hours": self.step_hours,
            "predicted_price": float(self.predicted_price),
            "current_price": float(self.current_price),
            "predicted_change": float(self.predicted_change),
            "predicted_return_pct": float(self.predicted_return_pct),
            "direction": self.direction,
            "confidence_p10": float(self.confidence_p10),
            "confidence_p50": float(self.confidence_p50),
            "confidence_p90": float(self.confidence_p90),
            "sentiment_adjusted": bool(self.sentiment_adjusted),
            "raw_predicted_price": (
                float(self.raw_predicted_price) if self.raw_predicted_price is not None else None
            ),
        }

    def is_bullish(self) -> bool:
        """Returns True if prediction direction is BULLISH."""
        return self.direction == "BULLISH"

    def is_bearish(self) -> bool:
        """Returns True if prediction direction is BEARISH."""
        return self.direction == "BEARISH"

    def uncertainty_spread(self) -> float:
        """Returns width of the 80% confidence interval (P90 - P10)."""
        return float(self.confidence_p90 - self.confidence_p10)

    @classmethod
    def from_point_and_quantiles(
        cls,
        horizon: str,
        step_hours: int,
        current_price: float,
        predicted_price: float,
        p10: float,
        p50: float,
        p90: float,
        sentiment_adjusted: bool = False,
        raw_predicted_price: float | None = None,
        epsilon: float = DIRECTION_THRESHOLD,
    ) -> HorizonPrediction:
        """Convenience factory computing delta, return percentage, and direction automatically."""
        change = float(predicted_price - current_price)
        return_pct = float((change / current_price) * 100.0) if current_price > 0.0 else 0.0

        if change > epsilon:
            direction = "BULLISH"
        elif change < -epsilon:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return cls(
            horizon=horizon,
            step_hours=step_hours,
            predicted_price=float(predicted_price),
            current_price=float(current_price),
            predicted_change=change,
            predicted_return_pct=return_pct,
            direction=direction,
            confidence_p10=float(p10),
            confidence_p50=float(p50),
            confidence_p90=float(p90),
            sentiment_adjusted=sentiment_adjusted,
            raw_predicted_price=raw_predicted_price,
        )


# ==============================================================================
# 2. BaseForecaster Abstract Base Class
# ==============================================================================

class BaseForecaster(ABC):
    """Abstract Base Class for Multi-Horizon Cryptocurrency Forecasters."""

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.name = model_id
        self.device = device

    @staticmethod
    def validate_context_prices(context_prices: Sequence[float] | np.ndarray) -> np.ndarray:
        """Validates and sanitizes context price series.

        Raises:
            ValueError: If context is empty, contains NaNs/Infs, or contains non-positive prices.
        """
        if context_prices is None or len(context_prices) == 0:
            raise ValueError("context_prices cannot be empty")

        prices = np.asarray(context_prices, dtype=np.float64).ravel()
        if len(prices) == 0:
            raise ValueError("context_prices cannot be empty")

        if np.any(np.isnan(prices)):
            raise ValueError("context_prices cannot contain NaNs")

        if np.any(~np.isfinite(prices)):
            raise ValueError("context_prices cannot contain infinite values")

        if np.any(prices <= 0.0):
            raise ValueError("context_prices must be strictly positive")

        return prices

    def _validate_context(self, context_prices: Sequence[float] | np.ndarray) -> np.ndarray:
        """Instance alias for validate_context_prices."""
        return self.validate_context_prices(context_prices)

    @abstractmethod
    def forecast_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_hours: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous point and quantile trajectories up to max_horizon_hours.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                point_forecasts: 1D array of shape (max_horizon_hours,)
                quantile_forecasts: 2D array of shape (max_horizon_hours, 3) where columns are [p10, p50, p90]
        """

    def get_full_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_steps: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility alias for forecast_trajectory matching PROJECT.md interface contract."""
        return self.forecast_trajectory(context_prices, max_horizon_hours=max_horizon_steps)

    def extract_discrete_horizons(
        self,
        current_price: float,
        point_trajectory: np.ndarray,
        quantile_trajectory: np.ndarray,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Extracts discrete horizon predictions from continuous trajectories.

        Maps discrete horizon identifiers to 0-based trajectory indices:
        index = step_hours - 1. Applies sentiment drift if polarity is provided.
        """
        n_steps = len(point_trajectory)
        if quantile_trajectory.ndim != 2 or quantile_trajectory.shape[0] != n_steps or quantile_trajectory.shape[1] != 3:
            raise ValueError(
                f"quantile_trajectory shape {quantile_trajectory.shape} must match (len(point_trajectory)={n_steps}, 3)"
            )

        predictions: dict[str, HorizonPrediction] = {}

        for h in horizons:
            if h not in HORIZON_TO_HOURS:
                raise ValueError(f"Unknown horizon identifier: '{h}'. Must be in HORIZON_TO_HOURS.")
            step_hours = HORIZON_TO_HOURS[h]
            idx = step_hours - 1

            if idx >= n_steps:
                raise ValueError(
                    f"Requested horizon '{h}' ({step_hours}h) exceeds trajectory length {n_steps}h"
                )

            pred_price = float(point_trajectory[idx])
            p10 = float(quantile_trajectory[idx, 0])
            p50 = float(quantile_trajectory[idx, 1])
            p90 = float(quantile_trajectory[idx, 2])

            pred = HorizonPrediction.from_point_and_quantiles(
                horizon=h,
                step_hours=step_hours,
                current_price=current_price,
                predicted_price=pred_price,
                p10=p10,
                p50=p50,
                p90=p90,
            )
            predictions[h] = pred

        # Apply sentiment drift modification if polarity provided
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

    def extract_horizon_predictions(
        self,
        point_trajectory: np.ndarray,
        quantile_trajectory: np.ndarray,
        current_price: float,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Alias for extract_discrete_horizons with alternative argument order."""
        return self.extract_discrete_horizons(
            current_price=current_price,
            point_trajectory=point_trajectory,
            quantile_trajectory=quantile_trajectory,
            horizons=horizons,
            sentiment_polarity=sentiment_polarity,
        )

    def forecast(
        self,
        context_prices: Sequence[float] | np.ndarray,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Generates multi-horizon point and quantile forecasts for the requested horizons."""
        prices = self.validate_context_prices(context_prices)
        current_price = float(prices[-1])

        # Determine trajectory depth required
        max_step = max(HORIZON_TO_HOURS.get(h, 1) for h in horizons) if horizons else 720
        trajectory_depth = max(max_step, 720)

        point_traj, quantile_traj = self.forecast_trajectory(prices, max_horizon_hours=trajectory_depth)
        return self.extract_discrete_horizons(
            current_price=current_price,
            point_trajectory=point_traj,
            quantile_trajectory=quantile_traj,
            horizons=horizons,
            sentiment_polarity=sentiment_polarity,
        )
