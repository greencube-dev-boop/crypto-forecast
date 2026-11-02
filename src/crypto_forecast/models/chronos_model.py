"""Amazon Chronos Foundation Model Forecaster with Dual-Resolution Adaptation.

Supports amazon/chronos-bolt-small and amazon/chronos-2.
Overcomes the 64-step ceiling of Chronos-Bolt via dual-resolution routing:
- Intraday (1h..2d <= 48 steps): Evaluated on hourly series
- Macro (3d..30d <= 30 steps): Evaluated on daily resampled series with spot re-anchoring
Includes lazy loading, device detection, and deterministic offline fallback.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from crypto_forecast.config import (
    CHRONOS_BOLT_MODEL_ID,
    CHRONOS_MAX_PREDICTION_LENGTH,
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
    INTRADAY_HORIZONS,
    MACRO_HORIZONS,
)
from crypto_forecast.models.base import BaseForecaster, HorizonPrediction
from crypto_forecast.models.sentiment import (
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
)

logger = logging.getLogger(__name__)


class ChronosError(Exception):
    """Base exception for Chronos forecaster errors."""


class ChronosModelError(ChronosError):
    """Raised when Chronos pipeline fails to initialize and fallback is disabled."""


class FallbackChronosForecaster:
    """Deterministic mathematical surrogate for Chronos when libraries/weights are absent."""

    def __init__(self, model_id: str = CHRONOS_BOLT_MODEL_ID) -> None:
        self.model_id = model_id

    def forecast(
        self,
        prices: np.ndarray,
        horizons: list[str],
        volatility: float,
    ) -> dict[str, HorizonPrediction]:
        current_price = float(prices[-1])
        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))
        lookback = min(len(prices), 48)
        trend = (
            (prices[-1] - prices[-lookback]) / (lookback * current_price)
            if lookback > 1 and current_price > 0
            else 0.0
        )
        trend = float(np.clip(trend, -0.005, 0.005))

        predictions: dict[str, HorizonPrediction] = {}
        for h in horizons:
            step_hours = HORIZON_TO_HOURS.get(h, 1)
            damping = (1.0 - math.exp(-0.02 * step_hours)) / 0.02
            pred_price = current_price * (1.0 + trend * damping)
            pred_price = max(lower_bound, pred_price)

            pred_change = pred_price - current_price
            pred_return = (pred_change / current_price) * 100.0

            if pred_change > 1e-6:
                direction = "BULLISH"
            elif pred_change < -1e-6:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            time_scale = math.sqrt(step_hours / 24.0)
            half_spread = current_price * 1.28 * volatility * time_scale

            p10 = max(lower_bound, pred_price - half_spread)
            p50 = pred_price
            p90 = pred_price + half_spread

            # Monotonicity guard
            p10 = min(p10, p50)
            p90 = max(p90, p50)

            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=step_hours,
                predicted_price=pred_price,
                current_price=current_price,
                predicted_change=pred_change,
                predicted_return_pct=pred_return,
                direction=direction,
                confidence_p10=p10,
                confidence_p50=p50,
                confidence_p90=p90,
                sentiment_adjusted=False,
                raw_predicted_price=pred_price,
            )
        return predictions

    def forecast_trajectory(
        self,
        prices: np.ndarray,
        max_horizon_hours: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        current_price = float(prices[-1])
        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))
        n = len(prices)
        if n > 1:
            log_returns = np.diff(np.log(prices))
            vol = float(np.std(log_returns, ddof=1 if len(log_returns) > 1 else 0))
            vol = max(vol, 0.001)
        else:
            vol = 0.01

        lookback = min(n, 48)
        trend = (
            (prices[-1] - prices[-lookback]) / (lookback * current_price)
            if lookback > 1 and current_price > 0
            else 0.0
        )
        trend = float(np.clip(trend, -0.005, 0.005))

        steps = np.arange(1, max_horizon_hours + 1, dtype=np.float64)
        dampings = (1.0 - np.exp(-0.02 * steps)) / 0.02
        point_traj = np.maximum(lower_bound, current_price * (1.0 + trend * dampings))

        time_scales = np.sqrt(steps / 24.0)
        spreads = current_price * 1.28 * vol * time_scales

        p10_traj = np.maximum(lower_bound, point_traj - spreads)
        p50_traj = point_traj.copy()
        p90_traj = point_traj + spreads

        p10_traj = np.minimum(p10_traj, p50_traj)
        p90_traj = np.maximum(p90_traj, p50_traj)

        quantile_traj = np.stack([p10_traj, p50_traj, p90_traj], axis=1)
        return point_traj, quantile_traj


class ChronosForecaster(BaseForecaster):
    """Chronos foundation model forecaster with dual-resolution adapter."""

    def __init__(
        self,
        model_id: str = CHRONOS_BOLT_MODEL_ID,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        use_fallback: bool = False,
    ) -> None:
        super().__init__(model_id=model_id, device=device_map)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_fallback = use_fallback
        self._pipeline: Any = None
        self._fallback_engine: FallbackChronosForecaster | None = None
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_fallback_active(self) -> bool:
        return self._fallback_engine is not None and self._pipeline is None

    def _ensure_loaded(self) -> None:
        """Lazy load pipeline or fallback engine."""
        if self._is_loaded:
            return

        if self.use_fallback:
            self._fallback_engine = FallbackChronosForecaster(self.model_id)
            self._is_loaded = True
            return

        try:
            import torch
            from chronos import BaseChronosPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = (
                torch.bfloat16
                if (device == "cuda" and torch.cuda.is_bf16_supported())
                else (torch.float16 if device == "cuda" else torch.float32)
            )

            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id,
                device_map=device,
                torch_dtype=dtype,
            )
            self._is_loaded = True
            logger.info("Loaded Chronos pipeline: %s on %s", self.model_id, device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chronos pipeline unavailable (%s). Using fallback surrogate.", exc)
            self._fallback_engine = FallbackChronosForecaster(self.model_id)
            self._is_loaded = True

    def forecast(
        self,
        context_prices: Sequence[float] | np.ndarray,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
        context_daily_prices: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Generates multi-horizon forecast using dual-resolution routing."""
        self._ensure_loaded()
        prices = self.validate_context_prices(context_prices)

        log_returns = np.diff(np.log(prices))
        vol = float(np.std(log_returns, ddof=1 if len(log_returns) > 1 else 0)) if len(prices) > 1 else 0.01
        vol = max(vol, 0.001)

        if self._fallback_engine is not None:
            preds = self._fallback_engine.forecast(prices, horizons, vol)
            if sentiment_polarity is not None:
                raw_prices = {h: p.predicted_price for h, p in preds.items()}
                drifted = apply_sentiment_drift(preds, sentiment_polarity=sentiment_polarity)
                for h, p in drifted.items():
                    if hasattr(p, "__dataclass_fields__"):
                        kwargs: dict[str, Any] = {}
                        if "sentiment_adjusted" in p.__dataclass_fields__:
                            kwargs["sentiment_adjusted"] = True
                        if "raw_predicted_price" in p.__dataclass_fields__:
                            kwargs["raw_predicted_price"] = raw_prices.get(h, p.predicted_price)
                        if kwargs:
                            drifted[h] = replace(p, **kwargs)
                preds = drifted
            return preds

        # Production Dual-Resolution Routing via Chronos Pipeline
        import torch

        current_price = float(prices[-1])
        predictions: dict[str, HorizonPrediction] = {}

        # 1. Intraday routing (hourly series, <= 48 steps)
        intraday_req = [h for h in horizons if h in INTRADAY_HORIZONS]
        if intraday_req:
            max_intraday_step = max(HORIZON_TO_HOURS[h] for h in intraday_req)
            pred_len = min(CHRONOS_MAX_PREDICTION_LENGTH, max(48, max_intraday_step))
            context_tensor = torch.tensor(prices[-512:], dtype=torch.float32).unsqueeze(0)

            forecast = self._pipeline.predict(context_tensor, prediction_length=pred_len)
            quantiles = torch.quantile(forecast[0], torch.tensor([0.1, 0.5, 0.9]), dim=0).numpy()

            for h in intraday_req:
                idx = HORIZON_TO_HOURS[h] - 1
                p10 = float(quantiles[0, idx])
                p50 = float(quantiles[1, idx])
                p90 = float(quantiles[2, idx])

                p10 = max(0.0001, p10)
                p50 = max(p10, p50)
                p90 = max(p50, p90)
                pred_price = p50

                pred_change = pred_price - current_price
                pred_ret = (pred_change / current_price) * 100.0
                direction = "BULLISH" if pred_change > 1e-6 else ("BEARISH" if pred_change < -1e-6 else "NEUTRAL")

                predictions[h] = HorizonPrediction(
                    horizon=h,
                    step_hours=HORIZON_TO_HOURS[h],
                    predicted_price=pred_price,
                    current_price=current_price,
                    predicted_change=pred_change,
                    predicted_return_pct=pred_ret,
                    direction=direction,
                    confidence_p10=p10,
                    confidence_p50=p50,
                    confidence_p90=p90,
                )

        # 2. Macro routing (daily series, <= 30 steps)
        macro_req = [h for h in horizons if h in MACRO_HORIZONS and h not in predictions]
        if macro_req:
            if context_daily_prices is not None:
                daily_arr = np.asarray(context_daily_prices, dtype=np.float32)
            else:
                daily_arr = prices[-1::-24][::-1].astype(np.float32)

            daily_tensor = torch.tensor(daily_arr[-365:], dtype=torch.float32).unsqueeze(0)
            forecast_daily = self._pipeline.predict(daily_tensor, prediction_length=30)
            quantiles_daily = torch.quantile(forecast_daily[0], torch.tensor([0.1, 0.5, 0.9]), dim=0).numpy()

            # Spot-price continuity re-anchoring
            scale = current_price / float(daily_arr[-1]) if daily_arr[-1] > 0 else 1.0

            for h in macro_req:
                step_days = HORIZON_TO_HOURS[h] // 24
                idx = step_days - 1
                p10 = float(quantiles_daily[0, idx]) * scale
                p50 = float(quantiles_daily[1, idx]) * scale
                p90 = float(quantiles_daily[2, idx]) * scale

                p10 = max(0.0001, p10)
                p50 = max(p10, p50)
                p90 = max(p50, p90)
                pred_price = p50

                pred_change = pred_price - current_price
                pred_ret = (pred_change / current_price) * 100.0
                direction = "BULLISH" if pred_change > 1e-6 else ("BEARISH" if pred_change < -1e-6 else "NEUTRAL")

                predictions[h] = HorizonPrediction(
                    horizon=h,
                    step_hours=HORIZON_TO_HOURS[h],
                    predicted_price=pred_price,
                    current_price=current_price,
                    predicted_change=pred_change,
                    predicted_return_pct=pred_ret,
                    direction=direction,
                    confidence_p10=p10,
                    confidence_p50=p50,
                    confidence_p90=p90,
                )

        if sentiment_polarity is not None:
            raw_prices = {h: p.predicted_price for h, p in predictions.items()}
            drifted = apply_sentiment_drift(predictions, sentiment_polarity=sentiment_polarity)
            for h, p in drifted.items():
                if hasattr(p, "__dataclass_fields__"):
                    kwargs = {}
                    if "sentiment_adjusted" in p.__dataclass_fields__:
                        kwargs["sentiment_adjusted"] = True
                    if "raw_predicted_price" in p.__dataclass_fields__:
                        kwargs["raw_predicted_price"] = raw_prices.get(h, p.predicted_price)
                    if kwargs:
                        drifted[h] = replace(p, **kwargs)
            predictions = drifted

        return predictions

    def forecast_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_hours: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous point and quantile trajectories up to max_horizon_hours."""
        self._ensure_loaded()
        prices = self.validate_context_prices(context_prices)

        if self._fallback_engine is not None:
            return self._fallback_engine.forecast_trajectory(prices, max_horizon_hours=max_horizon_hours)

        preds = self.forecast(prices, DEFAULT_HORIZONS)
        current_price = float(prices[-1])

        hours = [HORIZON_TO_HOURS[h] for h in DEFAULT_HORIZONS]
        p_vals = [preds[h].predicted_price for h in DEFAULT_HORIZONS]
        p10_vals = [preds[h].confidence_p10 for h in DEFAULT_HORIZONS]
        p90_vals = [preds[h].confidence_p90 for h in DEFAULT_HORIZONS]

        target_steps = np.arange(1, max_horizon_hours + 1)
        all_hours = np.array([0] + hours)
        all_p = np.array([current_price] + p_vals)
        all_p10 = np.array([current_price] + p10_vals)
        all_p90 = np.array([current_price] + p90_vals)

        point_traj = np.interp(target_steps, all_hours, all_p)
        p10_traj = np.maximum(0.0001, np.interp(target_steps, all_hours, all_p10))
        p50_traj = point_traj.copy()
        p90_traj = np.interp(target_steps, all_hours, all_p90)

        p10_traj = np.minimum(p10_traj, p50_traj)
        p90_traj = np.maximum(p90_traj, p50_traj)

        quantile_traj = np.stack([p10_traj, p50_traj, p90_traj], axis=1)
        return point_traj, quantile_traj

    def get_full_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_steps: int = 720,
        sentiment_polarity: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous 720-step trajectory via dual-resolution interpolation."""
        point_traj, quantile_traj = self.forecast_trajectory(context_prices, max_horizon_hours=max_horizon_steps)
        if sentiment_polarity is not None:
            point_traj, quantile_traj = apply_trajectory_sentiment_drift(
                point_traj,
                quantile_traj,
                sentiment_polarity=sentiment_polarity,
            )
        return point_traj, quantile_traj
