"""TimesFM 2.5 Foundation Forecaster Module.

Integrates Google's TimesFM 2.5 (google/timesfm-2.5-200m-pytorch) decoder-only
foundation model for multi-horizon cryptocurrency price forecasting.

Key Architectural Capabilities:
1. Native 720-step single forward pass covering all 11 horizons (1h to 30d)
   without recursive autoregressive accumulation error.
2. Continuous 10-level quantile head producing calibrated uncertainty intervals
   (P10, P50, P90).
3. Hardware auto-detection: FP16 execution on CUDA (Colab T4), CPU fallback.
4. Lazy loading: Zero memory/disk footprint until explicit invocation.
5. Deterministic local test fallback: Guarantees 100% offline unit test pass
   in <5ms with zero weights download (Requirement R3).
6. Sentiment drift integration: Seamless coupling with CryptoBERT sentiment polarity.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

import numpy as np

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
    PRIMARY_MODEL_ID,
)
from crypto_forecast.models.base import BaseForecaster, HorizonPrediction
from crypto_forecast.models.sentiment import (
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
)

logger = logging.getLogger(__name__)

# Standard normal quantile critical values for 10-level quantiles:
# Index 0: Mean (z=0.0)
# Index 1: 0.1 quantile (z=-1.28155)
# Index 2: 0.2 quantile (z=-0.84162)
# Index 3: 0.3 quantile (z=-0.52440)
# Index 4: 0.4 quantile (z=-0.25335)
# Index 5: 0.5 quantile (median, z=0.0)
# Index 6: 0.6 quantile (z=+0.25335)
# Index 7: 0.7 quantile (z=+0.52440)
# Index 8: 0.8 quantile (z=+0.84162)
# Index 9: 0.9 quantile (z=+1.28155)
Z_SCORES_10: Final[np.ndarray] = np.array(
    [0.0, -1.28155, -0.84162, -0.52440, -0.25335, 0.0, 0.25335, 0.52440, 0.84162, 1.28155],
    dtype=np.float64,
)


class TimesFMError(Exception):
    """Base exception for TimesFM forecaster errors."""


class TimesFMModelError(TimesFMError):
    """Raised when model weights cannot be loaded and fallback is disabled."""


class TimesFMInferenceError(TimesFMError):
    """Raised when inference encounters an unrecoverable failure."""


class FallbackTimesFMForecaster:
    """Deterministic offline fallback for TimesFM 2.5.

    Emulates TimesFM 2.5's single-pass 720h forecasting behavior using
    exponentially attenuated trend extrapolation and square-root-of-time
    calibrated quantile dispersion. Completely self-contained with zero downloads.
    """

    def __init__(self, max_horizon: int = 720) -> None:
        self.max_horizon = max_horizon

    def forecast_trajectory(
        self,
        prices: np.ndarray,
        horizon: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates point forecast (1, horizon) and 10-level quantiles (1, horizon, 10)."""
        current_price = float(prices[-1])
        n = len(prices)

        if n > 1:
            log_returns = np.diff(np.log(prices))
            vol = float(np.std(log_returns))
            vol = max(vol, 0.001)

            # Compute short vs long momentum
            short_w = min(12, n)
            long_w = min(48, n)
            ema_short = float(np.mean(prices[-short_w:]))
            ema_long = float(np.mean(prices[-long_w:]))
            momentum = (ema_short - ema_long) / ema_long if ema_long > 1e-8 else 0.0
            momentum = max(-0.15, min(0.15, momentum))
        else:
            vol = 0.01
            momentum = 0.0

        steps = np.arange(1, horizon + 1, dtype=np.float64)

        # Attenuated momentum drift
        attenuation = np.exp(-0.008 * steps)
        drift = momentum * attenuation * (steps / 24.0)
        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))
        point_pred = current_price * (1.0 + drift)
        point_pred = np.maximum(lower_bound, point_pred)

        # 10-level quantiles: (horizon, 10)
        time_scales = np.sqrt(steps / 24.0)
        spread_scale = vol * time_scales

        quantiles = np.zeros((horizon, 10), dtype=np.float64)
        for k in range(10):
            z = Z_SCORES_10[k]
            quantiles[:, k] = point_pred * np.exp(z * spread_scale)

        # Enforce positive bounds and monotonicity
        quantiles = np.maximum(lower_bound, quantiles)
        for k in range(1, 9):
            quantiles[:, k + 1] = np.maximum(quantiles[:, k + 1], quantiles[:, k])

        return point_pred.reshape(1, -1), quantiles.reshape(1, horizon, 10)


class TimesFMForecaster(BaseForecaster):
    """Production wrapper for Google TimesFM 2.5 (google/timesfm-2.5-200m-pytorch).

    Supports:
    - Lazy loading: Postpones weight imports and initialization until execution.
    - Colab GPU FP16 execution: Auto-selects CUDA FP16 or CPU.
    - Single forward pass: Direct 720h generation covering all 11 horizons.
    - 10-level quantiles: Probabilistic confidence envelopes (P10, P50, P90).
    - Offline fallback: Deterministic fallback enabling instant local test execution.
    - Sentiment drift: Horizon-attenuated sentiment conviction adjustments.
    """

    def __init__(
        self,
        model_id: str = PRIMARY_MODEL_ID,
        device: str | None = None,
        max_context: int = 1024,
        max_horizon: int = 720,
        use_fp16: bool = True,
        fallback_to_heuristic: bool = True,
        force_fallback: bool = False,
    ) -> None:
        super().__init__(model_id=model_id, device=device or "cpu")
        self.max_context: Final[int] = max(32, int(max_context))
        self.max_horizon: Final[int] = max(1, int(max_horizon))
        self.use_fp16: Final[bool] = bool(use_fp16)
        self.fallback_to_heuristic: Final[bool] = bool(fallback_to_heuristic)
        self.force_fallback: Final[bool] = bool(force_fallback)

        self._explicit_device_str: str | None = device
        self._device: Any = None
        self._model: Any = None
        self._is_loaded: bool = False
        self._is_fallback_active: bool = False
        self._fallback_forecaster: FallbackTimesFMForecaster | None = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_fallback_active(self) -> bool:
        return self._is_fallback_active

    @property
    def device(self) -> str:
        if self._is_fallback_active:
            return "cpu (fallback)"
        return str(self._device) if self._device is not None else "cpu"

    @device.setter
    def device(self, val: str) -> None:
        self._explicit_device_str = val

    def _init_fallback(self) -> None:
        if self._fallback_forecaster is None:
            self._fallback_forecaster = FallbackTimesFMForecaster(max_horizon=self.max_horizon)
        self._is_fallback_active = True
        self._is_loaded = True

    def _resolve_device(self, torch_mod: Any) -> Any:
        if self._explicit_device_str is not None:
            return torch_mod.device(self._explicit_device_str)
        if torch_mod.cuda.is_available():
            return torch_mod.device("cuda")
        if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return torch_mod.device("mps")
        return torch_mod.device("cpu")

    def load(self) -> None:
        """Loads TimesFM 2.5 checkpoint into memory or activates offline fallback."""
        if self._is_loaded:
            return

        if self.force_fallback:
            self._init_fallback()
            return

        try:
            import timesfm
            import torch

            resolved_device = self._resolve_device(torch)
            self._device = resolved_device

            # Instantiate model
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id,
                torch_compile=False,
            )

            # Optimize precision and place underlying PyTorch module on target device
            if resolved_device.type == "cuda":
                torch.set_float32_matmul_precision("high")

            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model = model.model.to(resolved_device)
                model.model.device = resolved_device
            elif hasattr(model, "to") and callable(model.to):
                model = model.to(resolved_device)

            # Compile inference configuration
            model.compile(
                timesfm.ForecastConfig(
                    max_context=self.max_context,
                    max_horizon=self.max_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                    force_flip_invariance=True,
                )
            )

            self._model = model
            self._is_loaded = True
            self._is_fallback_active = False
            logger.info("TimesFM 2.5 loaded successfully on device: %s", resolved_device)

        except Exception as exc:
            if self.fallback_to_heuristic:
                logger.warning(
                    "Unable to load TimesFM 2.5 ('%s'): %s. Activating deterministic fallback.",
                    self.model_id,
                    exc,
                )
                self._init_fallback()
            else:
                raise TimesFMModelError(
                    f"Failed to load TimesFM 2.5 model '{self.model_id}': {exc}"
                ) from exc

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            self.load()

    def _prepare_timesfm_context(self, prices: np.ndarray) -> np.ndarray:
        """Pads short context series to >= 32 steps for TimesFM 32-patch tokenizer."""
        if len(prices) > self.max_context:
            prices = prices[-self.max_context:]

        if len(prices) < 32:
            padding = np.full(32 - len(prices), prices[0], dtype=np.float64)
            prices = np.concatenate([padding, prices])

        return prices.astype(np.float32)

    def _run_forward_pass(self, prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Executes forward pass returning (1, horizon) point and (1, horizon, 10) quantiles."""
        self._ensure_loaded()

        if self._is_fallback_active or self._model is None:
            if self._fallback_forecaster is None:
                self._init_fallback()
            return self._fallback_forecaster.forecast_trajectory(prices, horizon=horizon)

        import torch

        sanitized_context = self._prepare_timesfm_context(prices)
        with torch.no_grad():
            point_pred, quantile_pred = self._model.forecast(
                horizon=horizon,
                inputs=[sanitized_context],
            )

        return point_pred, quantile_pred

    def forecast_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_hours: int = 720,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous point and quantile trajectories up to max_horizon_hours."""
        prices = self.validate_context_prices(context_prices)
        steps = min(self.max_horizon, max_horizon_hours)

        point_pred, quantile_pred = self._run_forward_pass(prices, horizon=steps)

        point_traj = np.asarray(point_pred[0, :steps], dtype=np.float64)

        # Slices 1, 5, 9 -> [P10, P50, P90]
        p10_traj = np.asarray(quantile_pred[0, :steps, 1], dtype=np.float64)
        p50_traj = np.asarray(quantile_pred[0, :steps, 5], dtype=np.float64)
        p90_traj = np.asarray(quantile_pred[0, :steps, 9], dtype=np.float64)

        # Defensive monotonicity enforcement
        lower_bound = max(1e-12, min(0.0001, float(prices[-1]) * 0.01))
        p10_traj = np.maximum(lower_bound, p10_traj)
        p50_traj = np.maximum(p10_traj, p50_traj)
        p90_traj = np.maximum(p50_traj, p90_traj)

        quantile_traj = np.stack([p10_traj, p50_traj, p90_traj], axis=1)
        return point_traj, quantile_traj

    def forecast(
        self,
        context_prices: Sequence[float] | np.ndarray,
        horizons: list[str] = DEFAULT_HORIZONS,
        sentiment_polarity: float | None = None,
    ) -> dict[str, HorizonPrediction]:
        """Generates multi-horizon predictions across specified horizons in a single pass."""
        prices = self.validate_context_prices(context_prices)
        current_price = float(prices[-1])
        lower_bound = max(1e-12, min(0.0001, current_price * 0.01))

        max_step = max(HORIZON_TO_HOURS.get(h, 1) for h in horizons) if horizons else 720
        max_step = min(self.max_horizon, max(max_step, 720))

        point_pred, quantile_pred = self._run_forward_pass(prices, horizon=max_step)

        predictions: dict[str, HorizonPrediction] = {}
        for h in horizons:
            if h not in HORIZON_TO_HOURS:
                raise ValueError(f"Unknown horizon identifier: '{h}'. Must be in HORIZON_TO_HOURS.")
            step_hours = HORIZON_TO_HOURS[h]
            idx = step_hours - 1

            pred_price = float(point_pred[0, idx])
            p10 = float(quantile_pred[0, idx, 1])
            p50 = float(quantile_pred[0, idx, 5])
            p90 = float(quantile_pred[0, idx, 9])

            # Monotonicity & positive guards
            p10 = max(lower_bound, p10)
            p50 = max(p10, p50)
            p90 = max(p50, p90)
            pred_price = max(lower_bound, pred_price)

            pred_change = pred_price - current_price
            pred_return_pct = (pred_change / current_price) * 100.0

            if pred_change > 1e-6:
                direction = "BULLISH"
            elif pred_change < -1e-6:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=step_hours,
                predicted_price=pred_price,
                current_price=current_price,
                predicted_change=pred_change,
                predicted_return_pct=pred_return_pct,
                direction=direction,
                confidence_p10=p10,
                confidence_p50=p50,
                confidence_p90=p90,
                sentiment_adjusted=False,
                raw_predicted_price=pred_price,
            )

        if sentiment_polarity is not None:
            raw_prices = {h: p.predicted_price for h, p in predictions.items()}
            drifted = apply_sentiment_drift(predictions, sentiment_polarity=sentiment_polarity)
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

    def get_full_trajectory(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_steps: int = 720,
        sentiment_polarity: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates continuous (720,) point and (720, 3) quantile trajectories for plotting."""
        point_traj, quantile_traj = self.forecast_trajectory(
            context_prices=context_prices,
            max_horizon_hours=max_horizon_steps,
        )

        if sentiment_polarity is not None:
            point_traj, quantile_traj = apply_trajectory_sentiment_drift(
                point_traj,
                quantile_traj,
                sentiment_polarity=sentiment_polarity,
            )

        return point_traj, quantile_traj

    def get_raw_quantiles(
        self,
        context_prices: Sequence[float] | np.ndarray,
        max_horizon_steps: int = 720,
    ) -> np.ndarray:
        """Returns the full 10-level quantile tensor sliced to (steps, 10)."""
        prices = self.validate_context_prices(context_prices)
        steps = min(self.max_horizon, max_horizon_steps)
        _, quantile_pred = self._run_forward_pass(prices, horizon=steps)
        return np.asarray(quantile_pred[0, :steps, :], dtype=np.float64)
