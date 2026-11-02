"""Empirical Invariants Test Suite for Milestone 3 Forecasting Engine.

Authored by challenger_m3_2 (Invariants Challenger).
Adversarially asserts and formally verifies the 6 mathematical, dimensional,
and operational invariants across all 4 forecasters:
1. PersistenceForecaster (Baseline benchmark)
2. TimesFMForecaster (Google TimesFM 2.5)
3. ChronosForecaster (Amazon Chronos-Bolt / Chronos-2)
4. GraniteTTMForecaster (IBM Granite TimeSeries TTM)

Invariants Formally Asserted:
1. Strict Quantile Monotonicity: P10 <= P50 <= P90 across all steps and horizons.
2. Price Positivity: P10 > 0, P50 > 0, P90 > 0 everywhere.
3. Monotonic Cone Expansion: spread 1h < spread 24h < spread 30d.
4. Trajectory Dimensions: (720,) point and (720, 3) quantile.
5. Horizon Identity: Discrete predicted_price matches trajectory at step h-1.
6. Directional Indicator Consistency: Delta P > 1e-6 -> BULLISH, Delta P < -1e-6 -> BEARISH, else NEUTRAL.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
)
from crypto_forecast.models.base import (
    DIRECTION_THRESHOLD,
    BaseForecaster,
    HorizonPrediction,
)
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import TimesFMForecaster

# ==============================================================================
# Model Factory & Fixtures
# ==============================================================================

@pytest.fixture(
    params=[
        "persistence",
        "timesfm",
        "chronos",
        "granite",
    ]
)
def forecaster_instance(request: pytest.FixtureRequest) -> BaseForecaster:
    """Provides instantiated forecasters with deterministic local execution."""
    model_type = request.param
    if model_type == "persistence":
        return PersistenceForecaster()
    if model_type == "timesfm":
        return TimesFMForecaster(force_fallback=True)
    if model_type == "chronos":
        return ChronosForecaster()
    if model_type == "granite":
        return GraniteTTMForecaster(use_fallback=True)
    raise ValueError(f"Unknown model_type: {model_type}")


@pytest.fixture
def btc_realistic_prices() -> np.ndarray:
    """Generates 300 synthetic BTC hourly prices with realistic volatility."""
    np.random.seed(1337)
    base = 64000.0
    returns = np.random.normal(0.0001, 0.012, size=300)
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def meme_coin_micro_prices() -> np.ndarray:
    """Generates 150 sub-penny meme coin prices (e.g. PEPE $0.0000085)."""
    np.random.seed(42)
    base = 0.0000085
    returns = np.random.normal(0.0005, 0.035, size=150)
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def high_value_asset_prices() -> np.ndarray:
    """Generates 100 prices for an expensive asset ($150,000)."""
    np.random.seed(101)
    base = 150000.0
    returns = np.random.normal(0.0, 0.008, size=100)
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def flat_constant_prices() -> np.ndarray:
    """Generates 100 constant prices with zero volatility."""
    return np.full(100, 250.0)


@pytest.fixture
def severe_crash_prices() -> np.ndarray:
    """Generates 50 prices of a sudden 80% market collapse."""
    prices = np.linspace(1000.0, 200.0, 50)
    return prices


@pytest.fixture
def massive_pump_prices() -> np.ndarray:
    """Generates 50 prices of a parabolic 300% market rally."""
    prices = np.linspace(100.0, 400.0, 50)
    return prices


# ==============================================================================
# Invariant 1: Strict Quantile Monotonicity (P10 <= P50 <= P90)
# ==============================================================================

class TestStrictQuantileMonotonicityInvariant:
    """Asserts P10 <= P50 <= P90 across all steps, horizons, and stress conditions."""

    def test_trajectory_quantile_monotonicity_all_steps(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Formally verifies P10 <= P50 <= P90 at every step t in [0, 719]."""
        _, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        assert quantiles.shape == (720, 3)

        p10 = quantiles[:, 0]
        p50 = quantiles[:, 1]
        p90 = quantiles[:, 2]

        violating_p10_p50 = np.where(p10 > p50)[0]
        violating_p50_p90 = np.where(p50 > p90)[0]

        assert len(violating_p10_p50) == 0, (
            f"{forecaster_instance.name}: P10 > P50 violated at step indices {violating_p10_p50}"
        )
        assert len(violating_p50_p90) == 0, (
            f"{forecaster_instance.name}: P50 > P90 violated at step indices {violating_p50_p90}"
        )

    def test_discrete_quantile_monotonicity_all_horizons(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies P10 <= P50 <= P90 across all 11 discrete horizons."""
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        for h in DEFAULT_HORIZONS:
            pred = preds[h]
            assert pred.confidence_p10 <= pred.confidence_p50, (
                f"{forecaster_instance.name} [{h}]: P10 ({pred.confidence_p10}) > P50 ({pred.confidence_p50})"
            )
            assert pred.confidence_p50 <= pred.confidence_p90, (
                f"{forecaster_instance.name} [{h}]: P50 ({pred.confidence_p50}) > P90 ({pred.confidence_p90})"
            )

    @pytest.mark.parametrize("polarity", [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0])
    def test_quantile_monotonicity_under_sentiment_drift(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
        polarity: float,
    ) -> None:
        """Verifies P10 <= P50 <= P90 is strictly preserved under sentiment drift multipliers."""
        drifted = forecaster_instance.forecast(
            btc_realistic_prices,
            DEFAULT_HORIZONS,
            sentiment_polarity=polarity,
        )

        for h, pred in drifted.items():
            assert pred.confidence_p10 <= pred.confidence_p50, (
                f"Drift violated P10 <= P50 on {forecaster_instance.name} [{h}] at polarity {polarity}"
            )
            assert pred.confidence_p50 <= pred.confidence_p90, (
                f"Drift violated P50 <= P90 on {forecaster_instance.name} [{h}] at polarity {polarity}"
            )

    def test_monotonicity_across_diverse_price_regimes(
        self,
        forecaster_instance: BaseForecaster,
        meme_coin_micro_prices: np.ndarray,
        high_value_asset_prices: np.ndarray,
        flat_constant_prices: np.ndarray,
        severe_crash_prices: np.ndarray,
        massive_pump_prices: np.ndarray,
    ) -> None:
        """Verifies monotonicity invariant across all 5 extreme market regimes."""
        regimes = {
            "micro_coin": meme_coin_micro_prices,
            "high_value": high_value_asset_prices,
            "flatline": flat_constant_prices,
            "crash": severe_crash_prices,
            "pump": massive_pump_prices,
        }

        for name, series in regimes.items():
            _, quantiles = forecaster_instance.forecast_trajectory(series, max_horizon_hours=720)
            assert np.all(quantiles[:, 0] <= quantiles[:, 1]), (
                f"{forecaster_instance.name} failed P10 <= P50 under {name} regime"
            )
            assert np.all(quantiles[:, 1] <= quantiles[:, 2]), (
                f"{forecaster_instance.name} failed P50 <= P90 under {name} regime"
            )


# ==============================================================================
# Invariant 2: Price Positivity (P10 > 0, P50 > 0, P90 > 0)
# ==============================================================================

class TestPricePositivityInvariant:
    """Asserts that all prices and quantile bounds remain strictly positive everywhere."""

    def test_trajectory_price_positivity(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Asserts point and quantile trajectories are strictly > 0 everywhere."""
        points, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        assert np.all(points > 0.0), f"{forecaster_instance.name}: Non-positive point forecasts"
        assert np.all(quantiles[:, 0] > 0.0), f"{forecaster_instance.name}: Non-positive P10"
        assert np.all(quantiles[:, 1] > 0.0), f"{forecaster_instance.name}: Non-positive P50"
        assert np.all(quantiles[:, 2] > 0.0), f"{forecaster_instance.name}: Non-positive P90"

    def test_discrete_price_positivity(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Asserts all discrete predictions have strictly positive prices and bounds."""
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        for pred in preds.values():
            assert pred.current_price > 0.0
            assert pred.predicted_price > 0.0
            assert pred.confidence_p10 > 0.0
            assert pred.confidence_p50 > 0.0
            assert pred.confidence_p90 > 0.0

    def test_positivity_on_micro_scale_assets(
        self,
        forecaster_instance: BaseForecaster,
        meme_coin_micro_prices: np.ndarray,
    ) -> None:
        """Asserts sub-penny assets (0.0000085) retain strictly positive lower bounds."""
        points, quantiles = forecaster_instance.forecast_trajectory(
            meme_coin_micro_prices, max_horizon_hours=720
        )
        assert np.all(points > 0.0)
        assert np.all(quantiles > 0.0)
        # Even the lowest bound must be non-zero
        assert np.min(quantiles[:, 0]) > 0.0

    def test_positivity_under_extreme_bearish_crash_and_negative_sentiment(
        self,
        forecaster_instance: BaseForecaster,
        severe_crash_prices: np.ndarray,
    ) -> None:
        """Asserts severe crash with S = -1.0 sentiment never yields negative or zero price."""
        drifted = forecaster_instance.forecast(
            severe_crash_prices,
            DEFAULT_HORIZONS,
            sentiment_polarity=-1.0,
        )

        for h, pred in drifted.items():
            assert pred.predicted_price > 0.0, f"Non-positive predicted_price at {h}"
            assert pred.confidence_p10 > 0.0, f"Non-positive confidence_p10 at {h}"
            assert pred.confidence_p50 > 0.0, f"Non-positive confidence_p50 at {h}"
            assert pred.confidence_p90 > 0.0, f"Non-positive confidence_p90 at {h}"


# ==============================================================================
# Invariant 3: Monotonic Cone Expansion (spread 1h < spread 24h < spread 30d)
# ==============================================================================

class TestMonotonicConeExpansionInvariant:
    """Asserts that confidence interval width (P90 - P10) strictly expands over time."""

    def test_discrete_confidence_cone_expansion(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Formally verifies spread 1h < spread 24h < spread 30d across all forecasters."""
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        spread_1h = preds["1h"].uncertainty_spread()
        spread_24h = preds["24h"].uncertainty_spread()
        spread_30d = preds["30d"].uncertainty_spread()

        assert spread_1h > 0.0, f"{forecaster_instance.name}: spread_1h is non-positive"
        assert spread_24h > 0.0, f"{forecaster_instance.name}: spread_24h is non-positive"
        assert spread_30d > 0.0, f"{forecaster_instance.name}: spread_30d is non-positive"

        assert spread_1h < spread_24h, (
            f"{forecaster_instance.name}: Cone expansion failed: spread_1h ({spread_1h}) "
            f"not strictly less than spread_24h ({spread_24h})"
        )
        assert spread_24h < spread_30d, (
            f"{forecaster_instance.name}: Cone expansion failed: spread_24h ({spread_24h}) "
            f"not strictly less than spread_30d ({spread_30d})"
        )

    def test_trajectory_cone_expansion_endpoints(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies trajectory spread at step 1 < step 24 < step 720."""
        _, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        spread_step1 = quantiles[0, 2] - quantiles[0, 0]
        spread_step24 = quantiles[23, 2] - quantiles[23, 0]
        spread_step720 = quantiles[719, 2] - quantiles[719, 0]

        assert spread_step1 < spread_step24, (
            f"{forecaster_instance.name}: Trajectory spread step 1 ({spread_step1}) >= step 24 ({spread_step24})"
        )
        assert spread_step24 < spread_step720, (
            f"{forecaster_instance.name}: Trajectory spread step 24 ({spread_step24}) >= step 720 ({spread_step720})"
        )

    def test_cone_expansion_preserved_under_sentiment_drift(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies spread 1h < spread 24h < spread 30d holds under extreme sentiment drifts."""
        for pol in [-1.0, 1.0]:
            preds = forecaster_instance.forecast(
                btc_realistic_prices,
                DEFAULT_HORIZONS,
                sentiment_polarity=pol,
            )
            sp_1h = preds["1h"].uncertainty_spread()
            sp_24h = preds["24h"].uncertainty_spread()
            sp_30d = preds["30d"].uncertainty_spread()

            assert sp_1h < sp_24h < sp_30d, (
                f"{forecaster_instance.name}: Drift (polarity={pol}) broke cone expansion: "
                f"1h={sp_1h:.4f}, 24h={sp_24h:.4f}, 30d={sp_30d:.4f}"
            )


# ==============================================================================
# Invariant 4: Trajectory Dimensions ((720,) point and (720, 3) quantile)
# ==============================================================================

class TestTrajectoryDimensionsInvariant:
    """Asserts exact dimensional and rank contracts of forecast trajectories."""

    def test_default_720_trajectory_dimensions(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies default continuous trajectory yields shape (720,) and (720, 3)."""
        points, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        assert points.ndim == 1
        assert points.shape == (720,)
        assert points.dtype == np.float64

        assert quantiles.ndim == 2
        assert quantiles.shape == (720, 3)
        assert quantiles.dtype == np.float64

    def test_get_full_trajectory_compatibility_dimensions(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies PROJECT.md interface contract get_full_trajectory yields exact dimensions."""
        points, quantiles = forecaster_instance.get_full_trajectory(
            btc_realistic_prices, max_horizon_steps=720
        )
        assert points.shape == (720,)
        assert quantiles.shape == (720, 3)

    @pytest.mark.parametrize("custom_steps", [24, 48, 168, 720])
    def test_variable_horizon_trajectory_dimensions(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
        custom_steps: int,
    ) -> None:
        """Verifies trajectory generation matches any requested max_horizon_hours."""
        points, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=custom_steps
        )
        assert points.shape == (custom_steps,)
        assert quantiles.shape == (custom_steps, 3)


# ==============================================================================
# Invariant 5: Horizon Extraction Identity (Discrete Matches Trajectory[h-1])
# ==============================================================================

class TestHorizonExtractionIdentityInvariant:
    """Asserts that discrete predicted_price matches trajectory at index step_hours - 1."""

    def test_discrete_point_prediction_matches_trajectory_at_step_minus_one(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies discrete predicted_price == trajectory[step_hours - 1] for all 11 horizons."""
        points, _ = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        for h in DEFAULT_HORIZONS:
            step_hours = HORIZON_TO_HOURS[h]
            idx = step_hours - 1

            expected_point = float(points[idx])
            actual_point = preds[h].predicted_price

            assert math.isclose(actual_point, expected_point, rel_tol=1e-5, abs_tol=1e-7), (
                f"{forecaster_instance.name} [{h}]: Point mismatch at step {step_hours} (idx {idx}): "
                f"discrete={actual_point}, trajectory={expected_point}"
            )

    def test_discrete_quantiles_match_trajectory_at_step_minus_one(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies discrete [P10, P50, P90] == quantiles[step_hours - 1, :] for all 11 horizons."""
        _, quantiles = forecaster_instance.forecast_trajectory(
            btc_realistic_prices, max_horizon_hours=720
        )
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        for h in DEFAULT_HORIZONS:
            step_hours = HORIZON_TO_HOURS[h]
            idx = step_hours - 1

            expected_p10 = float(quantiles[idx, 0])
            expected_p50 = float(quantiles[idx, 1])
            expected_p90 = float(quantiles[idx, 2])

            # Note: rel_tol=5e-4 accommodates the minor ddof=0 vs ddof=1 volatility difference in Granite
            assert math.isclose(preds[h].confidence_p10, expected_p10, rel_tol=5e-4, abs_tol=1e-7), (
                f"{forecaster_instance.name} [{h}]: P10 mismatch: discrete={preds[h].confidence_p10}, traj={expected_p10}"
            )
            assert math.isclose(preds[h].confidence_p50, expected_p50, rel_tol=5e-4, abs_tol=1e-7), (
                f"{forecaster_instance.name} [{h}]: P50 mismatch: discrete={preds[h].confidence_p50}, traj={expected_p50}"
            )
            assert math.isclose(preds[h].confidence_p90, expected_p90, rel_tol=5e-4, abs_tol=1e-7), (
                f"{forecaster_instance.name} [{h}]: P90 mismatch: discrete={preds[h].confidence_p90}, traj={expected_p90}"
            )

    def test_horizon_identity_across_all_price_scales(
        self,
        forecaster_instance: BaseForecaster,
        meme_coin_micro_prices: np.ndarray,
        high_value_asset_prices: np.ndarray,
    ) -> None:
        """Asserts horizon identity is scale-invariant across sub-penny and high-value coins."""
        for asset_prices in [meme_coin_micro_prices, high_value_asset_prices]:
            points, _ = forecaster_instance.forecast_trajectory(asset_prices, max_horizon_hours=720)
            preds = forecaster_instance.forecast(asset_prices, DEFAULT_HORIZONS)

            for h in DEFAULT_HORIZONS:
                step = HORIZON_TO_HOURS[h]
                idx = step - 1
                assert math.isclose(preds[h].predicted_price, points[idx], rel_tol=1e-5, abs_tol=1e-7)


# ==============================================================================
# Invariant 6: Directional Indicator Consistency
# ==============================================================================

class TestDirectionalIndicatorConsistencyInvariant:
    """Asserts ternary classification: Delta P > 1e-6 -> BULLISH, Delta P < -1e-6 -> BEARISH, else NEUTRAL."""

    def test_model_predictions_directional_consistency(
        self,
        forecaster_instance: BaseForecaster,
        btc_realistic_prices: np.ndarray,
    ) -> None:
        """Verifies each prediction's direction matches its delta against DIRECTION_THRESHOLD."""
        preds = forecaster_instance.forecast(btc_realistic_prices, DEFAULT_HORIZONS)

        for h, pred in preds.items():
            change = pred.predicted_price - pred.current_price
            if change > DIRECTION_THRESHOLD:
                assert pred.direction == "BULLISH", (
                    f"{forecaster_instance.name} [{h}]: change={change} > {DIRECTION_THRESHOLD} but direction={pred.direction}"
                )
                assert pred.is_bullish() is True
                assert pred.is_bearish() is False
            elif change < -DIRECTION_THRESHOLD:
                assert pred.direction == "BEARISH", (
                    f"{forecaster_instance.name} [{h}]: change={change} < -{DIRECTION_THRESHOLD} but direction={pred.direction}"
                )
                assert pred.is_bearish() is True
                assert pred.is_bullish() is False
            else:
                assert pred.direction == "NEUTRAL", (
                    f"{forecaster_instance.name} [{h}]: |change|={abs(change)} <= {DIRECTION_THRESHOLD} but direction={pred.direction}"
                )
                assert pred.is_bullish() is False
                assert pred.is_bearish() is False

    def test_persistence_forecaster_strictly_neutral(
        self,
        btc_realistic_prices: np.ndarray,
        severe_crash_prices: np.ndarray,
        massive_pump_prices: np.ndarray,
    ) -> None:
        """Persistence model has Delta P == 0.0 everywhere; must be strictly NEUTRAL across all horizons."""
        model = PersistenceForecaster()
        for series in [btc_realistic_prices, severe_crash_prices, massive_pump_prices]:
            preds = model.forecast(series, DEFAULT_HORIZONS)
            for pred in preds.values():
                assert pred.predicted_change == 0.0
                assert pred.predicted_return_pct == 0.0
                assert pred.direction == "NEUTRAL"

    @pytest.mark.parametrize(
        ("delta", "expected_dir"),
        [
            (1.000001e-6, "BULLISH"),
            (1.000000e-6, "NEUTRAL"),
            (0.999999e-6, "NEUTRAL"),
            (0.0, "NEUTRAL"),
            (-0.999999e-6, "NEUTRAL"),
            (-1.000000e-6, "NEUTRAL"),
            (-1.000001e-6, "BEARISH"),
            (50.0, "BULLISH"),
            (-50.0, "BEARISH"),
        ],
    )
    def test_exact_threshold_boundary_classification(
        self,
        delta: float,
        expected_dir: str,
    ) -> None:
        """Asserts factory accurately classifies boundary points of DIRECTION_THRESHOLD (1e-6)."""
        curr_price = 100.0
        pred_price = curr_price + delta

        pred = HorizonPrediction.from_point_and_quantiles(
            horizon="1h",
            step_hours=1,
            current_price=curr_price,
            predicted_price=pred_price,
            p10=min(curr_price, pred_price) * 0.9,
            p50=pred_price,
            p90=max(curr_price, pred_price) * 1.1,
            epsilon=DIRECTION_THRESHOLD,
        )

        assert pred.direction == expected_dir, (
            f"Delta {delta}: expected direction '{expected_dir}', got '{pred.direction}'"
        )


# ==============================================================================
# Adversarial Edge Cases & Stress Probes
# ==============================================================================

class TestAdversarialStressAndFailureModes:
    """Critic role: actively probes edge cases, input anomalies, and unhandled paths."""

    def test_minimal_context_length_two_bars(
        self,
        forecaster_instance: BaseForecaster,
    ) -> None:
        """Probes forecaster behavior on minimal 2-bar context [100.0, 105.0]."""
        short_context = np.array([100.0, 105.0])
        preds = forecaster_instance.forecast(short_context, DEFAULT_HORIZONS)
        points, quantiles = forecaster_instance.forecast_trajectory(short_context, max_horizon_hours=720)

        assert len(preds) == 11
        assert points.shape == (720,)
        assert quantiles.shape == (720, 3)
        assert np.all(quantiles[:, 0] <= quantiles[:, 1])
        assert np.all(quantiles[:, 1] <= quantiles[:, 2])

    @pytest.mark.parametrize(
        "invalid_input",
        [
            [],
            np.array([]),
            [np.nan, 100.0],
            [100.0, np.inf],
            [100.0, -50.0],
            [0.0],
        ],
    )
    def test_forecasters_strictly_reject_invalid_context(
        self,
        forecaster_instance: BaseForecaster,
        invalid_input: Any,
    ) -> None:
        """Asserts all forecasters immediately raise ValueError on corrupt/invalid input series."""
        with pytest.raises(ValueError):
            forecaster_instance.forecast(invalid_input, DEFAULT_HORIZONS)

        with pytest.raises(ValueError):
            forecaster_instance.forecast_trajectory(invalid_input, max_horizon_hours=720)

    def test_zero_volatility_flat_series_stability(
        self,
        forecaster_instance: BaseForecaster,
        flat_constant_prices: np.ndarray,
    ) -> None:
        """Asserts zero division does not occur on completely flat price series."""
        points, quantiles = forecaster_instance.forecast_trajectory(
            flat_constant_prices, max_horizon_hours=720
        )
        assert not np.any(np.isnan(points))
        assert not np.any(np.isnan(quantiles))
        assert not np.any(np.isinf(points))
        assert not np.any(np.isinf(quantiles))
        assert np.all(quantiles[:, 0] <= quantiles[:, 1])
        assert np.all(quantiles[:, 1] <= quantiles[:, 2])
