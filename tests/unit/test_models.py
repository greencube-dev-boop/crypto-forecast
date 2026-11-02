"""Unit Tests for Multi-Horizon Forecasting Engine.

Comprehensive unit test suite covering:
1. HorizonPrediction dataclass schema, validation invariants, and helper methods.
2. BaseForecaster abstract base class, input sanitization, and trajectory extraction.
3. PersistenceForecaster random-walk baseline model, volatility cones, and trajectory shape.
4. TimesFMForecaster single-pass 720h trajectory, 10-level quantiles, and fallback engine.
5. ChronosForecaster dual-resolution routing, 64-step ceiling handling, and fallback engine.
6. GraniteTTMForecaster adaptive patching, conformal intervals, and fallback engine.
7. Sentiment drift integration across all forecasters.
8. Model polymorphism, price scale resilience ($0.0000085 to $150,000), and boundary stress.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

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
    BaseForecaster,
    HorizonPrediction,
)
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import (
    ChronosForecaster,
)
from crypto_forecast.models.granite_model import (
    GraniteTTMForecaster,
)
from crypto_forecast.models.timesfm_model import (
    TimesFMForecaster,
)

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_prices_500() -> np.ndarray:
    """Generates 500 synthetic realistic price steps starting at 50,000."""
    np.random.seed(42)
    base = 50000.0
    returns = np.random.normal(0.0002, 0.015, size=500)
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def flat_prices_100() -> np.ndarray:
    """Generates constant price series with zero volatility."""
    return np.full(100, 100.0)


# ==============================================================================
# 1. Test HorizonPrediction Dataclass
# ==============================================================================

class TestHorizonPredictionDataclass:
    """Verifies schema, positional initialization, serialization, and helper methods."""

    def test_dataclass_fields_and_types(self):
        pred = HorizonPrediction(
            horizon="1h",
            step_hours=1,
            predicted_price=105.0,
            current_price=100.0,
            predicted_change=5.0,
            predicted_return_pct=5.0,
            direction="BULLISH",
            confidence_p10=95.0,
            confidence_p50=105.0,
            confidence_p90=115.0,
        )
        assert pred.horizon == "1h"
        assert pred.step_hours == 1
        assert pred.predicted_price == 105.0
        assert pred.current_price == 100.0
        assert pred.predicted_change == 5.0
        assert pred.predicted_return_pct == 5.0
        assert pred.direction == "BULLISH"
        assert pred.confidence_p10 == 95.0
        assert pred.confidence_p50 == 105.0
        assert pred.confidence_p90 == 115.0
        assert pred.sentiment_adjusted is False
        assert pred.raw_predicted_price is None

    def test_to_dict_structure_and_json_safety(self):
        pred = HorizonPrediction(
            horizon="24h",
            step_hours=24,
            predicted_price=105.5,
            current_price=100.0,
            predicted_change=5.5,
            predicted_return_pct=5.5,
            direction="BULLISH",
            confidence_p10=90.0,
            confidence_p50=105.5,
            confidence_p90=120.0,
            sentiment_adjusted=True,
            raw_predicted_price=103.0,
        )
        d = pred.to_dict()
        assert isinstance(d, dict)
        assert d["horizon"] == "24h"
        assert d["step_hours"] == 24
        assert d["predicted_price"] == 105.5
        assert d["current_price"] == 100.0
        assert d["direction"] == "BULLISH"
        assert d["sentiment_adjusted"] is True
        assert d["raw_predicted_price"] == 103.0

    def test_from_point_and_quantiles_bullish(self):
        pred = HorizonPrediction.from_point_and_quantiles(
            horizon="1h",
            step_hours=1,
            current_price=100.0,
            predicted_price=102.5,
            p10=98.0,
            p50=102.5,
            p90=107.0,
        )
        assert pred.direction == "BULLISH"
        assert math.isclose(pred.predicted_change, 2.5, abs_tol=1e-5)
        assert math.isclose(pred.predicted_return_pct, 2.5, abs_tol=1e-4)

    def test_from_point_and_quantiles_bearish(self):
        pred = HorizonPrediction.from_point_and_quantiles(
            horizon="1h",
            step_hours=1,
            current_price=100.0,
            predicted_price=97.0,
            p10=93.0,
            p50=97.0,
            p90=101.0,
        )
        assert pred.direction == "BEARISH"
        assert math.isclose(pred.predicted_change, -3.0, abs_tol=1e-5)
        assert math.isclose(pred.predicted_return_pct, -3.0, abs_tol=1e-4)

    def test_from_point_and_quantiles_neutral(self):
        pred = HorizonPrediction.from_point_and_quantiles(
            horizon="1h",
            step_hours=1,
            current_price=100.0,
            predicted_price=100.0000001,
            p10=95.0,
            p50=100.0,
            p90=105.0,
        )
        assert pred.direction == "NEUTRAL"
        assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-5)

    def test_uncertainty_spread(self):
        pred = HorizonPrediction(
            horizon="1h",
            step_hours=1,
            predicted_price=100.0,
            current_price=100.0,
            predicted_change=0.0,
            predicted_return_pct=0.0,
            direction="NEUTRAL",
            confidence_p10=95.0,
            confidence_p50=100.0,
            confidence_p90=105.0,
        )
        assert math.isclose(pred.uncertainty_spread(), 10.0, abs_tol=1e-5)

    def test_is_bullish_and_is_bearish_helpers(self):
        pred_bull = HorizonPrediction.from_point_and_quantiles(
            "1h", 1, 100.0, 105.0, 95.0, 105.0, 115.0
        )
        pred_bear = HorizonPrediction.from_point_and_quantiles(
            "1h", 1, 100.0, 95.0, 85.0, 95.0, 105.0
        )
        pred_neut = HorizonPrediction.from_point_and_quantiles(
            "1h", 1, 100.0, 100.0, 90.0, 100.0, 110.0
        )

        assert pred_bull.is_bullish() and not pred_bull.is_bearish()
        assert pred_bear.is_bearish() and not pred_bear.is_bullish()
        assert not pred_neut.is_bullish() and not pred_neut.is_bearish()


# ==============================================================================
# 2. Test HorizonPrediction Validations
# ==============================================================================

class TestHorizonPredictionValidations:
    """Verifies strict invariant checks on instantiation."""

    def test_non_positive_current_price_raises(self):
        with pytest.raises(ValueError, match="current_price must be strictly positive"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=10.0, current_price=0.0,
                predicted_change=10.0, predicted_return_pct=0.0, direction="BULLISH",
                confidence_p10=5.0, confidence_p50=10.0, confidence_p90=15.0,
            )

    def test_non_positive_predicted_price_raises(self):
        with pytest.raises(ValueError, match="predicted_price must be strictly positive"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=-1.0, current_price=10.0,
                predicted_change=-11.0, predicted_return_pct=-110.0, direction="BEARISH",
                confidence_p10=5.0, confidence_p50=10.0, confidence_p90=15.0,
            )

    def test_non_positive_p10_raises(self):
        with pytest.raises(ValueError, match="confidence_p10 must be strictly positive"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=10.0, current_price=10.0,
                predicted_change=0.0, predicted_return_pct=0.0, direction="NEUTRAL",
                confidence_p10=0.0, confidence_p50=10.0, confidence_p90=15.0,
            )

    def test_quantile_inversion_p10_gt_p50_raises(self):
        with pytest.raises(ValueError, match="Quantile monotonicity violated"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=10.0, current_price=10.0,
                predicted_change=0.0, predicted_return_pct=0.0, direction="NEUTRAL",
                confidence_p10=12.0, confidence_p50=10.0, confidence_p90=15.0,
            )

    def test_quantile_inversion_p50_gt_p90_raises(self):
        with pytest.raises(ValueError, match="Quantile monotonicity violated"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=10.0, current_price=10.0,
                predicted_change=0.0, predicted_return_pct=0.0, direction="NEUTRAL",
                confidence_p10=5.0, confidence_p50=16.0, confidence_p90=15.0,
            )

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="Invalid direction"):
            HorizonPrediction(
                horizon="1h", step_hours=1, predicted_price=10.0, current_price=10.0,
                predicted_change=0.0, predicted_return_pct=0.0, direction="SIDEWAYS",
                confidence_p10=5.0, confidence_p50=10.0, confidence_p90=15.0,
            )


# ==============================================================================
# 3. Test BaseForecaster Abstract Base Class & Contracts
# ==============================================================================

class TestBaseForecasterContract:
    """Verifies ABC abstraction and input validation methods."""

    def test_cannot_instantiate_abstract_base_forecaster(self):
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseForecaster("test")  # type: ignore

    def test_concrete_subclass_must_implement_forecast_trajectory(self):
        class IncompleteForecaster(BaseForecaster):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteForecaster("incomplete")  # type: ignore

    def test_context_validation_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BaseForecaster.validate_context_prices([])

        with pytest.raises(ValueError, match="cannot be empty"):
            BaseForecaster.validate_context_prices(np.array([]))

    def test_context_validation_nan_raises(self):
        with pytest.raises(ValueError, match="cannot contain NaNs"):
            BaseForecaster.validate_context_prices([100.0, np.nan, 102.0])

    def test_context_validation_inf_raises(self):
        with pytest.raises(ValueError, match="cannot contain infinite values"):
            BaseForecaster.validate_context_prices([100.0, np.inf, 102.0])

    def test_context_validation_non_positive_raises(self):
        with pytest.raises(ValueError, match="must be strictly positive"):
            BaseForecaster.validate_context_prices([100.0, 0.0, 105.0])

        with pytest.raises(ValueError, match="must be strictly positive"):
            BaseForecaster.validate_context_prices([100.0, -5.0, 105.0])

    def test_discrete_extraction_maps_indices_accurately(self):
        forecaster = PersistenceForecaster()
        point_traj = np.arange(1, 721, dtype=np.float64) + 100.0
        # Quantiles: p10 = p - 10, p50 = p, p90 = p + 10
        q_traj = np.column_stack([point_traj - 10.0, point_traj, point_traj + 10.0])

        preds = forecaster.extract_horizon_predictions(
            point_trajectory=point_traj,
            quantile_trajectory=q_traj,
            current_price=100.0,
            horizons=["1h", "24h", "30d"],
        )
        assert math.isclose(preds["1h"].predicted_price, 101.0)
        assert math.isclose(preds["24h"].predicted_price, 124.0)
        assert math.isclose(preds["30d"].predicted_price, 820.0)

    def test_discrete_extraction_unknown_horizon_raises(self):
        forecaster = PersistenceForecaster()
        point_traj = np.full(720, 100.0)
        q_traj = np.column_stack([point_traj - 5.0, point_traj, point_traj + 5.0])

        with pytest.raises(ValueError, match="Unknown horizon identifier"):
            forecaster.extract_horizon_predictions(
                point_trajectory=point_traj,
                quantile_trajectory=q_traj,
                current_price=100.0,
                horizons=["99d"],
            )

    def test_discrete_extraction_exceeds_trajectory_raises(self):
        forecaster = PersistenceForecaster()
        point_traj = np.full(10, 100.0)
        q_traj = np.column_stack([point_traj - 5.0, point_traj, point_traj + 5.0])

        with pytest.raises(ValueError, match="exceeds trajectory length"):
            forecaster.extract_horizon_predictions(
                point_trajectory=point_traj,
                quantile_trajectory=q_traj,
                current_price=100.0,
                horizons=["24h"],
            )


# ==============================================================================
# 4. Test PersistenceForecaster Baseline
# ==============================================================================

class TestPersistenceForecaster:
    """Verifies martingale persistence baseline properties."""

    def test_persistence_forecast_identity_and_neutral(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast(synthetic_prices_500)
        last_price = float(synthetic_prices_500[-1])

        for pred in preds.values():
            assert math.isclose(pred.predicted_price, last_price, abs_tol=1e-5)
            assert math.isclose(pred.current_price, last_price, abs_tol=1e-5)
            assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-5)
            assert math.isclose(pred.predicted_return_pct, 0.0, abs_tol=1e-5)
            assert pred.direction == "NEUTRAL"

    def test_persistence_all_11_horizons(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast(synthetic_prices_500)
        assert len(preds) == 11
        assert list(preds.keys()) == DEFAULT_HORIZONS

    def test_persistence_uncertainty_cone_widening(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast(synthetic_prices_500)

        spread_1h = preds["1h"].confidence_p90 - preds["1h"].confidence_p10
        spread_24h = preds["24h"].confidence_p90 - preds["24h"].confidence_p10
        spread_30d = preds["30d"].confidence_p90 - preds["30d"].confidence_p10

        assert spread_1h > 0.0
        assert spread_24h > spread_1h
        assert spread_30d > spread_24h

    def test_persistence_quantile_monotonicity(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast(synthetic_prices_500)

        for pred in preds.values():
            assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_persistence_trajectory_shape_and_values(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        pt, qt = forecaster.get_full_trajectory(synthetic_prices_500, max_horizon_steps=720)

        assert pt.shape == (720,)
        assert qt.shape == (720, 3)
        last_price = float(synthetic_prices_500[-1])
        assert np.allclose(pt, last_price)
        assert np.all(qt[:, 0] <= qt[:, 1])
        assert np.all(qt[:, 1] <= qt[:, 2])

    def test_persistence_single_price_volatility_fallback(self):
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast([100.0])
        assert len(preds) == 11
        assert preds["1h"].confidence_p10 < 100.0 < preds["1h"].confidence_p90

    def test_persistence_sentiment_drift_integration(self, synthetic_prices_500):
        forecaster = PersistenceForecaster()
        base = forecaster.forecast(synthetic_prices_500)
        bull = forecaster.forecast(synthetic_prices_500, sentiment_polarity=0.7)
        bear = forecaster.forecast(synthetic_prices_500, sentiment_polarity=-0.7)

        for h in ["1h", "24h", "7d", "30d"]:
            assert bull[h].predicted_price > base[h].predicted_price
            assert bear[h].predicted_price < base[h].predicted_price
            assert bull[h].sentiment_adjusted is True
            assert bull[h].raw_predicted_price == base[h].predicted_price
            assert bull[h].confidence_p10 <= bull[h].confidence_p50 <= bull[h].confidence_p90


# ==============================================================================
# 5. Test TimesFMForecaster Primary Model
# ==============================================================================

class TestTimesFMForecaster:
    """Verifies TimesFM 2.5 foundation model wrapper and offline fallback."""

    def test_timesfm_lazy_loading_and_properties(self):
        forecaster = TimesFMForecaster(force_fallback=True)
        assert forecaster.name == "google/timesfm-2.5-200m-pytorch"
        assert not forecaster.is_loaded

        forecaster.load()
        assert forecaster.is_loaded
        assert forecaster.is_fallback_active
        assert "fallback" in forecaster.device

    def test_timesfm_forecast_11_horizons(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        assert len(preds) == 11
        for h in DEFAULT_HORIZONS:
            assert h in preds
            p = preds[h]
            assert p.horizon == h
            assert p.step_hours == HORIZON_TO_HOURS[h]
            assert p.predicted_price > 0.0
            assert p.current_price == float(synthetic_prices_500[-1])
            assert p.direction in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_timesfm_single_pass_trajectory(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        pt, qt = forecaster.get_full_trajectory(synthetic_prices_500, max_horizon_steps=720)

        assert pt.shape == (720,)
        assert qt.shape == (720, 3)
        assert np.all(pt > 0.0)
        assert np.all(qt[:, 0] <= qt[:, 1])
        assert np.all(qt[:, 1] <= qt[:, 2])

    def test_timesfm_quantile_monotonicity(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        for pred in preds.values():
            assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_timesfm_uncertainty_cone_widening(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        spread_1h = preds["1h"].uncertainty_spread()
        spread_24h = preds["24h"].uncertainty_spread()
        spread_30d = preds["30d"].uncertainty_spread()

        assert spread_1h > 0.0
        assert spread_24h > spread_1h
        assert spread_30d > spread_24h

    def test_timesfm_flat_price_neutral(self, flat_prices_100):
        forecaster = TimesFMForecaster(force_fallback=True)
        preds = forecaster.forecast(flat_prices_100)

        for pred in preds.values():
            assert math.isclose(pred.predicted_price, 100.0, abs_tol=1e-4)
            assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-4)
            assert pred.direction == "NEUTRAL"

    def test_timesfm_short_context_padding(self):
        forecaster = TimesFMForecaster(force_fallback=True)
        short = np.array([50.0, 51.0, 52.0, 53.0])
        preds = forecaster.forecast(short)
        assert len(preds) == 11
        for pred in preds.values():
            assert pred.predicted_price > 0.0
            assert pred.current_price == 53.0

    def test_timesfm_sentiment_drift(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        base = forecaster.forecast(synthetic_prices_500)
        bull = forecaster.forecast(synthetic_prices_500, sentiment_polarity=0.8)
        bear = forecaster.forecast(synthetic_prices_500, sentiment_polarity=-0.8)

        for h in ["1h", "24h", "7d", "30d"]:
            assert bull[h].predicted_price > base[h].predicted_price
            assert bear[h].predicted_price < base[h].predicted_price
            assert bull[h].confidence_p10 <= bull[h].confidence_p50 <= bull[h].confidence_p90

    def test_timesfm_raw_quantiles(self, synthetic_prices_500):
        forecaster = TimesFMForecaster(force_fallback=True)
        raw_q = forecaster.get_raw_quantiles(synthetic_prices_500, max_horizon_steps=720)
        assert raw_q.shape == (720, 10)
        assert np.all(raw_q[:, 1] <= raw_q[:, 5])
        assert np.all(raw_q[:, 5] <= raw_q[:, 9])


# ==============================================================================
# 6. Test ChronosForecaster Dual-Resolution Model
# ==============================================================================

class TestChronosForecaster:
    """Verifies Chronos dual-resolution adapter and fallback."""

    def test_chronos_lazy_loading_and_fallback(self):
        forecaster = ChronosForecaster(use_fallback=True)
        assert not forecaster.is_loaded
        forecaster._ensure_loaded()
        assert forecaster.is_loaded
        assert forecaster.is_fallback_active

    def test_chronos_forecast_11_horizons(self, synthetic_prices_500):
        forecaster = ChronosForecaster(use_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        assert len(preds) == 11
        for h in DEFAULT_HORIZONS:
            assert h in preds
            p = preds[h]
            assert p.predicted_price > 0.0
            assert p.current_price == float(synthetic_prices_500[-1])
            assert p.confidence_p10 <= p.confidence_p50 <= p.confidence_p90

    def test_chronos_dual_resolution_trajectory(self, synthetic_prices_500):
        forecaster = ChronosForecaster(use_fallback=True)
        pt, qt = forecaster.get_full_trajectory(synthetic_prices_500, max_horizon_steps=720)

        assert pt.shape == (720,)
        assert qt.shape == (720, 3)
        assert np.all(pt > 0.0)
        assert np.all(qt[:, 0] <= qt[:, 1])
        assert np.all(qt[:, 1] <= qt[:, 2])

    def test_chronos_quantile_monotonicity(self, synthetic_prices_500):
        forecaster = ChronosForecaster(use_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        for pred in preds.values():
            assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_chronos_sentiment_drift(self, synthetic_prices_500):
        forecaster = ChronosForecaster(use_fallback=True)
        base = forecaster.forecast(synthetic_prices_500)
        bull = forecaster.forecast(synthetic_prices_500, sentiment_polarity=0.75)
        bear = forecaster.forecast(synthetic_prices_500, sentiment_polarity=-0.75)

        for h in ["1h", "24h", "7d", "30d"]:
            assert bull[h].predicted_price > base[h].predicted_price
            assert bear[h].predicted_price < base[h].predicted_price
            assert bull[h].confidence_p10 <= bull[h].confidence_p50 <= bull[h].confidence_p90

    def test_chronos_flat_price_neutral(self, flat_prices_100):
        forecaster = ChronosForecaster(use_fallback=True)
        preds = forecaster.forecast(flat_prices_100)

        for pred in preds.values():
            assert math.isclose(pred.predicted_price, 100.0, abs_tol=1e-4)
            assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-4)
            assert pred.direction == "NEUTRAL"


# ==============================================================================
# 7. Test GraniteTTMForecaster Model
# ==============================================================================

class TestGraniteTTMForecaster:
    """Verifies Granite TTM adapter and conformal quantiles."""

    def test_granite_lazy_loading_and_fallback(self):
        forecaster = GraniteTTMForecaster(use_fallback=True)
        assert not forecaster.is_loaded
        forecaster._ensure_loaded()
        assert forecaster.is_loaded
        assert forecaster.is_fallback_active

    def test_granite_forecast_11_horizons(self, synthetic_prices_500):
        forecaster = GraniteTTMForecaster(use_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        assert len(preds) == 11
        for h in DEFAULT_HORIZONS:
            assert h in preds
            p = preds[h]
            assert p.predicted_price > 0.0
            assert p.confidence_p10 <= p.confidence_p50 <= p.confidence_p90

    def test_granite_conformal_quantiles_monotonicity(self, synthetic_prices_500):
        forecaster = GraniteTTMForecaster(use_fallback=True)
        preds = forecaster.forecast(synthetic_prices_500)

        for pred in preds.values():
            assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
            assert pred.uncertainty_spread() > 0.0

    def test_granite_trajectory_shape(self, synthetic_prices_500):
        forecaster = GraniteTTMForecaster(use_fallback=True)
        pt, qt = forecaster.get_full_trajectory(synthetic_prices_500, max_horizon_steps=720)

        assert pt.shape == (720,)
        assert qt.shape == (720, 3)
        assert np.all(pt > 0.0)
        assert np.all(qt[:, 0] <= qt[:, 1])
        assert np.all(qt[:, 1] <= qt[:, 2])

    def test_granite_sentiment_drift(self, synthetic_prices_500):
        forecaster = GraniteTTMForecaster(use_fallback=True)
        base = forecaster.forecast(synthetic_prices_500)
        bull = forecaster.forecast(synthetic_prices_500, sentiment_polarity=0.7)
        bear = forecaster.forecast(synthetic_prices_500, sentiment_polarity=-0.7)

        for h in ["1h", "24h", "7d", "30d"]:
            assert bull[h].predicted_price > base[h].predicted_price
            assert bear[h].predicted_price < base[h].predicted_price
            assert bull[h].confidence_p10 <= bull[h].confidence_p50 <= bull[h].confidence_p90


# ==============================================================================
# 8. Test Model Polymorphism & Price Scale Invariance
# ==============================================================================

class TestModelPolymorphismAndBenchmarking:
    """Verifies polymorphic BaseForecaster adherence and scale resilience."""

    def test_all_models_inherit_from_base_forecaster(self):
        models = [
            PersistenceForecaster(),
            TimesFMForecaster(force_fallback=True),
            ChronosForecaster(use_fallback=True),
            GraniteTTMForecaster(use_fallback=True),
        ]
        for m in models:
            assert isinstance(m, BaseForecaster)
            assert hasattr(m, "forecast")
            assert hasattr(m, "get_full_trajectory")
            assert hasattr(m, "forecast_trajectory")

    def test_polymorphic_forecast_call(self, synthetic_prices_500):
        models = [
            PersistenceForecaster(),
            TimesFMForecaster(force_fallback=True),
            ChronosForecaster(use_fallback=True),
            GraniteTTMForecaster(use_fallback=True),
        ]
        for m in models:
            preds = m.forecast(synthetic_prices_500)
            assert len(preds) == 11
            assert set(preds.keys()) == set(DEFAULT_HORIZONS)

    def test_price_scale_invariance_subpenny_to_sixfigures(self):
        models = [
            PersistenceForecaster(),
            TimesFMForecaster(force_fallback=True),
            ChronosForecaster(use_fallback=True),
            GraniteTTMForecaster(use_fallback=True),
        ]
        subpenny = np.full(50, 0.0000085)
        sixfigures = np.full(50, 150000.0)

        for m in models:
            preds_sub = m.forecast(subpenny)
            preds_high = m.forecast(sixfigures)

            for h in DEFAULT_HORIZONS:
                assert preds_sub[h].predicted_price > 0.0
                assert preds_sub[h].confidence_p10 > 0.0
                assert preds_high[h].predicted_price > 100000.0
                assert preds_high[h].confidence_p90 >= preds_high[h].predicted_price
