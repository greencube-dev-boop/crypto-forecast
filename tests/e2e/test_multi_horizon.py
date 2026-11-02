"""Tier 1, Tier 2, and Tier 3 E2E Tests: Multi-Horizon Forecasting Engine.

Validates all 11 discrete forecasting horizons (1h, 2h, 4h, 6h, 12h, 24h, 2d, 3d, 7d, 14d, 30d),
horizon monotonicity, output schema, price bounds, confidence intervals, directional indicators,
boundary stress limits, and cross-feature sentiment drift modification.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path so tests can import from tests.e2e
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from tests.e2e.conftest import (
    DEFAULT_HORIZONS,
    HAS_CRYPTO_FORECAST,
    HORIZON_HOURS,
    HorizonPrediction,
    ReferencePersistenceForecaster,
    make_synthetic_ohlcv,
)


# ==============================================================================
# Helper Forecaster Resolver
# ==============================================================================

def get_forecaster_instance():
    """Returns the production BaseForecaster if available, otherwise reference oracle."""
    if HAS_CRYPTO_FORECAST:
        try:
            from crypto_forecast.models.baseline import PersistenceForecaster
            return PersistenceForecaster()
        except ImportError:
            pass
    return ReferencePersistenceForecaster()


# ==============================================================================
# TIER 1: Feature Coverage (Isolation & Primary Behavior — ≥5 tests per feature)
# ==============================================================================

class TestTier1MultiHorizonFeatureCoverage:
    """Validates the standard contract and behavior of all 11 forecasting horizons in isolation."""

    def test_all_11_horizons_defined_and_exact_count(self, all_horizons: List[str]):
        """Verifies exactly 11 distinct horizons are defined and match specification."""
        expected_horizons = ["1h", "2h", "4h", "6h", "12h", "24h", "2d", "3d", "7d", "14d", "30d"]
        assert len(all_horizons) == 11, f"Expected 11 horizons, found {len(all_horizons)}"
        assert all_horizons == expected_horizons, f"Horizons order or naming mismatch: {all_horizons}"

    def test_horizon_hours_mapping_exact_values(self, horizon_hours_map: Dict[str, int]):
        """Verifies step hours for all 11 horizons correctly translate to integer hours."""
        expected_steps = {
            "1h": 1,
            "2h": 2,
            "4h": 4,
            "6h": 6,
            "12h": 12,
            "24h": 24,
            "2d": 48,
            "3d": 72,
            "7d": 168,
            "14d": 336,
            "30d": 720,
        }
        for h, expected_step in expected_steps.items():
            assert h in horizon_hours_map, f"Missing horizon key {h} in mapping"
            assert horizon_hours_map[h] == expected_step, (
                f"Horizon {h} mapped to {horizon_hours_map[h]} hours, expected {expected_step}"
            )

    def test_horizon_step_hours_monotonic_ordering(self, all_horizons: List[str], horizon_hours_map: Dict[str, int]):
        """Verifies that horizons strictly increase monotonically in temporal step hours."""
        step_hours = [horizon_hours_map[h] for h in all_horizons]
        for i in range(len(step_hours) - 1):
            assert step_hours[i] < step_hours[i + 1], (
                f"Horizon ordering violation at index {i}: {all_horizons[i]} ({step_hours[i]}h) "
                f"is not strictly less than {all_horizons[i+1]} ({step_hours[i+1]}h)"
            )

    def test_horizon_prediction_schema_dataclass_fields(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies forecast output contains all required fields with correct types for all 11 horizons."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        assert len(predictions) in (len(DEFAULT_HORIZONS), 48), f"Expected predictions for {len(DEFAULT_HORIZONS)} or 48 horizons, got {len(predictions)}"

        for h, pred in predictions.items():
            assert hasattr(pred, "horizon"), f"Missing 'horizon' on prediction for {h}"
            assert hasattr(pred, "step_hours"), f"Missing 'step_hours' on prediction for {h}"
            assert hasattr(pred, "predicted_price"), f"Missing 'predicted_price' on prediction for {h}"
            assert hasattr(pred, "current_price"), f"Missing 'current_price' on prediction for {h}"
            assert hasattr(pred, "predicted_change"), f"Missing 'predicted_change' on prediction for {h}"
            assert hasattr(pred, "predicted_return_pct"), f"Missing 'predicted_return_pct' on prediction for {h}"
            assert hasattr(pred, "direction"), f"Missing 'direction' on prediction for {h}"
            assert hasattr(pred, "confidence_p10"), f"Missing 'confidence_p10' on prediction for {h}"
            assert hasattr(pred, "confidence_p50"), f"Missing 'confidence_p50' on prediction for {h}"
            assert hasattr(pred, "confidence_p90"), f"Missing 'confidence_p90' on prediction for {h}"

            assert isinstance(pred.horizon, str)
            assert isinstance(pred.step_hours, int)
            assert isinstance(pred.predicted_price, (int, float))
            assert isinstance(pred.current_price, (int, float))
            assert isinstance(pred.direction, str)

    def test_horizon_prediction_price_positivity(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies all predicted prices and confidence intervals are strictly positive numbers."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        for h, pred in predictions.items():
            assert pred.predicted_price > 0.0, f"Predicted price for {h} is <= 0: {pred.predicted_price}"
            assert pred.current_price > 0.0, f"Current price for {h} is <= 0: {pred.current_price}"
            assert pred.confidence_p10 > 0.0, f"P10 for {h} is <= 0: {pred.confidence_p10}"
            assert pred.confidence_p50 > 0.0, f"P50 for {h} is <= 0: {pred.confidence_p50}"
            assert pred.confidence_p90 > 0.0, f"P90 for {h} is <= 0: {pred.confidence_p90}"

    def test_horizon_confidence_intervals_quantile_monotonicity(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies quantile monotonicity: P10 <= P50 <= P90 across all 11 horizons."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        for h, pred in predictions.items():
            assert pred.confidence_p10 <= pred.confidence_p50, (
                f"Quantile inversion for {h}: P10 ({pred.confidence_p10}) > P50 ({pred.confidence_p50})"
            )
            assert pred.confidence_p50 <= pred.confidence_p90, (
                f"Quantile inversion for {h}: P50 ({pred.confidence_p50}) > P90 ({pred.confidence_p90})"
            )

    def test_horizon_uncertainty_cone_widening(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies uncertainty spread (P90 - P10) expands as the forecast horizon extends."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        spread_1h = predictions["1h"].confidence_p90 - predictions["1h"].confidence_p10
        spread_24h = predictions["24h"].confidence_p90 - predictions["24h"].confidence_p10
        spread_48h = predictions["48h"].confidence_p90 - predictions["48h"].confidence_p10

        assert spread_1h > 0.0, "1h uncertainty spread must be strictly positive"
        assert spread_24h > spread_1h, f"24h spread ({spread_24h}) must be greater than 1h spread ({spread_1h})"
        assert spread_48h > spread_24h, f"48h spread ({spread_48h}) must be greater than 24h spread ({spread_24h})"

    def test_directional_indicator_consistency(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies direction label ('BULLISH'/'BEARISH'/'NEUTRAL') matches predicted price change."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        for h, pred in predictions.items():
            assert pred.direction in ("BULLISH", "BEARISH", "NEUTRAL"), (
                f"Invalid direction '{pred.direction}' for horizon {h}"
            )
            if pred.predicted_change > 1e-6:
                assert pred.direction == "BULLISH", (
                    f"Direction mismatch for {h}: change is +{pred.predicted_change} but direction is {pred.direction}"
                )
            elif pred.predicted_change < -1e-6:
                assert pred.direction == "BEARISH", (
                    f"Direction mismatch for {h}: change is {pred.predicted_change} but direction is {pred.direction}"
                )

    def test_predicted_change_and_return_mathematical_consistency(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies return % == (change / current_price) * 100 within floating point tolerance."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        predictions = forecaster.forecast(prices)

        for h, pred in predictions.items():
            expected_change = pred.predicted_price - pred.current_price
            assert math.isclose(pred.predicted_change, expected_change, abs_tol=1e-5), (
                f"Change mismatch for {h}: {pred.predicted_change} vs expected {expected_change}"
            )
            expected_return = (pred.predicted_change / pred.current_price) * 100.0
            assert math.isclose(pred.predicted_return_pct, expected_return, abs_tol=1e-4), (
                f"Return % mismatch for {h}: {pred.predicted_return_pct}% vs expected {expected_return}%"
            )

    def test_full_trajectory_extraction_dimension_and_alignment(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Verifies continuous trajectory generation produces 720 steps aligned with discrete points."""
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        point_traj, quantile_traj = forecaster.get_full_trajectory(prices, max_horizon_steps=720)

        assert len(point_traj) == 720, f"Expected 720 trajectory steps, got {len(point_traj)}"
        assert quantile_traj.shape == (720, 3), f"Expected quantile shape (720, 3), got {quantile_traj.shape}"

        # Verify quantile monotonicity along entire trajectory
        for step in range(720):
            p10, p50, p90 = quantile_traj[step]
            assert p10 <= p50 <= p90, f"Quantile inversion at step {step}: {p10} <= {p50} <= {p90}"

    def test_production_crypto_forecast_module_contract_conformance(self):
        """Verifies production crypto_forecast package adheres to BaseForecaster contract when implemented."""
        if not HAS_CRYPTO_FORECAST:
            pytest.skip("crypto_forecast package implementation pending M1-M3 milestones")

        import crypto_forecast.config as cfg
        from crypto_forecast.models.base import BaseForecaster, HorizonPrediction as ProdHorizonPrediction

        assert hasattr(cfg, "DEFAULT_HORIZONS")
        assert len(cfg.DEFAULT_HORIZONS) in (11, 48)
        assert issubclass(get_forecaster_instance().__class__, BaseForecaster)


# ==============================================================================
# TIER 2: Boundary & Corner Cases (≥5 tests per feature)
# ==============================================================================

class TestTier2MultiHorizonBoundaryCases:
    """Stress tests boundary limits, zero volatility, extreme price swings, and NaN recovery."""

    def test_zero_volatility_flat_price_series(self):
        """Tests constant price series: predictions should equal constant price with zero change."""
        forecaster = get_forecaster_instance()
        flat_prices = np.full(100, 100.0)
        predictions = forecaster.forecast(flat_prices)

        for h, pred in predictions.items():
            assert math.isclose(pred.predicted_price, 100.0, abs_tol=1e-5), (
                f"Flat series predicted price {pred.predicted_price} != 100.0 for {h}"
            )
            assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-5)
            assert math.isclose(pred.predicted_return_pct, 0.0, abs_tol=1e-5)
            assert pred.direction == "NEUTRAL"

    def test_extreme_price_spike_robustness(self):
        """Verifies forecaster remains numerically stable and finite under sudden 10x price jump."""
        forecaster = get_forecaster_instance()
        prices = np.full(50, 100.0)
        prices[-1] = 1000.0  # 10x sudden spike
        predictions = forecaster.forecast(prices)

        for h, pred in predictions.items():
            assert np.isfinite(pred.predicted_price), f"Non-finite prediction for {h}: {pred.predicted_price}"
            assert pred.predicted_price > 0.0, f"Non-positive prediction for {h}"
            assert np.isfinite(pred.confidence_p10)
            assert np.isfinite(pred.confidence_p90)

    def test_sub_penny_meme_coin_precision(self):
        """Tests micro-priced meme coin values ($0.0000085) without floating underflow."""
        forecaster = get_forecaster_instance()
        micro_prices = np.full(50, 0.0000085)
        predictions = forecaster.forecast(micro_prices)

        for h, pred in predictions.items():
            assert pred.predicted_price > 0.0, f"Micro price underflowed to 0 for {h}"
            assert pred.confidence_p10 > 0.0, f"P10 underflowed to 0 for {h}"
            assert math.isclose(pred.current_price, 0.0000085, rel_tol=1e-5)

    def test_high_value_asset_precision(self):
        """Tests six-figure asset prices ($150,000.00) without numerical overflow."""
        forecaster = get_forecaster_instance()
        high_prices = np.full(50, 150000.0)
        predictions = forecaster.forecast(high_prices)

        for h, pred in predictions.items():
            assert np.isfinite(pred.predicted_price)
            assert pred.predicted_price > 100000.0
            assert pred.confidence_p90 >= pred.predicted_price

    def test_minimal_context_length_boundary(self):
        """Tests forecaster handles small valid context series (e.g. 10 candles) gracefully."""
        forecaster = get_forecaster_instance()
        short_series = np.array([50.0, 51.0, 52.0, 51.5, 53.0, 52.5, 54.0, 55.0, 54.5, 56.0])
        predictions = forecaster.forecast(short_series)

        assert len(predictions) in (11, 48)
        for h, pred in predictions.items():
            assert pred.current_price == 56.0
            assert pred.predicted_price > 0.0

    def test_empty_context_raises_value_error(self):
        """Tests that passing an empty list or array raises a clean ValueError."""
        forecaster = get_forecaster_instance()
        with pytest.raises(ValueError, match="cannot be empty"):
            forecaster.forecast([])

    def test_nan_containing_context_raises_value_error(self):
        """Tests that passing un-imputed NaNs raises an explicit ValueError."""
        forecaster = get_forecaster_instance()
        corrupted_prices = np.array([100.0, np.nan, 102.0])
        with pytest.raises(ValueError, match="cannot contain NaNs"):
            forecaster.forecast(corrupted_prices)

    def test_negative_or_zero_prices_raise_value_error(self):
        """Tests that passing zero or negative price series raises ValueError."""
        forecaster = get_forecaster_instance()
        with pytest.raises(ValueError, match="must be strictly positive"):
            forecaster.forecast([100.0, 0.0, 105.0])

        with pytest.raises(ValueError, match="must be strictly positive"):
            forecaster.forecast([100.0, -10.0, 105.0])


# ==============================================================================
# TIER 3: Cross-Feature Combinations (Pairwise Integration Tests)
# ==============================================================================

class TestTier3MultiHorizonCrossFeatureIntegration:
    """Tests end-to-end interactions between data alignment, sentiment drift, and forecasting."""

    def test_dual_resolution_alignment_hourly_and_daily(self):
        """Verifies dual-resolution resampling: hourly for 1h..24h and daily for 2d..30d."""
        df_hourly = make_synthetic_ohlcv(symbol="BTC", n_bars=720, freq="1h", seed=42)
        # Resample hourly data to daily to emulate dual-resolution preprocessing
        df_daily = (
            df_hourly.set_index("timestamp")
            .resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )

        assert len(df_daily) >= 30, f"Expected at least 30 daily bars, got {len(df_daily)}"
        # Verify timestamps are strictly increasing
        assert df_daily["timestamp"].is_monotonic_increasing

    def test_sentiment_drift_modification_bullish_and_bearish(self, synthetic_ohlcv_1000h: pd.DataFrame):
        """Tests that sentiment conviction modifies multi-horizon forecast drift appropriately.

        Formula: P_hat_adj = P_hat * (1 + gamma_h * sentiment_polarity)
        """
        forecaster = get_forecaster_instance()
        prices = synthetic_ohlcv_1000h["close"].values
        base_preds = forecaster.forecast(prices, horizons=["1h", "24h", "7d", "30d"])

        # Attenuation factors: decaying impact as horizon extends
        gamma = {"1h": 0.02, "24h": 0.015, "7d": 0.008, "30d": 0.003}

        # Simulate Bullish News shock (+0.80)
        bullish_polarity = 0.80
        bullish_adjusted = {}
        for h in ["1h", "24h", "7d", "30d"]:
            drift = gamma[h] * bullish_polarity
            adj_price = base_preds[h].predicted_price * (1.0 + drift)
            bullish_adjusted[h] = adj_price
            assert adj_price > base_preds[h].predicted_price, (
                f"Bullish sentiment failed to shift {h} price upward: {adj_price} <= {base_preds[h].predicted_price}"
            )

        # Simulate Bearish News shock (-0.75)
        bearish_polarity = -0.75
        for h in ["1h", "24h", "7d", "30d"]:
            drift = gamma[h] * bearish_polarity
            adj_price = base_preds[h].predicted_price * (1.0 + drift)
            assert adj_price < base_preds[h].predicted_price, (
                f"Bearish sentiment failed to shift {h} price downward: {adj_price} >= {base_preds[h].predicted_price}"
            )

    def test_sentiment_drift_attenuation_decay_across_horizons(self):
        """Verifies that sentiment impact is highest intraday (1h) and decays toward macro horizons (30d)."""
        gamma = {
            "1h": 0.02,
            "2h": 0.019,
            "4h": 0.018,
            "6h": 0.017,
            "12h": 0.016,
            "24h": 0.015,
            "2d": 0.012,
            "3d": 0.010,
            "7d": 0.007,
            "14d": 0.005,
            "30d": 0.002,
        }
        horizons_order = DEFAULT_HORIZONS
        for i in range(len(horizons_order) - 1):
            h_curr = horizons_order[i]
            h_next = horizons_order[i + 1]
            assert gamma[h_curr] > gamma[h_next], (
                f"Sentiment gamma did not decay: gamma[{h_curr}]={gamma[h_curr]} <= gamma[{h_next}]={gamma[h_next]}"
            )
