"""Adversarial Empirical Stress Testing Suite for Milestone 3 Forecasters.

Author: challenger_m3_1 (Empirical Stress Challenger)
Target Components:
- PersistenceForecaster (crypto_forecast.models.baseline)
- TimesFMForecaster & FallbackTimesFMForecaster (crypto_forecast.models.timesfm_model)
- ChronosForecaster & FallbackChronosForecaster (crypto_forecast.models.chronos_model)
- GraniteTTMForecaster & FallbackGraniteForecaster (crypto_forecast.models.granite_model)

Covers 5 Core Adversarial Stress Vectors:
1. Extreme Price Scales: Sub-penny meme coins (1e-8, 1e-12) to hyper-valued assets (1e6, 1e9).
2. Flash Crash & Parabolic Regimes: 50% single-bar crash, 99% collapse, 10x pump, 1000x surge, sawtooth.
3. Zero Volatility: Constant price series flatlines.
4. Input Validation & Short Context: Empty, NaNs, Infs, non-positive values, and short contexts (< 3 bars).
5. Sentiment Polarity Extremes: Full bullish (+1.0), full bearish (-1.0), out-of-bounds (+5.0, -10.0).

Also isolates and empirically asserts 5 specific architectural vulnerabilities:
- BUG-01: GraniteTTMForecaster / FallbackGraniteForecaster LinAlgError on single-bar context (len=1).
- BUG-02: BaseForecaster / PersistenceForecaster get_full_trajectory() missing sentiment_polarity kwarg.
- BUG-03: ChronosForecaster and GraniteTTMForecaster silent acceptance of invalid horizon identifiers.
- BUG-04: DIRECTION_THRESHOLD = 1e-6 absolute dollar blindness on sub-penny assets.
- BUG-05: FallbackTimesFMForecaster momentum suppression to 0.0 when asset price is below 1e-8.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Ensure src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest

from crypto_forecast.config import DEFAULT_HORIZONS
from crypto_forecast.models.base import BaseForecaster, HorizonPrediction
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import (
    FallbackTimesFMForecaster,
    TimesFMForecaster,
)


@pytest.fixture
def all_forecasters() -> list[tuple[str, BaseForecaster]]:
    """Returns instantiated forecasters in offline fallback execution mode."""
    return [
        ("PersistenceForecaster", PersistenceForecaster()),
        ("TimesFMForecaster", TimesFMForecaster(force_fallback=True)),
        ("ChronosForecaster", ChronosForecaster(use_fallback=True)),
        ("GraniteTTMForecaster", GraniteTTMForecaster(use_fallback=True)),
    ]


# ==============================================================================
# Suite 1: Extreme Price Scales
# ==============================================================================

class TestPriceScaleExtremes:
    """Probes numerical precision, underflows, overflows, and scale boundaries."""

    def test_subpenny_meme_coin_scale_positivity_and_monotonicity(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies sub-penny meme coin price scales ($1e-8, e.g. SHIB/PEPE) remain positive and monotonic."""
        p_subpenny = [1.0e-8, 1.05e-8, 1.02e-8] * 10  # 30 bars around $0.00000001
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_subpenny)
            assert len(preds) == len(DEFAULT_HORIZONS)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.predicted_price > 0.0, f"{name} {h} predicted_price non-positive: {pred.predicted_price}"
                assert pred.confidence_p10 > 0.0, f"{name} {h} p10 non-positive: {pred.confidence_p10}"
                assert (
                    pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                ), f"{name} {h} quantile inversion: {pred.confidence_p10} <= {pred.confidence_p50} <= {pred.confidence_p90}"
                assert np.isfinite(pred.predicted_price), f"{name} {h} non-finite predicted_price"

    def test_hyper_valued_asset_scale_stability(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies mega-cap price scales ($1e6 and $1e9) do not overflow or lose quantile ordering."""
        for scale in [1.0e6, 1.0e9]:
            p_megacap = [scale * 0.99, scale * 1.01, scale] * 15
            for name, forecaster in all_forecasters:
                preds = forecaster.forecast(p_megacap)
                for h in DEFAULT_HORIZONS:
                    pred = preds[h]
                    assert pred.predicted_price >= scale * 0.5, f"{name} {h} predicted price collapsed: {pred.predicted_price}"
                    assert (
                        pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                    ), f"{name} {h} quantile inversion on hyper scale: {pred.confidence_p10} <= {pred.confidence_p50} <= {pred.confidence_p90}"
                    assert np.isfinite(pred.confidence_p90), f"{name} {h} p90 overflowed to non-finite"

    def test_micro_subpenny_boundary_1e12(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies assets near the lower bound floor (1e-12) maintain strict quantile positivity and ordering."""
        p_micro = [1.0e-12] * 20
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_micro)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.predicted_price >= 1.0e-12
                assert pred.confidence_p10 >= 1.0e-12
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_subpenny_directional_threshold_insensitivity_flaw(self):
        """BUG-04: Fixed DIRECTION_THRESHOLD = 1e-6 causes massive relative moves on sub-penny coins to be marked NEUTRAL."""
        # A coin at $0.0000001 (1e-7) that jumps 500% to $0.0000006 (6e-7)
        current = 1.0e-7
        predicted = 6.0e-7  # +500% increase
        pred = HorizonPrediction.from_point_and_quantiles(
            horizon="1h",
            step_hours=1,
            current_price=current,
            predicted_price=predicted,
            p10=current,
            p50=predicted,
            p90=predicted * 1.2,
        )
        # Because delta is 5e-7 < 1e-6, direction is mistakenly marked NEUTRAL despite a 500% return
        assert pred.predicted_return_pct == 500.0
        assert pred.direction == "NEUTRAL", (
            "Empirically confirms BUG-04: absolute dollar threshold 1e-6 blinds directional tagging for sub-penny coins."
        )

    def test_timesfm_momentum_suppression_below_1e8_flaw(self):
        """BUG-05: FallbackTimesFMForecaster suppresses momentum drift to 0.0 when asset price is below 1e-8."""
        forecaster = FallbackTimesFMForecaster()
        # 5x rally on asset with price level 5e-9 (below 1e-8 threshold)
        p_sub = np.array([1.0e-9] * 30 + [5.0e-9] * 20)
        pred_sub, _ = forecaster.forecast_trajectory(p_sub, horizon=24)
        # Ratio of predicted price to current price is exactly 1.0 (drift suppressed)
        drift_ratio = float(pred_sub[0, -1] / p_sub[-1])
        assert math.isclose(drift_ratio, 1.0, abs_tol=1e-12), (
            "Empirically confirms BUG-05: ema_long > 1e-8 suppresses momentum continuation on sub-penny assets."
        )


# ==============================================================================
# Suite 2: Flash Crash and Parabolic Regimes
# ==============================================================================

class TestFlashCrashAndParabolicRegimes:
    """Stress tests massive single-bar volatility shocks, vertical pumps, and high-frequency swings."""

    def test_50_pct_flash_crash_cone_expansion(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies forecasters handle 50% single-bar crash without violating quantiles or negative prices."""
        p_crash = [100.0] * 50 + [50.0]
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_crash)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.confidence_p10 > 0.0, f"{name} {h} p10 non-positive on 50% crash"
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90, f"{name} {h} quantile inversion on crash"
                assert pred.current_price == 50.0

    def test_99_pct_catastrophic_crash_cone_expansion(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies forecasters survive a 99% crash (100.0 -> 1.0) with defensive lower bounds."""
        p_deep = [100.0] * 50 + [1.0]
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_deep)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.confidence_p10 >= 0.0001, f"{name} {h} lower bound violated on 99% crash"
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                assert np.isfinite(pred.confidence_p90)

    def test_10x_vertical_parabolic_pump(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies forecasters handle 10x vertical pump (10.0 -> 100.0) without quantile crossings."""
        p_pump = [10.0] * 50 + [100.0]
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_pump)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.current_price == 100.0
                assert pred.confidence_p10 > 0.0
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                assert pred.confidence_p90 >= 100.0, f"{name} {h} upper cone did not widen above pump price"

    def test_1000x_vertical_pump_extreme(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies extreme 1000x surge (0.001 -> 1000.0) remains numerically stable."""
        p_mega_pump = [0.001] * 30 + [1000.0]
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_mega_pump)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                assert np.isfinite(pred.predicted_price)

    def test_high_frequency_sawtooth_oscillations(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies violently alternating price series ([100.0, 1.0] * 25) maintains stability."""
        p_sawtooth = [100.0, 1.0] * 25
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_sawtooth)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90
                assert pred.confidence_p10 > 0.0

    def test_continuous_720_step_trajectory_monotonicity_under_shock(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies all 720 continuous trajectory steps maintain P10 <= P50 <= P90 under extreme crash and pump."""
        shock_scenarios = [
            ("50% crash", [100.0] * 50 + [50.0]),
            ("10x pump", [10.0] * 50 + [100.0]),
        ]
        for sc_name, prices in shock_scenarios:
            for name, forecaster in all_forecasters:
                point_traj, quantile_traj = forecaster.forecast_trajectory(prices, max_horizon_hours=720)
                assert len(point_traj) == 720
                assert quantile_traj.shape == (720, 3)

                inversions = np.where(~((quantile_traj[:, 0] <= quantile_traj[:, 1]) & (quantile_traj[:, 1] <= quantile_traj[:, 2])))[0]
                assert len(inversions) == 0, f"{name} {sc_name} trajectory inversions at steps: {inversions}"

                non_positives = np.where(quantile_traj[:, 0] <= 0.0)[0]
                assert len(non_positives) == 0, f"{name} {sc_name} non-positive P10 at steps: {non_positives}"


# ==============================================================================
# Suite 3: Zero Volatility & Constant Flat Series
# ==============================================================================

class TestZeroVolatilityAndFlatSeries:
    """Verifies that flat constant series avoid division-by-zero and yield neutral predictions."""

    def test_zero_volatility_flat_price_series_all_models(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies constant series produces exactly zero change and neutral direction across all models."""
        p_flat = [100.0] * 50
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_flat)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert math.isclose(pred.predicted_price, 100.0, abs_tol=1e-5), f"{name} {h} flat predicted_price != 100.0"
                assert math.isclose(pred.predicted_change, 0.0, abs_tol=1e-5), f"{name} {h} flat predicted_change != 0.0"
                assert math.isclose(pred.predicted_return_pct, 0.0, abs_tol=1e-5), f"{name} {h} flat predicted_return_pct != 0.0"
                assert pred.direction == "NEUTRAL", f"{name} {h} direction != NEUTRAL on flat series"
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_zero_volatility_subpenny_flat_price_series(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies constant sub-penny series ($0.0000085) preserves positivity and neutrality."""
        p_flat_sub = [0.0000085] * 50
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_flat_sub)
            for h in DEFAULT_HORIZONS:
                pred = preds[h]
                assert math.isclose(pred.predicted_price, 0.0000085, rel_tol=1e-5)
                assert pred.direction == "NEUTRAL"
                assert pred.confidence_p10 > 0.0


# ==============================================================================
# Suite 4: Input Validation & Boundary Context
# ==============================================================================

class TestInputValidationAndBoundaryContext:
    """Probes sanitization, NaN/Inf rejection, negative prices, and context length boundaries."""

    def test_empty_context_rejected(self, all_forecasters: list[tuple[str, BaseForecaster]]):
        """Verifies empty sequence raises ValueError across all models."""
        for name, forecaster in all_forecasters:
            with pytest.raises(ValueError, match="cannot be empty"):
                forecaster.forecast([])
            with pytest.raises(ValueError, match="cannot be empty"):
                forecaster.forecast(np.array([]))

    def test_nan_injections_rejected(self, all_forecasters: list[tuple[str, BaseForecaster]]):
        """Verifies NaN values in context prices raise ValueError across all models."""
        for name, forecaster in all_forecasters:
            with pytest.raises(ValueError, match="cannot contain NaNs"):
                forecaster.forecast([100.0, np.nan, 105.0])
            with pytest.raises(ValueError, match="cannot contain NaNs"):
                forecaster.forecast([np.nan])

    def test_inf_injections_rejected(self, all_forecasters: list[tuple[str, BaseForecaster]]):
        """Verifies infinite values raise ValueError across all models."""
        for name, forecaster in all_forecasters:
            with pytest.raises(ValueError, match="cannot contain infinite values"):
                forecaster.forecast([100.0, np.inf, 105.0])
            with pytest.raises(ValueError, match="cannot contain infinite values"):
                forecaster.forecast([100.0, -np.inf, 105.0])

    def test_negative_and_zero_prices_rejected(self, all_forecasters: list[tuple[str, BaseForecaster]]):
        """Verifies non-positive price values raise ValueError across all models."""
        for name, forecaster in all_forecasters:
            with pytest.raises(ValueError, match="strictly positive"):
                forecaster.forecast([-10.0, 50.0])
            with pytest.raises(ValueError, match="strictly positive"):
                forecaster.forecast([0.0, 50.0])

    def test_short_context_two_bars(self, all_forecasters: list[tuple[str, BaseForecaster]]):
        """Verifies all models can handle context length of exactly 2 bars ([100.0, 102.0])."""
        p_2bars = [100.0, 102.0]
        for name, forecaster in all_forecasters:
            preds = forecaster.forecast(p_2bars)
            assert len(preds) == len(DEFAULT_HORIZONS)
            point_traj, quant_traj = forecaster.forecast_trajectory(p_2bars, max_horizon_hours=720)
            assert len(point_traj) == 720
            assert quant_traj.shape == (720, 3)

    def test_granite_single_bar_context_crash_flaw(self):
        """BUG-01: GraniteTTMForecaster / FallbackGraniteForecaster crashes with LinAlgError on single-bar context."""
        forecaster = GraniteTTMForecaster(use_fallback=True)
        # BaseForecaster contract permits len(prices) >= 1, but Granite crashes due to polyfit deg=1 on len=1:
        with pytest.raises(np.linalg.LinAlgError, match="SVD did not converge in Linear Least Squares"):
            forecaster.forecast([100.0])

        with pytest.raises(np.linalg.LinAlgError, match="SVD did not converge in Linear Least Squares"):
            forecaster.forecast_trajectory([100.0], max_horizon_hours=720)

    def test_invalid_horizon_identifier_rejection_inconsistency_flaw(self):
        """BUG-03: ChronosForecaster & GraniteTTMForecaster fail to reject unknown horizon identifiers."""
        p = [100.0] * 10
        # Persistence and TimesFM properly raise ValueError on unknown horizon identifiers:
        forecaster_persist = PersistenceForecaster()
        with pytest.raises(ValueError, match="Unknown horizon identifier"):
            forecaster_persist.forecast(p, horizons=["1h", "bogus_horizon"])

        forecaster_timesfm = TimesFMForecaster(force_fallback=True)
        with pytest.raises(ValueError, match="Unknown horizon identifier"):
            forecaster_timesfm.forecast(p, horizons=["1h", "bogus_horizon"])

        # Chronos and Granite violate contract by silently accepting 'bogus_horizon' and defaulting step_hours=1:
        forecaster_chronos = ChronosForecaster(use_fallback=True)
        res_chronos = forecaster_chronos.forecast(p, horizons=["1h", "bogus_horizon"])
        assert "bogus_horizon" in res_chronos
        assert res_chronos["bogus_horizon"].step_hours == 1, (
            "Empirically confirms BUG-03: Chronos silently accepts unknown horizon identifiers."
        )

        forecaster_granite = GraniteTTMForecaster(use_fallback=True)
        res_granite = forecaster_granite.forecast(p, horizons=["1h", "bogus_horizon"])
        assert "bogus_horizon" in res_granite
        assert res_granite["bogus_horizon"].step_hours == 1, (
            "Empirically confirms BUG-03: Granite silently accepts unknown horizon identifiers."
        )


# ==============================================================================
# Suite 5: Sentiment Polarity Extremes
# ==============================================================================

class TestSentimentPolarityExtremes:
    """Stress tests directional conviction drift at polarity bounds and out-of-bounds inputs."""

    def test_full_bullish_and_bearish_polarity_drift(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies extreme sentiment (+1.0, -1.0) strictly dictates directional tags and preserves monotonicity."""
        p = [100.0] * 50
        for name, forecaster in all_forecasters:
            res_bull = forecaster.forecast(p, sentiment_polarity=1.0)
            res_bear = forecaster.forecast(p, sentiment_polarity=-1.0)

            for h in DEFAULT_HORIZONS:
                # Bullish polarity must yield predicted price > current price and BULLISH tag
                assert res_bull[h].predicted_price > 100.0, f"{name} {h} bull price <= 100"
                assert res_bull[h].direction == "BULLISH", f"{name} {h} direction != BULLISH on pol=+1.0"
                assert res_bull[h].confidence_p10 <= res_bull[h].confidence_p50 <= res_bull[h].confidence_p90

                # Bearish polarity must yield predicted price < current price and BEARISH tag
                assert res_bear[h].predicted_price < 100.0, f"{name} {h} bear price >= 100"
                assert res_bear[h].direction == "BEARISH", f"{name} {h} direction != BEARISH on pol=-1.0"
                assert res_bear[h].confidence_p10 <= res_bear[h].confidence_p50 <= res_bear[h].confidence_p90

    def test_out_of_bounds_sentiment_clipping(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies sentiment values outside [-1.0, 1.0] are safely clipped without explosion."""
        p = [100.0] * 50
        for name, forecaster in all_forecasters:
            res_extreme_bull = forecaster.forecast(p, sentiment_polarity=5.0)
            res_clamped_bull = forecaster.forecast(p, sentiment_polarity=1.0)

            res_extreme_bear = forecaster.forecast(p, sentiment_polarity=-10.0)
            res_clamped_bear = forecaster.forecast(p, sentiment_polarity=-1.0)

            for h in DEFAULT_HORIZONS:
                assert math.isclose(
                    res_extreme_bull[h].predicted_price, res_clamped_bull[h].predicted_price, rel_tol=1e-5
                ), f"{name} {h} out-of-bounds +5.0 was not clipped to +1.0"
                assert math.isclose(
                    res_extreme_bear[h].predicted_price, res_clamped_bear[h].predicted_price, rel_tol=1e-5
                ), f"{name} {h} out-of-bounds -10.0 was not clipped to -1.0"

    def test_subpenny_sentiment_drift_monotonicity(
        self, all_forecasters: list[tuple[str, BaseForecaster]]
    ):
        """Verifies extreme sentiment applied to sub-penny coin preserves positive lower bounds and monotonicity."""
        p_sub = [1.0e-8] * 50
        for name, forecaster in all_forecasters:
            res = forecaster.forecast(p_sub, sentiment_polarity=-1.0)
            for h in DEFAULT_HORIZONS:
                pred = res[h]
                assert pred.confidence_p10 > 0.0, f"{name} {h} p10 non-positive on subpenny bear sentiment"
                assert pred.confidence_p10 <= pred.confidence_p50 <= pred.confidence_p90

    def test_persistence_get_full_trajectory_sentiment_kwarg_flaw(self):
        """BUG-02: BaseForecaster / PersistenceForecaster get_full_trajectory() rejects sentiment_polarity kwarg."""
        forecaster = PersistenceForecaster()
        p = [100.0] * 20
        with pytest.raises(TypeError, match="unexpected keyword argument 'sentiment_polarity'"):
            forecaster.get_full_trajectory(p, max_horizon_steps=720, sentiment_polarity=0.5)

    def test_get_full_trajectory_continuous_sentiment_drift(self):
        """Verifies 720-step continuous trajectory drift preserves quantile ordering on TimesFM, Chronos, and Granite."""
        p = [100.0] * 50
        models = [
            ("TimesFM", TimesFMForecaster(force_fallback=True)),
            ("Chronos", ChronosForecaster(use_fallback=True)),
            ("Granite", GraniteTTMForecaster(use_fallback=True)),
        ]
        for pol in [-1.0, 1.0]:
            for name, m in models:
                point_traj, quantile_traj = m.get_full_trajectory(p, max_horizon_steps=720, sentiment_polarity=pol)
                assert len(point_traj) == 720
                assert quantile_traj.shape == (720, 3)
                inversions = np.where(~((quantile_traj[:, 0] <= quantile_traj[:, 1]) & (quantile_traj[:, 1] <= quantile_traj[:, 2])))[0]
                assert len(inversions) == 0, f"{name} pol={pol} trajectory inversions at steps: {inversions}"
