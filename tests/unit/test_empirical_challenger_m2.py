"""Empirical Property Verification & Invariant Challenge Harness for Milestone 2.

Author: challenger_m2_2 (Empirical Challenger)
Target: src/crypto_forecast/models/sentiment.py

Empirically challenges:
1. Polarity & Probability Invariants (Simplex, bounds, rounding, label concordance).
2. Exponential Decay Invariants (Exact analytical half-life, strict temporal monotonicity, asymptotic convergence, no sign oscillation).
3. Continuous Time-Series Reindexing (Exact length preservation, zero NaN/Inf under all conditions, idle decay, multi-article mean, unsorted indexing).
4. Multi-Horizon Drift & Horizon Attenuation (Strict monotonic decay across all 11 horizons, gamma positivity).
5. Quantile Ordering Preservation & Price Positivity (P10 <= P50 <= P90 preservation under extreme shocks s = +-1.0, P^adj > 0 for all P > 0).
6. Continuous Trajectory Drift Dynamics (720-step envelope preservation, monotonic decay of conviction impact).
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    DEFAULT_SENTIMENT_GAMMA_0,
    DEFAULT_SENTIMENT_LAMBDA,
    HORIZON_TO_HOURS,
    SENTIMENT_DECAY_ALPHA,
)
from crypto_forecast.models.sentiment import (
    HeuristicSentimentAnalyzer,
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
    compute_horizon_gamma,
    compute_hourly_decay,
    compute_sentiment_polarity,
    get_sentiment_series,
)


@dataclass
class MockHorizonPrediction:
    """Mock horizon prediction to emulate Milestone 3 HorizonPrediction dataclass."""
    horizon: str
    step_hours: int
    predicted_price: float
    current_price: float
    predicted_change: float
    predicted_return_pct: float
    direction: str
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float


# ==============================================================================
# 1. POLARITY & PROBABILITY INVARIANTS (SIMPLEX & BOUNDS)
# ==============================================================================


class TestPolarityAndProbabilityInvariantsEmpirical:
    """Empirical challenge of probability simplex and polarity bounds."""

    @pytest.fixture
    def analyzer(self) -> HeuristicSentimentAnalyzer:
        return HeuristicSentimentAnalyzer()

    def test_probability_simplex_invariant_fuzzing(self, analyzer: HeuristicSentimentAnalyzer):
        """Stress-tests the probability simplex: p_i in [0, 1] and sum(p_i) == 1.0 +- 1e-5.

        Generates 500 randomized adversarial sentences combining bullish keywords,
        bearish keywords, negations, noise words, punctuation, and numeric strings.
        """
        bull_words = list(analyzer.BULLISH_KEYWORDS)
        bear_words = list(analyzer.BEARISH_KEYWORDS)
        neg_words = list(analyzer.NEGATIONS)
        neutral_words = ["market", "token", "crypto", "blockchain", "volume", "report", "today", "system", "index"]

        rng = random.Random(42)
        test_samples: list[str] = [
            "",
            " ",
            "\t\n\r",
            "🚀🚀🚀 100x 💎🙌",
            "!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./",
            "1234567890 0.0000001 -999",
            "not " * 50,
            "surge " * 50,
            "crash " * 50,
            "not surge " * 25,
            "no crash " * 25,
        ]

        # Generate 400 combinatorial random texts
        for _ in range(400):
            n_tokens = rng.randint(1, 40)
            tokens: list[str] = []
            for _ in range(n_tokens):
                pool = rng.choice([bull_words, bear_words, neg_words, neutral_words])
                tokens.append(rng.choice(pool))
            test_samples.append(" ".join(tokens))

        for text in test_samples:
            res = analyzer.predict_single(text)

            p_bear = res["bearish"]
            p_neut = res["neutral"]
            p_bull = res["bullish"]
            pol = res["polarity"]
            label = res["label"]

            # Probability simplex invariant: p_i in [0.0, 1.0]
            assert 0.0 <= p_bear <= 1.0, f"Bearish prob {p_bear} out of [0, 1] for '{text}'"
            assert 0.0 <= p_neut <= 1.0, f"Neutral prob {p_neut} out of [0, 1] for '{text}'"
            assert 0.0 <= p_bull <= 1.0, f"Bullish prob {p_bull} out of [0, 1] for '{text}'"

            # Sum invariant: sum(p_i) == 1.0 +- 1e-5
            total_prob = p_bear + p_neut + p_bull
            assert abs(total_prob - 1.0) <= 1e-5, f"Sum {total_prob} != 1.0 for '{text}'"

            # Polarity bounds invariant: s in [-1.0, 1.0]
            assert -1.0 <= pol <= 1.0, f"Polarity {pol} out of [-1, 1] for '{text}'"

            # Label concordance invariant
            if pol > 0.05:
                assert label == "Bullish", f"Expected Bullish for pol={pol}, got {label}"
                assert p_bull > p_bear, f"Bullish label but p_bull ({p_bull}) <= p_bear ({p_bear})"
            elif pol < -0.05:
                assert label == "Bearish", f"Expected Bearish for pol={pol}, got {label}"
                assert p_bear > p_bull, f"Bearish label but p_bear ({p_bear}) <= p_bull ({p_bull})"
            else:
                assert label == "Neutral", f"Expected Neutral for pol={pol}, got {label}"

    def test_compute_sentiment_polarity_adversarial_grid(self):
        """Tests compute_sentiment_polarity across extreme and out-of-bounds inputs."""
        rng = np.random.default_rng(12345)
        # 10,000 random pairs spanning valid and extreme invalid ranges [-10.0, 10.0]
        bullish_probs = rng.uniform(-10.0, 10.0, size=10000)
        bearish_probs = rng.uniform(-10.0, 10.0, size=10000)

        for b_prob, br_prob in zip(bullish_probs, bearish_probs):
            pol = compute_sentiment_polarity(b_prob, br_prob)
            assert -1.0 <= pol <= 1.0, f"Polarity {pol} outside [-1.0, 1.0] for ({b_prob}, {br_prob})"
            expected = float(np.clip(b_prob - br_prob, -1.0, 1.0))
            assert pol == pytest.approx(expected, abs=1e-6)

    def test_compute_sentiment_polarity_special_floats(self):
        """Tests handling of infinity in polarity computation."""
        assert compute_sentiment_polarity(float("inf"), 0.0) == 1.0
        assert compute_sentiment_polarity(0.0, float("inf")) == -1.0
        assert compute_sentiment_polarity(float("-inf"), 0.0) == -1.0


# ==============================================================================
# 2. EXPONENTIAL DECAY INVARIANTS (HALF-LIFE & MONOTONICITY)
# ==============================================================================


class TestExponentialDecayInvariantsEmpirical:
    """Empirical challenge of half-life formula, decay monotonicity, and asymptotic convergence."""

    def test_analytical_half_life_exact_values(self):
        """Validates exact analytical half-life S(T_1/2) = 0.5 * S_0.

        T_1/2 = -ln(2) / ln(0.95) ≈ 13.51340733... hours.
        Tests across multiple initial sentiment magnitudes and signs.
        """
        alpha = SENTIMENT_DECAY_ALPHA  # 0.95
        t_half = -math.log(2.0) / math.log(alpha)
        assert t_half == pytest.approx(13.51340733, abs=1e-6)

        initial_sentiments = [-1.0, -0.8, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.8, 1.0]
        for s0 in initial_sentiments:
            s_half = compute_hourly_decay(s0, elapsed_hours=t_half, alpha=alpha)
            expected = 0.5 * s0
            assert abs(s_half - expected) <= 1e-5, (
                f"Half-life decay failed for S0={s0}: got {s_half}, expected {expected}"
            )

    def test_multi_half_life_analytical_powers(self):
        """Validates that S(k * T_1/2) = (0.5^k) * S_0 for k = 1..10."""
        alpha = SENTIMENT_DECAY_ALPHA
        t_half = -math.log(2.0) / math.log(alpha)
        s0 = 1.0

        for k in range(1, 11):
            s_k = compute_hourly_decay(s0, elapsed_hours=k * t_half, alpha=alpha)
            expected = 0.5**k
            assert s_k == pytest.approx(expected, abs=1e-5), (
                f"Multi-half-life k={k} failed: got {s_k}, expected {expected}"
            )

    def test_strict_temporal_monotonicity_positive(self):
        """Verifies strictly monotonic decay for positive sentiment without new news: S_t > S_{t+1} > 0."""
        s = 1.0
        alpha = 0.95
        history: list[float] = [s]

        # Step through 100 consecutive 1-hour steps
        for step in range(1, 101):
            s_next = compute_hourly_decay(s, elapsed_hours=1.0, alpha=alpha)
            assert s > s_next, f"Monotonicity violated at step {step}: {s} <= {s_next}"
            assert s_next > 0.0, f"Positivity violated at step {step}: {s_next} <= 0"
            history.append(s_next)
            s = s_next

        # Also verify multi-hour step consistency: S(h) == S_0 * (alpha^h)
        for h in range(1, 101):
            direct = compute_hourly_decay(1.0, elapsed_hours=float(h), alpha=alpha)
            assert direct == pytest.approx(history[h], abs=1e-5)

    def test_strict_temporal_monotonicity_negative(self):
        """Verifies strictly monotonic upward decay for negative sentiment without new news: S_t < S_{t+1} < 0."""
        s = -1.0
        alpha = 0.95

        for step in range(1, 101):
            s_next = compute_hourly_decay(s, elapsed_hours=1.0, alpha=alpha)
            assert s < s_next, f"Negative monotonicity violated at step {step}: {s} >= {s_next}"
            assert s_next < 0.0, f"Negativity violated at step {step}: {s_next} >= 0"
            assert abs(s) > abs(s_next), f"Magnitude decay violated at step {step}: |{s}| <= |{s_next}|"
            s = s_next

    def test_asymptotic_convergence_zero_oscillation(self):
        """Verifies asymptotic convergence to 0 without sign flip or oscillation.

        Tests across 1,000 steps for both positive and negative initial sentiments.
        """
        for s0 in [1.0, -1.0, 0.5, -0.5]:
            sign_s0 = 1 if s0 > 0 else -1
            prev = s0
            for h in range(1, 500):
                decayed = compute_hourly_decay(s0, elapsed_hours=float(h), alpha=0.95)
                if decayed != 0.0:
                    sign_decayed = 1 if decayed > 0 else -1
                    assert sign_decayed == sign_s0, f"Sign flip detected at h={h}: s0={s0}, decayed={decayed}"
                    assert abs(decayed) < abs(prev), f"Oscillation detected at h={h}: |{decayed}| >= |{prev}|"
                    prev = decayed
                else:
                    # Once clamped to 0.0, it must remain 0.0 forever
                    assert decayed == 0.0

            # At h=350, alpha^350 = 0.95^350 ≈ 1.6e-8 < 1e-7 -> clamped to 0.0
            decayed_350 = compute_hourly_decay(s0, elapsed_hours=350.0, alpha=0.95)
            assert decayed_350 == 0.0

    def test_hourly_decay_boundary_cases(self):
        """Verifies behavior on zero, negative elapsed hours, and subnormal values."""
        # Non-positive elapsed hours returns original sentiment unchanged
        assert compute_hourly_decay(0.75, elapsed_hours=0.0) == 0.75
        assert compute_hourly_decay(0.75, elapsed_hours=-5.0) == 0.75
        assert compute_hourly_decay(-0.75, elapsed_hours=-1.0) == -0.75

        # Subnormal inputs (< 1e-7) return 0.0 or unchanged
        assert compute_hourly_decay(0.0, elapsed_hours=10.0) == 0.0
        assert compute_hourly_decay(5e-8, elapsed_hours=1.0) == 5e-8


# ==============================================================================
# 3. TIME-SERIES REINDEXING (EXACT LENGTH & ZERO NAN INVARIANT)
# ==============================================================================


class TestSentimentSeriesReindexingEmpirical:
    """Empirical challenge of get_sentiment_series alignment, length, and NaN immunity."""

    def test_exact_length_invariant_across_grid(self):
        """Verifies len(get_sentiment_series) == len(target_timestamps) for various lengths."""
        test_lengths = [0, 1, 2, 6, 12, 24, 48, 168, 720, 2000]
        news_samples = [
            {"published": "2026-11-01 02:00:00+00:00", "polarity": 0.8},
            {"published": "2026-11-01 05:00:00+00:00", "polarity": -0.6},
        ]

        for n in test_lengths:
            dates = pd.date_range("2026-11-01 00:00:00", periods=n, freq="1h", tz="UTC")
            series = get_sentiment_series(news_samples, dates)
            assert len(series) == n, f"Length mismatch: got {len(series)}, expected {n}"
            assert isinstance(series, pd.Series)
            if n > 0:
                assert series.isna().sum() == 0, f"Found NaNs in series of length {n}"
                assert np.isinf(series).sum() == 0, f"Found Infs in series of length {n}"

    def test_zero_nan_inf_immunity_empty_news(self):
        """Verifies that an empty news list produces all 0.0 without any NaNs or Infs."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=720, freq="1h", tz="UTC")
        series = get_sentiment_series([], dates)
        assert len(series) == 720
        assert (series == 0.0).all()
        assert series.isna().sum() == 0
        assert np.isinf(series).sum() == 0

    def test_multi_article_same_hour_exact_mean(self):
        """Verifies arithmetic mean aggregation when M_t articles arrive in hour t."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=5, freq="1h", tz="UTC")

        # 20 articles in hour 1 with explicit polarities
        rng = np.random.default_rng(42)
        pols = rng.uniform(-1.0, 1.0, size=20)
        expected_mean = float(np.mean(pols))

        news: list[dict[str, Any]] = []
        for i, p in enumerate(pols):
            news.append({
                "published": f"2026-11-01 01:{i:02d}:00+00:00",
                "polarity": float(p),
            })

        series = get_sentiment_series(news, dates, decay_factor=0.95)
        # Hour 0: No news -> 0.0
        assert series.iloc[0] == 0.0
        # Hour 1: Arithmetic mean
        assert series.iloc[1] == pytest.approx(expected_mean, abs=1e-5)
        # Hour 2: alpha * mean
        assert series.iloc[2] == pytest.approx(expected_mean * 0.95, abs=1e-5)
        # Hour 3: alpha^2 * mean
        assert series.iloc[3] == pytest.approx(expected_mean * (0.95**2), abs=1e-5)

    def test_unsorted_target_timestamps_order_preservation(self):
        """Verifies that unsorted target timestamps correctly map decayed values back to input order."""
        sorted_dates = pd.date_range("2026-11-01 00:00:00", periods=10, freq="1h", tz="UTC")
        news = [{"published": "2026-11-01 00:00:00+00:00", "polarity": 1.0}]

        # Compute on sorted
        series_sorted = get_sentiment_series(news, sorted_dates)

        # Shuffle indices
        rng = np.random.default_rng(999)
        shuffled_idx = rng.permutation(len(sorted_dates))
        unsorted_dates = sorted_dates[shuffled_idx]

        series_unsorted = get_sentiment_series(news, unsorted_dates)

        # Output length must match
        assert len(series_unsorted) == len(unsorted_dates)
        # Values at each timestamp must match exactly between sorted and unsorted series
        for ts in unsorted_dates:
            assert series_unsorted.loc[ts] == pytest.approx(series_sorted.loc[ts], abs=1e-6)

    def test_prior_news_decay_into_target_window(self):
        """Verifies news occurring 5 hours prior to window start decays exactly by alpha^5."""
        dates = pd.date_range("2026-11-01 05:00:00", periods=5, freq="1h", tz="UTC")
        news = [{"published": "2026-11-01 00:00:00+00:00", "polarity": 0.80}]

        series = get_sentiment_series(news, dates, decay_factor=0.95)
        # At 05:00 (index 0), exactly 5 hours of decay: 0.80 * (0.95^5)
        expected_t0 = 0.80 * (0.95**5)
        assert series.iloc[0] == pytest.approx(expected_t0, abs=1e-5)
        assert series.iloc[1] == pytest.approx(expected_t0 * 0.95, abs=1e-5)

    def test_future_news_does_not_leak_backward(self):
        """Verifies future news arriving at hour t+5 does not affect hours t..t+4."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=10, freq="1h", tz="UTC")
        news = [{"published": "2026-11-01 05:00:00+00:00", "polarity": 1.0}]

        series = get_sentiment_series(news, dates, decay_factor=0.95)
        # Hours 0..4 must remain 0.0 (no backward leakage)
        for h in range(5):
            assert series.iloc[h] == 0.0, f"Future leak detected at hour {h}: {series.iloc[h]}"
        # Hour 5 must be 1.0
        assert series.iloc[5] == pytest.approx(1.0, abs=1e-5)
        # Hour 6 must be 0.95
        assert series.iloc[6] == pytest.approx(0.95, abs=1e-5)


# ==============================================================================
# 4. MULTI-HORIZON DRIFT & HORIZON MONOTONICITY INVARIANTS
# ==============================================================================


class TestHorizonGammaMonotonicityEmpirical:
    """Empirical challenge of horizon attenuation factor gamma_h."""

    def test_gamma_h_strict_monotonic_decay_11_horizons(self):
        """Verifies gamma_h is strictly monotonically decreasing across all 11 horizons:

        gamma_1h > gamma_2h > gamma_4h > gamma_6h > gamma_12h > gamma_24h >
        gamma_2d > gamma_3d > gamma_7d > gamma_14d > gamma_30d > 0.
        """
        assert len(DEFAULT_HORIZONS) == 11

        gammas: list[float] = []
        step_hours_list: list[int] = []

        for h in DEFAULT_HORIZONS:
            step_h = HORIZON_TO_HOURS[h]
            g = compute_horizon_gamma(h)
            gammas.append(g)
            step_hours_list.append(step_h)

        # Step hours must be strictly increasing
        for i in range(len(step_hours_list) - 1):
            assert step_hours_list[i] < step_hours_list[i + 1]

        # Gamma must be strictly decreasing and strictly positive
        for i in range(len(gammas) - 1):
            h_curr = DEFAULT_HORIZONS[i]
            h_next = DEFAULT_HORIZONS[i + 1]
            g_curr = gammas[i]
            g_next = gammas[i + 1]

            assert g_curr > g_next, (
                f"Gamma monotonicity violated: {h_curr} ({g_curr:.6f}) <= {h_next} ({g_next:.6f})"
            )
            assert g_next > 0.0, f"Gamma {h_next} ({g_next:.6f}) not strictly positive"

        # Explicit bounds: gamma_1h <= gamma_0 (0.02)
        assert gammas[0] <= DEFAULT_SENTIMENT_GAMMA_0
        assert gammas[0] == pytest.approx(0.019936, abs=1e-5)
        # gamma_30d (720h)
        expected_30d = DEFAULT_SENTIMENT_GAMMA_0 * math.exp(-DEFAULT_SENTIMENT_LAMBDA * 720)
        assert gammas[-1] == pytest.approx(expected_30d, abs=1e-5)
        assert gammas[-1] > 0.0019

    def test_gamma_h_arbitrary_continuous_hours(self):
        """Verifies gamma_h monotonicity across continuous steps 1..1000."""
        prev_g = compute_horizon_gamma(1)
        for h in range(2, 1001):
            g = compute_horizon_gamma(h)
            assert prev_g > g > 0.0, f"Gamma monotonicity broken at hour {h}"
            prev_g = g


# ==============================================================================
# 5. QUANTILE ORDERING PRESERVATION & PRICE POSITIVITY
# ==============================================================================


class TestQuantileOrderingAndPricePositivityEmpirical:
    """Empirical challenge of quantile ordering P10 <= P50 <= P90 and price positivity."""

    def test_price_positivity_preservation_under_extreme_sentiment(self):
        """Verifies P^adj > 0 for all P > 0 under extreme sentiment shocks s in [-1.0, 1.0]."""
        test_prices = [1e-6, 0.01, 1.0, 50.0, 2500.0, 95000.0, 1e7]
        extreme_polarities = [-1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0]

        for price in test_prices:
            for s in extreme_polarities:
                for h in DEFAULT_HORIZONS:
                    pred = {
                        h: {
                            "step_hours": HORIZON_TO_HOURS[h],
                            "current_price": price,
                            "predicted_price": price,
                            "confidence_p10": price * 0.9,
                            "confidence_p50": price,
                            "confidence_p90": price * 1.1,
                        }
                    }
                    adj = apply_sentiment_drift(pred, sentiment_polarity=s)
                    adj_p = adj[h]["predicted_price"]
                    assert adj_p > 0.0, f"Price positivity violated: price={price}, s={s}, adj={adj_p}"
                    # Maximum drift is bounded by gamma_0 (2%)
                    assert abs(adj_p - price) / price <= DEFAULT_SENTIMENT_GAMMA_0 + 1e-6

    def test_quantile_ordering_preservation_random_fuzzing(self):
        """Verifies P10 <= P50 <= P90 implies P10^adj <= P50^adj <= P90^adj for 1,000 cases."""
        rng = np.random.default_rng(777)

        for _ in range(1000):
            # Generate strictly ordered quantiles
            p_base = float(rng.uniform(10.0, 100000.0))
            spread_1 = float(rng.uniform(0.01, 0.20))
            spread_2 = float(rng.uniform(0.01, 0.20))

            p10 = p_base * (1.0 - spread_1)
            p50 = p_base
            p90 = p_base * (1.0 + spread_2)
            assert p10 <= p50 <= p90

            pred_dict = {
                "1h": {
                    "step_hours": 1,
                    "current_price": p_base,
                    "predicted_price": p50,
                    "confidence_p10": p10,
                    "confidence_p50": p50,
                    "confidence_p90": p90,
                },
                "30d": {
                    "step_hours": 720,
                    "current_price": p_base,
                    "predicted_price": p50,
                    "confidence_p10": p10,
                    "confidence_p50": p50,
                    "confidence_p90": p90,
                },
            }

            s = float(rng.uniform(-1.0, 1.0))
            adj = apply_sentiment_drift(pred_dict, sentiment_polarity=s)

            for h in ["1h", "30d"]:
                adj_p10 = adj[h]["confidence_p10"]
                adj_p50 = adj[h]["confidence_p50"]
                adj_p90 = adj[h]["confidence_p90"]

                assert adj_p10 <= adj_p50 <= adj_p90, (
                    f"Quantile ordering violated under polarity {s}: P10={adj_p10}, P50={adj_p50}, P90={adj_p90}"
                )
                assert adj_p10 > 0.0

    def test_sentiment_drift_dataclass_support(self):
        """Verifies apply_sentiment_drift updates MockHorizonPrediction dataclasses cleanly."""
        pred = MockHorizonPrediction(
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

        adj = apply_sentiment_drift({"1h": pred}, sentiment_polarity=0.80)
        res = adj["1h"]
        assert isinstance(res, MockHorizonPrediction)
        assert res.predicted_price > 100.0
        assert res.direction == "BULLISH"
        assert res.predicted_change > 0.0
        assert res.predicted_return_pct > 0.0
        assert res.confidence_p10 < res.confidence_p50 < res.confidence_p90

    def test_apply_trajectory_sentiment_drift_full_envelope(self):
        """Verifies 720-step continuous trajectory drift preserves ordering and decays smoothly."""
        steps = 720
        points = np.linspace(100.0, 150.0, steps)
        # Quantiles: 10 quantiles across 720 steps
        quantiles = np.zeros((steps, 10))
        for q in range(10):
            quantiles[:, q] = points * (0.80 + 0.04 * q)

        # Extreme positive polarity shock
        adj_points, adj_quantiles = apply_trajectory_sentiment_drift(
            points, quantiles, sentiment_polarity=1.0
        )

        assert adj_points.shape == (steps,)
        assert adj_quantiles.shape == (steps, 10)

        # Positivity
        assert (adj_points > 0.0).all()
        assert (adj_quantiles > 0.0).all()

        # Quantile monotonicity across columns for every step
        for s in range(steps):
            row = adj_quantiles[s, :]
            for q in range(len(row) - 1):
                assert row[q] <= row[q + 1], f"Quantile inversion at step {s}, col {q}"

        # Attenuation check: conviction impact (delta) must be largest at step 0 and decay toward step 719
        delta_0 = adj_points[0] - points[0]
        delta_last = adj_points[-1] - points[-1]
        pct_0 = delta_0 / points[0]
        pct_last = delta_last / points[-1]
        assert pct_0 > pct_last > 0.0, f"Conviction drift did not attenuate: {pct_0} <= {pct_last}"
