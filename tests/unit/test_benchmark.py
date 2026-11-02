"""Unit Tests for Benchmark Package (metrics, backtester, compatibility)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.benchmark.backtester import RollingWalkForwardBacktester
from crypto_forecast.benchmark.compatibility import (
    evaluate_model_performance,
    run_asset_compatibility_analysis,
)
from crypto_forecast.benchmark.metrics import (
    calculate_hit_rate,
    calculate_mae,
    calculate_rmse,
    calculate_skill_score,
    calculate_smape,
)
from crypto_forecast.models.baseline import PersistenceForecaster


class TestMetricsUnit:
    def test_mae_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        assert calculate_mae(y, y) == 0.0

    def test_rmse_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        assert calculate_rmse(y, y) == 0.0

    def test_smape_zero_error(self):
        y = np.array([10.0, 20.0, 30.0])
        assert calculate_smape(y, y) == 0.0

    def test_hit_rate_perfect(self):
        ret_t = np.array([0.01, -0.02, 0.05])
        ret_p = np.array([0.02, -0.01, 0.01])
        assert calculate_hit_rate(ret_t, ret_p) == 100.0

    def test_skill_score_calculation(self):
        # 20% improvement
        assert math.isclose(calculate_skill_score(80.0, 100.0), 20.0)
        # Worse model
        assert math.isclose(calculate_skill_score(120.0, 100.0), -20.0)


class TestBacktesterAndCompatibilityUnit:
    def test_evaluate_model_performance(self):
        prices = 100.0 + np.cumsum(np.random.normal(0, 1, 300))
        df = pd.DataFrame({"close": prices})
        model = PersistenceForecaster()
        res = evaluate_model_performance(model, df, horizons=["1h", "24h"])
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 2
        assert "mae_model" in res.columns

    def test_run_asset_compatibility_analysis(self):
        model = PersistenceForecaster()
        res = run_asset_compatibility_analysis([model], symbols=["BTC", "ETH"], horizons=["1h", "24h"])
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 4
        assert set(res["symbol"].unique()) == {"BTC", "ETH"}
