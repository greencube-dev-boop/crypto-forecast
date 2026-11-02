"""Benchmark Package for Multi-Horizon Crypto Forecasting."""

from crypto_forecast.benchmark.backtester import RollingWalkForwardBacktester
from crypto_forecast.benchmark.compatibility import (
    evaluate_model_performance,
    run_asset_compatibility_analysis,
)
from crypto_forecast.benchmark.metrics import (
    calc_hit_rate,
    calc_mae,
    calc_rmse,
    calc_skill_score,
    calc_smape,
    calculate_hit_rate,
    calculate_mae,
    calculate_rmse,
    calculate_skill_score,
    calculate_smape,
)

__all__ = [
    "calculate_mae",
    "calculate_rmse",
    "calculate_smape",
    "calculate_hit_rate",
    "calculate_skill_score",
    "calc_mae",
    "calc_rmse",
    "calc_smape",
    "calc_hit_rate",
    "calc_skill_score",
    "RollingWalkForwardBacktester",
    "evaluate_model_performance",
    "run_asset_compatibility_analysis",
]
