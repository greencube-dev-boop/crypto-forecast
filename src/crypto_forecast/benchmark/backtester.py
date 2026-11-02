"""Rolling Walk-Forward Backtester for Crypto Forecasting Models.

Executes expanding or rolling window evaluation slices across historical
time series, enforcing strict temporal isolation (zero lookahead bias)
to benchmark models against naive persistence.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from crypto_forecast.benchmark.metrics import (
    calculate_hit_rate,
    calculate_mae,
    calculate_rmse,
    calculate_skill_score,
    calculate_smape,
)
from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
)
from crypto_forecast.models.base import BaseForecaster
from crypto_forecast.models.baseline import PersistenceForecaster

logger = logging.getLogger(__name__)


class RollingWalkForwardBacktester:
    """Rolling walk-forward evaluation engine with temporal isolation."""

    def __init__(
        self,
        context_window: int = 168,
        step_stride: int = 24,
        horizons: list[str] | None = None,
    ) -> None:
        self.context_window = context_window
        self.step_stride = step_stride
        self.horizons = horizons or DEFAULT_HORIZONS

    def backtest(
        self,
        model: BaseForecaster,
        prices: Sequence[float] | np.ndarray,
        baseline_model: BaseForecaster | None = None,
    ) -> pd.DataFrame:
        """Runs rolling backtest across historical price series."""
        p_arr = np.asarray(prices, dtype=np.float64)
        if len(p_arr) < self.context_window + max(HORIZON_TO_HOURS[h] for h in self.horizons):
            raise ValueError(
                f"Price series length ({len(p_arr)}) is insufficient for "
                f"context window ({self.context_window}) + maximum horizon."
            )

        baseline = baseline_model or PersistenceForecaster()

        # Dictionary to accumulate true and predicted prices/returns per horizon
        eval_data: dict[str, dict[str, list[float]]] = {
            h: {
                "y_true": [],
                "y_model": [],
                "y_persist": [],
                "ret_true": [],
                "ret_model": [],
                "ret_persist": [],
            }
            for h in self.horizons
        }

        max_step = max(HORIZON_TO_HOURS[h] for h in self.horizons)
        eval_indices = range(self.context_window, len(p_arr) - max_step, self.step_stride)

        for t0 in eval_indices:
            context = p_arr[t0 - self.context_window : t0]
            current_p = context[-1]

            model_preds = model.forecast(context, horizons=self.horizons)
            persist_preds = baseline.forecast(context, horizons=self.horizons)

            for h in self.horizons:
                step = HORIZON_TO_HOURS[h]
                actual_target = p_arr[t0 + step]
                actual_return = (actual_target - current_p) / current_p

                m_pred = model_preds[h].predicted_price
                m_ret = model_preds[h].predicted_return_pct / 100.0

                p_pred = persist_preds[h].predicted_price
                p_ret = persist_preds[h].predicted_return_pct / 100.0

                eval_data[h]["y_true"].append(actual_target)
                eval_data[h]["y_model"].append(m_pred)
                eval_data[h]["y_persist"].append(p_pred)
                eval_data[h]["ret_true"].append(actual_return)
                eval_data[h]["ret_model"].append(m_ret)
                eval_data[h]["ret_persist"].append(p_ret)

        records = []
        for h in self.horizons:
            yt = eval_data[h]["y_true"]
            ym = eval_data[h]["y_model"]
            yp = eval_data[h]["y_persist"]
            rt = eval_data[h]["ret_true"]
            rm = eval_data[h]["ret_model"]

            mae_m = calculate_mae(yt, ym)
            mae_p = calculate_mae(yt, yp)
            rmse_m = calculate_rmse(yt, ym)
            rmse_p = calculate_rmse(yt, yp)
            smape_m = calculate_smape(yt, ym)
            hit_rate = calculate_hit_rate(rt, rm)
            skill_mae = calculate_skill_score(mae_m, mae_p)
            skill_rmse = calculate_skill_score(rmse_m, rmse_p)

            records.append(
                {
                    "horizon": h,
                    "step_hours": HORIZON_TO_HOURS[h],
                    "mae_model": mae_m,
                    "mae_persist": mae_p,
                    "rmse_model": rmse_m,
                    "rmse_persist": rmse_p,
                    "smape_pct": smape_m,
                    "hit_rate_pct": hit_rate,
                    "skill_mae_pct": skill_mae,
                    "skill_rmse_pct": skill_rmse,
                }
            )

        return pd.DataFrame(records)
