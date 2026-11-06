"""Retrospective 48-Hour Lookback Evaluation Math Engine.

Calculates multi-horizon retrospective forecasts strictly enforcing causal temporal
isolation (zero lookahead). For target time t and lookback k in [1, 48]:
- History is sliced up to t - k hours.
- The model generates predictions up to step k.
- Intermediate steps 1 .. k-1 are discarded.
- Step k prediction is recorded for time t.
- Ground truth settled close price at time t is matched alongside predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np
import pandas as pd

from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
)
from crypto_forecast.models.base import BaseForecaster

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW: Final[int] = 512
DEFAULT_MIN_CONTEXT: Final[int] = 32
DEFAULT_MAX_HORIZON: Final[int] = 48


def _get_close_price_at_timestamp(df_hourly: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Safely extracts the settled close price at a specific timestamp."""
    val = df_hourly.loc[ts, "close"]
    if isinstance(val, pd.Series):
        return float(val.iloc[-1])
    return float(val)


def evaluate_retrospective_horizon(
    model: BaseForecaster,
    df_hourly: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    k: int,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> float:
    """Evaluates a single retrospective horizon k for target_timestamp.

    Causal Isolation Rule:
        Historical data is sliced strictly at cutoff = target_timestamp - k hours.
        Data after cutoff is never visible to the model.

    Intermediate Step Discarding Rule:
        The model generates forecast up to step k.
        Intermediate steps 1 .. k-1 (predicting t - k + 1 .. t - 1) are ignored.
        Only the prediction at step k (predicting time t) is returned.

    Args:
        model: Candidate forecaster implementing BaseForecaster.
        df_hourly: Hourly OHLCV DataFrame with UTC DatetimeIndex.
        target_timestamp: The target ground-truth evaluation timestamp t.
        k: Lookback horizon in hours (1 to 48).
        context_window: Maximum history bars to feed into the model.
        min_context: Minimum required historical bars before cutoff.

    Returns:
        float: Step k predicted price for time t.
    """
    if k < 1 or k > DEFAULT_MAX_HORIZON:
        raise ValueError(f"Lookback horizon k must be in [1, {DEFAULT_MAX_HORIZON}], got {k}")

    ts = pd.to_datetime(target_timestamp, utc=True)
    cutoff_ts = ts - pd.DateOffset(hours=k)

    is_persistence = (
        getattr(model, "model_id", None) == "persistence"
        or model.__class__.__name__ == "PersistenceForecaster"
    )
    effective_min_context = 1 if is_persistence else min_context

    # Strict zero-lookahead slice: history <= cutoff_ts
    history_slice = df_hourly.loc[df_hourly.index <= cutoff_ts]
    if len(history_slice) < effective_min_context:
        raise ValueError(
            f"Insufficient historical bars ({len(history_slice)} < {effective_min_context}) "
            f"prior to cutoff {cutoff_ts} for horizon k={k}h."
        )

    context_prices = history_slice["close"].values[-context_window:].astype(np.float64)

    # Persistence optimization: point forecast is identically the latest context price
    if getattr(model, "model_id", None) == "persistence" or model.__class__.__name__ == "PersistenceForecaster":
        return float(context_prices[-1])

    # Model trajectory forward pass
    point_traj, _ = model.forecast_trajectory(context_prices, max_horizon_hours=max(k, 48))
    point_traj = np.asarray(point_traj, dtype=np.float64)

    if len(point_traj) < k:
        raise RuntimeError(
            f"Model {model.model_id} returned {len(point_traj)} trajectory steps; "
            f"required at least {k} steps for horizon {k}h."
        )

    # Step k is located at array index k - 1
    # Intermediate predictions (indices 0 .. k - 2) are intentionally discarded
    step_k_pred = float(point_traj[k - 1])
    return step_k_pred


def compute_retrospective_row(
    model: BaseForecaster,
    df_hourly: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> dict[str, Any]:
    """Computes a single 50-column retrospective evaluation row for target_timestamp.

    Row Format:
        'real time' -> ISO 8601 UTC timestamp string
        'value at time' -> ground truth settled close price at time t
        'value predicted 1h before' .. '48h before' -> retrospective forecasts
    """
    ts = pd.to_datetime(target_timestamp, utc=True)
    if ts not in df_hourly.index:
        raise KeyError(f"Target timestamp {ts} not found in df_hourly index.")

    ground_truth = _get_close_price_at_timestamp(df_hourly, ts)

    row: dict[str, Any] = {
        REAL_TIME_COL: ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        VALUE_AT_TIME_COL: ground_truth,
    }

    for k in range(1, max_horizon + 1):
        pred_k = evaluate_retrospective_horizon(
            model=model,
            df_hourly=df_hourly,
            target_timestamp=ts,
            k=k,
            context_window=context_window,
            min_context=min_context,
        )
        row[f"value predicted {k}h before"] = pred_k

    return row


def evaluate_historical_range_cached(
    model: BaseForecaster,
    df_hourly: pd.DataFrame,
    target_timestamps: list[pd.Timestamp],
    max_horizon: int = DEFAULT_MAX_HORIZON,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    min_context: int = DEFAULT_MIN_CONTEXT,
) -> pd.DataFrame:
    """Evaluates multiple target timestamps leveraging trajectory caching.

    Mathematical Invariance Guarantee:
        At any cutoff time tau, running 1 forward pass of 48 steps produces forecasts
        for tau + 1h, tau + 2h, ..., tau + 48h.
        For target t and horizon k, cutoff is tau = t - k hours.
        The step k forecast from cutoff tau is identically the 'value predicted kh before' for time t.
        
        Caching trajectories by unique cutoff timestamps reduces neural network forward passes
        by up to 48x while producing mathematically identical values to single-point evaluation.

    Args:
        model: Candidate forecaster.
        df_hourly: Hourly OHLCV DataFrame with UTC DatetimeIndex.
        target_timestamps: List of target evaluation timestamps t.
        max_horizon: Number of retrospective horizons (default 48).
        context_window: Context history length.
        min_context: Minimum required history length.

    Returns:
        pd.DataFrame: Complete 50-column evaluation DataFrame for the target timestamps.
    """
    if not target_timestamps:
        return pd.DataFrame(columns=ACCURACY_EVAL_COLUMNS)

    # Normalize target timestamps
    ts_list = [pd.to_datetime(t, utc=True) for t in target_timestamps]
    ts_list = sorted([t for t in ts_list if t in df_hourly.index])

    if not ts_list:
        return pd.DataFrame(columns=ACCURACY_EVAL_COLUMNS)

    is_persistence = (
        getattr(model, "model_id", None) == "persistence"
        or model.__class__.__name__ == "PersistenceForecaster"
    )

    # Collect all needed unique cutoffs
    needed_cutoffs: set[pd.Timestamp] = set()
    for t in ts_list:
        for k in range(1, max_horizon + 1):
            needed_cutoffs.add(t - pd.DateOffset(hours=k))

    sorted_cutoffs = sorted(needed_cutoffs)

    # Trajectory cache: cutoff_ts -> np.ndarray of shape (max_horizon,)
    traj_cache: dict[pd.Timestamp, np.ndarray] = {}

    effective_min_context = 1 if is_persistence else min_context

    for cutoff_ts in sorted_cutoffs:
        history_slice = df_hourly.loc[df_hourly.index <= cutoff_ts]
        if len(history_slice) < effective_min_context:
            continue

        ctx = history_slice["close"].values[-context_window:].astype(np.float64)

        if is_persistence:
            traj_cache[cutoff_ts] = np.full(max_horizon, float(ctx[-1]), dtype=np.float64)
        else:
            point_traj, _ = model.forecast_trajectory(ctx, max_horizon_hours=max(max_horizon, 48))
            traj_cache[cutoff_ts] = np.asarray(point_traj[:max_horizon], dtype=np.float64)

    # Build rows
    records: list[dict[str, Any]] = []
    for t in ts_list:
        gt = _get_close_price_at_timestamp(df_hourly, t)
        row_dict: dict[str, Any] = {
            REAL_TIME_COL: t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            VALUE_AT_TIME_COL: gt,
        }

        has_missing_horizon = False
        for k in range(1, max_horizon + 1):
            cutoff_k = t - pd.DateOffset(hours=k)
            col_name = f"value predicted {k}h before"
            if cutoff_k in traj_cache and len(traj_cache[cutoff_k]) >= k:
                # Step k is index k - 1
                row_dict[col_name] = float(traj_cache[cutoff_k][k - 1])
            else:
                row_dict[col_name] = np.nan
                has_missing_horizon = True

        if not has_missing_horizon:
            records.append(row_dict)
        else:
            logger.debug("Skipping target %s due to missing cutoff context", t)

    df_out = pd.DataFrame(records, columns=ACCURACY_EVAL_COLUMNS)
    return df_out
