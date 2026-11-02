"""Quantitative Metric Computations for Crypto Forecasting Models.

Provides vectorized implementations of:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Directional Hit-Rate (%)
- Relative Skill Score (%) against persistence baseline
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def calculate_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Mean Absolute Error (MAE): (1/N) * sum(|y - y_hat|)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays")
    return float(np.mean(np.abs(yt - yp)))


def calculate_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Root Mean Squared Error (RMSE): sqrt((1/N) * sum((y - y_hat)^2))."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def calculate_smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (sMAPE) in [0%, 200%]."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate sMAPE on empty arrays")
    denom = np.abs(yt) + np.abs(yp)
    mask = denom > 1e-12
    smape_vals = np.zeros_like(yt)
    smape_vals[mask] = 200.0 * np.abs(yp[mask] - yt[mask]) / denom[mask]
    return float(np.mean(smape_vals))


def calculate_hit_rate(y_true_returns: ArrayLike, y_pred_returns: ArrayLike) -> float:
    """Calculates Directional Hit-Rate percentage in [0%, 100%]."""
    yt = np.asarray(y_true_returns, dtype=float)
    yp = np.asarray(y_pred_returns, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate Hit-Rate on empty arrays")
    concordant = (np.sign(yt) == np.sign(yp)) & (np.sign(yt) != 0)
    concordant = concordant | ((np.sign(yt) == 0) & (np.sign(yp) == 0))
    return float(np.mean(concordant) * 100.0)


def calculate_skill_score(mae_model: float, mae_baseline: float) -> float:
    """Calculates relative skill score percentage: (1 - mae_model / mae_baseline) * 100%."""
    if mae_baseline <= 1e-12:
        return 0.0 if mae_model <= 1e-12 else -100.0
    return float((1.0 - mae_model / mae_baseline) * 100.0)


# Aliases for compatibility
calc_mae = calculate_mae
calc_rmse = calculate_rmse
calc_smape = calculate_smape
calc_hit_rate = calculate_hit_rate
calc_skill_score = calculate_skill_score
