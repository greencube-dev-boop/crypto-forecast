"""Evaluation schemas, column constants, and filename conventions for accuracy analysis."""

from __future__ import annotations

import re
from typing import Any, Final

import pandas as pd

# Core column names
REAL_TIME_COL: Final[str] = "real time"
VALUE_AT_TIME_COL: Final[str] = "value at time"

# 48 Retrospective prediction columns (Canonical format)
PREDICTION_COLUMNS: Final[list[str]] = [
    f"value predicted {k}h before" for k in range(1, 49)
]

# Complete 50-column canonical schema:
# real time | value at time | value predicted 1h before | ... | 48h before
ACCURACY_EVAL_COLUMNS: Final[list[str]] = [
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
] + PREDICTION_COLUMNS

# Shorthand columns format (as appeared in informal requirement summary)
# real time | value at time | value predicted 1h before | 2h before | ... | 48h before
PREDICTION_COLUMNS_SHORTHAND: Final[list[str]] = [
    "value predicted 1h before"
] + [f"{k}h before" for k in range(2, 49)]

ACCURACY_EVAL_COLUMNS_SHORTHAND: Final[list[str]] = [
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
] + PREDICTION_COLUMNS_SHORTHAND

# Model identifier prefixes
MODEL_PREFIXES: Final[list[str]] = [
    "timesfm",
    "chronos",
    "granite",
    "persistence",
]

MODEL_PREFIX_MAP: Final[dict[str, str]] = {
    "timesfm": "timesfm",
    "timesfmforecaster": "timesfm",
    "fallbacktimesfmforecaster": "timesfm",
    "google/timesfm-2.5-200m-pytorch": "timesfm",
    "chronos": "chronos",
    "chronosforecaster": "chronos",
    "fallbackchronosforecaster": "chronos",
    "amazon/chronos-bolt-small": "chronos",
    "amazon/chronos-2": "chronos",
    "granite": "granite",
    "granitettmforecaster": "granite",
    "fallbackgraniteforecaster": "granite",
    "ibm-granite/granite-timeseries-ttm-r2": "granite",
    "persistence": "persistence",
    "persistenceforecaster": "persistence",
}


def sanitize_pair_symbol(pair: str) -> str:
    """Sanitizes trading pair symbols for filesystem safe naming.

    Examples:
        'BTC/USDT' -> 'BTC_USDT'
        'ETH-USD'  -> 'ETH_USD'
        'sol/usdt' -> 'SOL_USDT'
    """
    clean = pair.strip().upper()
    clean = clean.replace("/", "_").replace("-", "_").replace(":", "_")
    return clean


def get_model_prefix(model_or_name: Any) -> str:
    """Resolves canonical model filename prefix (timesfm, chronos, granite, persistence)."""
    if isinstance(model_or_name, str):
        name = model_or_name.strip()
    elif hasattr(model_or_name, "model_id") and isinstance(model_or_name.model_id, str):
        name = model_or_name.model_id
    elif hasattr(model_or_name, "__name__"):
        name = model_or_name.__name__
    else:
        name = model_or_name.__class__.__name__

    name_lower = name.lower()
    if name_lower in MODEL_PREFIX_MAP:
        return MODEL_PREFIX_MAP[name_lower]

    for key, prefix in MODEL_PREFIX_MAP.items():
        if key in name_lower:
            return prefix

    # Generic fallback: clean up string
    prefix = (
        name_lower.replace("forecaster", "")
        .replace("_model", "")
        .replace("-", "_")
        .strip("_")
    )
    return prefix or "unknown_model"


def get_accuracy_csv_filename(model: Any, pair: str) -> str:
    """Returns the formatted accuracy evaluation CSV filename '{model}_{pair}.csv'.

    Args:
        model: Model instance or model name string (e.g. 'TimesFMForecaster', 'timesfm').
        pair: Symbol string (e.g. 'BTC/USDT').

    Returns:
        Formatted filename string (e.g. 'timesfm_BTC_USDT.csv').
    """
    prefix = get_model_prefix(model)
    clean_pair = sanitize_pair_symbol(pair)
    return f"{prefix}_{clean_pair}.csv"


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Transparently canonicalizes column names to standard ACCURACY_EVAL_COLUMNS.

    Maps shorthand variants (e.g. '2h before') and case differences to
    canonical 'value predicted {k}h before'.
    """
    rename_map: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()

        if col_lower == "real time":
            rename_map[col] = REAL_TIME_COL
            continue
        if col_lower in ("value at time", "ground truth", "real value"):
            rename_map[col] = VALUE_AT_TIME_COL
            continue

        # Check for 'value predicted Xh before' or 'Xh before'
        match = re.search(r"(\d+)\s*h\s*before", col_lower)
        if match:
            k = int(match.group(1))
            if 1 <= k <= 48:
                rename_map[col] = f"value predicted {k}h before"

    canonical_df = df.rename(columns=rename_map)
    return canonical_df


def validate_accuracy_df_schema(
    df: pd.DataFrame,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Validates that a DataFrame adheres to the 50-column accuracy evaluation schema.

    Returns:
        tuple (is_valid, list_of_errors)
    """
    errors: list[str] = []

    if df.shape[1] != 50:
        errors.append(f"Expected exactly 50 columns, found {df.shape[1]}")

    missing_cols = [c for c in ACCURACY_EVAL_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing {len(missing_cols)} required columns: {missing_cols[:5]}...")

    if strict and list(df.columns) != ACCURACY_EVAL_COLUMNS:
        errors.append("Column sequence does not strictly match canonical ACCURACY_EVAL_COLUMNS")

    is_valid = len(errors) == 0
    return is_valid, errors
