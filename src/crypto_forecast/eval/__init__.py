"""Accuracy evaluation and retrospective analysis subpackage.

Provides 48-hour retrospective lookback accuracy pipelines, 50-column CSV schemas,
Google Drive incremental persistence, and evaluation orchestrators.
"""

from __future__ import annotations

from crypto_forecast.eval.dataset import (
    COLAB_DRIVE_ACCURACY_DIR,
    LOCAL_FALLBACK_ACCURACY_DIR,
    get_accuracy_eval_dir,
    identify_pending_timestamps,
    inspect_latest_recorded_timestamp,
    mount_google_drive_safely,
    persist_accuracy_dataset,
)
from crypto_forecast.eval.pipeline import (
    DEFAULT_EVAL_PAIRS,
    resolve_eval_models,
    run_accuracy_evaluation,
)
from crypto_forecast.eval.retrospective import (
    compute_retrospective_row,
    evaluate_historical_range_cached,
    evaluate_retrospective_horizon,
)
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    ACCURACY_EVAL_COLUMNS_SHORTHAND,
    MODEL_PREFIX_MAP,
    MODEL_PREFIXES,
    PREDICTION_COLUMNS,
    PREDICTION_COLUMNS_SHORTHAND,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
    canonicalize_columns,
    get_accuracy_csv_filename,
    get_model_prefix,
    sanitize_pair_symbol,
    validate_accuracy_df_schema,
)

__all__ = [
    "ACCURACY_EVAL_COLUMNS",
    "ACCURACY_EVAL_COLUMNS_SHORTHAND",
    "REAL_TIME_COL",
    "VALUE_AT_TIME_COL",
    "PREDICTION_COLUMNS",
    "PREDICTION_COLUMNS_SHORTHAND",
    "MODEL_PREFIXES",
    "MODEL_PREFIX_MAP",
    "get_accuracy_csv_filename",
    "get_model_prefix",
    "sanitize_pair_symbol",
    "canonicalize_columns",
    "validate_accuracy_df_schema",
    "compute_retrospective_row",
    "evaluate_retrospective_horizon",
    "evaluate_historical_range_cached",
    "COLAB_DRIVE_ACCURACY_DIR",
    "LOCAL_FALLBACK_ACCURACY_DIR",
    "get_accuracy_eval_dir",
    "inspect_latest_recorded_timestamp",
    "identify_pending_timestamps",
    "mount_google_drive_safely",
    "persist_accuracy_dataset",
    "DEFAULT_EVAL_PAIRS",
    "resolve_eval_models",
    "run_accuracy_evaluation",
]
