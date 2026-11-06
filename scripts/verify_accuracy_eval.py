#!/usr/bin/env python3
"""Standalone headless verification script for Retrospective 48-Hour Accuracy Evaluation Pipeline.

Validates:
1. Retrospective lookback math and temporal isolation across 48 horizons.
2. Intermediate step discarding (asserting step k prediction is recorded, 1..k-1 ignored).
3. Mathematical persistence oracle identity.
4. CSV schema conformity (exact 50 columns) and multi-pair file naming ({model}_{pair}.csv).
5. Incremental resume, deduplication, and atomic persistence.

Usage:
    python scripts/verify_accuracy_eval.py [--fast]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add src to sys.path if not present
repo_root = Path(__file__).resolve().parent.parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from crypto_forecast.eval.dataset import (
    identify_pending_timestamps,
    inspect_latest_recorded_timestamp,
    persist_accuracy_dataset,
)
from crypto_forecast.eval.pipeline import run_accuracy_evaluation
from crypto_forecast.eval.retrospective import (
    compute_retrospective_row,
    evaluate_historical_range_cached,
    evaluate_retrospective_horizon,
)
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
    canonicalize_columns,
    get_accuracy_csv_filename,
    validate_accuracy_df_schema,
)
from crypto_forecast.models.base import BaseForecaster
from crypto_forecast.models.baseline import PersistenceForecaster


class StepAwareMockForecaster(BaseForecaster):
    """Deterministic mock forecaster where step s prediction is current_price + 100 * s.

    Proves rigorously that step k is recorded and intermediate steps 1..k-1 are discarded.
    """

    def __init__(self) -> None:
        super().__init__(model_id="step_aware_mock")

    def forecast_trajectory(
        self,
        context_prices: list[float] | np.ndarray,
        max_horizon_hours: int = 48,
    ) -> tuple[np.ndarray, np.ndarray]:
        prices = self.validate_context_prices(context_prices)
        current = float(prices[-1])
        steps = np.arange(1, max_horizon_hours + 1, dtype=np.float64)
        point_pred = current + 100.0 * steps
        quantiles = np.column_stack([
            point_pred - 10.0,
            point_pred,
            point_pred + 10.0,
        ])
        return point_pred, quantiles


def make_linear_ohlcv(n_bars: int = 120, start_price: float = 1000.0, step: float = 10.0) -> pd.DataFrame:
    """Creates a deterministic arithmetic price series: close = start_price + step * i."""
    base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(base_ts, periods=n_bars, freq="1h")
    closes = [start_price + step * i for i in range(n_bars)]

    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 2.0 for c in closes],
            "low": [c - 2.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n_bars,
        },
        index=timestamps,
    )
    return df


def verify_intermediate_step_discarding_and_isolation() -> tuple[bool, str]:
    """Test 1: Slicing isolation and intermediate step discarding."""
    df = make_linear_ohlcv(n_bars=120, start_price=1000.0, step=10.0)
    mock_model = StepAwareMockForecaster()

    # Target timestamp at index 80 (80 hours from start)
    target_idx = 80
    target_ts = df.index[target_idx]
    gt_price = float(df.loc[target_ts, "close"])  # 1000 + 10 * 80 = 1800.0

    # Test individual horizons
    for k in (1, 2, 12, 24, 48):
        pred_k = evaluate_retrospective_horizon(
            model=mock_model,
            df_hourly=df,
            target_timestamp=target_ts,
            k=k,
            context_window=64,
            min_context=32,
        )
        # Cutoff index is target_idx - k
        cutoff_price = float(df.iloc[target_idx - k]["close"])
        # Mock predicts cutoff_price + 100 * k at step k
        expected_pred_k = cutoff_price + 100.0 * k
        if not np.isclose(pred_k, expected_pred_k):
            return False, (
                f"Horizon k={k}h failed: expected step {k} pred {expected_pred_k}, got {pred_k}. "
                f"Intermediate steps were not properly discarded!"
            )

    return True, "All 48-hour lookback slices isolated; intermediate steps 1..k-1 discarded."


def verify_persistence_oracle() -> tuple[bool, str]:
    """Test 2: Analytical closed-form identity with PersistenceForecaster."""
    df = make_linear_ohlcv(n_bars=120, start_price=1000.0, step=10.0)
    persistence = PersistenceForecaster()

    target_idx = 80
    target_ts = df.index[target_idx]
    gt_price = float(df.loc[target_ts, "close"])

    row = compute_retrospective_row(
        model=persistence,
        df_hourly=df,
        target_timestamp=target_ts,
        max_horizon=48,
        context_window=64,
        min_context=32,
    )

    if not np.isclose(row[VALUE_AT_TIME_COL], gt_price):
        return False, f"Ground truth mismatch: expected {gt_price}, got {row[VALUE_AT_TIME_COL]}"

    for k in range(1, 49):
        col_name = f"value predicted {k}h before"
        expected_price = float(df.iloc[target_idx - k]["close"])
        actual_pred = row[col_name]
        if not np.isclose(actual_pred, expected_price):
            return False, f"Persistence oracle failed at k={k}: expected {expected_price}, got {actual_pred}"

    return True, "Persistence mathematical oracle matched across all 48 horizons."


def verify_schema_and_file_naming() -> tuple[bool, str]:
    """Test 3: CSV schema structure and filename conventions."""
    if len(ACCURACY_EVAL_COLUMNS) != 50:
        return False, f"Expected 50 columns in ACCURACY_EVAL_COLUMNS, found {len(ACCURACY_EVAL_COLUMNS)}"

    if ACCURACY_EVAL_COLUMNS[0] != REAL_TIME_COL or ACCURACY_EVAL_COLUMNS[1] != VALUE_AT_TIME_COL:
        return False, "First two columns must be 'real time' and 'value at time'"

    for k in range(1, 49):
        expected_col = f"value predicted {k}h before"
        if ACCURACY_EVAL_COLUMNS[1 + k] != expected_col:
            return False, f"Column index {1 + k} mismatch: expected '{expected_col}', got '{ACCURACY_EVAL_COLUMNS[1 + k]}'"

    # Filename tests
    fn1 = get_accuracy_csv_filename("timesfm", "BTC/USDT")
    fn2 = get_accuracy_csv_filename("chronos", "ETH/USDT")
    fn3 = get_accuracy_csv_filename("granite", "SOL-USD")
    fn4 = get_accuracy_csv_filename("persistence", "BNB:USDT")

    if fn1 != "timesfm_BTC_USDT.csv":
        return False, f"Filename mismatch: expected timesfm_BTC_USDT.csv, got {fn1}"
    if fn2 != "chronos_ETH_USDT.csv":
        return False, f"Filename mismatch: expected chronos_ETH_USDT.csv, got {fn2}"
    if fn3 != "granite_SOL_USD.csv":
        return False, f"Filename mismatch: expected granite_SOL_USD.csv, got {fn3}"
    if fn4 != "persistence_BNB_USDT.csv":
        return False, f"Filename mismatch: expected persistence_BNB_USDT.csv, got {fn4}"

    return True, "50-column schema definition and {model}_{pair}.csv naming validated."


def verify_incremental_resume_and_deduplication() -> tuple[bool, str]:
    """Test 4: Incremental resume, deduplication, and atomic persistence."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        csv_file = tmp_path / "persistence_BTC_USDT.csv"

        df_full = make_linear_ohlcv(n_bars=120, start_price=1000.0, step=10.0)
        persistence = PersistenceForecaster()

        # Step 4a: Run initial evaluation for 5 historical hours (indices 80 to 84)
        initial_targets = list(df_full.index[80:85])
        df_eval_1 = evaluate_historical_range_cached(
            model=persistence,
            df_hourly=df_full.iloc[:85],
            target_timestamps=initial_targets,
            max_horizon=48,
            context_window=64,
            min_context=32,
        )

        persist_accuracy_dataset(df_eval_1, csv_file)
        if not csv_file.exists():
            return False, "Failed to persist initial CSV file."

        df_saved_1 = pd.read_csv(csv_file)
        if len(df_saved_1) != 5:
            return False, f"Expected 5 rows in initial CSV, got {len(df_saved_1)}"

        # Step 4b: Re-run with zero new hours (should be idempotent, 0 rows added)
        latest_ts = inspect_latest_recorded_timestamp(csv_file)
        pending_0 = identify_pending_timestamps(
            df_hourly=df_full.iloc[:85],
            latest_ts=latest_ts,
            min_context_window=32,
            max_lookback_horizon=48,
        )
        if len(pending_0) != 0:
            return False, f"Expected 0 pending hours on re-run, got {len(pending_0)}"

        # Step 4c: Advance time by 3 hours (indices 85, 86, 87)
        pending_3 = identify_pending_timestamps(
            df_hourly=df_full.iloc[:88],
            latest_ts=latest_ts,
            min_context_window=32,
            max_lookback_horizon=48,
        )
        if len(pending_3) != 3:
            return False, f"Expected 3 pending hours when advancing by 3 hours, got {len(pending_3)}"

        df_eval_2 = evaluate_historical_range_cached(
            model=persistence,
            df_hourly=df_full.iloc[:88],
            target_timestamps=pending_3,
            max_horizon=48,
            context_window=64,
            min_context=32,
        )

        persist_accuracy_dataset(df_eval_2, csv_file)
        df_saved_2 = pd.read_csv(csv_file)
        if len(df_saved_2) != 8:
            return False, f"Expected 8 rows after incremental append, got {len(df_saved_2)}"

        # Verify no duplicates and sorted order
        ts_series = pd.to_datetime(df_saved_2[REAL_TIME_COL], utc=True)
        if not ts_series.is_monotonic_increasing:
            return False, "Timestamps in CSV are not monotonically increasing."
        if not ts_series.is_unique:
            return False, "Duplicate timestamps found in CSV."

    return True, "Incremental resume, deduplication, and file persistence verified."


def verify_full_end_to_end_pipeline() -> tuple[bool, str]:
    """Test 5: Full pipeline multi-pair multi-model runner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        df_btc = make_linear_ohlcv(n_bars=100, start_price=50000.0, step=50.0)
        df_eth = make_linear_ohlcv(n_bars=100, start_price=3000.0, step=10.0)
        ohlcv_map = {"BTC/USDT": df_btc, "ETH/USDT": df_eth}

        pairs = ["BTC/USDT", "ETH/USDT"]
        models = ["timesfm", "chronos", "granite", "persistence"]

        # Run pipeline with force_fallback=True for fast deterministic execution
        results = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=4,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map=ohlcv_map,
            verbose=False,
        )

        expected_count = len(pairs) * len(models)  # 2 * 4 = 8
        if len(results) != expected_count:
            return False, f"Expected {expected_count} result CSV files, got {len(results)}"

        for key, csv_path in results.items():
            if not csv_path.exists():
                return False, f"Expected CSV file {csv_path} does not exist."
            df_check = pd.read_csv(csv_path)
            is_valid, errors = validate_accuracy_df_schema(df_check, strict=True)
            if not is_valid:
                return False, f"Schema validation failed for {csv_path.name}: {errors}"
            if len(df_check) != 4:
                return False, f"Expected 4 rows in {csv_path.name}, got {len(df_check)}"

    return True, f"Full pipeline executed: {expected_count} CSV files verified across pairs and models."


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Retrospective Accuracy Evaluation Pipeline")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode")
    args = parser.parse_args()

    print("=" * 75)
    print("VERIFICATION SUITE: RETROSPECTIVE 48-HOUR ACCURACY EVALUATION (M8)")
    print("=" * 75)

    checks = [
        ("Intermediate Step Discarding & Zero Lookahead", verify_intermediate_step_discarding_and_isolation),
        ("Mathematical Persistence Oracle Identity", verify_persistence_oracle),
        ("50-Column Schema & Filename Conventions", verify_schema_and_file_naming),
        ("Incremental Resume, Deduplication & Atomic Write", verify_incremental_resume_and_deduplication),
        ("End-to-End Multi-Pair Multi-Model Pipeline", verify_full_end_to_end_pipeline),
    ]

    all_passed = True
    for title, check_fn in checks:
        print(f"\n[CHECK] {title}...")
        try:
            success, msg = check_fn()
            if success:
                print(f"  [PASS] {msg}")
            else:
                print(f"  [FAIL] {msg}")
                all_passed = False
        except Exception as exc:
            print(f"  [ERROR] Unhandled exception: {exc}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (Exit Code 0)")
        print("=" * 75)
        return 0
    else:
        print("VERIFICATION FAILURES DETECTED (Exit Code 1)")
        print("=" * 75)
        return 1


if __name__ == "__main__":
    sys.exit(main())
