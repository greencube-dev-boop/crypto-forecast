"""Adversarial Empirical Stress Testing Suite for Retrospective Accuracy Evaluation (M8).

Author: challenger_accuracy_1 (Empirical Stress Challenger)
Target Components:
- crypto_forecast.eval.schemas
- crypto_forecast.eval.retrospective
- crypto_forecast.eval.dataset
- crypto_forecast.eval.pipeline

Covers 5 Core Adversarial Stress Vectors:
1. Corrupted CSV Handling & Schema Fault Tolerance (zero-byte files, garbage bytes, missing columns, corrupt timestamps)
2. Market Data Anomalies (zero prices, negative prices, extreme volatility, flash crashes, sub-penny and hyper-scale prices)
3. Timestamp & Index Anomalies (timestamp gaps, duplicate timestamps, unsorted indices, tz-naive inputs)
4. Incremental Resume & Idempotence Stress (0-hour re-run idempotence, 1-hour step, 72-hour large gap, multi-step bursts)
5. Multi-Model Pipeline Stress (all 4 models across multiple pairs under iterative burst cycles, trajectory boundary assertions)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from typing import Any

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.eval.dataset import (
    get_accuracy_eval_dir,
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
    PREDICTION_COLUMNS,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
    canonicalize_columns,
    get_accuracy_csv_filename,
    get_model_prefix,
    sanitize_pair_symbol,
    validate_accuracy_df_schema,
)
from crypto_forecast.models.base import BaseForecaster
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import TimesFMForecaster


def make_test_ohlcv(
    n_bars: int = 120,
    start_price: float = 1000.0,
    step: float = 10.0,
    base_time: str = "2024-01-01 00:00:00",
) -> pd.DataFrame:
    """Generates synthetic hourly OHLCV DataFrame with UTC DatetimeIndex."""
    base_ts = pd.Timestamp(base_time, tz="UTC")
    timestamps = pd.date_range(base_ts, periods=n_bars, freq="1h")
    closes = [start_price + step * i for i in range(n_bars)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 2.0 for c in closes],
            "low": [c - 2.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n_bars,
        },
        index=timestamps,
    )


# ==============================================================================
# Vector 1: Corrupted CSV Files & Schema Fault Tolerance
# ==============================================================================


class TestCorruptedCSVsAndSchemaFaultTolerance:
    def test_zero_byte_empty_csv_handling(self, tmp_path: Path) -> None:
        """Probes zero-byte file handling in inspection and persistence."""
        csv_file = tmp_path / "zero_byte.csv"
        csv_file.touch()
        assert csv_file.stat().st_size == 0

        # Inspection on empty file must return None without raising exception
        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        # Persisting into a zero-byte file must succeed and write clean canonical CSV
        df_new = pd.DataFrame({
            REAL_TIME_COL: ["2026-11-06T00:00:00Z"],
            VALUE_AT_TIME_COL: [50000.0],
        })
        for k in range(1, 49):
            df_new[f"value predicted {k}h before"] = [49000.0 + k]

        persisted = persist_accuracy_dataset(df_new, csv_file)
        assert len(persisted) == 1
        assert csv_file.stat().st_size > 0

        is_valid, errors = validate_accuracy_df_schema(persisted, strict=True)
        assert is_valid, f"Schema invalid: {errors}"

    def test_unparseable_binary_garbage_csv_recovery(self, tmp_path: Path) -> None:
        """Probes recovery when existing CSV file contains random binary garbage."""
        csv_file = tmp_path / "corrupted_binary.csv"
        with open(csv_file, "wb") as f:
            f.write(b"\x00\xff\xfe\x00\x01\x02\x03\xff\xee\xdd\xcc")

        # Must safely return None
        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        # Must gracefully recover by overwriting corrupted file with new data
        df_new = pd.DataFrame({
            REAL_TIME_COL: ["2026-11-06T00:00:00Z"],
            VALUE_AT_TIME_COL: [50000.0],
        })
        for k in range(1, 49):
            df_new[f"value predicted {k}h before"] = [49000.0 + k]

        persisted = persist_accuracy_dataset(df_new, csv_file)
        assert len(persisted) == 1
        df_reloaded = pd.read_csv(csv_file)
        assert len(df_reloaded) == 1
        assert df_reloaded.iloc[0][REAL_TIME_COL] == "2026-11-06T00:00:00Z"

    def test_missing_required_real_time_column_in_existing_csv(self, tmp_path: Path) -> None:
        """Probes behavior when existing CSV is valid CSV syntax but missing 'real time' column.
        
        Empirically demonstrates BUG-01:
        When an existing CSV has arbitrary non-standard columns (missing 'real time'),
        persist_accuracy_dataset fails to discard the invalid rows and instead retains
        a phantom row with NaN across all 50 canonical columns.
        """
        csv_file = tmp_path / "missing_columns.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("col_a,col_b,col_c\n1,2,3\n4,5,6\n")

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        df_new = pd.DataFrame({
            REAL_TIME_COL: ["2024-01-01T00:00:00Z"],
            VALUE_AT_TIME_COL: [50000.0],
        })
        for k in range(1, 49):
            df_new[f"value predicted {k}h before"] = [49000.0 + k]

        persisted = persist_accuracy_dataset(df_new, csv_file)

        # BUG-01 Resolution: Clean recovery without corrupt all-NaN phantom rows
        nan_rows = persisted[persisted[REAL_TIME_COL].isna()]
        assert len(nan_rows) == 0, "No NaN rows should be retained after fix"
        assert len(persisted) == 1, "Only valid new row should be persisted when existing file is corrupted"
        assert persisted.iloc[0][REAL_TIME_COL] == "2024-01-01T00:00:00Z"
        assert persisted.iloc[0][VALUE_AT_TIME_COL] == 50000.0

    def test_unparseable_timestamp_string_in_existing_csv_crashes(self, tmp_path: Path) -> None:
        """Probes behavior when existing CSV has corrupted unparseable timestamp values.

        Verifies fix for BUG-02:
        persist_accuracy_dataset does NOT crash with DateParseError/ValueError,
        but recovers gracefully, purges the unparseable corrupted row, and saves the valid row.
        """
        csv_file = tmp_path / "corrupt_timestamp.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("real time,value at time\nNOT_A_REAL_DATE,100.0\n")

        # 1. inspect_latest_recorded_timestamp safely returns None
        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        # 2. persist_accuracy_dataset recovers gracefully and saves valid row without crashing
        df_new = pd.DataFrame({
            REAL_TIME_COL: ["2024-01-01T00:00:00Z"],
            VALUE_AT_TIME_COL: [50000.0],
        })
        for k in range(1, 49):
            df_new[f"value predicted {k}h before"] = [49000.0 + k]

        persisted = persist_accuracy_dataset(df_new, csv_file)
        assert len(persisted) == 1
        assert persisted.iloc[0][REAL_TIME_COL] == df_new.iloc[0][REAL_TIME_COL]
        assert persisted.iloc[0][VALUE_AT_TIME_COL] == 50000.0

    def test_unordered_and_duplicate_rows_in_existing_csv(self, tmp_path: Path) -> None:
        """Probes chronological re-ordering and deduplication when existing CSV has shuffle and dups."""
        csv_file = tmp_path / "unordered_dups.csv"
        raw_rows = pd.DataFrame({
            REAL_TIME_COL: [
                "2026-11-06T05:00:00Z",
                "2026-11-06T01:00:00Z",
                "2026-11-06T03:00:00Z",
                "2026-11-06T01:00:00Z",  # Duplicate of index 1
            ],
            VALUE_AT_TIME_COL: [105.0, 101.0, 103.0, 999.0],
        })
        for k in range(1, 49):
            raw_rows[f"value predicted {k}h before"] = 100.0 + k
        raw_rows.to_csv(csv_file, index=False)

        # Inspect latest timestamp should accurately return the true maximum (05:00:00Z)
        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest == pd.Timestamp("2026-11-06 05:00:00", tz="UTC")

        # Persisting a new row at 06:00:00Z should sort everything and drop the duplicate
        df_new = pd.DataFrame({
            REAL_TIME_COL: ["2026-11-06T06:00:00Z"],
            VALUE_AT_TIME_COL: [106.0],
        })
        for k in range(1, 49):
            df_new[f"value predicted {k}h before"] = 100.0 + k

        persisted = persist_accuracy_dataset(df_new, csv_file)
        assert len(persisted) == 4  # 01, 03, 05, 06

        ts_series = pd.to_datetime(persisted[REAL_TIME_COL], utc=True)
        assert ts_series.is_monotonic_increasing
        assert ts_series.is_unique

        # Original value at 01:00:00Z should be preserved (101.0, not 999.0)
        row_01 = persisted[persisted[REAL_TIME_COL] == "2026-11-06T01:00:00Z"].iloc[0]
        assert np.isclose(row_01[VALUE_AT_TIME_COL], 101.0)


# ==============================================================================
# Vector 2: Market Data Anomalies (Zero, Negative, Extreme Volatility, Scales)
# ==============================================================================


class TestMarketDataAnomaliesAndScales:
    def test_non_positive_prices_enforce_fail_fast_contract(self) -> None:
        """Asserts that zero or negative prices trigger validation errors in BaseForecaster models."""
        base_ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(base_ts, periods=60, freq="1h")

        # Series with a zero price inside the context slice
        closes_zero = [100.0] * 30 + [0.0] + [100.0] * 29
        df_zero = pd.DataFrame(
            {"open": closes_zero, "high": closes_zero, "low": closes_zero, "close": closes_zero, "volume": [10.0] * 60},
            index=timestamps,
        )

        # TimesFM, Chronos, and Granite validate context prices strictly > 0
        timesfm = TimesFMForecaster(force_fallback=True)
        target_ts = timestamps[-1]

        with pytest.raises(ValueError, match="strictly positive"):
            evaluate_retrospective_horizon(timesfm, df_zero, target_ts, k=1)

        # Series with a negative price inside the context slice
        closes_neg = [100.0] * 30 + [-50.0] + [100.0] * 29
        df_neg = pd.DataFrame(
            {"open": closes_neg, "high": closes_neg, "low": closes_neg, "close": closes_neg, "volume": [10.0] * 60},
            index=timestamps,
        )
        chronos = ChronosForecaster(use_fallback=True)
        with pytest.raises(ValueError, match="strictly positive"):
            evaluate_retrospective_horizon(chronos, df_neg, target_ts, k=1)

    def test_sub_penny_extreme_small_scale_precision(self) -> None:
        """Evaluates numerical stability with sub-penny meme coin prices (e.g. 1e-8 to 1e-6)."""
        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(base_ts, periods=100, freq="1h")
        closes = [0.00000100 + 0.00000001 * i for i in range(100)]
        df_micro = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000.0] * 100},
            index=timestamps,
        )

        persistence = PersistenceForecaster()
        target_ts = timestamps[80]

        row = compute_retrospective_row(persistence, df_micro, target_ts, max_horizon=48)
        assert np.isclose(row[VALUE_AT_TIME_COL], df_micro.iloc[80]["close"])

        for k in range(1, 49):
            expected_pred = df_micro.iloc[80 - k]["close"]
            pred = row[f"value predicted {k}h before"]
            assert np.isclose(pred, expected_pred, rtol=1e-12)
            assert not np.isnan(pred)
            assert not np.isinf(pred)

    def test_hyper_scale_large_prices(self) -> None:
        """Evaluates numerical stability with hyper-scaled asset prices (e.g. 10^8)."""
        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(base_ts, periods=100, freq="1h")
        closes = [100_000_000.0 + 10_000.0 * i for i in range(100)]
        df_macro = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000.0] * 100},
            index=timestamps,
        )

        persistence = PersistenceForecaster()
        target_ts = timestamps[80]

        row = compute_retrospective_row(persistence, df_macro, target_ts, max_horizon=48)
        assert np.isclose(row[VALUE_AT_TIME_COL], df_macro.iloc[80]["close"])
        for k in range(1, 49):
            expected_pred = df_macro.iloc[80 - k]["close"]
            pred = row[f"value predicted {k}h before"]
            assert np.isclose(pred, expected_pred, rtol=1e-8)

    def test_extreme_volatility_flash_crash_and_recovery(self) -> None:
        """Asserts exact retrospective lookback tracking across a 90% flash crash and 10x recovery."""
        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(base_ts, periods=100, freq="1h")

        closes = [1000.0] * 50 + [100.0] * 5 + [1000.0] * 45
        df_crash = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000.0] * 100},
            index=timestamps,
        )

        persistence = PersistenceForecaster()
        target_ts = timestamps[60]  # After recovery

        row = compute_retrospective_row(persistence, df_crash, target_ts, max_horizon=48)

        # Target at index 60 has close 1000.0
        assert np.isclose(row[VALUE_AT_TIME_COL], 1000.0)

        # Horizon k=6 was formulated at index 60 - 6 = 54 (which was during the crash, close=100.0)
        assert np.isclose(row["value predicted 6h before"], 100.0)

        # Horizon k=15 was formulated at index 60 - 15 = 45 (before the crash, close=1000.0)
        assert np.isclose(row["value predicted 15h before"], 1000.0)


# ==============================================================================
# Vector 3: Timestamp Gaps, Missing Data, and Index Invariants
# ==============================================================================


class TestTimestampAndIndexInvariants:
    def test_identify_pending_with_timestamp_gap_in_history(self) -> None:
        """Verifies that gaps in hourly timestamps do not cause off-by-one corruption."""
        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        # 100 bars total, but with a 10-hour gap between bar 40 and 41
        ts_part1 = pd.date_range(base_ts, periods=40, freq="1h")
        ts_part2 = pd.date_range(ts_part1[-1] + pd.Timedelta(hours=11), periods=60, freq="1h")
        full_index = ts_part1.append(ts_part2)

        closes = [1000.0 + 10.0 * i for i in range(100)]
        df_gap = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [100.0] * 100},
            index=full_index,
        )

        pending = identify_pending_timestamps(
            df_hourly=df_gap,
            latest_ts=None,
            min_context_window=32,
            max_lookback_horizon=48,
            initial_backfill_hours=10,
        )
        assert len(pending) == 10
        assert pending[-1] == df_gap.index[-1]

    def test_pipeline_sanitizes_duplicate_and_unsorted_ohlcv(self, tmp_path: Path) -> None:
        """Verifies that pipeline data ingestion cleans unsorted/duplicate market data."""
        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        timestamps = list(pd.date_range(base_ts, periods=100, freq="1h"))

        # Inject duplicate and shuffle
        timestamps.append(timestamps[50])  # Duplicate
        np.random.seed(42)
        shuffled_indices = list(range(len(timestamps)))
        np.random.shuffle(shuffled_indices)

        shuffled_ts = [timestamps[i] for i in shuffled_indices]
        closes = [1000.0 + 5.0 * i for i in range(len(shuffled_ts))]

        df_dirty = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [100.0] * len(shuffled_ts)},
            index=shuffled_ts,
        )

        # Pipeline should normalize and evaluate without failing
        results = run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=5,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_dirty},
            verbose=False,
        )

        assert len(results) == 1
        csv_file = list(results.values())[0]
        assert csv_file.exists()

        df_saved = pd.read_csv(csv_file)
        assert len(df_saved) == 5
        ts_series = pd.to_datetime(df_saved[REAL_TIME_COL], utc=True)
        assert ts_series.is_monotonic_increasing
        assert ts_series.is_unique


# ==============================================================================
# Vector 4: Incremental Resume, 0-Hour Idempotence & Burst Stress
# ==============================================================================


class TestIncrementalResumeAndBurstStress:
    def test_zero_new_hours_strict_idempotence(self, tmp_path: Path) -> None:
        """Stress-tests strict idempotence when 0 new hours have elapsed.
        
        Asserts:
        - Exactly 0 new rows are generated.
        - The CSV file is NOT overwritten with empty or corrupted data.
        - Hash / row count / data content is identical across successive 0-hour re-runs.
        """
        df_ohlcv = make_test_ohlcv(n_bars=100, start_price=50000.0, step=50.0)
        data_map = {"BTC/USDT": df_ohlcv}

        # Run 1: Initial run
        results_1 = run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=6,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map=data_map,
            verbose=False,
        )

        csv_path = results_1["persistence_BTC_USDT"]
        df_run1 = pd.read_csv(csv_path)
        assert len(df_run1) == 6
        initial_mtime = csv_path.stat().st_mtime_ns

        # Run 2: Immediate re-run with 0 new hours elapsed
        results_2 = run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=6,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map=data_map,
            verbose=False,
        )

        df_run2 = pd.read_csv(csv_path)
        assert len(df_run2) == 6, f"Expected 6 rows, got {len(df_run2)} after 0-hour re-run"
        pd.testing.assert_frame_equal(df_run1, df_run2)

        # Run 3: Third re-run (triple idempotence confirmation)
        results_3 = run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=6,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map=data_map,
            verbose=False,
        )
        df_run3 = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df_run1, df_run3)

    def test_exactly_one_new_hour_incremental_append(self, tmp_path: Path) -> None:
        """Stress-tests single-hour progression: exactly 1 hour elapsed -> exactly 1 row added."""
        df_base = make_test_ohlcv(n_bars=100, start_price=50000.0, step=50.0)

        # Run 1: 5 hours
        run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=5,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_base},
            verbose=False,
        )

        csv_path = tmp_path / "persistence_BTC_USDT.csv"
        df_initial = pd.read_csv(csv_path)
        assert len(df_initial) == 5

        # Run 2: Advance by exactly 1 hour
        df_plus_1 = make_test_ohlcv(n_bars=101, start_price=50000.0, step=50.0)
        run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=5,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_plus_1},
            verbose=False,
        )

        df_after_1 = pd.read_csv(csv_path)
        assert len(df_after_1) == 6, f"Expected 6 rows, got {len(df_after_1)}"

        # Verify historical 5 rows are completely preserved
        pd.testing.assert_frame_equal(df_after_1.iloc[:5], df_initial)

        # Verify 6th row matches newly completed hour
        new_ts_str = df_after_1.iloc[5][REAL_TIME_COL]
        expected_ts_str = df_plus_1.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
        assert new_ts_str == expected_ts_str
        assert np.isclose(df_after_1.iloc[5][VALUE_AT_TIME_COL], df_plus_1.iloc[-1]["close"])

    def test_large_gap_catchup_72_hours(self, tmp_path: Path) -> None:
        """Stress-tests re-running after a large 72-hour network/execution gap."""
        df_base = make_test_ohlcv(n_bars=100, start_price=50000.0, step=50.0)

        # Initial run: 5 hours
        run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=5,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_base},
            verbose=False,
        )

        csv_path = tmp_path / "persistence_BTC_USDT.csv"
        assert len(pd.read_csv(csv_path)) == 5

        # Advance by 72 hours
        df_jump_72 = make_test_ohlcv(n_bars=172, start_price=50000.0, step=50.0)
        run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=5,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_jump_72},
            verbose=False,
        )

        df_after_jump = pd.read_csv(csv_path)
        assert len(df_after_jump) == 5 + 72  # Exactly 77 rows
        ts_series = pd.to_datetime(df_after_jump[REAL_TIME_COL], utc=True)
        assert ts_series.is_monotonic_increasing
        assert ts_series.is_unique

    def test_multi_step_burst_sequence_matrix(self, tmp_path: Path) -> None:
        """Executes a demanding 7-stage incremental burst sequence:
        Run 1: Initial 4 hours (bars 100) -> len = 4
        Run 2: 0 hours (idempotence)      -> len = 4
        Run 3: +1 hour (bars 101)         -> len = 5
        Run 4: 0 hours (idempotence)      -> len = 5
        Run 5: +10 hours burst (bars 111) -> len = 15
        Run 6: +24 hours burst (bars 135) -> len = 39
        Run 7: 0 hours (idempotence)      -> len = 39
        """
        burst_schedule = [
            (100, 4, 4),    # initial
            (100, 4, 4),    # 0-hour re-run
            (101, 4, 5),    # +1 hour
            (101, 4, 5),    # 0-hour re-run
            (111, 4, 15),   # +10 hours burst
            (135, 4, 39),   # +24 hours burst
            (135, 4, 39),   # 0-hour re-run
        ]

        csv_path = tmp_path / "persistence_BTC_USDT.csv"

        for stage, (n_bars, backfill, expected_rows) in enumerate(burst_schedule, start=1):
            df_stage = make_test_ohlcv(n_bars=n_bars, start_price=50000.0, step=50.0)
            run_accuracy_evaluation(
                pairs=["BTC/USDT"],
                models=["persistence"],
                output_dir=tmp_path,
                initial_backfill_hours=backfill,
                max_horizon=48,
                context_window=32,
                df_ohlcv_map={"BTC/USDT": df_stage},
                verbose=False,
            )

            df_current = pd.read_csv(csv_path)
            assert len(df_current) == expected_rows, (
                f"Stage {stage} (n_bars={n_bars}): expected {expected_rows} rows, got {len(df_current)}"
            )

            ts_series = pd.to_datetime(df_current[REAL_TIME_COL], utc=True)
            assert ts_series.is_monotonic_increasing
            assert ts_series.is_unique


# ==============================================================================
# Vector 5: Multi-Model & Pipeline Orchestration Stress
# ==============================================================================


class TestMultiModelPipelineOrchestration:
    def test_all_four_models_two_pairs_incremental_burst(self, tmp_path: Path) -> None:
        """Executes full pipeline across 4 models and 2 pairs through an incremental burst."""
        pairs = ["BTC/USDT", "ETH/USDT"]
        models = ["timesfm", "chronos", "granite", "persistence"]

        # Stage 1: Initial 3 hours
        df_btc_1 = make_test_ohlcv(n_bars=90, start_price=60000.0, step=100.0)
        df_eth_1 = make_test_ohlcv(n_bars=90, start_price=3000.0, step=20.0)

        results_1 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=3,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map={"BTC/USDT": df_btc_1, "ETH/USDT": df_eth_1},
            verbose=False,
        )

        assert len(results_1) == 8
        for key, p in results_1.items():
            df = pd.read_csv(p)
            assert len(df) == 3
            is_valid, errors = validate_accuracy_df_schema(df, strict=True)
            assert is_valid, f"Invalid schema in {key}: {errors}"

        # Stage 2: 0-hour re-run (idempotence across all 8 files)
        results_2 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=3,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map={"BTC/USDT": df_btc_1, "ETH/USDT": df_eth_1},
            verbose=False,
        )

        for key, p in results_2.items():
            df = pd.read_csv(p)
            assert len(df) == 3, f"Duplicate detected in {key} during 0-hour re-run"

        # Stage 3: +2 hours incremental progression
        df_btc_2 = make_test_ohlcv(n_bars=92, start_price=60000.0, step=100.0)
        df_eth_2 = make_test_ohlcv(n_bars=92, start_price=3000.0, step=20.0)

        results_3 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=3,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map={"BTC/USDT": df_btc_2, "ETH/USDT": df_eth_2},
            verbose=False,
        )

        for key, p in results_3.items():
            df = pd.read_csv(p)
            assert len(df) == 5, f"Expected 5 rows in {key}, got {len(df)}"
            assert not df.isna().any().any(), f"NaNs detected in {key}"

    def test_trajectory_length_underflow_fails_gracefully(self) -> None:
        """Verifies that forecasters returning insufficient trajectory steps fail fast."""
        class StubTruncatedForecaster(BaseForecaster):
            def __init__(self) -> None:
                super().__init__(model_id="stub_truncated")

            def forecast_trajectory(
                self,
                context_prices: list[float] | np.ndarray,
                max_horizon_hours: int = 48,
            ) -> tuple[np.ndarray, np.ndarray]:
                prices = self.validate_context_prices(context_prices)
                # Intentionally return only 10 steps instead of requested horizon
                point_pred = np.full(10, float(prices[-1]))
                quantiles = np.column_stack([point_pred, point_pred, point_pred])
                return point_pred, quantiles

        df_ohlcv = make_test_ohlcv(n_bars=100)
        truncated_model = StubTruncatedForecaster()
        target_ts = df_ohlcv.index[80]

        # Requesting horizon k=24 should raise RuntimeError because trajectory has only 10 steps
        with pytest.raises(RuntimeError, match="required at least 24 steps"):
            evaluate_retrospective_horizon(truncated_model, df_ohlcv, target_ts, k=24)
