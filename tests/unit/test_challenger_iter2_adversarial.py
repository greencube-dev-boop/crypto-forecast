"""Adversarial Verification Suite for BUG-01 and BUG-02 Fixes in Dataset Persistence.

Authored by: challenger_accuracy_iter2_1
Focus:
- BUG-01: No all-NaN phantom row injection under any corrupt CSV structure (missing real time, whitespace, null rows).
- BUG-02: No DateParseError or unhandled crash on malformed/unparseable timestamps.
- Data Preservation: Existing valid rows are strictly preserved and sorted chronologically.
- Incremental Pipeline Healing: Pipeline auto-recovers from corrupted CSVs without manual intervention.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_forecast.eval.dataset import (
    inspect_latest_recorded_timestamp,
    persist_accuracy_dataset,
)
from crypto_forecast.eval.pipeline import run_accuracy_evaluation
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
    validate_accuracy_df_schema,
)


def _make_sample_row(ts: str = "2026-11-06T00:00:00Z", val: float = 50000.0) -> pd.DataFrame:
    df = pd.DataFrame({
        REAL_TIME_COL: [ts],
        VALUE_AT_TIME_COL: [val],
    })
    for k in range(1, 49):
        df[f"value predicted {k}h before"] = [val - 100.0 + k]
    return df


class TestBUG01AdversarialVariations:
    """Rigorous probes ensuring zero all-NaN phantom rows under all corrupt CSV topologies."""

    def test_existing_csv_with_arbitrary_columns_and_no_real_time(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "arbitrary_cols.csv"
        csv_file.write_text("foo,bar,baz\n10,20,30\n40,50,60\n", encoding="utf-8")

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        df_new = _make_sample_row("2026-11-06T01:00:00Z", 65000.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        # Invariant: exactly 1 row, 0 NaN rows, valid schema
        assert len(persisted) == 1
        assert not persisted[REAL_TIME_COL].isna().any()
        assert not persisted.isna().all(axis=1).any()
        is_valid, errors = validate_accuracy_df_schema(persisted, strict=True)
        assert is_valid, f"Invalid schema: {errors}"
        assert persisted.iloc[0][REAL_TIME_COL] == "2026-11-06T01:00:00Z"
        assert persisted.iloc[0][VALUE_AT_TIME_COL] == 65000.0

    def test_existing_csv_with_real_time_column_all_nan(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "all_nan_real_time.csv"
        csv_file.write_text("real time,value at time\n,\n,\n", encoding="utf-8")

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        df_new = _make_sample_row("2026-11-06T01:00:00Z", 65000.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        assert len(persisted) == 1
        assert persisted.iloc[0][REAL_TIME_COL] == "2026-11-06T01:00:00Z"
        assert not persisted[REAL_TIME_COL].isna().any()
        assert not persisted.isna().all(axis=1).any()

    def test_existing_csv_with_mixed_valid_and_all_nan_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "mixed_nan_rows.csv"
        # Row 1 is valid, Row 2 is empty/NaN, Row 3 is valid
        csv_file.write_text(
            "real time,value at time\n"
            "2026-11-06T01:00:00Z,100.0\n"
            ",\n"
            "2026-11-06T02:00:00Z,102.0\n",
            encoding="utf-8",
        )

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest == pd.Timestamp("2026-11-06T02:00:00Z")

        df_new = _make_sample_row("2026-11-06T03:00:00Z", 103.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        # Invariant: 3 valid rows (01, 02, 03), zero NaN rows
        assert len(persisted) == 3
        assert not persisted[REAL_TIME_COL].isna().any()
        assert not persisted.isna().all(axis=1).any()
        assert list(persisted[REAL_TIME_COL]) == [
            "2026-11-06T01:00:00Z",
            "2026-11-06T02:00:00Z",
            "2026-11-06T03:00:00Z",
        ]

    def test_existing_csv_with_whitespace_and_blank_lines(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "blank_lines.csv"
        csv_file.write_text(
            "\n\nreal time,value at time\n\n2026-11-06T01:00:00Z,100.0\n\n\n",
            encoding="utf-8",
        )

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest == pd.Timestamp("2026-11-06T01:00:00Z")

        df_new = _make_sample_row("2026-11-06T02:00:00Z", 102.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        assert len(persisted) == 2
        assert not persisted[REAL_TIME_COL].isna().any()
        assert not persisted.isna().all(axis=1).any()


class TestBUG02AdversarialVariations:
    """Rigorous probes ensuring DateParseError resilience across corrupt timestamp formats."""

    def test_unparseable_random_strings_purged_safely(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "corrupt_dates.csv"
        csv_file.write_text(
            "real time,value at time\n"
            "NOT_A_DATE,100.0\n"
            "2026-11-06T01:00:00Z,101.0\n"
            "???INVALID###,102.0\n"
            "99999-99-99,103.0\n"
            "2026-11-06T04:00:00Z,104.0\n",
            encoding="utf-8",
        )

        # inspect_latest_recorded_timestamp must find maximum valid date (04:00:00Z) without crashing
        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest == pd.Timestamp("2026-11-06T04:00:00Z")

        df_new = _make_sample_row("2026-11-06T05:00:00Z", 105.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        # Invariant: 3 valid rows (01, 04, 05), all corrupted rows cleanly purged
        assert len(persisted) == 3
        assert list(persisted[REAL_TIME_COL]) == [
            "2026-11-06T01:00:00Z",
            "2026-11-06T04:00:00Z",
            "2026-11-06T05:00:00Z",
        ]
        assert not persisted[REAL_TIME_COL].isna().any()

    def test_all_rows_corrupt_timestamps_recovers_to_new_data(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "all_corrupt_dates.csv"
        csv_file.write_text(
            "real time,value at time\n"
            "JUNK_1,10.0\n"
            "JUNK_2,20.0\n",
            encoding="utf-8",
        )

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        df_new = _make_sample_row("2026-11-06T01:00:00Z", 100.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        assert len(persisted) == 1
        assert persisted.iloc[0][REAL_TIME_COL] == "2026-11-06T01:00:00Z"
        assert not persisted[REAL_TIME_COL].isna().any()


class TestPipelineSelfHealingAndDeduplication:
    """Validates end-to-end pipeline recovery when pre-existing CSV is corrupted."""

    def test_pipeline_recovers_cleanly_from_corrupt_preexisting_csv(self, tmp_path: Path) -> None:
        # Pre-seed a corrupt CSV for BTC/USDT persistence
        corrupt_csv = tmp_path / "persistence_BTC_USDT.csv"
        corrupt_csv.write_text("bad_col1,bad_col2\nfoo,bar\n", encoding="utf-8")

        base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
        timestamps = pd.date_range(base_ts, periods=90, freq="1h")
        closes = [1000.0 + 10.0 * i for i in range(90)]
        df_ohlcv = pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [100.0] * 90},
            index=timestamps,
        )

        # Pipeline should not crash; it should heal the file
        results = run_accuracy_evaluation(
            pairs=["BTC/USDT"],
            models=["persistence"],
            output_dir=tmp_path,
            initial_backfill_hours=3,
            max_horizon=48,
            context_window=32,
            df_ohlcv_map={"BTC/USDT": df_ohlcv},
            verbose=False,
        )

        assert len(results) == 1
        df_saved = pd.read_csv(corrupt_csv)
        assert len(df_saved) == 3
        is_valid, errors = validate_accuracy_df_schema(df_saved, strict=True)
        assert is_valid, f"Pipeline healed CSV has invalid schema: {errors}"
        assert not df_saved[REAL_TIME_COL].isna().any()
        assert not df_saved.isna().all(axis=1).any()

    def test_df_new_internal_corrupted_rows_purged(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "df_new_corrupt.csv"
        # df_new contains 1 valid row, 1 row with NaN timestamp, 1 row with unparseable timestamp
        df_dirty_new = pd.DataFrame({
            REAL_TIME_COL: ["2026-11-06T01:00:00Z", None, "NOT_A_VALID_DATE", "2026-11-06T02:00:00Z"],
            VALUE_AT_TIME_COL: [100.0, 200.0, 300.0, 400.0],
        })
        for k in range(1, 49):
            df_dirty_new[f"value predicted {k}h before"] = 99.0

        persisted = persist_accuracy_dataset(df_dirty_new, csv_file)
        assert len(persisted) == 2
        assert list(persisted[REAL_TIME_COL]) == ["2026-11-06T01:00:00Z", "2026-11-06T02:00:00Z"]
        assert not persisted[REAL_TIME_COL].isna().any()

    def test_existing_row_preservation_over_new_duplicate(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "preservation_test.csv"
        ts = "2026-11-06T01:00:00Z"
        df_existing = _make_sample_row(ts=ts, val=111.1)
        persist_accuracy_dataset(df_existing, csv_file)

        # Attempt to append the same timestamp with different value
        df_new_dup = _make_sample_row(ts=ts, val=999.9)
        persisted = persist_accuracy_dataset(df_new_dup, csv_file)

        assert len(persisted) == 1
        # Existing value 111.1 MUST be preserved (keep="first")
        assert np.isclose(persisted.iloc[0][VALUE_AT_TIME_COL], 111.1)

    def test_existing_csv_null_bytes_recovery(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "null_bytes.csv"
        csv_file.write_bytes(b"\x00" * 256)

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is None

        df_new = _make_sample_row("2026-11-06T01:00:00Z", 555.0)
        persisted = persist_accuracy_dataset(df_new, csv_file)

        assert len(persisted) == 1
        assert persisted.iloc[0][REAL_TIME_COL] == "2026-11-06T01:00:00Z"
        assert not persisted[REAL_TIME_COL].isna().any()

