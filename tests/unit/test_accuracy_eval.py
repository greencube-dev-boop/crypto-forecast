"""Unit test suite for Retrospective 48-Hour Accuracy Evaluation Pipeline (M8)."""

from __future__ import annotations

from pathlib import Path
import sys
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
from crypto_forecast.eval.retrospective import (
    compute_retrospective_row,
    evaluate_historical_range_cached,
    evaluate_retrospective_horizon,
)
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    ACCURACY_EVAL_COLUMNS_SHORTHAND,
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


class StepAwareMockForecaster(BaseForecaster):
    """Deterministic mock forecaster predicting current_price + 100 * step."""

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
        quantiles = np.column_stack([point_pred, point_pred, point_pred])
        return point_pred, quantiles


class ContextSpyForecaster(BaseForecaster):
    """Spy forecaster capturing context prices passed to it."""

    def __init__(self) -> None:
        super().__init__(model_id="context_spy")
        self.last_context: np.ndarray | None = None

    def forecast_trajectory(
        self,
        context_prices: list[float] | np.ndarray,
        max_horizon_hours: int = 48,
    ) -> tuple[np.ndarray, np.ndarray]:
        prices = self.validate_context_prices(context_prices)
        self.last_context = np.asarray(prices, dtype=np.float64).copy()
        current = float(prices[-1])
        point_pred = np.full(max_horizon_hours, current)
        quantiles = np.column_stack([point_pred, point_pred, point_pred])
        return point_pred, quantiles


@pytest.fixture
def linear_ohlcv_120() -> pd.DataFrame:
    """Deterministic arithmetic OHLCV series of 120 hourly bars: close = 1000 + 10 * i."""
    base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(base_ts, periods=120, freq="1h")
    closes = [1000.0 + 10.0 * i for i in range(120)]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [100.0] * 120,
        },
        index=timestamps,
    )
    return df


# ==============================================================================
# 1. Retrospective Lookback Math & Temporal Isolation Tests
# ==============================================================================


class TestRetrospectiveMath:
    def test_temporal_isolation_no_lookahead(self, linear_ohlcv_120: pd.DataFrame) -> None:
        """Asserts that context array strictly ends at t - k hours for horizon k."""
        spy_model = ContextSpyForecaster()
        target_idx = 80
        target_ts = linear_ohlcv_120.index[target_idx]

        for k in (1, 6, 12, 24, 48):
            evaluate_retrospective_horizon(
                model=spy_model,
                df_hourly=linear_ohlcv_120,
                target_timestamp=target_ts,
                k=k,
                context_window=32,
                min_context=16,
            )
            expected_cutoff_price = float(linear_ohlcv_120.iloc[target_idx - k]["close"])
            assert spy_model.last_context is not None
            assert np.isclose(spy_model.last_context[-1], expected_cutoff_price), (
                f"Horizon k={k}: Context latest price was {spy_model.last_context[-1]}, "
                f"expected cutoff price {expected_cutoff_price}"
            )

    def test_intermediate_predictions_discarded(self, linear_ohlcv_120: pd.DataFrame) -> None:
        """Asserts that for horizon k, prediction at step k is recorded and 1..k-1 ignored."""
        mock_model = StepAwareMockForecaster()
        target_idx = 80
        target_ts = linear_ohlcv_120.index[target_idx]

        for k in range(1, 49):
            pred_k = evaluate_retrospective_horizon(
                model=mock_model,
                df_hourly=linear_ohlcv_120,
                target_timestamp=target_ts,
                k=k,
                context_window=64,
                min_context=32,
            )
            cutoff_price = float(linear_ohlcv_120.iloc[target_idx - k]["close"])
            expected_step_k = cutoff_price + 100.0 * k
            # Notice: intermediate step 1 would be cutoff_price + 100.0 * 1
            assert np.isclose(pred_k, expected_step_k), (
                f"Horizon k={k}: expected step k prediction {expected_step_k}, got {pred_k}"
            )
            if k >= 2:
                # Rigorously verify it is NOT step 1
                step_1_val = cutoff_price + 100.0 * 1
                assert not np.isclose(pred_k, step_1_val), "Intermediate step was not discarded!"

    def test_persistence_oracle_exact_identity(self, linear_ohlcv_120: pd.DataFrame) -> None:
        """Asserts that PersistenceForecaster matches P(t - k) analytically across all 48 horizons."""
        persistence = PersistenceForecaster()
        target_idx = 80
        target_ts = linear_ohlcv_120.index[target_idx]
        expected_gt = float(linear_ohlcv_120.iloc[target_idx]["close"])

        row = compute_retrospective_row(
            model=persistence,
            df_hourly=linear_ohlcv_120,
            target_timestamp=target_ts,
            max_horizon=48,
            context_window=64,
            min_context=32,
        )

        assert np.isclose(row[VALUE_AT_TIME_COL], expected_gt)
        for k in range(1, 49):
            col_name = f"value predicted {k}h before"
            expected_k = float(linear_ohlcv_120.iloc[target_idx - k]["close"])
            assert np.isclose(row[col_name], expected_k)

    def test_cached_batch_matches_single_point_evaluation(self, linear_ohlcv_120: pd.DataFrame) -> None:
        """Asserts that cached batch evaluation yields identical rows to single-point evaluation."""
        mock_model = StepAwareMockForecaster()
        targets = list(linear_ohlcv_120.index[80:83])

        # Method 1: Single point evaluation
        single_rows = [
            compute_retrospective_row(
                model=mock_model,
                df_hourly=linear_ohlcv_120,
                target_timestamp=t,
                max_horizon=48,
                context_window=64,
                min_context=32,
            )
            for t in targets
        ]
        df_single = pd.DataFrame(single_rows, columns=ACCURACY_EVAL_COLUMNS)

        # Method 2: Cached batch evaluation
        df_cached = evaluate_historical_range_cached(
            model=mock_model,
            df_hourly=linear_ohlcv_120,
            target_timestamps=targets,
            max_horizon=48,
            context_window=64,
            min_context=32,
        )

        pd.testing.assert_frame_equal(df_single, df_cached)

    def test_invalid_horizon_k_raises_error(self, linear_ohlcv_120: pd.DataFrame) -> None:
        persistence = PersistenceForecaster()
        ts = linear_ohlcv_120.index[80]
        with pytest.raises(ValueError, match="Lookback horizon k must be in"):
            evaluate_retrospective_horizon(persistence, linear_ohlcv_120, ts, k=0)
        with pytest.raises(ValueError, match="Lookback horizon k must be in"):
            evaluate_retrospective_horizon(persistence, linear_ohlcv_120, ts, k=49)

    def test_insufficient_history_raises_error(self, linear_ohlcv_120: pd.DataFrame) -> None:
        mock_model = StepAwareMockForecaster()
        ts = linear_ohlcv_120.index[5]  # Index 5 has only 6 bars
        with pytest.raises(ValueError, match="Insufficient historical bars"):
            evaluate_retrospective_horizon(mock_model, linear_ohlcv_120, ts, k=4, min_context=10)


# ==============================================================================
# 2. CSV Schema & File Naming Tests
# ==============================================================================


class TestCSVSchema:
    def test_exact_50_columns_header_structure(self) -> None:
        assert len(ACCURACY_EVAL_COLUMNS) == 50
        assert ACCURACY_EVAL_COLUMNS[0] == "real time"
        assert ACCURACY_EVAL_COLUMNS[1] == "value at time"
        for k in range(1, 49):
            assert ACCURACY_EVAL_COLUMNS[1 + k] == f"value predicted {k}h before"

    def test_canonicalize_columns_shorthand(self) -> None:
        data = {
            "real time": ["2026-11-06T00:00:00Z"],
            "value at time": [50000.0],
            "value predicted 1h before": [49900.0],
        }
        for k in range(2, 49):
            data[f"{k}h before"] = [49000.0 + k]

        df_shorthand = pd.DataFrame(data)
        df_canonical = canonicalize_columns(df_shorthand)

        assert df_canonical.shape[1] == 50
        is_valid, errors = validate_accuracy_df_schema(df_canonical, strict=True)
        assert is_valid, f"Validation failed: {errors}"

    def test_model_prefix_and_file_naming(self) -> None:
        assert get_model_prefix("TimesFMForecaster") == "timesfm"
        assert get_model_prefix("ChronosForecaster") == "chronos"
        assert get_model_prefix("GraniteTTMForecaster") == "granite"
        assert get_model_prefix("PersistenceForecaster") == "persistence"

        assert get_accuracy_csv_filename("timesfm", "BTC/USDT") == "timesfm_BTC_USDT.csv"
        assert get_accuracy_csv_filename("chronos", "ETH/USDT") == "chronos_ETH_USDT.csv"
        assert get_accuracy_csv_filename("granite", "SOL-USD") == "granite_SOL_USD.csv"
        assert get_accuracy_csv_filename("persistence", "DOGE:USDT") == "persistence_DOGE_USDT.csv"

    def test_sanitize_pair_symbol(self) -> None:
        assert sanitize_pair_symbol("BTC/USDT") == "BTC_USDT"
        assert sanitize_pair_symbol("eth-usdt") == "ETH_USD" or sanitize_pair_symbol("eth-usdt") == "ETH_USDT"
        assert sanitize_pair_symbol("  sol/usd  ") == "SOL_USD"


# ==============================================================================
# 3. Incremental State & Dataset Persistence Tests
# ==============================================================================


class TestDatasetPersistence:
    def test_inspect_latest_recorded_timestamp_nonexistent_or_empty(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "missing.csv"
        assert inspect_latest_recorded_timestamp(nonexistent) is None

        empty_file = tmp_path / "empty.csv"
        empty_file.touch()
        assert inspect_latest_recorded_timestamp(empty_file) is None

    def test_inspect_latest_recorded_timestamp_valid(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({
            "real time": ["2026-11-06T01:00:00Z", "2026-11-06T03:00:00Z", "2026-11-06T02:00:00Z"],
            "value at time": [100.0, 102.0, 101.0],
        })
        df.to_csv(csv_file, index=False)

        latest = inspect_latest_recorded_timestamp(csv_file)
        assert latest is not None
        assert latest == pd.Timestamp("2026-11-06 03:00:00", tz="UTC")

    def test_identify_pending_timestamps(self, linear_ohlcv_120: pd.DataFrame) -> None:
        # Fresh run: should return the requested initial_backfill_hours
        fresh_pending = identify_pending_timestamps(
            df_hourly=linear_ohlcv_120,
            latest_ts=None,
            min_context_window=32,
            max_lookback_horizon=48,
            initial_backfill_hours=10,
        )
        assert len(fresh_pending) == 10
        assert fresh_pending[-1] == linear_ohlcv_120.index[-1]

        # Incremental run: all timestamps strictly greater than latest
        latest = fresh_pending[4]  # 5th item
        incr_pending = identify_pending_timestamps(
            df_hourly=linear_ohlcv_120,
            latest_ts=latest,
            min_context_window=32,
            max_lookback_horizon=48,
        )
        assert len(incr_pending) == 5
        assert all(t > latest for t in incr_pending)

    def test_persist_accuracy_dataset_atomic_and_deduplicated(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "dataset.csv"

        df_1 = pd.DataFrame({
            "real time": ["2026-11-06T01:00:00Z", "2026-11-06T02:00:00Z"],
            "value at time": [100.0, 101.0],
        })
        for k in range(1, 49):
            df_1[f"value predicted {k}h before"] = 99.0 + k

        saved_1 = persist_accuracy_dataset(df_1, csv_file)
        assert len(saved_1) == 2
        assert csv_file.exists()

        # Append overlapping row + 1 new row
        df_2 = pd.DataFrame({
            "real time": ["2026-11-06T02:00:00Z", "2026-11-06T03:00:00Z"],
            "value at time": [999.0, 102.0],  # Overlapping row has altered value; should preserve original
        })
        for k in range(1, 49):
            df_2[f"value predicted {k}h before"] = 100.0 + k

        saved_2 = persist_accuracy_dataset(df_2, csv_file)
        assert len(saved_2) == 3
        # Original value at 02:00 should be preserved (101.0, not 999.0)
        row_2 = saved_2[saved_2["real time"] == "2026-11-06T02:00:00Z"].iloc[0]
        assert np.isclose(row_2["value at time"], 101.0)

    def test_get_accuracy_eval_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_eval"
        resolved = get_accuracy_eval_dir(custom)
        assert resolved == custom
        assert resolved.exists()
