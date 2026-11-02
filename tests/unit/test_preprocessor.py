"""Unit tests for crypto_forecast.data.preprocessor module.

Covers Feature 4 (Dual-Resolution Alignment) and Feature 5 (Gap Imputation),
Target Horizon Generation, and Chronological Splitting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import numpy as np
import pandas as pd
import pytest

from crypto_forecast.data.preprocessor import (
    DEFAULT_HORIZONS,
    INTRADAY_HORIZONS,
    MACRO_HORIZONS,
    add_target_horizons,
    create_dual_resolution_targets,
    detect_timestamp_gaps,
    get_dual_resolution_data,
    get_horizon_steps,
    normalize_datetime_index,
    reindex_and_impute,
    resample_dual_resolution,
    resample_to_daily,
    resample_to_hourly,
    train_val_test_split,
)


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def sample_hourly_ohlcv() -> pd.DataFrame:
  """Generates 120 contiguous hourly candles from 2026-01-01 00:00:00 UTC."""
  index = pd.date_range(
      start="2026-01-01 00:00:00", periods=120, freq="1h", tz="UTC"
  )
  np.random.seed(42)
  base_price = 70000.0
  returns = np.random.normal(0.0002, 0.005, size=len(index))
  close = base_price * np.exp(np.cumsum(returns))
  open_p = np.roll(close, 1)
  open_p[0] = base_price
  high = np.maximum(open_p, close) + np.abs(np.random.normal(50, 10, len(index)))
  low = np.minimum(open_p, close) - np.abs(np.random.normal(50, 10, len(index)))
  volume = np.random.uniform(10.0, 200.0, size=len(index))

  return pd.DataFrame(
      {
          "open": open_p,
          "high": high,
          "low": low,
          "close": close,
          "volume": volume,
      },
      index=index,
  )


@pytest.fixture
def sample_hourly_with_gaps(sample_hourly_ohlcv: pd.DataFrame) -> pd.DataFrame:
  """Drops specific rows to simulate exchange outages and missing bars."""
  df = sample_hourly_ohlcv.copy()
  # Drop single hour at index 15
  df = df.drop(df.index[15])
  # Drop multi-hour gap from index 40 to 44 (4 hours missing)
  drop_range = df.index[40:44]
  df = df.drop(drop_range)
  return df


# ============================================================================
# Tier 1: Feature Coverage Tests
# ============================================================================


class TestTier1FeatureCoverage:
  """Tests verifying standard operation and core requirements."""

  def test_get_horizon_steps(self):
    """Verifies horizon step conversion for hourly and daily resolutions."""
    assert get_horizon_steps("1h", "1h") == 1
    assert get_horizon_steps("24h", "1h") == 24
    assert get_horizon_steps("2d", "1h") == 48
    assert get_horizon_steps("30d", "1h") == 720
    assert get_horizon_steps("1d", "1d") == 1
    assert get_horizon_steps("24h", "1d") == 1
    assert get_horizon_steps("7d", "1d") == 7
    assert get_horizon_steps("30d", "1d") == 30

  def test_normalize_datetime_index_from_column(self):
    """Verifies that string/epoch timestamp columns are converted to UTC DatetimeIndex."""
    data = {
        "timestamp": [
            "2026-01-01 00:00:00",
            "2026-01-01 01:00:00",
            "2026-01-01 02:00:00",
        ],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [10.0, 15.0, 20.0],
    }
    df = pd.DataFrame(data)
    norm = normalize_datetime_index(df)

    assert isinstance(norm.index, pd.DatetimeIndex)
    assert str(norm.index.tz) == "UTC"
    assert norm.index.name == "timestamp"
    assert "timestamp" not in norm.columns
    assert len(norm) == 3

  def test_reindex_and_impute_single_gap(
      self, sample_hourly_with_gaps: pd.DataFrame
  ):
    """Verifies that missing bars are reindexed and imputed according to financial rules."""
    gaps = detect_timestamp_gaps(sample_hourly_with_gaps, freq="1h")
    assert len(gaps) >= 2

    imputed = reindex_and_impute(
        sample_hourly_with_gaps, freq="1h", add_imputed_flag=True
    )

    # Verifies index is fully continuous
    expected_len = 120
    assert len(imputed) == expected_len
    assert imputed.isna().sum().sum() == 0

    # Verifies imputed bar at hour 15 has open=high=low=close and volume=0.0
    gap_ts = sample_hourly_with_gaps.index[0] + pd.Timedelta(15, unit="h")
    row = imputed.loc[gap_ts]
    prev_row = imputed.loc[gap_ts - pd.Timedelta(1, unit="h")]

    assert row["is_imputed"] is True or row["is_imputed"] == 1
    assert row["close"] == prev_row["close"]
    assert row["open"] == row["close"]
    assert row["high"] == row["close"]
    assert row["low"] == row["close"]
    assert row["volume"] == 0.0

  def test_resample_to_daily_ohlcv_aggregation(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies 1h to 1d resampling aggregation math: open=first, high=max, low=min, close=last, volume=sum."""
    daily = resample_to_daily(sample_hourly_ohlcv)

    # 120 hours = 5 days (from 2026-01-01 to 2026-01-05)
    assert len(daily) == 5
    first_day = sample_hourly_ohlcv.iloc[:24]
    daily_first = daily.iloc[0]

    assert daily_first["open"] == first_day["open"].iloc[0]
    assert daily_first["high"] == first_day["high"].max()
    assert daily_first["low"] == first_day["low"].min()
    assert daily_first["close"] == first_day["close"].iloc[-1]
    assert np.isclose(daily_first["volume"], first_day["volume"].sum())

  def test_resample_dual_resolution_structure(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies resample_dual_resolution outputs both hourly and daily streams."""
    df_h, df_d = resample_dual_resolution(sample_hourly_ohlcv)

    assert isinstance(df_h, pd.DataFrame)
    assert isinstance(df_d, pd.DataFrame)
    assert len(df_h) == 120
    assert len(df_d) == 5
    assert df_h.index.freqstr in ("1h", "h", "1H")
    assert df_d.index.freqstr in ("1D", "D", "1d")

  def test_add_target_horizons_intraday(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies target prices and relative returns for intraday horizons."""
    df_targets = add_target_horizons(sample_hourly_ohlcv, resolution="1h")

    for h in INTRADAY_HORIZONS:
      steps = get_horizon_steps(h, "1h")
      price_col = f"target_price_{h}"
      return_col = f"target_return_{h}"
      dir_col = f"target_direction_{h}"

      assert price_col in df_targets.columns
      assert return_col in df_targets.columns
      assert dir_col in df_targets.columns

      # Verify math at t=0
      expected_future_price = sample_hourly_ohlcv["close"].iloc[steps]
      expected_return = (
          expected_future_price - sample_hourly_ohlcv["close"].iloc[0]
      ) / sample_hourly_ohlcv["close"].iloc[0]
      assert np.isclose(df_targets[price_col].iloc[0], expected_future_price)
      assert np.isclose(df_targets[return_col].iloc[0], expected_return)

  def test_create_dual_resolution_targets(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies decorating both hourly and daily streams with native horizons."""
    df_h, df_d = resample_dual_resolution(sample_hourly_ohlcv)
    df_h_targets, df_d_targets = create_dual_resolution_targets(df_h, df_d)

    for h in INTRADAY_HORIZONS:
      assert f"target_price_{h}" in df_h_targets.columns
    for h in MACRO_HORIZONS:
      assert f"target_price_{h}" in df_d_targets.columns

  def test_train_val_test_split_ratios(self, sample_hourly_ohlcv: pd.DataFrame):
    """Verifies chronological 70/15/15 train/val/test splitting."""
    train, val, test = train_val_test_split(
        sample_hourly_ohlcv, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    n = len(sample_hourly_ohlcv)
    assert len(train) == int(n * 0.70)
    assert len(val) == int(n * 0.15)
    assert len(test) == n - len(train) - len(val)

    # Strictly chronological
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

  def test_get_dual_resolution_data_mocked(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies get_dual_resolution_data facade orchestration."""
    with patch(
        "crypto_forecast.data.ingestion.fetch_ohlcv",
        return_value=sample_hourly_ohlcv,
    ):
      df_h, df_d = get_dual_resolution_data("BTC/USDT")
      assert len(df_h) == len(sample_hourly_ohlcv)
      assert len(df_d) == 5


# ============================================================================
# Tier 2: Boundary & Corner Cases Tests
# ============================================================================


class TestTier2BoundaryCases:
  """Tests verifying edge cases, boundary limits, and error handling."""

  def test_empty_dataframe_raises(self):
    """Verifies that empty DataFrames raise descriptive ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
      normalize_datetime_index(empty_df)
    with pytest.raises(ValueError, match="empty"):
      reindex_and_impute(empty_df)
    with pytest.raises(ValueError, match="empty"):
      train_val_test_split(empty_df)
    with pytest.raises(ValueError, match="empty"):
      add_target_horizons(empty_df)

  def test_single_row_dataframe(self):
    """Verifies graceful handling of single-row DataFrame."""
    idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [50.0],
        },
        index=idx,
    )
    norm = normalize_datetime_index(df)
    assert len(norm) == 1
    gaps = detect_timestamp_gaps(norm, freq="1h")
    assert gaps == []

  def test_unsorted_and_duplicate_timestamps(self):
    """Verifies sorting and deduplication of disordered timestamps."""
    dates = pd.to_datetime(
        ["2026-01-01 02:00:00", "2026-01-01 00:00:00", "2026-01-01 01:00:00"]
    )
    df = pd.DataFrame({"close": [102.0, 100.0, 101.0]}, index=dates)
    norm = normalize_datetime_index(df)

    assert norm.index.is_monotonic_increasing
    assert norm["close"].iloc[0] == 100.0
    assert norm["close"].iloc[-1] == 102.0

  def test_target_horizons_nan_tail(self, sample_hourly_ohlcv: pd.DataFrame):
    """Verifies that exactly the last k rows contain NaN for horizon h (zero lookahead)."""
    df_targets = add_target_horizons(
        sample_hourly_ohlcv, resolution="1h", horizons=["4h"]
    )
    steps = 4
    # The last 4 rows must be NaN in target_price_4h
    assert df_targets["target_price_4h"].iloc[-steps:].isna().all()
    # The row right before must NOT be NaN
    assert not np.isnan(df_targets["target_price_4h"].iloc[-(steps + 1)])

  def test_constant_price_zero_return(self):
    """Verifies flat prices produce zero returns without division by zero or errors."""
    idx = pd.date_range("2026-01-01", periods=20, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 10.0,
        },
        index=idx,
    )
    df_targets = add_target_horizons(
        df, resolution="1h", horizons=["1h", "2h", "4h"]
    )
    assert (df_targets["target_return_1h"].dropna() == 0.0).all()
    assert (df_targets["target_direction_1h"].dropna() == 1.0).all()

  def test_chronological_split_embargo_isolation(
      self, sample_hourly_ohlcv: pd.DataFrame
  ):
    """Verifies embargo buffer creates an unshared temporal gap between train and val."""
    embargo = 12
    train, val, test = train_val_test_split(
        sample_hourly_ohlcv,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        embargo_steps=embargo,
    )

    # Time delta between last train bar and first val bar must be >= embargo hours
    gap_delta = val.index.min() - train.index.max()
    assert gap_delta >= pd.Timedelta(embargo, unit="h")

  def test_drop_incomplete_days(self, sample_hourly_ohlcv: pd.DataFrame):
    """Verifies drop_incomplete_days drops calendar days that do not have 24 hourly bars."""
    # Slice 50 hours: Day 1 has 24h, Day 2 has 24h, Day 3 has 2h (incomplete)
    df_partial = sample_hourly_ohlcv.iloc[:50]
    daily_all = resample_to_daily(df_partial, drop_incomplete_days=False)
    daily_complete = resample_to_daily(df_partial, drop_incomplete_days=True)

    assert len(daily_all) == 3
    assert len(daily_complete) == 2

  def test_invalid_split_ratios_raise(self, sample_hourly_ohlcv: pd.DataFrame):
    """Verifies ValueError on invalid split ratios not summing to 1.0."""
    with pytest.raises(ValueError, match="sum to 1.0"):
      train_val_test_split(
          sample_hourly_ohlcv, train_ratio=0.5, val_ratio=0.2, test_ratio=0.1
      )
