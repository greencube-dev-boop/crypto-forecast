"""Unit Test Suite for Technical Indicator Vectorized Formulations.

Verifies mathematical correctness, zero-division immunity, boundary invariance,
and non-destructive row preservation across:
- Tier 1: Mathematical correctness & known analytical test vectors
- Tier 2: Extreme boundary conditions (zero volatility, flat prices, zero volume, NaNs)
- Tier 3: Invariant contracts (bounds, band orderings, ATR positivity)
- Tier 4: Unified interface validation and warm-up preservation
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.data.indicators import (
    add_technical_indicators,
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_volume_zscore,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
  """Deterministic 50-row OHLCV fixture with continuous datetime index."""
  dates = pd.date_range("2026-01-01 00:00:00", periods=50, freq="1h")
  np.random.seed(42)
  # Generate synthetic random walk with drift
  returns = np.random.normal(0.001, 0.01, size=50)
  price = 100.0 * np.exp(np.cumsum(returns))

  df = pd.DataFrame(
      {
          "open": price * (1.0 - np.random.uniform(0.0, 0.005, size=50)),
          "high": price * (1.0 + np.random.uniform(0.001, 0.01, size=50)),
          "low": price * (1.0 - np.random.uniform(0.001, 0.01, size=50)),
          "close": price,
          "volume": np.random.uniform(100.0, 5000.0, size=50),
      },
      index=dates,
  )
  # Ensure high >= low and bounds contain open and close
  df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.1
  df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.1
  return df


# ============================================================================
# Tier 1: Mathematical Correctness & Known Test Vectors
# ============================================================================


def test_rsi_monotonic_increasing() -> None:
  """RSI of a strictly increasing price series must converge to 100.0."""
  series = pd.Series(np.linspace(100.0, 200.0, 30))
  rsi = compute_rsi(series, period=14)
  # After warm-up, all price changes are gains -> RSI must be 100.0
  valid_rsi = rsi.dropna()
  assert len(valid_rsi) > 0
  np.testing.assert_allclose(valid_rsi.values, 100.0, atol=1e-5)


def test_rsi_monotonic_decreasing() -> None:
  """RSI of a strictly decreasing price series must converge to 0.0."""
  series = pd.Series(np.linspace(200.0, 100.0, 30))
  rsi = compute_rsi(series, period=14)
  valid_rsi = rsi.dropna()
  assert len(valid_rsi) > 0
  np.testing.assert_allclose(valid_rsi.values, 0.0, atol=1e-5)


def test_macd_constant_price() -> None:
  """MACD on a flat price series must be exactly 0.0 for line, signal, and hist."""
  series = pd.Series([100.0] * 50)
  macd, signal, hist = compute_macd(series, fast=12, slow=26, signal=9)
  np.testing.assert_allclose(macd.values, 0.0, atol=1e-9)
  np.testing.assert_allclose(signal.values, 0.0, atol=1e-9)
  np.testing.assert_allclose(hist.values, 0.0, atol=1e-9)


def test_bollinger_bands_known_values() -> None:
  """Bollinger Bands must compute exact moving average and population standard deviation."""
  # Series with constant increment: [1, 2, ..., 20]
  series = pd.Series(np.arange(1.0, 21.0))
  upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(
      series, period=20, num_std=2.0
  )

  # Analytical values at index 19 (row 20):
  # Mean = (1 + 20) / 2 = 10.5
  # Var = (20^2 - 1) / 12 = 399 / 12 = 33.25 -> Std = sqrt(33.25) ≈ 5.76628
  expected_mean = 10.5
  expected_std = np.sqrt(33.25)
  expected_upper = expected_mean + 2.0 * expected_std
  expected_lower = expected_mean - 2.0 * expected_std

  assert np.isclose(middle.iloc[-1], expected_mean, atol=1e-5)
  assert np.isclose(upper.iloc[-1], expected_upper, atol=1e-5)
  assert np.isclose(lower.iloc[-1], expected_lower, atol=1e-5)
  assert np.isclose(
      pct_b.iloc[-1],
      (20.0 - expected_lower) / (expected_upper - expected_lower),
      atol=1e-5,
  )


def test_atr_fixed_range() -> None:
  """ATR on bars with constant range (H-L=10, no gap) must equal 10.0."""
  n = 30
  df = pd.DataFrame({
      "high": [105.0] * n,
      "low": [95.0] * n,
      "close": [100.0] * n,
  })
  atr = compute_atr(df, period=14)
  valid_atr = atr.dropna()
  assert len(valid_atr) > 0
  np.testing.assert_allclose(valid_atr.values, 10.0, atol=1e-5)


def test_obv_known_sequence() -> None:
  """OBV must accumulate volume in strict accordance with Granville's formula."""
  df = pd.DataFrame({
      "close": [100.0, 105.0, 102.0, 102.0, 110.0],
      "volume": [1000.0, 500.0, 300.0, 200.0, 800.0],
  })
  # Step-by-step:
  # t=0: OBV = 1000.0
  # t=1: Close 105 > 100 -> OBV = 1000 + 500 = 1500.0
  # t=2: Close 102 < 105 -> OBV = 1500 - 300 = 1200.0
  # t=3: Close 102 == 102 -> OBV = 1200.0
  # t=4: Close 110 > 102 -> OBV = 1200 + 800 = 2000.0
  expected_obv = [1000.0, 1500.0, 1200.0, 1200.0, 2000.0]
  obv = compute_obv(df)
  np.testing.assert_allclose(obv.values, expected_obv, atol=1e-7)


def test_volume_zscore_known_surge() -> None:
  """Volume Z-score must correctly measure standard deviation divergence."""
  # 19 bars of 100.0 volume, 20th bar surges to 300.0
  vol = [100.0] * 19 + [300.0]
  series = pd.Series(vol)
  z = compute_volume_zscore(series, period=20)

  # Mean of 19*100 + 300 = 2200 / 20 = 110.0
  # Variance = (19*(100-110)^2 + (300-110)^2) / 20 = (19*100 + 36100) / 20 = 38000 / 20 = 1900.0
  # Std = sqrt(1900) ≈ 43.588989
  # Z = (300 - 110) / 43.588989 ≈ 4.3588989
  expected_z = (300.0 - 110.0) / np.sqrt(1900.0)
  assert np.isclose(z.iloc[-1], expected_z, atol=1e-5)


# ============================================================================
# Tier 2: Boundary & Corner Cases (Numerical Stability & Zero-Division)
# ============================================================================


def test_rsi_zero_volatility_neutral_50() -> None:
  """RSI must evaluate to exactly 50.0 without ZeroDivisionError when price is constant."""
  series = pd.Series([100.0] * 30)
  rsi = compute_rsi(series, period=14)
  valid_rsi = rsi.dropna()
  assert len(valid_rsi) > 0
  np.testing.assert_allclose(valid_rsi.values, 50.0, atol=1e-7)


def test_bollinger_bands_zero_volatility() -> None:
  """Bollinger Bands on constant price must yield upper == middle == lower, %B == 0.5, Bandwidth == 0.0."""
  series = pd.Series([50.0] * 30)
  upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(
      series, period=20
  )
  valid_pct_b = pct_b.dropna()
  valid_bw = bandwidth.dropna()

  assert len(valid_pct_b) > 0
  np.testing.assert_allclose(upper.dropna().values, 50.0, atol=1e-7)
  np.testing.assert_allclose(middle.dropna().values, 50.0, atol=1e-7)
  np.testing.assert_allclose(lower.dropna().values, 50.0, atol=1e-7)
  np.testing.assert_allclose(valid_pct_b.values, 0.5, atol=1e-7)
  np.testing.assert_allclose(valid_bw.values, 0.0, atol=1e-7)


def test_volume_zscore_zero_variance() -> None:
  """Volume Z-score must return 0.0 without ZeroDivisionError when volume is completely constant."""
  series = pd.Series([250.0] * 30)
  z = compute_volume_zscore(series, period=20)
  valid_z = z.dropna()
  assert len(valid_z) > 0
  np.testing.assert_allclose(valid_z.values, 0.0, atol=1e-7)


def test_short_series_insufficient_length() -> None:
  """Indicator functions must gracefully return all NaNs when len(data) < period."""
  short_series = pd.Series([10.0, 11.0, 12.0])
  short_df = pd.DataFrame({
      "open": [10.0, 11.0, 12.0],
      "high": [10.5, 11.5, 12.5],
      "low": [9.5, 10.5, 11.5],
      "close": [10.0, 11.0, 12.0],
      "volume": [100.0, 200.0, 150.0],
  })

  rsi = compute_rsi(short_series, period=14)
  assert len(rsi) == 3
  assert rsi.isna().all()

  macd, signal, hist = compute_macd(short_series, slow=26)
  assert len(macd) == 3
  assert macd.isna().all()

  upper, middle, lower, pct_b, bw = compute_bollinger_bands(
      short_series, period=20
  )
  assert upper.isna().all()

  atr = compute_atr(short_df, period=14)
  assert atr.isna().all()

  z = compute_volume_zscore(short_df["volume"], period=20)
  assert z.isna().all()


def test_single_row_input() -> None:
  """Indicators must handle single-row inputs without crashing."""
  df_single = pd.DataFrame({
      "open": [100.0],
      "high": [105.0],
      "low": [95.0],
      "close": [100.0],
      "volume": [500.0],
  })
  obv = compute_obv(df_single)
  assert len(obv) == 1
  assert obv.iloc[0] == 500.0


def test_column_name_case_insensitivity(sample_ohlcv_df: pd.DataFrame) -> None:
  """Indicator suite must accept both lowercase and title-cased column names."""
  df_title = sample_ohlcv_df.rename(
      columns={
          "open": "Open",
          "high": "High",
          "low": "Low",
          "close": "Close",
          "volume": "Volume",
      }
  )
  out = add_technical_indicators(df_title)
  assert "rsi_14" in out.columns
  assert "bb_upper" in out.columns
  assert "atr_14" in out.columns


def test_missing_column_raises_keyerror(sample_ohlcv_df: pd.DataFrame) -> None:
  """Missing required column in add_technical_indicators must raise KeyError."""
  df_corrupted = sample_ohlcv_df.drop(columns=["close"])
  with pytest.raises(KeyError, match="Missing required OHLCV column 'close'"):
    add_technical_indicators(df_corrupted)


# ============================================================================
# Tier 3: Invariant & Contract Assertions
# ============================================================================


def test_rsi_bounds(sample_ohlcv_df: pd.DataFrame) -> None:
  """RSI must be strictly bounded in [0.0, 100.0]."""
  rsi = compute_rsi(sample_ohlcv_df["close"], period=14).dropna()
  assert (rsi >= 0.0).all()
  assert (rsi <= 100.0).all()


def test_bollinger_bands_ordering(sample_ohlcv_df: pd.DataFrame) -> None:
  """Bollinger Bands must satisfy upper >= middle >= lower at every valid point."""
  upper, middle, lower, pct_b, bw = compute_bollinger_bands(
      sample_ohlcv_df["close"], period=20
  )
  valid_idx = middle.dropna().index
  assert (upper.loc[valid_idx] >= middle.loc[valid_idx]).all()
  assert (middle.loc[valid_idx] >= lower.loc[valid_idx]).all()


def test_atr_positivity(sample_ohlcv_df: pd.DataFrame) -> None:
  """ATR must be strictly non-negative at every valid point."""
  atr = compute_atr(sample_ohlcv_df, period=14).dropna()
  assert (atr >= 0.0).all()


# ============================================================================
# Tier 4: Unified Interface & Non-Destructive Preservation
# ============================================================================


def test_add_technical_indicators_row_count_and_columns(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
  """add_technical_indicators must append all 12 indicators without dropping rows."""
  original_len = len(sample_ohlcv_df)
  original_index = sample_ohlcv_df.index.copy()

  out = add_technical_indicators(sample_ohlcv_df)

  # Assert exact row count and index preservation
  assert len(out) == original_len
  assert out.index.equals(original_index)

  # Assert all required contract columns are present
  expected_cols = [
      "rsi_14",
      "macd",
      "macd_signal",
      "macd_hist",
      "bb_upper",
      "bb_middle",
      "bb_lower",
      "bb_pct_b",
      "bb_bandwidth",
      "atr_14",
      "obv",
      "volume_zscore",
  ]
  for col in expected_cols:
    assert col in out.columns, f"Missing expected column: {col}"
    assert out[col].dtype == np.float64, f"Column {col} must have float64 dtype"


def test_add_technical_indicators_non_mutating(
    sample_ohlcv_df: pd.DataFrame,
) -> None:
  """add_technical_indicators with copy=True must not mutate the input DataFrame."""
  cols_before = set(sample_ohlcv_df.columns)
  _ = add_technical_indicators(sample_ohlcv_df, copy=True)
  cols_after = set(sample_ohlcv_df.columns)
  assert cols_before == cols_after
