"""Dual-resolution resampling, continuous reindexing, gap imputation,
multi-horizon target engineering, and chronological splitting for crypto time
series.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
    INTRADAY_HORIZONS,
    MACRO_HORIZONS,
    OHLCV_COLUMNS,
)

logger = logging.getLogger(__name__)

# Step mappings based on series frequency
HOURLY_HORIZON_STEPS: dict[str, int] = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "6h": 6,
    "12h": 12,
    "24h": 24,
    "1d": 24,
    "2d": 48,
    "3d": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

DAILY_HORIZON_STEPS: dict[str, int] = {
    "24h": 1,
    "1d": 1,
    "48h": 2,
    "2d": 2,
    "3d": 3,
    "7d": 7,
    "14d": 14,
    "30d": 30,
}

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def get_horizon_steps(horizon: str, resolution: str = "1h") -> int:
  """Converts a horizon string (e.g. '1h', '24h', '7d', '30d') to integer steps

  based on the series resolution ('1h' or '1d').

  Parameters
  ----------
  horizon : str
      Horizon string representation, e.g. '1h', '24h', '2d', '7d', '30d'.
  resolution : str
      Series resolution, either '1h' (hourly) or '1d' (daily).

  Returns
  -------
  int
      Discrete number of forward steps corresponding to the horizon.
  """
  h = horizon.strip().lower()
  res = resolution.strip().lower()

  if res in ("1h", "h", "hourly"):
    if h in HOURLY_HORIZON_STEPS:
      return HOURLY_HORIZON_STEPS[h]
    if h.endswith("h"):
      return int(h[:-1])
    if h.endswith("d"):
      return int(h[:-1]) * 24
  elif res in ("1d", "d", "daily"):
    if h in DAILY_HORIZON_STEPS:
      return DAILY_HORIZON_STEPS[h]
    if h.endswith("d"):
      return int(h[:-1])
    if h.endswith("h"):
      hours = int(h[:-1])
      return max(1, hours // 24)
  else:
    raise ValueError(
        f"Unsupported resolution '{resolution}'. Must be '1h' or '1d'."
    )

  raise ValueError(
      f"Unrecognized horizon format '{horizon}' for resolution '{resolution}'."
  )


def normalize_datetime_index(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
  """Ensures DataFrame has a monotonic, ascending, UTC-localized DatetimeIndex

  named 'timestamp' with no duplicate indices.

  Parameters
  ----------
  df : pd.DataFrame
      Raw input OHLCV DataFrame.
  timestamp_col : str
      Column name containing timestamp values if not already in index.

  Returns
  -------
  pd.DataFrame
      Cleaned DataFrame with UTC DatetimeIndex and numeric OHLCV columns.
  """
  if df.empty:
    raise ValueError("Input DataFrame is empty.")

  df_clean = df.copy()

  # 1. Handle timestamp column or index
  if timestamp_col in df_clean.columns:
    df_clean[timestamp_col] = pd.to_datetime(
        df_clean[timestamp_col], utc=True, errors="coerce"
    )
    df_clean = df_clean.dropna(subset=[timestamp_col])
    df_clean = df_clean.set_index(timestamp_col)
  elif not isinstance(df_clean.index, pd.DatetimeIndex):
    df_clean.index = pd.to_datetime(df_clean.index, utc=True, errors="coerce")
    df_clean = df_clean[df_clean.index.notna()]

  # 2. Enforce UTC timezone
  if df_clean.index.tz is None:
    df_clean.index = df_clean.index.tz_localize("UTC")
  else:
    df_clean.index = df_clean.index.tz_convert("UTC")

  df_clean.index.name = "timestamp"

  # 3. Deduplicate timestamps (keep latest)
  if df_clean.index.duplicated().any():
    dup_count = df_clean.index.duplicated().sum()
    logger.warning("Dropping %d duplicate timestamp rows.", dup_count)
    df_clean = df_clean[~df_clean.index.duplicated(keep="last")]

  # 4. Sort chronologically
  df_clean = df_clean.sort_index()

  # 5. Ensure numeric OHLCV types
  for col in REQUIRED_OHLCV_COLUMNS:
    if col in df_clean.columns:
      df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

  return df_clean


def _parse_timedelta(freq: str | pd.Timedelta) -> pd.Timedelta:
  """Safely converts frequency string into pd.Timedelta without numpy 2.x generic unit deprecation."""
  if isinstance(freq, pd.Timedelta):
    return freq
  f = str(freq).strip().lower()
  if f in ("1h", "h", "hourly"):
    return pd.Timedelta(1, unit="h")
  if f in ("1d", "d", "daily"):
    return pd.Timedelta(1, unit="D")
  return pd.Timedelta(freq)


def detect_timestamp_gaps(
    df: pd.DataFrame,
    freq: str = "1h",
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
  """Detects missing timestamp intervals in a time series.

  Parameters
  ----------
  df : pd.DataFrame
      DataFrame with a DatetimeIndex.
  freq : str
      Expected frequency, e.g. '1h' or '1d'.

  Returns
  -------
  list of tuple[pd.Timestamp, pd.Timestamp, int]
      List of detected gaps as (gap_start, gap_end, missing_bars_count).
  """
  df_norm = normalize_datetime_index(df)
  if len(df_norm) < 2:
    return []

  expected_delta = _parse_timedelta(freq)
  deltas = df_norm.index.to_series().diff()

  gap_mask = deltas > expected_delta
  gap_indices = df_norm.index[gap_mask]

  gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
  for end_ts in gap_indices:
    loc = df_norm.index.get_loc(end_ts)
    start_ts = df_norm.index[loc - 1]
    missing_count = int(round((end_ts - start_ts) / expected_delta)) - 1
    if missing_count > 0:
      gaps.append((start_ts, end_ts, missing_count))

  return gaps


def reindex_and_impute(
    df: pd.DataFrame,
    freq: str = "1h",
    fill_initial: bool = True,
    add_imputed_flag: bool = False,
) -> pd.DataFrame:
  """Reindexes a DataFrame onto a continuous DatetimeIndex and imputes missing bars

  under standard 24/7 financial market rules.

  Imputation Policy:
  - Close: Forward-filled (with optional back-fill for initial leading NaNs).
  - Open, High, Low: Filled with the imputed Close price (zero-volatility flat
  bar).
  - Volume: Filled with 0.0 (no trading volume occurred).

  Parameters
  ----------
  df : pd.DataFrame
      Input OHLCV DataFrame.
  freq : str
      Target frequency ('1h' for hourly, '1d' or '1D' for daily).
  fill_initial : bool
      Whether to backward-fill leading NaNs if the dataset starts with missing
      bars.
  add_imputed_flag : bool
      Whether to include an 'is_imputed' boolean column in the returned
      DataFrame.

  Returns
  -------
  pd.DataFrame
      Continuous, fully imputed DataFrame with zero NaNs in OHLCV columns.
  """
  df_clean = normalize_datetime_index(df)

  # Generate complete date range spanning min to max timestamp
  full_index = pd.date_range(
      start=df_clean.index.min(),
      end=df_clean.index.max(),
      freq=freq,
      tz="UTC",
      name="timestamp",
  )

  # Track imputed positions
  is_imputed = None
  if add_imputed_flag:
    is_imputed = ~full_index.isin(df_clean.index)

  # Reindex onto complete range
  df_reindexed = df_clean.reindex(full_index)

  # 1. Forward-fill close price
  if "close" in df_reindexed.columns:
    df_reindexed["close"] = df_reindexed["close"].ffill()
    if fill_initial and df_reindexed["close"].isna().any():
      df_reindexed["close"] = df_reindexed["close"].bfill()

  # 2. Impute open, high, low with imputed close
  for col in ("open", "high", "low"):
    if col in df_reindexed.columns:
      df_reindexed[col] = df_reindexed[col].fillna(df_reindexed["close"])

  # 3. Impute volume with 0.0
  if "volume" in df_reindexed.columns:
    df_reindexed["volume"] = df_reindexed["volume"].fillna(0.0)

  # 4. Handle supplementary exchange columns if present
  for col in df_reindexed.columns:
    if col in (
        "quote_asset_volume",
        "trades_count",
        "taker_buy_base",
        "taker_buy_quote",
    ):
      df_reindexed[col] = df_reindexed[col].fillna(0.0)
    elif col not in ("open", "high", "low", "close", "volume"):
      df_reindexed[col] = df_reindexed[col].ffill()
      if fill_initial and df_reindexed[col].isna().any():
        df_reindexed[col] = df_reindexed[col].bfill()

  if add_imputed_flag and is_imputed is not None:
    df_reindexed["is_imputed"] = is_imputed

  df_reindexed.index.name = "timestamp"
  return df_reindexed


def resample_to_hourly(df_raw: pd.DataFrame) -> pd.DataFrame:
  """Converts raw or sub-hourly market data into a clean, continuous hourly stream.

  Parameters
  ----------
  df_raw : pd.DataFrame
      Input DataFrame (raw hourly or sub-hourly e.g. 15m, 30m).

  Returns
  -------
  pd.DataFrame
      Clean, imputed hourly DataFrame.
  """
  df_clean = normalize_datetime_index(df_raw)

  # If data is sub-hourly, aggregate to 1h
  deltas = df_clean.index.to_series().diff().dropna()
  median_delta = deltas.median() if not deltas.empty else _parse_timedelta("1h")

  if median_delta < _parse_timedelta("1h"):
    agg_rules: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in df_clean.columns:
      if col in (
          "quote_asset_volume",
          "trades_count",
          "taker_buy_base",
          "taker_buy_quote",
      ):
        agg_rules[col] = "sum"
      elif col not in agg_rules:
        agg_rules[col] = "last"

    df_clean = df_clean.resample("1h").agg(agg_rules)

  # Reindex and impute missing hourly bars
  return reindex_and_impute(df_clean, freq="1h")


def resample_to_daily(
    df_hourly: pd.DataFrame,
    drop_incomplete_days: bool = False,
) -> pd.DataFrame:
  """Aggregates a clean hourly DataFrame into daily candles (00:00:00 to 23:59:59

  UTC).

  Aggregation Rules:
  - Open: First hourly open of the day.
  - High: Maximum hourly high of the day.
  - Low: Minimum hourly low of the day.
  - Close: Last hourly close of the day.
  - Volume: Sum of hourly volumes.

  Parameters
  ----------
  df_hourly : pd.DataFrame
      Hourly OHLCV DataFrame.
  drop_incomplete_days : bool
      Whether to drop calendar days with fewer than 24 hours of data.

  Returns
  -------
  pd.DataFrame
      Continuous, imputed daily DataFrame.
  """
  df_clean = normalize_datetime_index(df_hourly)

  if drop_incomplete_days:
    # Keep only days with exactly 24 hourly candles
    day_counts = df_clean.groupby(df_clean.index.date).size()
    complete_dates = set(day_counts[day_counts == 24].index)
    mask = df_clean.index.map(lambda ts: ts.date() in complete_dates)
    df_clean = df_clean[mask]
    if df_clean.empty:
      raise ValueError("No complete 24-hour days found in dataset.")

  agg_rules: dict[str, str] = {
      "open": "first",
      "high": "max",
      "low": "min",
      "close": "last",
      "volume": "sum",
  }
  for col in df_clean.columns:
    if col in (
        "quote_asset_volume",
        "trades_count",
        "taker_buy_base",
        "taker_buy_quote",
    ):
      agg_rules[col] = "sum"
    elif col not in agg_rules:
      agg_rules[col] = "last"

  df_daily = df_clean.resample("1D").agg(agg_rules)

  # Reindex and impute missing daily bars
  return reindex_and_impute(df_daily, freq="1D")


def resample_dual_resolution(
    df_hourly_raw: pd.DataFrame,
    df_daily_raw: pd.DataFrame | None = None,
    drop_incomplete_days: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Constructs the dual-resolution time series pair: (df_hourly, df_daily).

  Parameters
  ----------
  df_hourly_raw : pd.DataFrame
      Raw hourly (or sub-hourly) DataFrame.
  df_daily_raw : pd.DataFrame | None
      Optional pre-fetched raw daily DataFrame from exchange. If None,
      df_daily is resampled directly from df_hourly.
  drop_incomplete_days : bool
      Whether to drop days with fewer than 24 hours during daily resampling.

  Returns
  -------
  tuple[pd.DataFrame, pd.DataFrame]
      (clean_df_hourly, clean_df_daily)
  """
  df_hourly = resample_to_hourly(df_hourly_raw)

  if df_daily_raw is not None and not df_daily_raw.empty:
    df_daily = reindex_and_impute(df_daily_raw, freq="1D")
  else:
    df_daily = resample_to_daily(
        df_hourly, drop_incomplete_days=drop_incomplete_days
    )

  return df_hourly, df_daily


def add_target_horizons(
    df: pd.DataFrame,
    resolution: str = "1h",
    horizons: list[str] | None = None,
    include_log_returns: bool = False,
    include_direction: bool = True,
    price_col: str = "close",
) -> pd.DataFrame:
  """Constructs future target prices and returns for specified forecast horizons.

  For each horizon h with step offset k:
  - target_price_{h}: P_{t+k} = close.shift(-k)
  - target_return_{h}: R_{t, t+k} = (P_{t+k} - P_t) / P_t
  - target_direction_{h}: 1.0 if P_{t+k} >= P_t else 0.0 (NaN if future price is
  NaN)

  Parameters
  ----------
  df : pd.DataFrame
      Clean OHLCV DataFrame with continuous DatetimeIndex.
  resolution : str
      Resolution of the series ('1h' or '1d').
  horizons : list[str] | None
      List of horizons to compute. Defaults to INTRADAY_HORIZONS for '1h' and
      MACRO_HORIZONS for '1d'.
  include_log_returns : bool
      Whether to compute target_log_return_{h} = ln(P_{t+k} / P_t).
  include_direction : bool
      Whether to compute target_direction_{h} (1 for Bullish/Flat, 0 for
      Bearish).
  price_col : str
      Column name representing price. Default 'close'.

  Returns
  -------
  pd.DataFrame
      DataFrame decorated with target columns. The last k rows contain NaN for
      horizon h.
  """
  if df.empty:
    raise ValueError("Input DataFrame is empty.")
  if price_col not in df.columns:
    raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

  df_out = df.copy()
  norm_res = resolution.lower()

  if horizons is None:
    if norm_res in ("1h", "h", "hourly"):
      horizons = list(INTRADAY_HORIZONS)
    elif norm_res in ("1d", "d", "daily"):
      horizons = list(MACRO_HORIZONS)
    else:
      horizons = list(DEFAULT_HORIZONS)

  prices = df_out[price_col]

  for h in horizons:
    steps = get_horizon_steps(h, norm_res)
    future_price = prices.shift(-steps)

    # 1. Absolute Future Target Price
    df_out[f"target_price_{h}"] = future_price

    # 2. Relative Target Return: (P_{t+k} - P_t) / P_t
    df_out[f"target_return_{h}"] = (future_price - prices) / prices

    # 3. Optional Log Return: ln(P_{t+k} / P_t)
    if include_log_returns:
      df_out[f"target_log_return_{h}"] = np.log(future_price / prices)

    # 4. Target Direction: 1.0 for UP/Flat, 0.0 for DOWN, NaN if future unknown
    if include_direction:
      direction = np.where(future_price >= prices, 1.0, 0.0)
      df_out[f"target_direction_{h}"] = np.where(
          future_price.isna(), np.nan, direction
      )

  return df_out


def create_dual_resolution_targets(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    include_log_returns: bool = False,
    include_direction: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Decorates both hourly and daily streams with their native target horizons:

  - Hourly: ['1h', '2h', '4h', '6h', '12h', '24h', '2d']
  - Daily: ['1d', '2d', '3d', '7d', '14d', '30d']
  """
  df_h_targets = add_target_horizons(
      df_hourly,
      resolution="1h",
      horizons=INTRADAY_HORIZONS,
      include_log_returns=include_log_returns,
      include_direction=include_direction,
  )
  df_d_targets = add_target_horizons(
      df_daily,
      resolution="1d",
      horizons=MACRO_HORIZONS,
      include_log_returns=include_log_returns,
      include_direction=include_direction,
  )
  return df_h_targets, df_d_targets


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    embargo_steps: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Chronologically partitions a time series into Train, Validation, and Test sets

  with strict zero-lookahead ordering and optional horizon embargo buffers.

  Parameters
  ----------
  df : pd.DataFrame
      Input DataFrame with DatetimeIndex.
  train_ratio : float
      Fraction of data assigned to training (e.g. 0.70).
  val_ratio : float
      Fraction of data assigned to validation (e.g. 0.15).
  test_ratio : float
      Fraction of data assigned to test (e.g. 0.15).
  embargo_steps : int
      Number of rows purged between splits to prevent multi-step target leakage.

  Returns
  -------
  tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
      (train_df, val_df, test_df)
  """
  if df.empty:
    raise ValueError("Cannot split an empty DataFrame.")

  if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-5):
    raise ValueError(
        f"Split ratios must sum to 1.0 (got"
        f" {train_ratio + val_ratio + test_ratio:.5f})."
    )

  if train_ratio <= 0 or test_ratio <= 0 or val_ratio < 0:
    raise ValueError("train_ratio and test_ratio must be > 0; val_ratio >= 0.")

  if embargo_steps < 0:
    raise ValueError("embargo_steps must be non-negative.")

  n = len(df)
  min_required = 3 * max(1, embargo_steps)
  if n < min_required:
    raise ValueError(
        f"DataFrame length {n} is too short for split with embargo_steps"
        f" {embargo_steps} (minimum {min_required} required)."
    )

  n_train = int(n * train_ratio)
  n_val = int(n * val_ratio)

  train_end = n_train
  val_start = n_train + embargo_steps
  val_end = n_train + n_val
  test_start = n_train + n_val + embargo_steps

  if val_start >= val_end or test_start >= n:
    raise ValueError(
        f"embargo_steps={embargo_steps} is too large for dataset size {n},"
        " resulting in empty validation or test partitions."
    )

  train_df = df.iloc[:train_end].copy()
  val_df = df.iloc[val_start:val_end].copy()
  test_df = df.iloc[test_start:].copy()

  return train_df, val_df, test_df


def get_dual_resolution_data(
    symbol: str,
    hourly_limit: int = 1000,
    daily_limit: int = 365,
    source: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Fetches and processes dual-resolution market data (hourly and daily).

  Adheres strictly to M1 <-> M3 / M4 contract in PROJECT.md:
  Returns (df_hourly, df_daily) properly sanitized and normalized.

  Parameters
  ----------
  symbol : str
      Cryptocurrency symbol, e.g. 'BTC/USDT'.
  hourly_limit : int
      Candle limit for hourly stream.
  daily_limit : int
      Candle limit for daily stream.
  source : str
      Data source ('auto', 'binance', or 'yahoo').

  Returns
  -------
  tuple[pd.DataFrame, pd.DataFrame]
      (clean_df_hourly, clean_df_daily)
  """
  from crypto_forecast.data.ingestion import fetch_ohlcv

  df_hourly_raw = fetch_ohlcv(
      symbol=symbol, interval="1h", limit=hourly_limit, source=source
  )
  df_daily_raw = None
  try:
    df_daily_raw = fetch_ohlcv(
        symbol=symbol, interval="1d", limit=daily_limit, source=source
    )
  except Exception as exc:
    logger.info(
        "Direct daily fetch failed (%s); will resample from hourly stream.", exc
    )

  return resample_dual_resolution(
      df_hourly_raw=df_hourly_raw, df_daily_raw=df_daily_raw
  )
