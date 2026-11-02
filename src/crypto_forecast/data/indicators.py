"""Technical Indicator Suite for Cryptocurrency Price Forecasting.

Provides pure vectorized NumPy and Pandas implementations of core technical indicators:
- Relative Strength Index (RSI) via Wilder's Smoothing
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands (Upper, Middle, Lower, %B, Bandwidth)
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Rolling Volume Z-Score

All functions feature strict numerical stability guards to prevent ZeroDivisionError,
floating-point overflows, or NaN leakage during flat market regimes or exchange halts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
  """Compute Relative Strength Index (RSI) using Wilder's exponential smoothing.

  Formulation:
      ΔP = P_t - P_{t-1}
      U_t = max(ΔP_t, 0)
      D_t = max(-ΔP_t, 0)
      AvgU_t = EMA(U_t, alpha=1/period, adjust=False)
      AvgD_t = EMA(D_t, alpha=1/period, adjust=False)
      RSI_t = 100 * AvgU_t / (AvgU_t + AvgD_t)

  Numerical Stability:
      - When total movement (AvgU + AvgD) == 0 (flat price series), RSI is set to
      50.0.
      - When AvgD == 0 and AvgU > 0, RSI evaluates cleanly to 100.0 without
      division by zero.
      - Output is strictly bounded in [0.0, 100.0].

  Args:
      series: Pandas Series of close prices.
      period: Lookback smoothing period (default: 14).

  Returns:
      pd.Series containing RSI values aligned with the input index.
  """
  if len(series) < period:
    return pd.Series(np.nan, index=series.index, dtype=np.float64)

  diff = series.diff()
  up = diff.clip(lower=0.0)
  down = (-diff).clip(lower=0.0)

  # Wilder's smoothing corresponds to alpha = 1 / period
  avg_gain = up.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
  avg_loss = down.ewm(
      alpha=1.0 / period, min_periods=period, adjust=False
  ).mean()

  total_movement = avg_gain + avg_loss

  rsi = pd.Series(np.nan, index=series.index, dtype=np.float64)
  valid_mask = total_movement.notna()

  # Zero volatility guard: flat prices exhibit neutral momentum (50.0)
  flat_mask = valid_mask & (total_movement == 0.0)
  normal_mask = valid_mask & (total_movement > 0.0)

  rsi[flat_mask] = 50.0
  rsi[normal_mask] = 100.0 * (
      avg_gain[normal_mask] / total_movement[normal_mask]
  )

  return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
  """Compute Moving Average Convergence Divergence (MACD).

  Formulation:
      FastEMA_t = EMA(P_t, span=fast, adjust=False)
      SlowEMA_t = EMA(P_t, span=slow, adjust=False)
      MACD_t = FastEMA_t - SlowEMA_t
      Signal_t = EMA(MACD_t, span=signal, adjust=False)
      Hist_t = MACD_t - Signal_t

  Args:
      series: Pandas Series of close prices.
      fast: Fast EMA span (default: 12).
      slow: Slow EMA span (default: 26).
      signal: Signal line EMA span (default: 9).

  Returns:
      tuple[pd.Series, pd.Series, pd.Series]: (macd_line, signal_line, histogram)
  """
  if len(series) < slow:
    nan_s = pd.Series(np.nan, index=series.index, dtype=np.float64)
    return nan_s, nan_s.copy(), nan_s.copy()

  fast_ema = series.ewm(span=fast, adjust=False).mean()
  slow_ema = series.ewm(span=slow, adjust=False).mean()

  macd_line = fast_ema - slow_ema
  signal_line = macd_line.ewm(span=signal, adjust=False).mean()
  hist = macd_line - signal_line

  return macd_line, signal_line, hist


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
  """Compute Bollinger Bands, Percent B (%B), and Bandwidth.

  Formulation:
      MiddleBand = SMA(P, period)
      σ = RollingStd(P, period, ddof=0)
      UpperBand = MiddleBand + (num_std * σ)
      LowerBand = MiddleBand - (num_std * σ)
      %B = (P - LowerBand) / (UpperBand - LowerBand)
      Bandwidth = (UpperBand - LowerBand) / MiddleBand

  Numerical Stability:
      - When σ == 0 (flat price, upper == lower), %B is set to 0.5 (centered).
      - When MiddleBand == 0, Bandwidth is set to 0.0.

  Args:
      series: Pandas Series of close prices.
      period: Rolling window length (default: 20).
      num_std: Number of standard deviations (default: 2.0).

  Returns:
      tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
          (upper, middle, lower, pct_b, bandwidth)
  """
  if len(series) < period:
    nan_s = pd.Series(np.nan, index=series.index, dtype=np.float64)
    return nan_s, nan_s.copy(), nan_s.copy(), nan_s.copy(), nan_s.copy()

  middle = series.rolling(window=period, min_periods=period).mean()
  std = series.rolling(window=period, min_periods=period).std(ddof=0)

  upper = middle + (num_std * std)
  lower = middle - (num_std * std)
  band_width_diff = upper - lower

  # %B Calculation
  pct_b = pd.Series(np.nan, index=series.index, dtype=np.float64)
  valid_diff = band_width_diff.notna()
  zero_width = valid_diff & (band_width_diff == 0.0)
  normal_width = valid_diff & (band_width_diff > 0.0)

  pct_b[zero_width] = 0.5
  pct_b[normal_width] = (series[normal_width] - lower[normal_width]) / (
      band_width_diff[normal_width]
  )

  # Bandwidth Calculation
  bandwidth = pd.Series(np.nan, index=series.index, dtype=np.float64)
  valid_mb = valid_diff & middle.notna()
  zero_mb = valid_mb & (middle == 0.0)
  normal_mb = valid_mb & (middle != 0.0)

  bandwidth[zero_mb] = 0.0
  bandwidth[normal_mb] = band_width_diff[normal_mb] / middle[normal_mb]

  return upper, middle, lower, pct_b, bandwidth


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
  """Compute Average True Range (ATR) using Wilder's exponential smoothing.

  Formulation:
      TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
      For t=0: TR_0 = High_0 - Low_0
      ATR_t = EMA(TR_t, alpha=1/period, adjust=False)

  Numerical Stability:
      - Since High >= Low, ATR is strictly non-negative.
      - Zero-price volatility produces ATR = 0.0 without division.

  Args:
      df: DataFrame containing 'high', 'low', 'close' columns (case-insensitive).
      period: Smoothing period (default: 14).

  Returns:
      pd.Series containing ATR values aligned with df.index.
  """
  col_map = {c.lower(): c for c in df.columns}
  for req in ["high", "low", "close"]:
    if req not in col_map:
      raise KeyError(f"Missing required column '{req}' for ATR computation.")

  high = df[col_map["high"]]
  low = df[col_map["low"]]
  close = df[col_map["close"]]

  if len(df) < period:
    return pd.Series(np.nan, index=df.index, dtype=np.float64)

  prev_close = close.shift(1)
  tr1 = high - low
  tr2 = (high - prev_close).abs()
  tr3 = (low - prev_close).abs()

  # Vectorized max across components; row 0 defaults to tr1 because tr2, tr3 are NaN
  tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

  atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
  return atr


def compute_obv(df: pd.DataFrame) -> pd.Series:
  """Compute On-Balance Volume (OBV).

  Formulation:
      OBV_0 = Volume_0
      OBV_t = OBV_{t-1} + Volume_t  if Close_t > Close_{t-1}
      OBV_t = OBV_{t-1} - Volume_t  if Close_t < Close_{t-1}
      OBV_t = OBV_{t-1}             if Close_t == Close_{t-1}

  Args:
      df: DataFrame containing 'close' and 'volume' columns (case-insensitive).

  Returns:
      pd.Series containing cumulative OBV aligned with df.index.
  """
  col_map = {c.lower(): c for c in df.columns}
  for req in ["close", "volume"]:
    if req not in col_map:
      raise KeyError(f"Missing required column '{req}' for OBV computation.")

  close = df[col_map["close"]]
  volume = df[col_map["volume"]]

  if len(df) == 0:
    return pd.Series(dtype=np.float64)
  if len(df) == 1:
    return pd.Series(volume.iloc[0], index=df.index, dtype=np.float64)

  close_diff = close.diff()
  direction = pd.Series(0.0, index=df.index, dtype=np.float64)
  direction[close_diff > 0.0] = 1.0
  direction[close_diff < 0.0] = -1.0
  direction.iloc[0] = 1.0  # Initializes OBV_0 = Volume_0

  volume_flow = direction * volume
  obv = volume_flow.cumsum()
  return obv


def compute_volume_zscore(series: pd.Series, period: int = 20) -> pd.Series:
  """Compute rolling Volume Z-Score.

  Formulation:
      Z_t = (Volume_t - RollingMean(Volume, period)) / RollingStd(Volume,
      period, ddof=0)

  Numerical Stability:
      - When RollingStd == 0.0 (constant volume), Z-score evaluates to 0.0.
      - Preserves NaN during warm-up period.

  Args:
      series: Pandas Series of trading volumes.
      period: Rolling window length (default: 20).

  Returns:
      pd.Series containing volume Z-scores aligned with series.index.
  """
  if len(series) < period:
    return pd.Series(np.nan, index=series.index, dtype=np.float64)

  mean_v = series.rolling(window=period, min_periods=period).mean()
  std_v = series.rolling(window=period, min_periods=period).std(ddof=0)

  zscore = pd.Series(np.nan, index=series.index, dtype=np.float64)
  valid_mask = std_v.notna()
  zero_std = valid_mask & (std_v == 0.0)
  normal = valid_mask & (std_v > 0.0)

  zscore[zero_std] = 0.0
  zscore[normal] = (series[normal] - mean_v[normal]) / std_v[normal]

  return zscore


def add_technical_indicators(
    df: pd.DataFrame, copy: bool = True
) -> pd.DataFrame:
  """Calculate and append the complete technical indicator suite to an OHLCV DataFrame.

  Engineered Columns:
      - 'rsi_14': Relative Strength Index (14 periods)
      - 'macd': MACD line (12, 26, 9)
      - 'macd_signal': MACD signal line (9 periods)
      - 'macd_hist': MACD histogram
      - 'bb_upper': Bollinger Upper Band (20 periods, 2 std)
      - 'bb_middle': Bollinger Middle Band (20 periods)
      - 'bb_lower': Bollinger Lower Band (20 periods, 2 std)
      - 'bb_pct_b': Bollinger %B
      - 'bb_bandwidth': Bollinger Bandwidth
      - 'atr_14': Average True Range (14 periods)
      - 'obv': On-Balance Volume
      - 'volume_zscore': Volume Z-Score (20 periods)

  Warm-up Behavior:
      Preserves all rows in `df`. Warm-up periods contain NaNs without dropping
      rows,
      ensuring strict index synchronization with upstream dual-resolution
      candles.

  Args:
      df: DataFrame containing 'open', 'high', 'low', 'close', 'volume' columns.
      copy: Whether to operate on a copy (default: True).

  Returns:
      pd.DataFrame with added indicator feature columns.
  """
  out = df.copy() if copy else df

  col_map = {c.lower(): c for c in out.columns}
  required = ["open", "high", "low", "close", "volume"]
  for req in required:
    if req not in col_map:
      raise KeyError(
          f"Missing required OHLCV column '{req}'. Present: {list(out.columns)}"
      )

  c_close = out[col_map["close"]]
  c_vol = out[col_map["volume"]]

  # 1. RSI
  out["rsi_14"] = compute_rsi(c_close, period=14)

  # 2. MACD
  macd, signal, hist = compute_macd(c_close, fast=12, slow=26, signal=9)
  out["macd"] = macd
  out["macd_signal"] = signal
  out["macd_hist"] = hist

  # 3. Bollinger Bands
  upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(
      c_close, period=20, num_std=2.0
  )
  out["bb_upper"] = upper
  out["bb_middle"] = middle
  out["bb_lower"] = lower
  out["bb_pct_b"] = pct_b
  out["bb_bandwidth"] = bandwidth

  # 4. ATR
  out["atr_14"] = compute_atr(out, period=14)

  # 5. OBV
  out["obv"] = compute_obv(out)

  # 6. Volume Z-Score
  out["volume_zscore"] = compute_volume_zscore(c_vol, period=20)

  return out
