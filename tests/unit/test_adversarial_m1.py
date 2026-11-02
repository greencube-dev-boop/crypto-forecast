"""Milestone 1 Adversarial & Stress Verification Test Suite.

Adversarially stresses:
1. `indicators.py`:
   - 1,000-bar constant price zero-volatility regime (RSI=50.0, BB %B=0.5, Bandwidth=0.0, Volume Z-Score=0.0).
   - Massive geometric spikes (1,000,000x) and scale bounds.
   - Monotonic 100% downward decay (RSI asymptoting to 0.0).
   - Zero-price and negative/inverted spreads.
   - Exact index/length preservation with NaN warm-up retention.
2. `preprocessor.py`:
   - Heavy missing gaps (90% to 99.8% timestamps missing).
   - Scrambled unsorted timestamps with high duplication.
   - High-frequency aggregation integrity and target horizon NaN tails.
3. `ingestion.py`:
   - Persistent HTTP 429 retry exhaustion and backoff jitter.
   - Automatic failover from Binance 429 saturation to Yahoo Finance fallback.
   - IP weight saturation pacing and dual-source failure escalation.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest.mock as mock

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest
import requests

from crypto_forecast.data.indicators import (
    add_technical_indicators,
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_volume_zscore,
)
from crypto_forecast.data.ingestion import (
    DataIngestionError,
    RateLimitExceededError,
    YahooFinanceError,
    _compute_backoff_sleep,
    _inspect_binance_weight,
    fetch_binance_klines,
    fetch_ohlcv,
    fetch_yahoo_ohlcv,
)
from crypto_forecast.data.preprocessor import (
    add_target_horizons,
    detect_timestamp_gaps,
    get_horizon_steps,
    normalize_datetime_index,
    reindex_and_impute,
    resample_dual_resolution,
    resample_to_daily,
    train_val_test_split,
)


# ==============================================================================
# 1. Technical Indicators Adversarial Stress Tests
# ==============================================================================


class TestIndicatorsAdversarial:
  """Adversarial stress testing for technical indicator calculation routines."""

  def test_adv_rsi_constant_price_1000_bars(self):
    """Stress-test RSI with 1,000 bars of strictly zero price changes.

    Assumptions challenged:
    - total_movement (AvgGain + AvgLoss) == 0 must not trigger ZeroDivisionError.
    - Output must cleanly evaluate to 50.0 (neutral momentum) after the 14-period
      warm-up.
    """
    n = 1000
    prices = pd.Series([123.456] * n, index=pd.date_range("2026-01-01", periods=n, freq="1h"))
    rsi = compute_rsi(prices, period=14)

    # First 14 periods (indices 0..13) should be NaN (warm-up for period=14)
    assert rsi.iloc[:14].isna().all(), "Warm-up periods must be NaN"
    # From index 14 onwards, all values must be exactly 50.0
    valid_rsi = rsi.iloc[14:]
    assert (valid_rsi == 50.0).all(), "Zero-volatility RSI must be strictly 50.0"
    assert not np.isinf(rsi).any(), "RSI must never be infinite"
    assert not rsi.iloc[14:].isna().any(), "RSI must have no NaNs after warm-up"

  def test_adv_bollinger_bands_zero_volatility_1000_bars(self):
    """Stress-test Bollinger Bands with 1,000 bars of constant price.

    Assumptions challenged:
    - Upper - Lower == 0 must not cause ZeroDivisionError in %B.
    - %B must evaluate to 0.5 (perfect center of degenerate band).
    - Bandwidth must evaluate to 0.0 without division by zero.
    """
    n = 1000
    prices = pd.Series([500.0] * n, index=pd.date_range("2026-01-01", periods=n, freq="1h"))
    upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(prices, period=20, num_std=2.0)

    # Check warm-up (first 19 values should be NaN)
    assert upper.iloc[:19].isna().all()
    assert pct_b.iloc[:19].isna().all()

    # Valid range
    v_upper = upper.iloc[19:]
    v_middle = middle.iloc[19:]
    v_lower = lower.iloc[19:]
    v_pct_b = pct_b.iloc[19:]
    v_bw = bandwidth.iloc[19:]

    assert np.allclose(v_upper, 500.0)
    assert np.allclose(v_middle, 500.0)
    assert np.allclose(v_lower, 500.0)
    assert (v_pct_b == 0.5).all(), "Bollinger %B on zero variance must be exactly 0.5"
    assert (v_bw == 0.0).all(), "Bollinger Bandwidth on zero variance must be exactly 0.0"

  def test_adv_bollinger_bands_zero_price_middle_band_division(self):
    """Stress-test Bollinger Bands when middle band is 0.0 (all prices zero).

    Bandwidth = (Upper - Lower) / MiddleBand. MiddleBand == 0 must not raise ZeroDivisionError.
    """
    n = 50
    prices = pd.Series([0.0] * n)
    upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(prices, period=20)
    v_bw = bandwidth.iloc[19:]
    assert (v_bw == 0.0).all(), "Bandwidth must be 0.0 when MiddleBand is 0.0"

  def test_adv_volume_zscore_constant_volume_1000_bars(self):
    """Stress-test Volume Z-Score with 1,000 bars of constant volume.

    Standard deviation is 0.0. Division by zero std must be caught and evaluated to 0.0.
    """
    n = 1000
    volumes = pd.Series([10000.0] * n)
    z = compute_volume_zscore(volumes, period=20)

    assert z.iloc[:19].isna().all(), "Warm-up must be NaN"
    assert (z.iloc[19:] == 0.0).all(), "Zero-variance volume Z-score must be 0.0"

  def test_adv_indicators_massive_geometric_spike(self):
    """Stress-test indicators under a 1,000,000x sudden price flash spike and collapse."""
    n = 200
    base = np.full(n, 100.0)
    # Inject 1,000,000x spike at index 100
    base[100] = 100.0 * 1_000_000.0
    series = pd.Series(base)

    # 1. RSI
    rsi = compute_rsi(series, period=14)
    assert not np.isinf(rsi).any(), "RSI must never be infinite during massive spike"
    assert (rsi.dropna() >= 0.0).all() and (rsi.dropna() <= 100.0).all(), "RSI must respect [0, 100]"

    # 2. MACD
    macd, sig, hist = compute_macd(series)
    assert not np.isinf(macd).any()
    assert not np.isinf(hist).any()

    # 3. Bollinger Bands
    upper, middle, lower, pct_b, bw = compute_bollinger_bands(series, period=20)
    assert not np.isinf(upper).any()
    assert not np.isinf(pct_b).any()

  def test_adv_rsi_monotonic_geometric_decay_asymptote(self):
    """Stress-test RSI with monotonic geometric decay (P_t = 1000 * 0.9^t).

    AvgGain must decay towards 0, causing RSI to asymptote directly towards 0.0.
    """
    n = 100
    prices = pd.Series([1000.0 * (0.9**t) for t in range(n)])
    rsi = compute_rsi(prices, period=14)

    # After warm-up, because all diffs are strictly negative (down), gain is 0.
    # Therefore RSI must be exactly 0.0!
    assert (rsi.iloc[14:] == 0.0).all(), "Monotonic strictly decreasing series must yield RSI = 0.0"

  def test_adv_atr_zero_spread_regime(self):
    """Stress-test ATR with zero-spread flat candles (High == Low == Close).

    Under zero price movement, ATR must evaluate cleanly to 0.0 without errors.
    """
    n = 100
    df_flat = pd.DataFrame({
        "open": [10.0] * n,
        "high": [10.0] * n,
        "low": [10.0] * n,
        "close": [10.0] * n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="1h"))
    atr = compute_atr(df_flat, period=14)
    # Check warm-up (first 13 rows NaN for period=14)
    assert atr.iloc[:13].isna().all()
    # From index 13 onwards, ATR must be exactly 0.0
    assert (atr.iloc[13:] == 0.0).all(), "ATR on zero-spread series must evaluate to 0.0"

  def test_adv_obv_directional_steps_and_flat_invariance(self):
    """Stress-test OBV: verify zero volume flow on flat bars and step accuracy."""
    df = pd.DataFrame({
        "close": [10.0, 10.0, 10.0, 12.0, 12.0, 9.0],
        "volume": [100.0, 50.0, 50.0, 200.0, 100.0, 300.0],
    })
    obv = compute_obv(df)
    # Expected:
    # row 0: close 10.0, vol 100 -> OBV = 100.0
    # row 1: close 10.0 == 10.0 -> flat, OBV = 100.0
    # row 2: close 10.0 == 10.0 -> flat, OBV = 100.0
    # row 3: close 12.0 > 10.0 -> up, OBV = 100.0 + 200.0 = 300.0
    # row 4: close 12.0 == 12.0 -> flat, OBV = 300.0
    # row 5: close 9.0 < 12.0 -> down, OBV = 300.0 - 300.0 = 0.0
    expected = [100.0, 100.0, 100.0, 300.0, 300.0, 0.0]
    assert np.allclose(obv.values, expected), f"OBV failed: {obv.values} vs {expected}"

  def test_adv_add_technical_indicators_extreme_inputs(self):
    """Stress-test `add_technical_indicators` on an adversarial DataFrame."""
    n = 500
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    # Mix flat periods, massive spikes, zero volumes
    closes = np.full(n, 200.0)
    closes[250] = 2_000_000.0  # Spike
    closes[300:350] = 50.0     # Flat lower plateau

    vols = np.full(n, 1000.0)
    vols[100:150] = 0.0        # Zero volume zone

    df = pd.DataFrame({
        "open": closes,
        "high": closes + 5.0,
        "low": closes - 5.0,
        "close": closes,
        "volume": vols,
    }, index=dates)

    out = add_technical_indicators(df, copy=True)

    # 1. Shape preservation
    assert len(out) == n, "Output row count must match input exactly"
    assert (out.index == df.index).all(), "Index alignment must be perfectly preserved"

    # 2. Check all 12 indicators are present
    expected_cols = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth",
        "atr_14", "obv", "volume_zscore",
    ]
    for col in expected_cols:
      assert col in out.columns, f"Indicator column {col} missing"

    # 3. No infinite values anywhere in indicators
    for col in expected_cols:
      assert not np.isinf(out[col]).any(), f"Infinite value found in column {col}"


# ==============================================================================
# 2. Preprocessor Adversarial Stress Tests
# ==============================================================================


class TestPreprocessorAdversarial:
  """Adversarial stress testing for resampling, gap imputation, and target engineering."""

  def test_adv_preprocessor_heavy_missing_gaps_90_percent(self):
    """Stress-test `reindex_and_impute` when 90% of timestamps are missing.

    Scenario: A series spanning 1,000 hours where 900 random timestamps are dropped.
    The reindexer must:
    - Reconstruct all 1,000 continuous hourly intervals.
    - Impute all missing close prices by forward-filling.
    - Zero-fill all missing volumes.
    - Have 0 NaNs across all OHLCV columns.
    """
    np.random.seed(42)
    full_dates = pd.date_range("2026-01-01", periods=1000, freq="1h", tz="UTC")
    # Keep only 10% (100 rows)
    kept_indices = np.sort(np.random.choice(len(full_dates), size=100, replace=False))
    # Ensure first and last are kept so full range is preserved
    kept_indices[0] = 0
    kept_indices[-1] = 999
    kept_indices = np.unique(kept_indices)

    df_sparse = pd.DataFrame({
        "open": 100.0 + np.random.randn(len(kept_indices)),
        "high": 105.0 + np.random.randn(len(kept_indices)),
        "low": 95.0 + np.random.randn(len(kept_indices)),
        "close": 100.0 + np.random.randn(len(kept_indices)),
        "volume": 500.0 + np.abs(np.random.randn(len(kept_indices)) * 100),
    }, index=full_dates[kept_indices])

    # Detect gaps before imputation
    gaps = detect_timestamp_gaps(df_sparse, freq="1h")
    assert len(gaps) > 0, "Sparse series must have detected timestamp gaps"
    total_missing = sum(g[2] for g in gaps)
    assert total_missing == 1000 - len(df_sparse)

    # Impute
    df_imputed = reindex_and_impute(df_sparse, freq="1h", add_imputed_flag=True)

    # Invariants verification
    assert len(df_imputed) == 1000, f"Expected 1,000 rows, got {len(df_imputed)}"
    assert (df_imputed.index == full_dates).all(), "Index must exactly match continuous hourly date_range"
    assert not df_imputed["close"].isna().any(), "Close price must have 0 NaNs"
    assert not df_imputed["open"].isna().any(), "Open price must have 0 NaNs"
    assert not df_imputed["high"].isna().any(), "High price must have 0 NaNs"
    assert not df_imputed["low"].isna().any(), "Low price must have 0 NaNs"
    assert not df_imputed["volume"].isna().any(), "Volume must have 0 NaNs"

    # Verify imputed rows have volume == 0.0
    imputed_mask = df_imputed["is_imputed"]
    assert (df_imputed.loc[imputed_mask, "volume"] == 0.0).all(), "All imputed bars must have volume == 0.0"

  def test_adv_preprocessor_extreme_sparse_boundary_2_points(self):
    """Stress-test when dataset contains only 2 points separated by 500 hours (99.6% missing)."""
    t0 = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    t1 = pd.Timestamp("2026-01-21 20:00:00", tz="UTC")  # exactly 500 hours later

    df = pd.DataFrame({
        "open": [100.0, 200.0],
        "high": [110.0, 210.0],
        "low": [90.0, 190.0],
        "close": [105.0, 205.0],
        "volume": [10.0, 20.0],
    }, index=pd.DatetimeIndex([t0, t1]))

    df_imputed = reindex_and_impute(df, freq="1h")
    assert len(df_imputed) == 501
    assert df_imputed["close"].iloc[0] == 105.0
    # Imputed rows between 1 and 499 must carry forward close = 105.0
    assert (df_imputed["close"].iloc[1:500] == 105.0).all()
    # Imputed rows must have volume = 0.0
    assert (df_imputed["volume"].iloc[1:500] == 0.0).all()
    assert df_imputed["close"].iloc[500] == 205.0
    assert df_imputed["volume"].iloc[500] == 20.0

  def test_adv_preprocessor_unsorted_and_duplicate_timestamps(self):
    """Stress-test `normalize_datetime_index` with scrambled timestamps and conflicting duplicates."""
    t1 = pd.Timestamp("2026-01-01 01:00:00")
    t2 = pd.Timestamp("2026-01-01 02:00:00")
    t3 = pd.Timestamp("2026-01-01 03:00:00")

    # Scrambled order with t2 appearing twice with different close prices
    df_messy = pd.DataFrame({
        "timestamp": [t3, t1, t2, t2],
        "close": [300.0, 100.0, 200.0, 250.0],  # Latest t2 has 250.0
        "open": [300.0, 100.0, 200.0, 250.0],
        "high": [300.0, 100.0, 200.0, 250.0],
        "low": [300.0, 100.0, 200.0, 250.0],
        "volume": [30.0, 10.0, 20.0, 25.0],
    })

    df_clean = normalize_datetime_index(df_messy)

    # 1. Monotonic ascending index
    assert df_clean.index.is_monotonic_increasing
    assert len(df_clean) == 3, "Duplicate t2 must be collapsed"
    # 2. Kept latest duplicate (250.0)
    t2_utc = pd.Timestamp("2026-01-01 02:00:00", tz="UTC")
    assert df_clean.loc[t2_utc, "close"] == 250.0
    # 3. UTC localization
    assert str(df_clean.index.tz) == "UTC"

  def test_adv_preprocessor_target_horizons_constant_prices(self):
    """Stress-test target engineering under zero price changes."""
    n = 100
    df = pd.DataFrame({
        "close": [100.0] * n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"))

    out = add_target_horizons(df, resolution="1h", horizons=["1h", "24h"], include_direction=True)

    # Check 1h horizon: target_return must be 0.0 for rows 0..98, and NaN for row 99
    assert (out["target_return_1h"].iloc[:99] == 0.0).all()
    assert np.isnan(out["target_return_1h"].iloc[99])
    # Target direction: 1.0 for flat (>= current price)
    assert (out["target_direction_1h"].iloc[:99] == 1.0).all()
    assert np.isnan(out["target_direction_1h"].iloc[99])

    # Check 24h horizon: exactly 24 NaN rows at the tail
    assert out["target_price_24h"].iloc[-24:].isna().all()
    assert out["target_price_24h"].iloc[:-24].notna().all()


# ==============================================================================
# 3. Ingestion Adversarial Stress Tests
# ==============================================================================


class TestIngestionAdversarial:
  """Adversarial stress testing for API retry policies, rate limit backoff, and fallback failovers."""

  @mock.patch("time.sleep", return_value=None)
  def test_adv_binance_klines_429_chain_exhaustion(self, mock_sleep):
    """Stress-test Binance HTTP 429 response chain: verify max retries exhaustion and RateLimitExceededError."""
    session = mock.MagicMock(spec=requests.Session)
    mock_resp = mock.MagicMock(spec=requests.Response)
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "0.1"}
    session.get.return_value = mock_resp

    with pytest.raises(RateLimitExceededError) as exc_info:
      fetch_binance_klines("BTC/USDT", limit=10, max_retries=4, session=session)

    assert "Exhausted 4 retries" in str(exc_info.value)
    assert session.get.call_count == 4
    assert mock_sleep.call_count == 4

  @mock.patch("time.sleep", return_value=None)
  @mock.patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv")
  def test_adv_fetch_ohlcv_429_chain_automatic_fallback_to_yahoo(self, mock_yahoo, mock_sleep):
    """Stress-test unified `fetch_ohlcv`: when Binance fails with persistent 429 chain, fall back to Yahoo."""
    # Setup Binance failure
    mock_session = mock.MagicMock(spec=requests.Session)
    mock_resp_429 = mock.MagicMock(spec=requests.Response)
    mock_resp_429.status_code = 429
    mock_resp_429.headers = {}
    mock_session.get.return_value = mock_resp_429

    # Setup Yahoo success
    dates = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    yahoo_df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0] * 10,
        "high": [105.0] * 10,
        "low": [95.0] * 10,
        "close": [102.0] * 10,
        "volume": [1000.0] * 10,
    }, index=dates)
    mock_yahoo.return_value = yahoo_df

    with mock.patch("requests.Session", return_value=mock_session):
      result = fetch_ohlcv("BTC/USDT", interval="1h", limit=10, source="auto")

    assert result is not None
    assert len(result) == 10
    mock_yahoo.assert_called_once_with(symbol="BTC/USDT", interval="1h")

  @mock.patch("time.sleep", return_value=None)
  def test_adv_binance_weight_header_saturation_pacing(self, mock_sleep):
    """Stress-test proactive rate pacing when X-MBX-USED-WEIGHT-1M approaches 6000 threshold."""
    # Near threshold 5500 (warn threshold is 5000)
    headers = {"X-MBX-USED-WEIGHT-1M": "5500"}
    _inspect_binance_weight(headers)
    assert mock_sleep.call_count == 1
    sleep_arg = mock_sleep.call_args[0][0]
    # Sleep should be paced proportionally
    assert 0.2 <= sleep_arg <= 2.0

  @mock.patch("time.sleep", return_value=None)
  @mock.patch("requests.Session")
  def test_adv_fetch_ohlcv_both_sources_fail_raises_data_ingestion_error(self, mock_session_cls, mock_sleep):
    """Stress-test total failure when both Binance and Yahoo Finance fail."""
    mock_session = mock.MagicMock()
    mock_session_cls.return_value = mock_session
    # Binance fails with 500
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.headers = {}
    mock_session.get.return_value = mock_resp

    with mock.patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv", side_effect=YahooFinanceError("Yahoo down")):
      with pytest.raises(DataIngestionError) as exc_info:
        fetch_ohlcv("BTC/USDT", source="auto")

      assert "All data sources failed" in str(exc_info.value)

  def test_adv_backoff_jitter_bounds(self):
    """Verify exponential backoff sleep calculation remains strictly within bounded range with jitter."""
    for attempt in range(6):
      sleep_val = _compute_backoff_sleep(attempt=attempt, base=0.5, factor=2.0, max_sleep=10.0, jitter=0.5)
      assert sleep_val >= 0.1, "Backoff sleep must be at least 0.1s"
      # Maximum possible sleep for max_sleep=10.0 and jitter=0.5 is 10.0 + 0.5 = 10.5
      assert sleep_val <= 11.0, f"Backoff sleep exceeded upper limit: {sleep_val}"

  def test_adv_alternating_oscillation_wave(self):
    """Stress-test indicators under high-frequency alternating square wave (+1, -1, +1, -1)."""
    n = 200
    pattern = np.array([100.0, 102.0] * (n // 2))
    df = pd.DataFrame({
        "open": pattern,
        "high": pattern + 1.0,
        "low": pattern - 1.0,
        "close": pattern,
        "volume": [100.0] * n,
    }, index=pd.date_range("2026-01-01", periods=n, freq="1h"))

    out = add_technical_indicators(df)
    assert not out["rsi_14"].dropna().isna().any()
    # High frequency oscillation centered at 50
    assert 40.0 <= out["rsi_14"].iloc[-1] <= 60.0
    assert not np.isinf(out["macd"]).any()
    assert not np.isinf(out["bb_pct_b"]).any()

  def test_adv_preprocessor_99_percent_missing_large_scale(self):
    """Stress-test reindex_and_impute on 5,000 hourly candles with 99% missing data."""
    dates = pd.date_range("2025-01-01", periods=5000, freq="1h", tz="UTC")
    # Sample only 50 points
    idx = np.sort(np.random.choice(5000, size=50, replace=False))
    idx[0] = 0
    idx[-1] = 4999
    sparse = pd.DataFrame({
        "open": [100.0] * len(idx),
        "high": [105.0] * len(idx),
        "low": [95.0] * len(idx),
        "close": [100.0] * len(idx),
        "volume": [1000.0] * len(idx),
    }, index=dates[idx])

    reindexed = reindex_and_impute(sparse, freq="1h")
    assert len(reindexed) == 5000
    assert not reindexed["close"].isna().any()
    assert (reindexed["volume"].iloc[1:4999] == 0.0).sum() > 4900

