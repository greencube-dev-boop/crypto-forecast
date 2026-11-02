"""Unit tests for crypto_forecast.data.ingestion module.

Covers Tier 1 (Coverage) and Tier 2 (Boundary & Error) test requirements for:
- Feature 1: Binance Public Klines Ingestion
- Feature 2: Binance 24hr Ticker & Spot Price
- Feature 3: Yahoo Finance Fallback Fetcher
- Unified OHLCV failover routing
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure yfinance module exists in sys.modules so patch("yfinance.download") works offline
if "yfinance" not in sys.modules:
  sys.modules["yfinance"] = MagicMock()

import numpy as np
import pandas as pd
import pytest
import requests

from crypto_forecast.config import (
    DEFAULT_SYMBOLS,
    OHLCV_COLUMNS,
)
from crypto_forecast.data.ingestion import (
    BinanceAPIError,
    BinanceGeoBlockedError,
    canonicalize_symbol,
    DataIngestionError,
    fetch_binance_klines,
    fetch_binance_ticker,
    fetch_ohlcv,
    fetch_yahoo_ohlcv,
    normalize_symbol_for_binance,
    normalize_symbol_for_yahoo,
    RateLimitExceededError,
    YahooFinanceError,
)

# ==============================================================================
# Helper Mock Generators
# ==============================================================================


def make_raw_kline(
    open_time: int,
    open_p: float = 78000.0,
    high_p: float = 78500.0,
    low_p: float = 77800.0,
    close_p: float = 78200.0,
    vol: float = 120.5,
) -> list[Any]:
  """Constructs a single 12-element Binance kline list."""
  return [
      open_time,
      f"{open_p:.2f}",
      f"{high_p:.2f}",
      f"{low_p:.2f}",
      f"{close_p:.2f}",
      f"{vol:.4f}",
      open_time + 3599999,
      "9423100.00",
      1542,
      "60.2",
      "4711550.00",
      "0",
  ]


def make_kline_batch(
    count: int, start_time: int = 1700000000000, interval_ms: int = 3600000
) -> list[list[Any]]:
  """Generates a contiguous sequence of raw Binance candles."""
  return [
      make_raw_kline(start_time + i * interval_ms, open_p=78000.0 + i * 10)
      for i in range(count)
  ]


def make_ticker_json(
    symbol: str = "BTCUSDT", last_price: float = 78500.0
) -> dict[str, Any]:
  """Constructs a mock 24hr ticker JSON payload."""
  return {
      "symbol": symbol,
      "priceChange": "150.00",
      "priceChangePercent": "0.19",
      "weightedAvgPrice": "78350.00",
      "prevClosePrice": "78350.00",
      "lastPrice": f"{last_price:.2f}",
      "openPrice": "78350.00",
      "highPrice": "79000.00",
      "lowPrice": "77800.00",
      "volume": "14500.25",
      "quoteVolume": "1136094575.00",
      "openTime": 1700000000000,
      "closeTime": 1700086400000,
      "count": 89450,
  }


# ==============================================================================
# Symbol Normalization Tests
# ==============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC/USDT", "BTCUSDT"),
        ("btc/usdt", "BTCUSDT"),
        ("BTC-USD", "BTCUSDT"),
        ("BTCUSDT", "BTCUSDT"),
        ("ETH/USDT", "ETHUSDT"),
        ("SOL/USDT", "SOLUSDT"),
    ],
)
def test_normalize_symbol_binance(raw: str, expected: str):
  assert normalize_symbol_for_binance(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC/USDT", "BTC-USD"),
        ("BTCUSDT", "BTC-USD"),
        ("ETH/USDT", "ETH-USD"),
        ("SOL/USDT", "SOL-USD"),
        ("DOGE/USDT", "DOGE-USD"),
    ],
)
def test_normalize_symbol_yahoo(raw: str, expected: str):
  assert normalize_symbol_for_yahoo(raw) == expected


def test_canonicalize_symbol():
  assert canonicalize_symbol("btcusdt") == "BTC/USDT"
  assert canonicalize_symbol("BTC-USD") == "BTC/USDT"
  assert canonicalize_symbol("ETH/USDT") == "ETH/USDT"


# ==============================================================================
# Feature 1: Binance Public Klines Tests (Tier 1 & Tier 2)
# ==============================================================================


def test_binance_klines_happy_path_schema():
  mock_resp = MagicMock()
  mock_resp.status_code = 200
  mock_resp.headers = {"X-MBX-USED-WEIGHT-1M": "45"}
  mock_resp.json.return_value = make_kline_batch(100)

  with patch("requests.Session.get", return_value=mock_resp):
    df = fetch_binance_klines("BTC/USDT", interval="1h", limit=100)

  assert list(df.columns) == OHLCV_COLUMNS
  assert len(df) == 100
  assert isinstance(df.index, pd.DatetimeIndex)
  assert df.index.tz is not None


def test_binance_klines_data_types():
  mock_resp = MagicMock()
  mock_resp.status_code = 200
  mock_resp.headers = {"X-MBX-USED-WEIGHT-1M": "10"}
  mock_resp.json.return_value = make_kline_batch(10)

  with patch("requests.Session.get", return_value=mock_resp):
    df = fetch_binance_klines("BTC/USDT", interval="1h", limit=10)

  assert df["open"].dtype == np.float64
  assert df["high"].dtype == np.float64
  assert df["low"].dtype == np.float64
  assert df["close"].dtype == np.float64
  assert df["volume"].dtype == np.float64
  assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_binance_klines_forward_pagination():
  batch1 = make_kline_batch(1000, start_time=1700000000000)
  batch2 = make_kline_batch(1000, start_time=1700000000000 + 1000 * 3600000)
  batch3 = make_kline_batch(500, start_time=1700000000000 + 2000 * 3600000)

  mock_resp1 = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=batch1)
  )
  mock_resp2 = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=batch2)
  )
  mock_resp3 = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=batch3)
  )

  with patch(
      "requests.Session.get", side_effect=[mock_resp1, mock_resp2, mock_resp3]
  ) as mock_get:
    df = fetch_binance_klines(
        "BTC/USDT", interval="1h", limit=2500, start_time=1700000000000
    )

  assert len(df) == 2500
  assert mock_get.call_count == 3
  # Check startTime advancement in pagination calls
  args2 = mock_get.call_args_list[1][1]["params"]
  assert args2["startTime"] == batch1[-1][6] + 1


def test_binance_klines_backward_pagination():
  batch1 = make_kline_batch(1000, start_time=2000000000000)
  batch2 = make_kline_batch(500, start_time=1500000000000)

  mock_resp1 = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=batch1)
  )
  mock_resp2 = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=batch2)
  )

  with patch(
      "requests.Session.get", side_effect=[mock_resp1, mock_resp2]
  ) as mock_get:
    df = fetch_binance_klines("BTC/USDT", interval="1h", limit=1500)

  assert len(df) == 1500
  assert mock_get.call_count == 2
  args2 = mock_get.call_args_list[1][1]["params"]
  assert args2["endTime"] == batch1[0][0] - 1


@pytest.mark.parametrize("interval", ["1h", "2h", "4h", "1d"])
def test_binance_klines_interval_support(interval: str):
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_kline_batch(5)),
  )
  with patch("requests.Session.get", return_value=mock_resp) as mock_get:
    df = fetch_binance_klines("BTC/USDT", interval=interval, limit=5)
  assert len(df) == 5
  assert mock_get.call_args[1]["params"]["interval"] == interval


def test_binance_klines_http_429_backoff_and_recover():
  resp_429 = MagicMock(
      status_code=429, headers={"Retry-After": "0.01"}, text="Too Many Requests"
  )
  resp_200 = MagicMock(
      status_code=200,
      headers={"X-MBX-USED-WEIGHT-1M": "100"},
      json=MagicMock(return_value=make_kline_batch(10)),
  )

  with (
      patch("requests.Session.get", side_effect=[resp_429, resp_200]),
      patch("time.sleep") as mock_sleep,
  ):
    df = fetch_binance_klines("BTC/USDT", limit=10)

  assert len(df) == 10
  assert mock_sleep.called


def test_binance_klines_weight_header_pacing():
  resp_high_weight = MagicMock(
      status_code=200,
      headers={"X-MBX-USED-WEIGHT-1M": "5500"},
      json=MagicMock(return_value=make_kline_batch(10)),
  )

  with (
      patch("requests.Session.get", return_value=resp_high_weight),
      patch("time.sleep") as mock_sleep,
  ):
    df = fetch_binance_klines("BTC/USDT", limit=10)

  assert len(df) == 10
  assert mock_sleep.called


def test_binance_klines_http_418_immediate_abort():
  resp_418 = MagicMock(
      status_code=418, headers={}, text="I'm a teapot / IP banned"
  )

  with patch("requests.Session.get", return_value=resp_418):
    with pytest.raises(BinanceAPIError) as exc_info:
      fetch_binance_klines("BTC/USDT", limit=10)
    assert exc_info.value.status_code == 418


def test_binance_klines_empty_response():
  mock_resp = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=[])
  )
  with patch("requests.Session.get", return_value=mock_resp):
    df = fetch_binance_klines("BTC/USDT", limit=10)
  assert df.empty
  assert list(df.columns) == OHLCV_COLUMNS


def test_binance_klines_http_400_invalid_symbol():
  mock_resp = MagicMock(
      status_code=400,
      headers={},
      json=MagicMock(
          return_value={"code": -1121, "msg": "Invalid symbol: FOOBAR"}
      ),
      text="Invalid symbol",
  )
  with patch("requests.Session.get", return_value=mock_resp):
    with pytest.raises(BinanceAPIError) as exc_info:
      fetch_binance_klines("FOOBAR/USDT", limit=10)
    assert exc_info.value.error_code == -1121


def test_binance_klines_timeout_exhaustion():
  with (
      patch(
          "requests.Session.get",
          side_effect=requests.Timeout("Connection timeout"),
      ),
      patch("time.sleep"),
  ):
    with pytest.raises(BinanceAPIError):
      fetch_binance_klines("BTC/USDT", limit=10, max_retries=2)


# ==============================================================================
# Feature 2: Binance 24hr Ticker Tests (Tier 1 & Tier 2)
# ==============================================================================


def test_binance_ticker_happy_path():
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json("BTCUSDT", 78500.0)),
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker("BTC/USDT")

  assert ticker["symbol"] == "BTC/USDT"
  assert ticker["last_price"] == 78500.0
  assert "high_price" in ticker
  assert "low_price" in ticker
  assert "volume" in ticker
  assert "open_time" in ticker
  assert "close_time" in ticker


def test_binance_ticker_numeric_conversions():
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json("ETHUSDT", 2500.5)),
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker("ETH/USDT")

  assert isinstance(ticker["last_price"], float)
  assert isinstance(ticker["volume"], float)
  assert isinstance(ticker["price_change_percent"], float)
  assert isinstance(ticker["trades_count"], int)


def test_binance_ticker_datetime_conversions():
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json()),
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker("BTC/USDT")

  assert isinstance(ticker["open_time"], pd.Timestamp)
  assert ticker["open_time"].tz is not None


@pytest.mark.parametrize("symbol", DEFAULT_SYMBOLS)
def test_binance_ticker_all_default_symbols(symbol: str):
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json()),
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker(symbol)
  assert ticker["symbol"] == canonicalize_symbol(symbol)


def test_binance_ticker_trades_count_int():
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json()),
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker("BTC/USDT")
  assert isinstance(ticker["trades_count"], int)


def test_binance_ticker_http_429_retry():
  resp_429 = MagicMock(status_code=429, headers={}, text="Rate limit")
  resp_200 = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_ticker_json()),
  )
  with (
      patch("requests.Session.get", side_effect=[resp_429, resp_200]),
      patch("time.sleep"),
  ):
    ticker = fetch_binance_ticker("BTC/USDT", max_retries=3)
  assert ticker["last_price"] == 78500.0


def test_binance_ticker_http_400_invalid_symbol():
  mock_resp = MagicMock(
      status_code=400, headers={}, text="Unknown symbol: INVALID"
  )
  with patch("requests.Session.get", return_value=mock_resp):
    with pytest.raises(BinanceAPIError):
      fetch_binance_ticker("INVALID/USDT")


def test_binance_ticker_zero_volume():
  payload = make_ticker_json()
  payload["volume"] = "0.00"
  payload["quoteVolume"] = "0.00"
  mock_resp = MagicMock(
      status_code=200, headers={}, json=MagicMock(return_value=payload)
  )
  with patch("requests.Session.get", return_value=mock_resp):
    ticker = fetch_binance_ticker("BTC/USDT")
  assert ticker["volume"] == 0.0


def test_binance_ticker_malformed_json():
  mock_resp = MagicMock(status_code=200, headers={})
  mock_resp.json.side_effect = ValueError("Malformed JSON")
  with patch("requests.Session.get", return_value=mock_resp):
    with pytest.raises(RateLimitExceededError):
      fetch_binance_ticker("BTC/USDT", max_retries=1)


def test_binance_ticker_network_failure():
  with (
      patch(
          "requests.Session.get",
          side_effect=requests.ConnectionError("Connection aborted"),
      ),
      patch("time.sleep"),
  ):
    with pytest.raises(BinanceAPIError):
      fetch_binance_ticker("BTC/USDT", max_retries=2)


# ==============================================================================
# Feature 3: Yahoo Finance Fallback Tests (Tier 1 & Tier 2)
# ==============================================================================


def test_yahoo_ohlcv_happy_path():
  dates = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [78000.0] * 10,
          "High": [78500.0] * 10,
          "Low": [77800.0] * 10,
          "Close": [78200.0] * 10,
          "Volume": [100.0] * 10,
      },
      index=dates,
  )

  with patch("yfinance.download", return_value=raw_df):
    df = fetch_yahoo_ohlcv("BTC/USDT", interval="1h")

  assert list(df.columns) == OHLCV_COLUMNS
  assert len(df) == 10
  assert df.index.tz is not None


def test_yahoo_ohlcv_timezone_normalization():
  dates_unaware = pd.date_range("2026-01-01", periods=5, freq="1h")
  raw_df = pd.DataFrame(
      {
          "Open": [78000.0] * 5,
          "High": [78500.0] * 5,
          "Low": [77800.0] * 5,
          "Close": [78200.0] * 5,
          "Volume": [100.0] * 5,
      },
      index=dates_unaware,
  )

  with patch("yfinance.download", return_value=raw_df):
    df = fetch_yahoo_ohlcv("BTC/USDT", interval="1h")

  assert df.index.tz is not None
  assert str(df.index.tz) == "UTC"


def test_yahoo_ohlcv_multiindex_flattening():
  dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
  tuples = [
      ("Open", "BTC-USD"),
      ("High", "BTC-USD"),
      ("Low", "BTC-USD"),
      ("Close", "BTC-USD"),
      ("Volume", "BTC-USD"),
  ]
  multi_cols = pd.MultiIndex.from_tuples(tuples)
  raw_df = pd.DataFrame(
      [[78000.0, 78500.0, 77500.0, 78200.0, 50.0]] * 5,
      index=dates,
      columns=multi_cols,
  )

  with patch("yfinance.download", return_value=raw_df):
    df = fetch_yahoo_ohlcv("BTC/USDT", interval="1h")

  assert list(df.columns) == OHLCV_COLUMNS
  assert len(df) == 5


def test_yahoo_ohlcv_direct_chart_fallback():
  chart_response = {
      "chart": {
          "result": [{
              "timestamp": [1700000000, 1700003600],
              "indicators": {
                  "quote": [{
                      "open": [78000.0, 78100.0],
                      "high": [78500.0, 78600.0],
                      "low": [77800.0, 77900.0],
                      "close": [78200.0, 78300.0],
                      "volume": [100.0, 150.0],
                  }]
              },
          }]
      }
  }
  mock_resp = MagicMock(status_code=200, json=MagicMock(return_value=chart_response))

  with (
      patch("yfinance.download", side_effect=ImportError("No yfinance")),
      patch("requests.Session.get", return_value=mock_resp),
  ):
    df = fetch_yahoo_ohlcv("BTC/USDT", interval="1h")

  assert len(df) == 2
  assert list(df.columns) == OHLCV_COLUMNS


def test_yahoo_ohlcv_period_clamping():
  dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [78000.0] * 5,
          "High": [78500.0] * 5,
          "Low": [77800.0] * 5,
          "Close": [78200.0] * 5,
          "Volume": [100.0] * 5,
      },
      index=dates,
  )

  with patch("yfinance.download", return_value=raw_df) as mock_yf:
    _ = fetch_yahoo_ohlcv("BTC/USDT", interval="1h", period="1000d")
  assert mock_yf.call_args[1]["period"] == "730d"


def test_yahoo_ohlcv_nan_imputation():
  dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [78000.0, np.nan, 78100.0, 78200.0, np.nan],
          "High": [78500.0, np.nan, 78600.0, 78700.0, 78800.0],
          "Low": [77800.0, 77900.0, np.nan, 78000.0, 78100.0],
          "Close": [78200.0, np.nan, 78300.0, np.nan, 78500.0],
          "Volume": [100.0, np.nan, 150.0, np.nan, 200.0],
      },
      index=dates,
  )

  with patch("yfinance.download", return_value=raw_df):
    df = fetch_yahoo_ohlcv("BTC/USDT")

  assert not df.isna().any().any()
  # Verify volume nan was zero-filled
  assert df.loc[dates[1], "volume"] == 0.0


def test_yahoo_ohlcv_empty_dataframe_raises():
  with patch("yfinance.download", return_value=pd.DataFrame()):
    with patch("requests.Session.get", side_effect=requests.ConnectionError):
      with pytest.raises(YahooFinanceError):
        fetch_yahoo_ohlcv("BTC/USDT")


def test_yahoo_ohlcv_all_nan_series_raises():
  dates = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [np.nan] * 5,
          "High": [np.nan] * 5,
          "Low": [np.nan] * 5,
          "Close": [np.nan] * 5,
          "Volume": [np.nan] * 5,
      },
      index=dates,
  )
  with patch("yfinance.download", return_value=raw_df):
    with pytest.raises(YahooFinanceError):
      fetch_yahoo_ohlcv("BTC/USDT")


def test_yahoo_ohlcv_zero_volume_fill():
  dates = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [100.0, 101.0, 102.0],
          "High": [105.0, 106.0, 107.0],
          "Low": [95.0, 96.0, 97.0],
          "Close": [101.0, 102.0, 103.0],
          "Volume": [50.0, np.nan, 70.0],
      },
      index=dates,
  )
  with patch("yfinance.download", return_value=raw_df):
    df = fetch_yahoo_ohlcv("BTC/USDT")
  assert df["volume"].iloc[1] == 0.0


def test_yahoo_ohlcv_missing_columns_raises():
  dates = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
  raw_df = pd.DataFrame(
      {
          "Open": [100.0, 101.0, 102.0],
          "High": [105.0, 106.0, 107.0],
          "Low": [95.0, 96.0, 97.0],
          # Missing Close and Volume
      },
      index=dates,
  )
  with patch("yfinance.download", return_value=raw_df):
    with pytest.raises(YahooFinanceError):
      fetch_yahoo_ohlcv("BTC/USDT")


# ==============================================================================
# Unified Ingestion & Failover Tests (Tier 1 & Tier 2)
# ==============================================================================


def test_fetch_ohlcv_primary_binance_success():
  binance_mock = MagicMock(
      return_value=pd.DataFrame({
          "timestamp": pd.date_range(
              "2026-01-01", periods=10, freq="1h", tz="UTC"
          ),
          "open": [78000.0] * 10,
          "high": [78500.0] * 10,
          "low": [77800.0] * 10,
          "close": [78200.0] * 10,
          "volume": [100.0] * 10,
      })
  )
  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines", binance_mock
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv"
    ) as yahoo_mock:
      df = fetch_ohlcv("BTC/USDT", interval="1h", limit=10)

  assert len(df) == 10
  assert binance_mock.called
  assert not yahoo_mock.called


def test_fetch_ohlcv_fallback_on_binance_429():
  yahoo_df = pd.DataFrame({
      "timestamp": pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC"),
      "open": [78000.0] * 10,
      "high": [78500.0] * 10,
      "low": [77800.0] * 10,
      "close": [78200.0] * 10,
      "volume": [100.0] * 10,
  })
  yahoo_df.index = pd.DatetimeIndex(yahoo_df["timestamp"], tz="UTC")

  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines",
      side_effect=RateLimitExceededError("429 exhausted"),
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv",
        return_value=yahoo_df,
    ) as yahoo_mock:
      df = fetch_ohlcv("BTC/USDT", interval="1h", limit=10)

  assert len(df) == 10
  assert yahoo_mock.called


def test_fetch_ohlcv_fallback_on_binance_418():
  yahoo_df = pd.DataFrame({
      "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
      "open": [78000.0] * 5,
      "high": [78500.0] * 5,
      "low": [77800.0] * 5,
      "close": [78200.0] * 5,
      "volume": [50.0] * 5,
  })
  yahoo_df.index = pd.DatetimeIndex(yahoo_df["timestamp"], tz="UTC")

  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines",
      side_effect=BinanceAPIError("HTTP 418 IP banned", status_code=418),
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv",
        return_value=yahoo_df,
    ) as yahoo_mock:
      df = fetch_ohlcv("BTC/USDT", interval="1h", limit=5)

  assert len(df) == 5
  assert yahoo_mock.called


def test_fetch_ohlcv_fallback_on_binance_timeout():
  yahoo_df = pd.DataFrame({
      "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
      "open": [78000.0] * 5,
      "high": [78500.0] * 5,
      "low": [77800.0] * 5,
      "close": [78200.0] * 5,
      "volume": [50.0] * 5,
  })
  yahoo_df.index = pd.DatetimeIndex(yahoo_df["timestamp"], tz="UTC")

  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines",
      side_effect=BinanceAPIError("Timeout"),
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv",
        return_value=yahoo_df,
    ) as yahoo_mock:
      df = fetch_ohlcv("BTC/USDT", interval="1h", limit=5)

  assert len(df) == 5
  assert yahoo_mock.called


def test_fetch_ohlcv_force_source_binance():
  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines",
      side_effect=BinanceAPIError("Binance Error", status_code=500),
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv"
    ) as yahoo_mock:
      with pytest.raises(BinanceAPIError):
        fetch_ohlcv("BTC/USDT", source="binance")
    assert not yahoo_mock.called


def test_fetch_ohlcv_all_sources_fail_raises():
  with patch(
      "crypto_forecast.data.ingestion.fetch_binance_klines",
      side_effect=RateLimitExceededError("Binance 429"),
  ):
    with patch(
        "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv",
        side_effect=YahooFinanceError("Yahoo Down"),
    ):
      with pytest.raises(DataIngestionError):
        fetch_ohlcv("BTC/USDT", source="auto")


def test_binance_klines_http_451_immediate_abort_zero_delay():
  """Asserts HTTP 451 immediately raises BinanceGeoBlockedError with status_code=451 and 0s sleep."""
  resp_451 = MagicMock(
      status_code=451,
      headers={},
      text="Service unavailable from a restricted location",
  )
  with (
      patch("requests.Session.get", return_value=resp_451) as mock_get,
      patch("time.sleep") as mock_sleep,
  ):
    with pytest.raises(BinanceGeoBlockedError) as exc_info:
      fetch_binance_klines("BTC/USDT", limit=10, max_retries=5)

    assert exc_info.value.status_code == 451
    # Must abort on the very first attempt without looping
    assert mock_get.call_count == 1
    # Must NOT call time.sleep (zero backoff delay)
    mock_sleep.assert_not_called()


def test_binance_ticker_http_451_immediate_abort_zero_delay():
  """Asserts HTTP 451 on ticker immediately raises BinanceGeoBlockedError with 0s sleep."""
  resp_451 = MagicMock(
      status_code=451,
      headers={},
      text="Service unavailable from a restricted location",
  )
  with (
      patch("requests.Session.get", return_value=resp_451) as mock_get,
      patch("time.sleep") as mock_sleep,
  ):
    with pytest.raises(BinanceGeoBlockedError) as exc_info:
      fetch_binance_ticker("BTC/USDT", max_retries=5)

    assert exc_info.value.status_code == 451
    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


def test_fetch_binance_klines_custom_base_url():
  """Asserts base_url parameter properly directs request to Binance US."""
  mock_resp = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_kline_batch(5)),
  )
  with patch("requests.Session.get", return_value=mock_resp) as mock_get:
    df = fetch_binance_klines(
        "BTC/USDT", limit=5, base_url="https://api.binance.us"
    )

  assert len(df) == 5
  called_url = mock_get.call_args[0][0]
  assert called_url == "https://api.binance.us/api/v3/klines"


def test_fetch_ohlcv_http_451_fallback_to_binance_us():
  """Asserts HTTP 451 on Binance Global immediately routes to Binance US with 0s delay."""
  resp_451 = MagicMock(
      status_code=451,
      headers={},
      text="Geo-blocked",
  )
  resp_us_200 = MagicMock(
      status_code=200,
      headers={},
      json=MagicMock(return_value=make_kline_batch(10)),
  )

  def mock_get(url, *args, **kwargs):
    if "api.binance.com" in url:
      return resp_451
    if "api.binance.us" in url:
      return resp_us_200
    raise ValueError(f"Unexpected url: {url}")

  with (
      patch("requests.Session.get", side_effect=mock_get) as mock_req,
      patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv") as mock_yahoo,
      patch("time.sleep") as mock_sleep,
  ):
    df = fetch_ohlcv("BTC/USDT", interval="1h", limit=10, source="auto")

  assert len(df) == 10
  assert mock_req.call_count == 2
  # Binance Global called first, Binance US called second
  assert "api.binance.com" in mock_req.call_args_list[0][0][0]
  assert "api.binance.us" in mock_req.call_args_list[1][0][0]
  # Yahoo Finance was bypassed because Binance US succeeded
  assert not mock_yahoo.called
  # Zero backoff sleep was invoked
  mock_sleep.assert_not_called()


def test_fetch_ohlcv_http_451_cascade_to_yahoo_finance():
  """Asserts HTTP 451 on both Binance Global and Binance US immediately cascades to Yahoo Finance."""
  resp_451 = MagicMock(
      status_code=451,
      headers={},
      text="Geo-blocked",
  )
  yahoo_df = pd.DataFrame({
      "timestamp": pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC"),
      "open": [78000.0] * 10,
      "high": [78500.0] * 10,
      "low": [77800.0] * 10,
      "close": [78200.0] * 10,
      "volume": [100.0] * 10,
  })
  yahoo_df.index = pd.DatetimeIndex(yahoo_df["timestamp"], tz="UTC")

  with (
      patch("requests.Session.get", return_value=resp_451) as mock_req,
      patch(
          "crypto_forecast.data.ingestion.fetch_yahoo_ohlcv",
          return_value=yahoo_df,
      ) as mock_yahoo,
      patch("time.sleep") as mock_sleep,
  ):
    df = fetch_ohlcv("BTC/USDT", interval="1h", limit=10, source="auto")

  assert len(df) == 10
  # Binance Global called first, Binance US called second, then Yahoo
  assert mock_req.call_count == 2
  assert mock_yahoo.called
  mock_sleep.assert_not_called()


def test_fetch_ohlcv_force_source_binance_us():
  """Asserts source='binance_us' directly queries Binance US and raises on failure."""
  resp_500 = MagicMock(
      status_code=500,
      headers={},
      text="Internal Server Error",
  )
  with (
      patch("requests.Session.get", return_value=resp_500),
      patch("time.sleep"),
      patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv") as mock_yahoo,
  ):
    with pytest.raises(BinanceAPIError):
      fetch_ohlcv("BTC/USDT", source="binance_us")

    assert not mock_yahoo.called

