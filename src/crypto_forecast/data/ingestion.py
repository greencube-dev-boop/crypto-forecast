"""Market Data Ingestion Module.

Delivers robust, keyless historical and real-time OHLCV market data ingestion
via:
1. Primary engine: Binance REST (/api/v3/klines and /api/v3/ticker/24hr)
   - 1000-candle paginated retrieval
   - Rate limit pacing via X-MBX-USED-WEIGHT-1M header inspection
   - Exponential backoff with jitter on HTTP 429
   - HTTP 418 ban prevention with immediate fallback
2. Secondary fallback: Yahoo Finance (yfinance / direct chart API)
   - Mandatory .ffill().bfill() data sanitization
   - MultiIndex column flattening and timezone normalization to UTC
3. Unified entry point: fetch_ohlcv() with automatic source failover
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Mapping
import numpy as np
import pandas as pd
import requests

from crypto_forecast.config import (
    BINANCE_BASE_URL,
    BINANCE_KLINES_PATH,
    BINANCE_SYMBOL_MAP,
    BINANCE_TICKER_24HR_PATH,
    BINANCE_WEIGHT_LIMIT_1M,
    BINANCE_WEIGHT_WARN_THRESHOLD,
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BACKOFF_JITTER,
    DEFAULT_BACKOFF_MAX,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    OHLCV_COLUMNS,
    YAHOO_CHART_BASE_URL,
    YAHOO_SYMBOL_MAP,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Exceptions
# ==============================================================================


class DataIngestionError(Exception):
  """Base exception for all market data ingestion failures."""

  pass


class BinanceAPIError(DataIngestionError):
  """Raised when Binance returns an unrecoverable HTTP status or error payload."""

  def __init__(
      self,
      message: str,
      status_code: int | None = None,
      error_code: int | None = None,
  ) -> None:
    super().__init__(message)
    self.status_code = status_code
    self.error_code = error_code


class RateLimitExceededError(BinanceAPIError):
  """Raised when HTTP 429 persists after exhausting all retries."""

  pass


class YahooFinanceError(DataIngestionError):
  """Raised when Yahoo Finance retrieval fails or returns empty/corrupted data."""

  pass


# ==============================================================================
# Symbol Normalization Helpers
# ==============================================================================


def canonicalize_symbol(symbol: str) -> str:
  """Normalizes arbitrary cryptocurrency symbol string into canonical 'BASE/QUOTE' format.

  Examples:
      'btcusdt' -> 'BTC/USDT'
      'BTCUSDT' -> 'BTC/USDT'
      'BTC-USD' -> 'BTC/USDT'
      'BTC/USDT' -> 'BTC/USDT'
  """
  clean = symbol.strip().upper()
  if "/" in clean:
    return clean
  if "-" in clean:
    base, quote = clean.split("-", 1)
    if quote in ("USD", "USDT"):
      return f"{base}/USDT"
    return f"{base}/{quote}"
  for quote in ("USDT", "BUSD", "USDC", "USD", "BTC", "ETH"):
    if clean.endswith(quote):
      base = clean[: -len(quote)]
      return f"{base}/USDT" if quote in ("USD", "USDT") else f"{base}/{quote}"
  return f"{clean}/USDT"


def normalize_symbol_for_binance(symbol: str) -> str:
  """Maps a canonical or raw symbol to Binance format (e.g. 'BTC/USDT' -> 'BTCUSDT')."""
  canon = canonicalize_symbol(symbol)
  if canon in BINANCE_SYMBOL_MAP:
    return BINANCE_SYMBOL_MAP[canon]
  return canon.replace("/", "").replace("-", "")


def normalize_symbol_for_yahoo(symbol: str) -> str:
  """Maps a canonical or raw symbol to Yahoo Finance format (e.g. 'BTC/USDT' -> 'BTC-USD')."""
  canon = canonicalize_symbol(symbol)
  if canon in YAHOO_SYMBOL_MAP:
    return YAHOO_SYMBOL_MAP[canon]
  base, quote = canon.split("/", 1)
  quote_mapped = "USD" if quote in ("USDT", "USD") else quote
  return f"{base}-{quote_mapped}"


# ==============================================================================
# Rate Limit & Backoff Utilities
# ==============================================================================


def _compute_backoff_sleep(
    attempt: int,
    base: float = DEFAULT_BACKOFF_BASE,
    factor: float = DEFAULT_BACKOFF_FACTOR,
    max_sleep: float = DEFAULT_BACKOFF_MAX,
    jitter: float = DEFAULT_BACKOFF_JITTER,
) -> float:
  """Calculates exponential backoff with full jitter."""
  calculated = min(max_sleep, base * (factor**attempt))
  sleep_time = random.uniform(0.5 * calculated, calculated) + random.uniform(
      0.0, jitter
  )
  return max(0.1, sleep_time)


def _inspect_binance_weight(headers: Mapping[str, Any] | None) -> None:
  """Checks Binance rate limit headers and sleeps proactively if near 6000 threshold."""
  if not headers:
    return
  for header_key in (
      "X-MBX-USED-WEIGHT-1M",
      "x-mbx-used-weight-1m",
      "X-MBX-USED-WEIGHT",
  ):
    if header_key in headers:
      try:
        used_weight = int(headers[header_key])
        if used_weight >= BINANCE_WEIGHT_WARN_THRESHOLD:
          pace_sleep = min(
              2.0,
              (used_weight - BINANCE_WEIGHT_WARN_THRESHOLD) / 1000.0 * 1.5 + 0.2,
          )
          logger.warning(
              "Binance IP weight high (%d/%d). Throttling for %.2fs",
              used_weight,
              BINANCE_WEIGHT_LIMIT_1M,
              pace_sleep,
          )
          time.sleep(pace_sleep)
      except (ValueError, TypeError):
        pass
      break


# ==============================================================================
# Core Ingestion Functions
# ==============================================================================


def fetch_binance_klines(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    session: requests.Session | None = None,
) -> pd.DataFrame:
  """Fetches historical and real-time OHLCV candles from Binance public REST API.

  Supports automatic pagination for requests requesting >1000 candles or wide
  timeframes.

  Args:
      symbol: Cryptocurrency symbol (e.g. 'BTC/USDT', 'BTCUSDT').
      interval: Kline interval ('1m', '1h', '2h', '4h', '1d', '3d', '1w').
      limit: Total number of candles to return.
      start_time: Optional start timestamp in milliseconds.
      end_time: Optional end timestamp in milliseconds.
      max_retries: Max retry attempts on network error or HTTP 429.
      session: Optional requests.Session instance for connection reuse.

  Returns:
      pd.DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close',
      'volume'],
      indexed by a continuous UTC DatetimeIndex.

  Raises:
      BinanceAPIError: If parameters are invalid or exchange errors out.
      RateLimitExceededError: If rate limit cannot be resolved within
      max_retries.
  """
  req_session = session or requests.Session()
  binance_symbol = normalize_symbol_for_binance(symbol)
  url = f"{BINANCE_BASE_URL}{BINANCE_KLINES_PATH}"

  all_raw_candles: list[list[Any]] = []
  total_needed = max(1, limit)

  def _make_request(params: dict[str, Any]) -> list[list[Any]]:
    for attempt in range(max_retries):
      try:
        resp = req_session.get(
            url, params=params, timeout=DEFAULT_REQUEST_TIMEOUT
        )
        _inspect_binance_weight(resp.headers)

        if resp.status_code == 200:
          data = resp.json()
          if not isinstance(data, list):
            raise BinanceAPIError(f"Unexpected response format: {data}")
          return data

        if resp.status_code == 418:
          # Immediate IP ban indicator - abort retries immediately to allow fallback
          raise BinanceAPIError(
              "Binance HTTP 418 IP auto-ban detected. Aborting.",
              status_code=418,
          )

        if resp.status_code == 429:
          retry_after = resp.headers.get("Retry-After")
          sleep_sec = (
              float(retry_after)
              if retry_after
              else _compute_backoff_sleep(attempt)
          )
          logger.warning(
              "Binance HTTP 429 hit. Sleeping %.2fs (attempt %d/%d)",
              sleep_sec,
              attempt + 1,
              max_retries,
          )
          time.sleep(sleep_sec)
          continue

        if resp.status_code == 400:
          data = resp.json() if resp.text else {}
          msg = (
              data.get("msg", resp.text)
              if isinstance(data, dict)
              else resp.text
          )
          code = data.get("code") if isinstance(data, dict) else None
          raise BinanceAPIError(
              f"Binance HTTP 400: {msg}", status_code=400, error_code=code
          )

        # 5xx Server Errors or other statuses
        sleep_sec = _compute_backoff_sleep(attempt)
        logger.warning(
            "Binance HTTP %d: %s. Retrying in %.2fs",
            resp.status_code,
            resp.text,
            sleep_sec,
        )
        time.sleep(sleep_sec)

      except (requests.Timeout, requests.ConnectionError) as exc:
        if attempt == max_retries - 1:
          raise BinanceAPIError(
              f"Binance connection failed after {max_retries} retries: {exc}"
          ) from exc
        sleep_sec = _compute_backoff_sleep(attempt)
        logger.warning(
            "Network error: %s. Retrying in %.2fs", str(exc), sleep_sec
        )
        time.sleep(sleep_sec)

    raise RateLimitExceededError(
        f"Exhausted {max_retries} retries for Binance klines request."
    )

  # Pagination loop
  if start_time is not None:
    # Forward pagination starting from start_time
    curr_start = start_time
    while len(all_raw_candles) < total_needed:
      batch_limit = min(1000, total_needed - len(all_raw_candles))
      params: dict[str, Any] = {
          "symbol": binance_symbol,
          "interval": interval,
          "limit": batch_limit,
          "startTime": curr_start,
      }
      if end_time is not None:
        params["endTime"] = end_time

      batch = _make_request(params)
      if not batch:
        break
      all_raw_candles.extend(batch)
      last_close_time = batch[-1][6]
      curr_start = last_close_time + 1
      if end_time is not None and curr_start >= end_time:
        break
      if len(batch) < batch_limit:
        break
  else:
    # Backward pagination when retrieving latest N candles without start_time
    curr_end = end_time
    collected_batches: list[list[list[Any]]] = []
    candles_accumulated = 0

    while candles_accumulated < total_needed:
      batch_limit = min(1000, total_needed - candles_accumulated)
      params = {
          "symbol": binance_symbol,
          "interval": interval,
          "limit": batch_limit,
      }
      if curr_end is not None:
        params["endTime"] = curr_end

      batch = _make_request(params)
      if not batch:
        break
      collected_batches.append(batch)
      candles_accumulated += len(batch)
      earliest_open_time = batch[0][0]
      curr_end = earliest_open_time - 1
      if len(batch) < batch_limit:
        break

    # Assemble in chronological order
    for batch in reversed(collected_batches):
      all_raw_candles.extend(batch)

  if not all_raw_candles:
    empty_df = pd.DataFrame(columns=OHLCV_COLUMNS)
    empty_df["timestamp"] = pd.to_datetime([], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
      empty_df[col] = pd.Series([], dtype=np.float64)
    empty_df.index = pd.DatetimeIndex([], tz="UTC")
    return empty_df

  # Parse 12-element Binance candles
  timestamps = [
      pd.to_datetime(c[0], unit="ms", utc=True) for c in all_raw_candles
  ]
  opens = [float(c[1]) for c in all_raw_candles]
  highs = [float(c[2]) for c in all_raw_candles]
  lows = [float(c[3]) for c in all_raw_candles]
  closes = [float(c[4]) for c in all_raw_candles]
  volumes = [float(c[5]) for c in all_raw_candles]

  df = pd.DataFrame({
      "timestamp": timestamps,
      "open": opens,
      "high": highs,
      "low": lows,
      "close": closes,
      "volume": volumes,
  })

  # Deduplicate on timestamp and sort ascending
  df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
  if len(df) > total_needed:
    df = df.iloc[-total_needed:]
  df = df.reset_index(drop=True)
  df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC")
  return df


def fetch_binance_ticker(
    symbol: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    session: requests.Session | None = None,
) -> dict[str, Any]:
  """Fetches 24-hour rolling ticker statistics and real-time spot price from Binance.

  Args:
      symbol: Cryptocurrency symbol (e.g. 'BTC/USDT').
      max_retries: Retry attempts.
      session: Optional requests session.

  Returns:
      Standardized dictionary with float metrics and UTC timestamps:
      {
          'symbol': 'BTC/USDT',
          'last_price': float,
          'price_change': float,
          'price_change_percent': float,
          'weighted_avg_price': float,
          'prev_close_price': float,
          'open_price': float,
          'high_price': float,
          'low_price': float,
          'volume': float,
          'quote_volume': float,
          'open_time': pd.Timestamp,
          'close_time': pd.Timestamp,
          'trades_count': int
      }
  """
  req_session = session or requests.Session()
  binance_symbol = normalize_symbol_for_binance(symbol)
  canonical = canonicalize_symbol(symbol)
  url = f"{BINANCE_BASE_URL}{BINANCE_TICKER_24HR_PATH}"

  for attempt in range(max_retries):
    try:
      resp = req_session.get(
          url,
          params={"symbol": binance_symbol},
          timeout=DEFAULT_REQUEST_TIMEOUT,
      )
      _inspect_binance_weight(resp.headers)

      if resp.status_code == 200:
        try:
          data = resp.json()
        except (ValueError, TypeError) as exc:
          if attempt == max_retries - 1:
            raise RateLimitExceededError(
                f"Failed to parse Binance ticker response: {exc}"
            ) from exc
          time.sleep(_compute_backoff_sleep(attempt))
          continue

        return {
            "symbol": canonical,
            "last_price": float(data.get("lastPrice", 0.0)),
            "price_change": float(data.get("priceChange", 0.0)),
            "price_change_percent": float(data.get("priceChangePercent", 0.0)),
            "weighted_avg_price": float(data.get("weightedAvgPrice", 0.0)),
            "prev_close_price": float(data.get("prevClosePrice", 0.0)),
            "open_price": float(data.get("openPrice", 0.0)),
            "high_price": float(data.get("highPrice", 0.0)),
            "low_price": float(data.get("lowPrice", 0.0)),
            "volume": float(data.get("volume", 0.0)),
            "quote_volume": float(data.get("quoteVolume", 0.0)),
            "open_time": pd.to_datetime(
                data.get("openTime", 0), unit="ms", utc=True
            ),
            "close_time": pd.to_datetime(
                data.get("closeTime", 0), unit="ms", utc=True
            ),
            "trades_count": int(data.get("count", 0)),
        }

      if resp.status_code == 429:
        sleep_sec = _compute_backoff_sleep(attempt)
        time.sleep(sleep_sec)
        continue

      if resp.status_code == 400:
        raise BinanceAPIError(
            f"Invalid symbol '{symbol}' for Binance ticker: {resp.text}",
            status_code=400,
        )

      sleep_sec = _compute_backoff_sleep(attempt)
      time.sleep(sleep_sec)

    except (requests.Timeout, requests.ConnectionError) as exc:
      if attempt == max_retries - 1:
        raise BinanceAPIError(
            f"Binance ticker failed after {max_retries} attempts: {exc}"
        ) from exc
      time.sleep(_compute_backoff_sleep(attempt))

  raise RateLimitExceededError(
      f"Failed to fetch Binance ticker for {symbol} after {max_retries} attempts."
  )


def fetch_yahoo_ohlcv(
    symbol: str,
    interval: str = "1h",
    period: str = "730d",
    start: str | None = None,
    end: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
  """Fallback data fetcher retrieving OHLCV data from Yahoo Finance.

  Executes mandatory .ffill().bfill() data sanitization to guarantee 0 NaN
  values.

  Args:
      symbol: Canonical or raw crypto symbol.
      interval: '1h' or '1d'.
      period: Lookback window ('730d' max for 1h; 'max' or multi-year for 1d).
      start: Optional start date string ('YYYY-MM-DD').
      end: Optional end date string ('YYYY-MM-DD').
      session: Optional requests session.

  Returns:
      pd.DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close',
      'volume'],
      indexed by a continuous UTC DatetimeIndex.
  """
  yahoo_symbol = normalize_symbol_for_yahoo(symbol)

  # Check lookback limit for 1h interval
  if interval == "1h" and period not in (
      "1d",
      "5d",
      "7d",
      "1mo",
      "3mo",
      "6mo",
      "1y",
      "2y",
      "730d",
  ):
    logger.warning(
        "Yahoo Finance caps 1h interval at 730d lookback. Clamping period to"
        " 730d."
    )
    period = "730d"

  df: pd.DataFrame | None = None

  # Try yfinance package first
  try:
    import yfinance as yf

    data = yf.download(
        tickers=yahoo_symbol,
        interval=interval,
        period=period if start is None else None,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )
    if data is not None and not data.empty:
      df = data
  except Exception as e:
    logger.info(
        "yfinance library call failed (%s), attempting direct Yahoo chart API.",
        e,
    )

  # Direct HTTP fallback to Yahoo Chart API if yfinance library is unavailable or returns empty
  if df is None or df.empty:
    req_session = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    chart_url = f"{YAHOO_CHART_BASE_URL}/{yahoo_symbol}"
    params = {"interval": interval, "range": period}
    try:
      resp = req_session.get(
          chart_url,
          params=params,
          headers=headers,
          timeout=DEFAULT_REQUEST_TIMEOUT,
      )
      if resp.status_code == 200:
        res_json = resp.json()
        result = res_json.get("chart", {}).get("result")
        if result and len(result) > 0:
          timestamps_raw = result[0].get("timestamp", [])
          quote = result[0].get("indicators", {}).get("quote", [{}])[0]
          if timestamps_raw and quote:
            df = pd.DataFrame(
                {
                    "Open": quote.get("open", []),
                    "High": quote.get("high", []),
                    "Low": quote.get("low", []),
                    "Close": quote.get("close", []),
                    "Volume": quote.get("volume", []),
                },
                index=pd.to_datetime(timestamps_raw, unit="s", utc=True),
            )
    except Exception as exc:
      raise YahooFinanceError(
          f"Direct Yahoo Chart API failed for {yahoo_symbol}: {exc}"
      ) from exc

  if df is None or df.empty:
    raise YahooFinanceError(
        f"Yahoo Finance returned empty DataFrame for {yahoo_symbol}."
    )

  # Handle potential MultiIndex columns from newer yfinance
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df.rename(columns=str.lower)
  required_cols = ["open", "high", "low", "close", "volume"]
  for col in required_cols:
    if col not in df.columns:
      raise YahooFinanceError(
          f"Missing expected column '{col}' in Yahoo Finance output."
      )

  df = df[required_cols].copy()

  # Mandatory data sanitization: forward-fill then back-fill price columns
  df["close"] = df["close"].ffill().bfill()
  df["open"] = df["open"].fillna(df["close"])
  df["high"] = df["high"].fillna(df[["open", "close"]].max(axis=1))
  df["low"] = df["low"].fillna(df[["open", "close"]].min(axis=1))
  df["volume"] = df["volume"].fillna(0.0)

  # Check if all values are NaN
  if df["close"].isna().all():
    raise YahooFinanceError(f"All price values are NaN for {yahoo_symbol}.")

  # Ensure UTC timezone
  if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
  else:
    df.index = df.index.tz_convert("UTC")

  df["timestamp"] = df.index
  df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
  df = df.reset_index(drop=True)
  df.index = pd.DatetimeIndex(df["timestamp"], tz="UTC")
  return df[OHLCV_COLUMNS]


def fetch_ohlcv(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    start_time: int | None = None,
    end_time: int | None = None,
    source: str = "auto",
) -> pd.DataFrame:
  """Unified entry point for OHLCV candlestick data with automated exchange fallback.

  Adheres strictly to M1 <-> M3 / M4 interface contract in PROJECT.md:
  Returns pd.DataFrame with columns ['timestamp', 'open', 'high', 'low',
  'close', 'volume']
  indexed by a continuous UTC DatetimeIndex.

  Args:
      symbol: Canonical cryptocurrency symbol (e.g. 'BTC/USDT').
      interval: Resolution ('1h', '1d', etc.).
      limit: Requested candle count.
      start_time: Optional start timestamp (epoch ms).
      end_time: Optional end timestamp (epoch ms).
      source: Ingestion source: 'auto', 'binance', or 'yahoo'.

  Returns:
      Sanitized, verified DataFrame with strict OHLCV columns and UTC
      DatetimeIndex.

  Raises:
      DataIngestionError: If all candidate sources fail.
  """
  clean_source = source.strip().lower()

  if clean_source in ("auto", "binance"):
    try:
      df = fetch_binance_klines(
          symbol=symbol,
          interval=interval,
          limit=limit,
          start_time=start_time,
          end_time=end_time,
      )
      if not df.empty:
        return df
    except Exception as exc:
      if clean_source == "binance":
        raise
      logger.warning(
          "Primary Binance fetch failed for %s (%s). Falling back to Yahoo"
          " Finance.",
          symbol,
          exc,
      )

  if clean_source in ("auto", "yahoo"):
    try:
      df_yahoo = fetch_yahoo_ohlcv(symbol=symbol, interval=interval)
      if len(df_yahoo) > limit:
        df_yahoo = df_yahoo.iloc[-limit:].copy()
        df_yahoo.index = pd.DatetimeIndex(df_yahoo["timestamp"], tz="UTC")
      return df_yahoo
    except Exception as exc:
      logger.error("Yahoo Finance fallback also failed for %s: %s", symbol, exc)
      raise DataIngestionError(
          f"All data sources failed for {symbol}: {exc}"
      ) from exc

  raise DataIngestionError(f"Unsupported source provider: '{source}'")
