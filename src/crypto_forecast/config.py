"""Crypto Forecast Configuration Module.

Centralizes all system constants, discrete forecasting horizons, supported
crypto pairs, candidate foundation models, API endpoints, and network pacing
settings.
"""

from __future__ import annotations

from typing import Final

# ==============================================================================
# 1. Multi-Horizon Forecasting Horizons
# ==============================================================================

# 48 Continuous Hourly Horizons specified in Requirement R3 & ORIGINAL_REQUEST.md
HOURLY_48_HORIZONS: Final[list[str]] = [f"{i}h" for i in range(1, 49)]
DISCRETE_11_HORIZONS: Final[list[str]] = [
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "24h",
    "2d",
    "3d",
    "7d",
    "14d",
    "30d",
]

DEFAULT_HORIZONS: Final[list[str]] = HOURLY_48_HORIZONS

# Discrete mapping from horizon identifier to equivalent integer hours (1h..48h and legacy multi-day)
HORIZON_TO_HOURS: Final[dict[str, int]] = {
    **{f"{i}h": i for i in range(1, 49)},
    "1d": 24,
    "2d": 48,
    "3d": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}


def get_horizon_hours(horizon: str) -> int:
  """Converts a horizon string ('1h'..'48h', '1d'..'30d') into integer hours."""
  if horizon in HORIZON_TO_HOURS:
    return HORIZON_TO_HOURS[horizon]
  clean = horizon.strip().lower()
  if clean.endswith("h") and clean[:-1].isdigit():
    return int(clean[:-1])
  raise ValueError(f"Unknown horizon identifier: '{horizon}'")


# Dual-resolution routing:
# Intraday pipeline (1-hour candles): horizons up to 48 hours (48 steps <= 64 Chronos-Bolt cap)
INTRADAY_HORIZONS: Final[list[str]] = HOURLY_48_HORIZONS + ["1d", "2d"]

# Macro pipeline (1-day candles): horizons from 1 day to 30 days (1 to 30 steps <= 64 Chronos-Bolt cap)
MACRO_HORIZONS: Final[list[str]] = [
    "1d",
    "2d",
    "3d",
    "7d",
    "14d",
    "30d",
]

# Discrete step mappings per resolution
INTRADAY_STEPS: Final[list[int]] = list(range(1, 49))
MACRO_STEPS: Final[list[int]] = [1, 2, 3, 7, 14, 30]

# ==============================================================================
# 2. Supported Cryptocurrency Symbols & Canonical Mappings
# ==============================================================================

DEFAULT_SYMBOLS: Final[list[str]] = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "XRP/USDT",
]

BENCHMARK_SYMBOL: Final[str] = "BTC/USDT"

# Binance REST symbol format (e.g. BTCUSDT)
BINANCE_SYMBOL_MAP: Final[dict[str, str]] = {
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
    "BNB/USDT": "BNBUSDT",
    "ADA/USDT": "ADAUSDT",
    "DOGE/USDT": "DOGEUSDT",
    "XRP/USDT": "XRPUSDT",
}

# Yahoo Finance symbol format (e.g. BTC-USD)
YAHOO_SYMBOL_MAP: Final[dict[str, str]] = {
    "BTC/USDT": "BTC-USD",
    "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD",
    "BNB/USDT": "BNB-USD",
    "ADA/USDT": "ADA-USD",
    "DOGE/USDT": "DOGE-USD",
    "XRP/USDT": "XRP-USD",
}

# ==============================================================================
# 3. Model Identifiers & Registry
# ==============================================================================

PRIMARY_MODEL_ID: Final[str] = "google/timesfm-2.5-200m-pytorch"
CHRONOS_BOLT_MODEL_ID: Final[str] = "amazon/chronos-bolt-small"
CHRONOS_2_MODEL_ID: Final[str] = "amazon/chronos-2"
GRANITE_TTM_MODEL_ID: Final[str] = "ibm-granite/granite-timeseries-ttm-r2"
SENTIMENT_MODEL_ID: Final[str] = "ElKulako/cryptobert"
BASELINE_MODEL_ID: Final[str] = "persistence"

CANDIDATE_MODELS: Final[list[str]] = [
    PRIMARY_MODEL_ID,
    CHRONOS_BOLT_MODEL_ID,
    CHRONOS_2_MODEL_ID,
    GRANITE_TTM_MODEL_ID,
    BASELINE_MODEL_ID,
]

# Chronos-Bolt maximum prediction horizon length in a single non-autoregressive patch
CHRONOS_MAX_PREDICTION_LENGTH: Final[int] = 64

# ==============================================================================
# 4. API Endpoints & Public Feeds
# ==============================================================================

BINANCE_BASE_URL: Final[str] = "https://api.binance.com"
BINANCE_US_BASE_URL: Final[str] = "https://api.binance.us"
BINANCE_KLINES_PATH: Final[str] = "/api/v3/klines"
BINANCE_TICKER_24HR_PATH: Final[str] = "/api/v3/ticker/24hr"
BINANCE_TICKER_PRICE_PATH: Final[str] = "/api/v3/ticker/price"

YAHOO_CHART_BASE_URL: Final[str] = (
    "https://query1.finance.yahoo.com/v8/finance/chart"
)

COINTELEGRAPH_RSS_URL: Final[str] = "https://cointelegraph.com/rss"
COINDESK_RSS_URL: Final[str] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
)

# ==============================================================================
# 5. Rate Limits, Timeouts & Backoff Settings
# ==============================================================================

BINANCE_WEIGHT_LIMIT_1M: Final[int] = 6000
BINANCE_WEIGHT_WARN_THRESHOLD: Final[int] = 5000
DEFAULT_REQUEST_TIMEOUT: Final[int] = 10
DEFAULT_MAX_RETRIES: Final[int] = 5
DEFAULT_BACKOFF_BASE: Final[float] = 1.0
DEFAULT_BACKOFF_FACTOR: Final[float] = 2.0
DEFAULT_BACKOFF_MAX: Final[float] = 32.0
DEFAULT_BACKOFF_JITTER: Final[float] = 0.5

# ==============================================================================
# 6. Data Schema & Feature Engineering Constants
# ==============================================================================

OHLCV_COLUMNS: Final[list[str]] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

DEFAULT_HOURLY_LIMIT: Final[int] = 1000
DEFAULT_DAILY_LIMIT: Final[int] = 365
YAHOO_HOURLY_MAX_PERIOD_DAYS: Final[int] = 730

# Sentiment settings
SENTIMENT_DECAY_ALPHA: Final[float] = 0.95
CRYPTOBERT_MAX_LENGTH: Final[int] = 128
SENTIMENT_LABELS: Final[dict[int, str]] = {
    0: "Bearish",
    1: "Neutral",
    2: "Bullish",
}

# RSS Feed Configurations
DEFAULT_RSS_FEEDS: Final[dict[str, str]] = {
    "CoinTelegraph": COINTELEGRAPH_RSS_URL,
    "CoinDesk": COINDESK_RSS_URL,
}
DEFAULT_RSS_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_RSS_HEADERS: Final[dict[str, str]] = {
    "User-Agent": DEFAULT_RSS_USER_AGENT,
    "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_MAX_NEWS_ARTICLES: Final[int] = 50
TITLE_SIMILARITY_THRESHOLD: Final[float] = 0.85

# Sentiment Conviction Drift Settings
DEFAULT_SENTIMENT_GAMMA_0: Final[float] = 0.02
DEFAULT_SENTIMENT_LAMBDA: Final[float] = 0.0032

