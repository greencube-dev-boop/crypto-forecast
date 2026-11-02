"""Market Data Ingestion, Dual-Resolution Preprocessing & Technical Indicators."""

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
    BinanceAPIError,
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
from crypto_forecast.data.preprocessor import (
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

__all__ = [
    # Ingestion
    "fetch_ohlcv",
    "fetch_binance_klines",
    "fetch_binance_ticker",
    "fetch_yahoo_ohlcv",
    "canonicalize_symbol",
    "normalize_symbol_for_binance",
    "normalize_symbol_for_yahoo",
    "DataIngestionError",
    "BinanceAPIError",
    "RateLimitExceededError",
    "YahooFinanceError",
    # Preprocessor
    "get_dual_resolution_data",
    "resample_dual_resolution",
    "resample_to_hourly",
    "resample_to_daily",
    "reindex_and_impute",
    "normalize_datetime_index",
    "detect_timestamp_gaps",
    "get_horizon_steps",
    "add_target_horizons",
    "create_dual_resolution_targets",
    "train_val_test_split",
    # Indicators
    "add_technical_indicators",
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_obv",
    "compute_volume_zscore",
]
