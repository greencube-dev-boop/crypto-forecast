"""Models Package - Time-Series Foundation Models & Sentiment Analysis.

Includes:
- Multi-horizon foundation forecasters (TimesFM, Chronos, Granite TTM, Baseline)
- CryptoBERT sentiment engine, RSS news scraper, polarity scoring, decay dynamics
"""

from __future__ import annotations

from crypto_forecast.models.base import (
    DIRECTION_THRESHOLD,
    BaseForecaster,
    HorizonPrediction,
)
from crypto_forecast.models.baseline import (
    PersistenceForecaster,
)
from crypto_forecast.models.chronos_model import (
    ChronosError,
    ChronosForecaster,
    ChronosModelError,
    FallbackChronosForecaster,
)
from crypto_forecast.models.granite_model import (
    FallbackGraniteForecaster,
    GraniteError,
    GraniteModelError,
    GraniteTTMForecaster,
)
from crypto_forecast.models.sentiment import (
    CryptoBertClassifier,
    HeuristicSentimentAnalyzer,
    RSSFeedError,
    SentimentError,
    SentimentInferenceError,
    SentimentModelError,
    analyze_sentiment,
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
    clean_article_url,
    clean_html_text,
    compute_horizon_gamma,
    compute_hourly_decay,
    compute_sentiment_polarity,
    compute_title_similarity,
    deduplicate_articles,
    fetch_crypto_news,
    fetch_latest_crypto_news,
    get_default_classifier,
    get_sentiment_series,
    normalize_rss_date,
    parse_rss_xml,
)
from crypto_forecast.models.timesfm_model import (
    FallbackTimesFMForecaster,
    TimesFMError,
    TimesFMForecaster,
    TimesFMInferenceError,
    TimesFMModelError,
)

__all__ = [
    "DIRECTION_THRESHOLD",
    "BaseForecaster",
    "ChronosError",
    "ChronosForecaster",
    "ChronosModelError",
    "CryptoBertClassifier",
    "FallbackChronosForecaster",
    "FallbackGraniteForecaster",
    "FallbackTimesFMForecaster",
    "GraniteError",
    "GraniteModelError",
    "GraniteTTMForecaster",
    "HeuristicSentimentAnalyzer",
    "HorizonPrediction",
    "PersistenceForecaster",
    "RSSFeedError",
    "SentimentError",
    "SentimentInferenceError",
    "SentimentModelError",
    "TimesFMError",
    "TimesFMForecaster",
    "TimesFMInferenceError",
    "TimesFMModelError",
    "analyze_sentiment",
    "apply_sentiment_drift",
    "apply_trajectory_sentiment_drift",
    "clean_article_url",
    "clean_html_text",
    "compute_horizon_gamma",
    "compute_hourly_decay",
    "compute_sentiment_polarity",
    "compute_title_similarity",
    "deduplicate_articles",
    "fetch_crypto_news",
    "fetch_latest_crypto_news",
    "get_default_classifier",
    "get_sentiment_series",
    "normalize_rss_date",
    "parse_rss_xml",
]
