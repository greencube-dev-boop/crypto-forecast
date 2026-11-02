"""Shared pytest fixtures, mock data generators, and reference oracles for E2E tests.

Provides deterministic synthetic OHLCV data across 1000 hourly candles, mock news items,
exchange API response mocks, reference metric functions, and baseline forecasting models.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Check for production crypto_forecast availability
try:
    import crypto_forecast
    HAS_CRYPTO_FORECAST = True
except ImportError:
    HAS_CRYPTO_FORECAST = False


# ==============================================================================
# Domain Constants
# ==============================================================================

DEFAULT_HORIZONS: List[str] = [
    "1h", "2h", "4h", "6h", "12h", "24h", "2d", "3d", "7d", "14d", "30d"
]

HORIZON_HOURS: Dict[str, int] = {
    "1h": 1,
    "2h": 2,
    "4h": 4,
    "6h": 6,
    "12h": 12,
    "24h": 24,
    "2d": 48,
    "3d": 72,
    "7d": 168,
    "14d": 336,
    "30d": 720,
}

SUPPORTED_ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "XRP"]

ASSET_BASE_PRICES: Dict[str, float] = {
    "BTC": 65000.0,
    "ETH": 3500.0,
    "SOL": 145.0,
    "BNB": 580.0,
    "ADA": 0.45,
    "DOGE": 0.12,
    "XRP": 0.55,
}

ASSET_VOLATILITIES: Dict[str, float] = {
    "BTC": 0.02,
    "ETH": 0.025,
    "SOL": 0.04,
    "BNB": 0.022,
    "ADA": 0.035,
    "DOGE": 0.05,
    "XRP": 0.03,
}


# ==============================================================================
# Interface Contracts & Reference Dataclasses
# ==============================================================================

@dataclass
class HorizonPrediction:
    """Standardized prediction dataclass defined in PROJECT.md interface contracts."""
    horizon: str
    step_hours: int
    predicted_price: float
    current_price: float
    predicted_change: float
    predicted_return_pct: float
    direction: str  # 'BULLISH', 'BEARISH', or 'NEUTRAL'
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "step_hours": self.step_hours,
            "predicted_price": self.predicted_price,
            "current_price": self.current_price,
            "predicted_change": self.predicted_change,
            "predicted_return_pct": self.predicted_return_pct,
            "direction": self.direction,
            "confidence_p10": self.confidence_p10,
            "confidence_p50": self.confidence_p50,
            "confidence_p90": self.confidence_p90,
        }


# ==============================================================================
# Reference Metric Implementations (Mathematical Oracles)
# ==============================================================================

def calc_mae(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Vectorized Mean Absolute Error (MAE): (1/N) * sum(|y - y_hat|)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate MAE on empty arrays")
    return float(np.mean(np.abs(yt - yp)))


def calc_rmse(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Vectorized Root Mean Squared Error (RMSE): sqrt((1/N) * sum((y - y_hat)^2))."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate RMSE on empty arrays")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def calc_smape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Vectorized Symmetric Mean Absolute Percentage Error (sMAPE) in [0%, 200%]."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate sMAPE on empty arrays")
    denom = np.abs(yt) + np.abs(yp)
    # Handle zero/zero case safely
    mask = denom > 1e-12
    smape_vals = np.zeros_like(yt)
    smape_vals[mask] = 200.0 * np.abs(yp[mask] - yt[mask]) / denom[mask]
    return float(np.mean(smape_vals))


def calc_hit_rate(y_true_returns: Union[List[float], np.ndarray], y_pred_returns: Union[List[float], np.ndarray]) -> float:
    """Directional Hit-Rate percentage in [0%, 100%]: percentage of concordant signs."""
    yt = np.asarray(y_true_returns, dtype=float)
    yp = np.asarray(y_pred_returns, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise ValueError("Cannot calculate Hit-Rate on empty arrays")
    # Concordant when signs match (positive return with positive return, or negative with negative)
    concordant = (np.sign(yt) == np.sign(yp)) & (np.sign(yt) != 0)
    # Both zero also concordant
    concordant = concordant | ((np.sign(yt) == 0) & (np.sign(yp) == 0))
    return float(np.mean(concordant) * 100.0)


def calc_skill_score(mae_model: float, mae_baseline: float) -> float:
    """Skill score percentage: (1 - mae_model / mae_baseline) * 100%."""
    if mae_baseline <= 1e-12:
        return 0.0 if mae_model <= 1e-12 else -100.0
    return float((1.0 - mae_model / mae_baseline) * 100.0)


# ==============================================================================
# Reference Forecaster Oracle (Persistence Baseline Forecaster)
# ==============================================================================

class ReferencePersistenceForecaster:
    """Reference implementation of Naive Persistence Baseline Forecaster (y_hat_{t+h} = y_t)."""

    def __init__(self, name: str = "persistence"):
        self.name = name

    def forecast(
        self,
        context_prices: Union[List[float], np.ndarray],
        horizons: Optional[List[str]] = None,
    ) -> Dict[str, HorizonPrediction]:
        prices = np.asarray(context_prices, dtype=float)
        if len(prices) == 0:
            raise ValueError("context_prices cannot be empty")
        if np.any(np.isnan(prices)):
            raise ValueError("context_prices cannot contain NaNs")
        if np.any(prices <= 0):
            raise ValueError("context_prices must be strictly positive")

        if horizons is None:
            horizons = DEFAULT_HORIZONS

        current_price = float(prices[-1])
        # Compute empirical return volatility for uncertainty cones
        if len(prices) > 1:
            log_returns = np.diff(np.log(prices))
            vol = float(np.std(log_returns))
            vol = max(vol, 0.001)
        else:
            vol = 0.01

        predictions: Dict[str, HorizonPrediction] = {}
        for h in horizons:
            step_hours = HORIZON_HOURS.get(h, 1)
            pred_price = current_price  # persistence forecast
            pred_change = 0.0
            pred_ret_pct = 0.0
            direction = "NEUTRAL"

            # Cone expands with sqrt(step_hours / 24)
            time_scale = math.sqrt(step_hours / 24.0)
            half_spread = current_price * 1.28 * vol * time_scale
            p10 = max(0.0001, current_price - half_spread)
            p50 = current_price
            p90 = current_price + half_spread

            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=step_hours,
                predicted_price=pred_price,
                current_price=current_price,
                predicted_change=pred_change,
                predicted_return_pct=pred_ret_pct,
                direction=direction,
                confidence_p10=p10,
                confidence_p50=p50,
                confidence_p90=p90,
            )
        return predictions

    def get_full_trajectory(
        self,
        context_prices: Union[List[float], np.ndarray],
        max_horizon_steps: int = 720,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prices = np.asarray(context_prices, dtype=float)
        current_price = float(prices[-1])
        if len(prices) > 1:
            vol = float(np.std(np.diff(np.log(prices))))
            vol = max(vol, 0.001)
        else:
            vol = 0.01

        steps = np.arange(1, max_horizon_steps + 1)
        point_predictions = np.full(max_horizon_steps, current_price)

        # Quantiles envelope: (p10, p50, p90)
        time_scales = np.sqrt(steps / 24.0)
        spreads = current_price * 1.28 * vol * time_scales
        p10_traj = np.maximum(0.0001, current_price - spreads)
        p50_traj = point_predictions
        p90_traj = current_price + spreads

        quantile_predictions = np.stack([p10_traj, p50_traj, p90_traj], axis=1)
        return point_predictions, quantile_predictions


# ==============================================================================
# Synthetic Data Generators & Fixtures
# ==============================================================================

def make_synthetic_ohlcv(
    symbol: str = "BTC",
    n_bars: int = 1000,
    freq: str = "1h",
    start_price: Optional[float] = None,
    volatility: Optional[float] = None,
    seed: int = 42,
    start_time: str = "2026-01-01 00:00:00",
) -> pd.DataFrame:
    """Generates realistic synthetic OHLCV data adhering strictly to domain properties."""
    rng = np.random.default_rng(seed)
    base_p = start_price or ASSET_BASE_PRICES.get(symbol, 100.0)
    sigma = volatility or ASSET_VOLATILITIES.get(symbol, 0.02)

    # Hourly or daily delta t
    dt_factor = 1.0 / 24.0 if freq.endswith("h") else 1.0
    mu = 0.0002 * dt_factor
    sigma_adj = sigma * math.sqrt(dt_factor)

    # Geometric Brownian Motion close prices
    shocks = rng.normal(loc=(mu - 0.5 * sigma_adj**2), scale=sigma_adj, size=n_bars)
    log_prices = np.cumsum(shocks)
    close_prices = base_p * np.exp(log_prices)

    # Generate open, high, low, volume
    open_prices = np.empty(n_bars)
    open_prices[0] = base_p
    open_prices[1:] = close_prices[:-1]

    high_prices = np.maximum(open_prices, close_prices) * (1.0 + np.abs(rng.normal(0, sigma_adj * 0.4, n_bars)))
    low_prices = np.minimum(open_prices, close_prices) * (1.0 - np.abs(rng.normal(0, sigma_adj * 0.4, n_bars)))
    low_prices = np.maximum(low_prices, 1e-6)  # Strictly positive

    # Log-normal volume
    base_vol = 1000.0 * (100.0 / base_p)
    volumes = base_vol * np.exp(rng.normal(0, 0.5, n_bars))

    start_ts = pd.Timestamp(start_time, tz="UTC")
    timestamps = pd.date_range(start=start_ts, periods=n_bars, freq=freq)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes,
        }
    )
    return df


@pytest.fixture(scope="session")
def all_horizons() -> List[str]:
    return list(DEFAULT_HORIZONS)


@pytest.fixture(scope="session")
def horizon_hours_map() -> Dict[str, int]:
    return dict(HORIZON_HOURS)


@pytest.fixture(scope="session")
def supported_assets() -> List[str]:
    return list(SUPPORTED_ASSETS)


@pytest.fixture(scope="session")
def synthetic_ohlcv_1000h() -> pd.DataFrame:
    """1000 hourly candles for BTC starting 2026-01-01."""
    return make_synthetic_ohlcv(symbol="BTC", n_bars=1000, freq="1h", seed=42)


@pytest.fixture(scope="session")
def synthetic_ohlcv_daily_365d() -> pd.DataFrame:
    """365 daily candles for BTC starting 2025-01-01."""
    return make_synthetic_ohlcv(symbol="BTC", n_bars=365, freq="1D", seed=101, start_time="2025-01-01 00:00:00")


@pytest.fixture(scope="session")
def multi_asset_ohlcv_factory() -> Callable[[str, int, str, int], pd.DataFrame]:
    """Factory fixture to generate realistic OHLCV DataFrames for any supported asset."""
    def _factory(symbol: str = "BTC", n_bars: int = 1000, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
        return make_synthetic_ohlcv(symbol=symbol, n_bars=n_bars, freq=freq, seed=seed)
    return _factory


@pytest.fixture(scope="session")
def mock_crypto_news() -> List[Dict[str, Any]]:
    """Diverse crypto news dataset spanning positive, negative, and neutral market events."""
    base_time = dt.datetime(2026, 1, 15, 12, 0, tzinfo=dt.timezone.utc)
    return [
        {
            "title": "Bitcoin ETF inflows surge to new record high as institutions buy",
            "published": base_time - dt.timedelta(hours=1),
            "source": "CoinDesk",
            "sentiment_label": "BULLISH",
            "polarity": 0.85,
        },
        {
            "title": "Solana breaks through key resistance after major network throughput upgrade",
            "published": base_time - dt.timedelta(hours=3),
            "source": "CoinTelegraph",
            "sentiment_label": "BULLISH",
            "polarity": 0.72,
        },
        {
            "title": "SEC issues new regulatory guidance on decentralized exchange staking",
            "published": base_time - dt.timedelta(hours=5),
            "source": "CoinDesk",
            "sentiment_label": "NEUTRAL",
            "polarity": -0.05,
        },
        {
            "title": "Exchange hack reports trigger flash selloff across meme coins including DOGE",
            "published": base_time - dt.timedelta(hours=8),
            "source": "CoinTelegraph",
            "sentiment_label": "BEARISH",
            "polarity": -0.78,
        },
        {
            "title": "Ethereum staking yield holds steady as validator queue clears",
            "published": base_time - dt.timedelta(hours=12),
            "source": "CoinDesk",
            "sentiment_label": "NEUTRAL",
            "polarity": 0.10,
        },
        {
            "title": "Ripple XRP wins cross-border payments partnership with Asian banking consortium",
            "published": base_time - dt.timedelta(hours=15),
            "source": "CoinTelegraph",
            "sentiment_label": "BULLISH",
            "polarity": 0.65,
        },
        {
            "title": "Binance completes quarterly token burn removing 2 million BNB from circulation",
            "published": base_time - dt.timedelta(hours=20),
            "source": "CoinDesk",
            "sentiment_label": "BULLISH",
            "polarity": 0.50,
        },
        {
            "title": "Macro volatility increases as Federal Reserve maintains restrictive rates",
            "published": base_time - dt.timedelta(hours=24),
            "source": "CoinTelegraph",
            "sentiment_label": "BEARISH",
            "polarity": -0.45,
        },
    ]


@pytest.fixture(scope="session")
def mock_binance_klines_raw(synthetic_ohlcv_1000h: pd.DataFrame) -> List[List[Any]]:
    """Simulates the raw JSON list-of-lists response from Binance REST /api/v3/klines."""
    klines = []
    for _, row in synthetic_ohlcv_1000h.iterrows():
        open_time = int(row["timestamp"].timestamp() * 1000)
        close_time = open_time + 3600000 - 1
        kline = [
            open_time,
            f"{row['open']:.2f}",
            f"{row['high']:.2f}",
            f"{row['low']:.2f}",
            f"{row['close']:.2f}",
            f"{row['volume']:.4f}",
            close_time,
            f"{row['close'] * row['volume']:.2f}",  # Quote asset volume
            1500,  # Number of trades
            f"{row['volume'] * 0.5:.4f}",  # Taker buy base asset volume
            f"{row['close'] * row['volume'] * 0.5:.2f}",  # Taker buy quote asset volume
            "0",  # Ignore
        ]
        klines.append(kline)
    return klines


@pytest.fixture(scope="session")
def mock_binance_ticker_raw() -> Dict[str, Any]:
    """Simulates Binance REST /api/v3/ticker/24hr response."""
    return {
        "symbol": "BTCUSDT",
        "priceChange": "1250.50",
        "priceChangePercent": "1.95",
        "weightedAvgPrice": "65120.40",
        "prevClosePrice": "64100.00",
        "lastPrice": "65350.50",
        "bidPrice": "65350.00",
        "askPrice": "65351.00",
        "openPrice": "64100.00",
        "highPrice": "65800.00",
        "lowPrice": "63900.00",
        "volume": "14520.35",
        "quoteVolume": "945560120.00",
        "openTime": 1700000000000,
        "closeTime": 1700086400000,
        "count": 312000,
    }


@pytest.fixture
def mock_exchange_session(mock_binance_klines_raw: List[List[Any]], mock_binance_ticker_raw: Dict[str, Any]):
    """Intercepts requests.get to provide realistic responses for Binance and Yahoo Finance."""
    with patch("requests.get") as mock_get:
        def _side_effect(url: str, *args, **kwargs):
            response = MagicMock()
            response.status_code = 200

            if "api.binance.com/api/v3/klines" in url:
                response.json.return_value = mock_binance_klines_raw
                return response
            elif "api.binance.com/api/v3/ticker/24hr" in url:
                response.json.return_value = mock_binance_ticker_raw
                return response
            elif "finance.yahoo.com" in url or "query1.finance.yahoo.com" in url:
                # Yahoo Finance mock chart format
                timestamps = [k[0] // 1000 for k in mock_binance_klines_raw[-100:]]
                closes = [float(k[4]) for k in mock_binance_klines_raw[-100:]]
                response.json.return_value = {
                    "chart": {
                        "result": [
                            {
                                "timestamp": timestamps,
                                "indicators": {
                                    "quote": [
                                        {
                                            "open": closes,
                                            "high": [c * 1.005 for c in closes],
                                            "low": [c * 0.995 for c in closes],
                                            "close": closes,
                                            "volume": [1000.0] * len(closes),
                                        }
                                    ]
                                },
                            }
                        ],
                        "error": None,
                    }
                }
                return response

            # Default 200 empty dict
            response.json.return_value = {}
            return response

        mock_get.side_effect = _side_effect
        yield mock_get


@pytest.fixture(scope="session")
def reference_persistence_forecaster() -> ReferencePersistenceForecaster:
    """Provides reference implementation of Naive Persistence Baseline Forecaster."""
    return ReferencePersistenceForecaster()
