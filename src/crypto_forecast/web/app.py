"""Gradio Blocks Web GUI for Multi-Horizon Cryptocurrency Forecasting.

Provides interactive dashboard for selecting cryptocurrencies (BTC, ETH, SOL, BNB, ADA, DOGE, XRP),
forecasting engine (TimesFM 2.5, Chronos-Bolt, Granite TTM, Persistence), viewing Plotly uncertainty
cones, 11-horizon breakdown table, and benchmark scoreboard.
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd

from crypto_forecast.benchmark.compatibility import _generate_synthetic_prices
from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    DEFAULT_SYMBOLS,
)
from crypto_forecast.data.ingestion import fetch_ohlcv
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.sentiment import analyze_sentiment, fetch_crypto_news
from crypto_forecast.models.timesfm_model import TimesFMForecaster
from crypto_forecast.web.charts import create_forecast_cone_chart
from crypto_forecast.web.components import (
    build_forecast_table_rows,
    build_scoreboard_markdown,
)

logger = logging.getLogger(__name__)

SUPPORTED_COINS: list[str] = ["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "XRP"]
MODEL_CHOICES: list[str] = [
    "TimesFM 2.5 (Primary)",
    "Chronos-Bolt",
    "Granite TTM",
    "Persistence (Baseline)",
]


def resolve_forecaster(model_choice: str):
    """Instantiates the selected forecasting model."""
    mc = model_choice.lower()
    if "timesfm" in mc:
        return TimesFMForecaster()
    if "chronos" in mc:
        return ChronosForecaster()
    if "granite" in mc or "ttm" in mc:
        return GraniteTTMForecaster()
    return PersistenceForecaster()


def run_forecast_pipeline(
    coin: str,
    model_choice: str,
    apply_sentiment: bool = True,
) -> tuple[Any, list[list[Any]], str]:
    """Generates chart, tabular breakdown, and scoreboard for UI."""
    clean_sym = coin.strip().upper()
    full_sym = f"{clean_sym}/USDT"

    # 1. Fetch market data with synthetic fallback
    try:
        df = fetch_ohlcv(full_sym, interval="1h", limit=200)
    except Exception as e:
        logger.info(f"Using synthetic prices for {coin}: {e}")
        prices = _generate_synthetic_prices(clean_sym, n_bars=200)
        now = pd.Timestamp.utcnow()
        timestamps = pd.date_range(end=now, periods=len(prices), freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": prices,
                "high": prices * 1.005,
                "low": prices * 0.995,
                "close": prices,
                "volume": 1000.0,
            }
        )

    prices = df["close"].values

    # 2. Sentiment drift
    polarity = None
    if apply_sentiment:
        try:
            articles = fetch_crypto_news()
            if articles:
                titles = [a["title"] for a in articles[:10] if "title" in a]
                if titles:
                    sentiments = analyze_sentiment(titles)
                    pols = [float(s["polarity"]) for s in sentiments if "polarity" in s]
                    polarity = float(np.mean(pols)) if pols else 0.0
        except Exception:
            polarity = 0.0

    # 3. Model prediction
    forecaster = resolve_forecaster(model_choice)
    predictions = forecaster.forecast(prices, horizons=DEFAULT_HORIZONS, sentiment_polarity=polarity)

    # 4. Format outputs
    fig = create_forecast_cone_chart(df, predictions, symbol=clean_sym, model_name=model_choice)
    table_rows = build_forecast_table_rows(predictions)
    scoreboard_md = build_scoreboard_markdown(clean_sym, model_choice, predictions)

    return fig, table_rows, scoreboard_md


def build_reference_gradio_app() -> gr.Blocks:
    """Builds the primary Gradio Blocks application."""
    with gr.Blocks(title="Multi-Horizon Cryptocurrency Forecasting System") as demo:
        gr.Markdown("# 🚀 Multi-Horizon Cryptocurrency Price Forecasting")
        gr.Markdown(
            "Predict cryptocurrency prices across 11 discrete horizons "
            "(1h, 2h, 4h, 6h, 12h, 24h, 2d, 3d, 7d, 14d, 30d) using SOTA Foundation Transformers."
        )

        with gr.Row():
            coin_selector = gr.Dropdown(
                choices=SUPPORTED_COINS,
                value="BTC",
                label="Select Cryptocurrency",
            )
            model_selector = gr.Dropdown(
                choices=MODEL_CHOICES,
                value="TimesFM 2.5 (Primary)",
                label="Forecasting Engine",
            )
            sentiment_toggle = gr.Checkbox(
                value=True,
                label="Enable CryptoBERT Sentiment Drift",
            )
            forecast_btn = gr.Button("Generate Forecast", variant="primary")

        with gr.Row():
            plot_component = gr.Plot(label="Price Action & Uncertainty Cones")

        with gr.Row():
            forecast_table = gr.Dataframe(
                headers=[
                    "Horizon",
                    "Step (hrs)",
                    "Predicted Price",
                    "Return %",
                    "Direction",
                    "P10 Bound",
                    "P90 Bound",
                ],
                label="Multi-Horizon Forecast Breakdown Table",
            )

        with gr.Row():
            scoreboard = gr.Markdown("### Accuracy Scoreboard vs Persistence Baseline")

        forecast_btn.click(
            fn=run_forecast_pipeline,
            inputs=[coin_selector, model_selector, sentiment_toggle],
            outputs=[plot_component, forecast_table, scoreboard],
        )

    # Concurrency limit = 1 to protect GPU VRAM on Google Colab T4
    demo.queue(default_concurrency_limit=1)
    return demo


create_app = build_reference_gradio_app
