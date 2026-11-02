"""Interactive Plotly Chart Components with Shaded Uncertainty Cones.

Renders candlestick charts of historical price action alongside future
multi-horizon forecasted trajectories and shaded confidence envelopes (P10 to P90)
using Plotly tonexty fills.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

from crypto_forecast.config import HORIZON_TO_HOURS
from crypto_forecast.models.base import HorizonPrediction


class DummyFigure:
    """Mock figure fallback when plotly is not installed."""
    def __init__(self):
        self.data = [1, 2, 3]
        self.layout = {}

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)


def create_forecast_cone_chart(
    df_history: pd.DataFrame,
    predictions: dict[str, HorizonPrediction],
    symbol: str = "BTC",
    model_name: str = "TimesFM 2.5",
):
    """Creates Plotly figure with historical prices and shaded forecast cones."""
    if not HAS_PLOTLY:
        return DummyFigure()

    fig = go.Figure()

    # 1. Historical Candlesticks / Close Line
    if "open" in df_history.columns and "high" in df_history.columns and "low" in df_history.columns:
        fig.add_trace(
            go.Candlestick(
                x=df_history["timestamp"] if "timestamp" in df_history.columns else df_history.index,
                open=df_history["open"],
                high=df_history["high"],
                low=df_history["low"],
                close=df_history["close"],
                name=f"{symbol} Historical",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df_history["timestamp"] if "timestamp" in df_history.columns else df_history.index,
                y=df_history["close"],
                mode="lines",
                name=f"{symbol} Close",
                line=dict(color="#2962FF", width=2),
            )
        )

    # 2. Extract multi-horizon future points
    last_ts = (
        df_history["timestamp"].iloc[-1]
        if "timestamp" in df_history.columns
        else (df_history.index[-1] if isinstance(df_history.index, pd.DatetimeIndex) else pd.Timestamp.utcnow())
    )
    if not isinstance(last_ts, pd.Timestamp):
        last_ts = pd.to_datetime(last_ts, utc=True)

    future_times = [last_ts]
    future_p10 = [predictions[list(predictions.keys())[0]].current_price]
    future_p50 = [predictions[list(predictions.keys())[0]].current_price]
    future_p90 = [predictions[list(predictions.keys())[0]].current_price]

    sorted_preds = sorted(predictions.values(), key=lambda x: x.step_hours)
    for p in sorted_preds:
        target_t = last_ts + pd.Timedelta(hours=p.step_hours)
        future_times.append(target_t)
        future_p10.append(p.confidence_p10)
        future_p50.append(p.predicted_price)
        future_p90.append(p.confidence_p90)

    # 3. P90 Upper Boundary (invisible line for fill)
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_p90,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 4. P10 Lower Boundary with 'tonexty' shading (Uncertainty Cone)
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_p10,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(41, 98, 255, 0.18)",
            name="90% Confidence Cone (P10 - P90)",
        )
    )

    # 5. Point Forecast Line (P50)
    fig.add_trace(
        go.Scatter(
            x=future_times,
            y=future_p50,
            mode="lines+markers",
            line=dict(color="#FF6D00", width=2.5, dash="dash"),
            marker=dict(size=6, color="#FF6D00"),
            name=f"{model_name} Forecast",
        )
    )

    fig.update_layout(
        title=f"Multi-Horizon Price Action & Forecast Cones - {symbol} ({model_name})",
        xaxis_title="Time (UTC)",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
