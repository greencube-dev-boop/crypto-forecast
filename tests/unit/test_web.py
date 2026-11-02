"""Unit Tests for Web GUI Package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import pytest

from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.web.app import build_reference_gradio_app, resolve_forecaster, run_forecast_pipeline
from crypto_forecast.web.charts import create_forecast_cone_chart
from crypto_forecast.web.components import (
    build_forecast_table_rows,
    build_scoreboard_markdown,
    format_price,
)


class TestWebComponents:
    def test_format_price(self):
        assert format_price(65000.5) == "$65,000.50"
        assert format_price(0.000456) == "$0.000456"

    def test_build_forecast_table_rows(self):
        model = PersistenceForecaster()
        preds = model.forecast([100.0, 105.0])
        rows = build_forecast_table_rows(preds)
        assert len(rows) == 11
        assert rows[0][0] == "1h"
        assert rows[0][1] == 1

    def test_build_scoreboard_markdown(self):
        model = PersistenceForecaster()
        preds = model.forecast([100.0, 105.0])
        md = build_scoreboard_markdown("BTC", "TimesFM 2.5", preds)
        assert "Accuracy Scoreboard" in md
        assert "BTC" in md

    def test_create_forecast_cone_chart(self):
        model = PersistenceForecaster()
        preds = model.forecast([100.0, 105.0])
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"), "close": [100.0] * 10})
        fig = create_forecast_cone_chart(df, preds, symbol="BTC")
        assert fig is not None
        assert len(fig.data) >= 3

    def test_run_forecast_pipeline(self):
        fig, table_rows, scoreboard = run_forecast_pipeline("BTC", "Persistence (Baseline)", apply_sentiment=False)
        assert fig is not None
        assert len(table_rows) == 11
        assert "Accuracy Scoreboard" in scoreboard
