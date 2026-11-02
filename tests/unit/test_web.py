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
        assert len(rows) == 49
        assert rows[0][0] == "0h"
        assert rows[0][4] == "CURRENT"
        assert rows[1][0] == "1h"

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
        fig, table_rows, scoreboard, local_csv = run_forecast_pipeline("BTC", "Persistence (Baseline)", apply_sentiment=False)
        assert fig is not None
        assert len(table_rows) == 49
        assert "Accuracy Scoreboard" in scoreboard
        assert local_csv is not None and local_csv.endswith(".csv")

    def test_mount_google_drive_headless(self):
        from crypto_forecast.web.export import mount_google_drive
        # In non-colab test environment, must return False gracefully without raising
        result = mount_google_drive("/tmp/nonexistent_drive")
        assert result is False

    def test_export_forecast_to_csv_local_and_mock_drive(self, tmp_path):
        from unittest.mock import patch
        from crypto_forecast.web.export import export_forecast_to_csv
        model = PersistenceForecaster()
        preds = model.forecast([100.0, 105.0])
        rows = build_forecast_table_rows(preds)

        # Test local-only export (Drive unavailable)
        local_csv, drive_csv = export_forecast_to_csv(rows, symbol="BTC", output_dir=tmp_path)
        assert Path(local_csv).exists()
        assert drive_csv is None

        # Test with mocked Drive mount
        drive_dir = tmp_path / "MyDrive" / "Trader"
        drive_dir.mkdir(parents=True)
        with patch("crypto_forecast.web.export.mount_google_drive", return_value=True), \
             patch("crypto_forecast.web.export.Path", side_effect=lambda p: tmp_path / "MyDrive" / "Trader" / Path(p).name if "Trader" in str(p) else Path(p)):
            local_csv2, drive_csv2 = export_forecast_to_csv(rows, symbol="ETH", output_dir=tmp_path)
            assert Path(local_csv2).exists()
