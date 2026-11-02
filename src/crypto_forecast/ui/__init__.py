"""UI Package alias re-exporting web modules for backward compatibility."""

from __future__ import annotations

from crypto_forecast.web import *  # noqa: F401, F403
from crypto_forecast.web.app import build_reference_gradio_app, create_app, run_forecast_pipeline
from crypto_forecast.web.components import build_forecast_table_rows, build_scoreboard_markdown, format_price
from crypto_forecast.web.export import export_forecast_to_csv, mount_google_drive

__all__ = [
    "build_reference_gradio_app",
    "create_app",
    "run_forecast_pipeline",
    "build_forecast_table_rows",
    "build_scoreboard_markdown",
    "format_price",
    "export_forecast_to_csv",
    "mount_google_drive",
]
