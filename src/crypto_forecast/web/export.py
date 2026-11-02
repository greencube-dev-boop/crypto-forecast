"""CSV Export and Google Drive Integration Module.

Provides robust export of 48-hour continuous forecast tables to local storage
and Google Drive (/content/drive/MyDrive/Trader/<timestamp>.csv).
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_forecast.config import HOURLY_48_HORIZONS
from crypto_forecast.models.base import HorizonPrediction
from crypto_forecast.web.components import format_price

logger = logging.getLogger(__name__)


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Attempts automatic Google Drive mounting on Google Colab.

    If google.colab is unavailable, or running headlessly/without user authentication,
    logs a warning and safely returns False without raising exceptions.
    """
    mydrive_path = os.path.join(mount_point, "MyDrive")
    if os.path.exists(mydrive_path):
        return True

    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point, force_remount=False)
        return os.path.exists(mydrive_path)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "Google Colab drive module unavailable (%s). Continuing without Google Drive sync.",
            exc,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Google Drive could not be mounted automatically (%s). Continuing without Google Drive sync.",
            exc,
        )
        return False


def create_forecast_dataframe(
    predictions: dict[str, HorizonPrediction],
    current_time: pd.Timestamp | None = None,
    current_price: float | None = None,
) -> pd.DataFrame:
    """Creates a standardized 49-row DataFrame from predictions (Row 0: 0h, Rows 1..48: 1h..48h)."""
    now = current_time or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    if current_price is None:
        if predictions:
            first_pred = next(iter(predictions.values()))
            current_price = first_pred.current_price
        else:
            current_price = 0.0

    rows: list[dict[str, Any]] = [
        {
            "Step": "0h",
            "Timestamp (UTC)": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Price / Prediction": format_price(current_price),
            "Return %": "0.00%",
            "Direction": "CURRENT",
            "Prediction Interval [P10, P90]": f"[{format_price(current_price)}, {format_price(current_price)}]",
            "predicted_price_raw": current_price,
            "confidence_p10_raw": current_price,
            "confidence_p90_raw": current_price,
        }
    ]

    for h in HOURLY_48_HORIZONS:
        if h not in predictions:
            continue
        p = predictions[h]
        future_ts = now + pd.Timedelta(hours=int(p.step_hours))
        rows.append(
            {
                "Step": p.horizon,
                "Timestamp (UTC)": future_ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "Price / Prediction": format_price(p.predicted_price),
                "Return %": f"{p.predicted_return_pct:+.2f}%",
                "Direction": p.direction,
                "Prediction Interval [P10, P90]": f"[{format_price(p.confidence_p10)}, {format_price(p.confidence_p90)}]",
                "predicted_price_raw": p.predicted_price,
                "confidence_p10_raw": p.confidence_p10,
                "confidence_p90_raw": p.confidence_p90,
            }
        )

    return pd.DataFrame(rows)


def export_forecast_to_csv(
    df_table: pd.DataFrame | list[list[Any]] | dict[str, HorizonPrediction],
    symbol: str = "BTC",
    timestamp: datetime | None = None,
    output_dir: str | Path | None = None,
    current_time: pd.Timestamp | None = None,
    current_price: float | None = None,
) -> tuple[str, str | None]:
    """Exports 48-hour forecast table (all 49 rows) to local CSV and Google Drive.

    Saves to /content/drive/MyDrive/Trader/<timestamp>.csv if Drive is available.
    Always saves a local copy in output_dir (defaults to 'exports/') for UI download.

    Args:
        df_table: Forecast table as DataFrame, list of row lists, or dict of HorizonPrediction.
        symbol: Cryptocurrency symbol.
        timestamp: Optional timestamp for file naming.
        output_dir: Directory for local CSV storage.
        current_time: Cutoff timestamp.
        current_price: Spot price.

    Returns:
        tuple[str, str | None]: (local_csv_path, drive_csv_path_or_none)
    """
    ts = timestamp or datetime.now(timezone.utc)
    ts_str = ts.strftime("%Y%m%d_%H%M%S")

    # Normalize df_table to DataFrame
    if isinstance(df_table, pd.DataFrame):
        df = df_table.copy()
    elif isinstance(df_table, dict):
        df = create_forecast_dataframe(df_table, current_time=current_time, current_price=current_price)
    elif isinstance(df_table, list):
        columns = [
            "Step",
            "Timestamp (UTC)",
            "Price / Prediction",
            "Return %",
            "Direction",
            "Prediction Interval [P10, P90]",
        ]
        if df_table and len(df_table[0]) == len(columns):
            df = pd.DataFrame(df_table, columns=columns)
        else:
            df = pd.DataFrame(df_table)
    else:
        df = pd.DataFrame(df_table)

    # 1. Local CSV Export
    local_dir = Path(output_dir or "exports")
    local_dir.mkdir(parents=True, exist_ok=True)
    clean_sym = symbol.replace("/", "").replace("-", "").upper()
    local_filename = f"{clean_sym}_forecast_{ts_str}.csv"
    local_csv_path = str(local_dir / local_filename)

    df.to_csv(local_csv_path, index=False)
    logger.info("Saved local forecast CSV: %s", local_csv_path)

    # 2. Google Drive Export
    drive_csv_path: str | None = None
    drive_mounted = mount_google_drive("/content/drive")

    if drive_mounted:
        try:
            drive_dir = Path("/content/drive/MyDrive/Trader")
            drive_dir.mkdir(parents=True, exist_ok=True)
            drive_filename = f"{ts_str}.csv"
            target_drive_path = drive_dir / drive_filename
            df.to_csv(target_drive_path, index=False)
            drive_csv_path = str(target_drive_path)
            logger.info("Successfully exported forecast CSV to Google Drive: %s", drive_csv_path)
        except Exception as exc:
            logger.warning(
                "Failed writing CSV to Google Drive (%s). Local CSV remains accessible.",
                exc,
            )
            drive_csv_path = None
    else:
        logger.warning(
            "Google Drive not mounted or inaccessible. CSV saved locally at %s.",
            local_csv_path,
        )

    return local_csv_path, drive_csv_path
