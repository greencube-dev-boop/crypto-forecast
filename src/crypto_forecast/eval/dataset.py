"""Incremental Google Drive Dataset Persistence & State Management.

Handles discovering existing evaluation progress, identifying newly completed hours,
deduplicating historical rows, and atomically persisting 50-column accuracy datasets.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Final
import uuid

import pandas as pd

from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    REAL_TIME_COL,
    canonicalize_columns,
)

logger = logging.getLogger(__name__)

COLAB_DRIVE_ACCURACY_DIR: Final[str] = "/content/drive/MyDrive/Trader/accuracy_eval"
LOCAL_FALLBACK_ACCURACY_DIR: Final[str] = "./accuracy_eval"


def mount_google_drive_safely(mount_point: str = "/content/drive") -> bool:
    """Attempts to mount Google Drive if running in Google Colab.

    Returns True if /content/drive/MyDrive exists or is successfully mounted;
    returns False if running locally or outside Google Colab.
    """
    try:
        mydrive = os.path.join(mount_point, "MyDrive")
        if os.path.exists(mydrive):
            return True
        # Try importing Colab drive module
        from google.colab import drive  # type: ignore[import-not-found]
        drive.mount(mount_point, force_remount=False)
        return os.path.exists(mydrive)
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception as exc:
        logger.debug("Google Drive mount attempt skipped: %s", exc)
        return False


def get_accuracy_eval_dir(
    output_dir: str | Path | None = None,
    mount_point: str = "/content/drive",
) -> Path:
    """Resolves output directory for accuracy evaluation datasets.

    Precedence:
    1. Explicit custom output_dir if provided
    2. Google Drive directory (/content/drive/MyDrive/Trader/accuracy_eval) if Colab Drive mounted
    3. Local fallback directory (./accuracy_eval)
    """
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target

    # Check Google Drive
    mydrive_dir = Path(mount_point) / "MyDrive"
    if mydrive_dir.exists() or mount_google_drive_safely(mount_point):
        target = Path(COLAB_DRIVE_ACCURACY_DIR)
        target.mkdir(parents=True, exist_ok=True)
        return target

    # Local fallback
    fallback = Path(LOCAL_FALLBACK_ACCURACY_DIR)
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def inspect_latest_recorded_timestamp(csv_path: Path | str) -> pd.Timestamp | None:
    """Inspects existing CSV file and returns the latest evaluated timestamp, or None if new/empty."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        df = pd.read_csv(path)
        if df.empty or REAL_TIME_COL not in df.columns:
            return None

        clean_series = df[REAL_TIME_COL].dropna()
        if clean_series.empty:
            return None

        timestamps = pd.to_datetime(
            clean_series,
            utc=True,
            format="mixed",
            errors="coerce",
        ).dropna()
        if timestamps.empty:
            return None

        max_ts = timestamps.max()
        return None if pd.isna(max_ts) else max_ts
    except Exception as exc:
        logger.warning("Could not read or parse existing CSV at %s: %s", path, exc)
        return None


def identify_pending_timestamps(
    df_hourly: pd.DataFrame,
    latest_ts: pd.Timestamp | None,
    min_context_window: int = 512,
    max_lookback_horizon: int = 48,
    initial_backfill_hours: int = 48,
    min_context_floor: int = 32,
) -> list[pd.Timestamp]:
    """Identifies newly completed hourly timestamps requiring evaluation.

    Args:
        df_hourly: Hourly OHLCV DataFrame with UTC DatetimeIndex.
        latest_ts: Latest recorded timestamp from existing CSV (or None if fresh run).
        min_context_window: Desired context history window.
        max_lookback_horizon: Maximum retrospective lookback in hours (48).
        initial_backfill_hours: Hours to evaluate on a fresh run.
        min_context_floor: Absolute minimum required historical bars (32).

    Returns:
        list[pd.Timestamp]: Sorted list of target timestamps to evaluate.
    """
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise ValueError("df_hourly must have a UTC DatetimeIndex")

    # Guard: Ensure index is timezone-aware UTC
    idx = df_hourly.index
    if idx.tz is None:
        df_hourly = df_hourly.copy()
        df_hourly.index = idx.tz_localize("UTC")
    elif str(idx.tz) != "UTC":
        df_hourly = df_hourly.copy()
        df_hourly.index = idx.tz_convert("UTC")

    # Filter out in-progress (unclosed) candles if live
    now_utc = pd.Timestamp.now(tz="UTC")
    closed_candles = df_hourly[df_hourly.index <= now_utc.floor("h")]

    # If df_hourly is historical (e.g. testing), use all bars
    if len(closed_candles) == 0:
        closed_candles = df_hourly

    eff_min_floor = min(min_context_floor, min_context_window)
    min_required_bars = eff_min_floor + max_lookback_horizon

    if len(closed_candles) < min_required_bars:
        logger.warning(
            "Insufficient data length (%d bars) for evaluation (need at least %d bars).",
            len(closed_candles),
            min_required_bars,
        )
        return []

    # Valid evaluation targets have at least min_required_bars history preceding them
    valid_targets = closed_candles.index[min_required_bars - 1:]

    if latest_ts is not None:
        latest_utc = pd.to_datetime(latest_ts, utc=True)
        pending = [ts for ts in valid_targets if ts > latest_utc]
    else:
        pending = list(valid_targets[-initial_backfill_hours:])

    return sorted(list(dict.fromkeys(pending)))


def persist_accuracy_dataset(
    df_new: pd.DataFrame,
    csv_path: Path | str,
) -> pd.DataFrame:
    """Atomically persists evaluation rows to CSV, appending and deduplicating.

    Guarantees:
    - Preserves existing rows without modification or overwrite.
    - Drops duplicates by 'real time' keeping existing (first) values across both initial creation and appends.
    - Sorts output chronologically by 'real time'.
    - Atomic file write via unique temporary file to prevent corruption on crash or race conditions.

    Args:
        df_new: New DataFrame of evaluated rows.
        csv_path: Destination CSV path.

    Returns:
        pd.DataFrame: Merged, deduplicated, and persisted DataFrame.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_new_canonical = canonicalize_columns(df_new)

    if path.exists() and path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(path)
            df_existing_canonical = canonicalize_columns(df_existing)
            if REAL_TIME_COL not in df_existing_canonical.columns:
                raise ValueError(
                    f"Existing CSV at {path} missing required column '{REAL_TIME_COL}'"
                )
            df_existing_clean = df_existing_canonical.dropna(subset=[REAL_TIME_COL])
            df_combined = pd.concat([df_existing_clean, df_new_canonical], ignore_index=True)
        except Exception as exc:
            logger.warning("Error reading existing CSV at %s, overwriting with new: %s", path, exc)
            df_combined = df_new_canonical
    else:
        df_combined = df_new_canonical

    if not df_combined.empty and REAL_TIME_COL in df_combined.columns:
        try:
            # Purge rows missing REAL_TIME_COL
            df_combined = df_combined.dropna(subset=[REAL_TIME_COL])

            # Coerced timestamp parsing to prevent DateParseError on corrupted strings
            df_combined["_sort_ts"] = pd.to_datetime(
                df_combined[REAL_TIME_COL],
                utc=True,
                format="mixed",
                errors="coerce",
            )
            # Drop any rows where _sort_ts is NaT
            df_combined = df_combined.dropna(subset=["_sort_ts"])

            # Universal deduplication (keep="first" preserves existing rows when appended,
            # and deduplicates internal duplicates on fresh writes or within df_new)
            df_combined = df_combined.drop_duplicates(subset=[REAL_TIME_COL], keep="first")

            # Sort chronologically
            df_combined = df_combined.sort_values(by="_sort_ts").drop(columns=["_sort_ts"])
            df_combined = df_combined.reset_index(drop=True)
        except Exception as exc:
            logger.warning(
                "Error parsing timestamps or sorting dataset at %s: %s",
                path,
                exc,
            )
            df_combined = df_new_canonical
            if not df_combined.empty and REAL_TIME_COL in df_combined.columns:
                df_combined = df_combined.drop_duplicates(subset=[REAL_TIME_COL], keep="first")

    # Reorder columns to canonical schema if all present
    if set(ACCURACY_EVAL_COLUMNS).issubset(df_combined.columns):
        df_combined = df_combined[ACCURACY_EVAL_COLUMNS]

    # Defense-in-depth: purge any rows that are all-NaN
    df_combined = df_combined.dropna(how="all").reset_index(drop=True)

    # Atomic write with unique temporary file and guaranteed cleanup
    unique_suffix = f".tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_path = path.with_suffix(unique_suffix)
    try:
        df_combined.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    logger.info("Persisted %d rows to %s", len(df_combined), path)
    return df_combined
