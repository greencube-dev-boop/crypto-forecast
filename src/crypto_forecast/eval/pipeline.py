"""Multi-Pair Multi-Model Accuracy Evaluation Pipeline Orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

from crypto_forecast.data.ingestion import fetch_ohlcv
from crypto_forecast.data.preprocessor import normalize_datetime_index, reindex_and_impute
from crypto_forecast.eval.dataset import (
    get_accuracy_eval_dir,
    identify_pending_timestamps,
    inspect_latest_recorded_timestamp,
    persist_accuracy_dataset,
)
from crypto_forecast.eval.retrospective import evaluate_historical_range_cached
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    get_accuracy_csv_filename,
    get_model_prefix,
    sanitize_pair_symbol,
)
from crypto_forecast.models.base import BaseForecaster
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import TimesFMForecaster

logger = logging.getLogger(__name__)

DEFAULT_EVAL_PAIRS: Final[list[str]] = ["BTC/USDT", "ETH/USDT"]


def _instantiate_model(name_or_instance: str | BaseForecaster, force_fallback: bool = False) -> tuple[str, BaseForecaster]:
    """Helper to resolve a model string or instance into (prefix, BaseForecaster)."""
    if isinstance(name_or_instance, BaseForecaster):
        prefix = get_model_prefix(name_or_instance)
        return prefix, name_or_instance

    name_lower = str(name_or_instance).lower().strip()
    if "timesfm" in name_lower:
        return "timesfm", TimesFMForecaster(force_fallback=force_fallback)
    elif "chronos" in name_lower:
        return "chronos", ChronosForecaster(use_fallback=force_fallback)
    elif "granite" in name_lower:
        return "granite", GraniteTTMForecaster(use_fallback=force_fallback)
    elif "persistence" in name_lower or "baseline" in name_lower:
        return "persistence", PersistenceForecaster()
    else:
        raise ValueError(f"Unknown model identifier: '{name_or_instance}'")


def resolve_eval_models(
    models: dict[str, BaseForecaster] | list[str | BaseForecaster] | None = None,
    force_fallback: bool = False,
) -> dict[str, BaseForecaster]:
    """Resolves model configurations into a dictionary of prefix -> BaseForecaster."""
    if models is None:
        return {
            "timesfm": TimesFMForecaster(force_fallback=force_fallback),
            "chronos": ChronosForecaster(use_fallback=force_fallback),
            "granite": GraniteTTMForecaster(use_fallback=force_fallback),
            "persistence": PersistenceForecaster(),
        }

    resolved: dict[str, BaseForecaster] = {}
    if isinstance(models, dict):
        for k, v in models.items():
            prefix, inst = _instantiate_model(v, force_fallback=force_fallback)
            resolved[k if k in ("timesfm", "chronos", "granite", "persistence") else prefix] = inst
    elif isinstance(models, (list, tuple)):
        for item in models:
            prefix, inst = _instantiate_model(item, force_fallback=force_fallback)
            resolved[prefix] = inst
    else:
        raise TypeError(f"Invalid models specification type: {type(models)}")

    return resolved


def run_accuracy_evaluation(
    pairs: list[str] | None = None,
    models: dict[str, BaseForecaster] | list[str | BaseForecaster] | None = None,
    output_dir: str | Path | None = None,
    initial_backfill_hours: int = 48,
    max_horizon: int = 48,
    context_window: int = 512,
    force_fallback: bool = False,
    df_ohlcv_map: dict[str, pd.DataFrame] | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Runs the 48-hour retrospective accuracy evaluation pipeline across pairs and models.

    Args:
        pairs: List of cryptocurrency pair symbols (default: ['BTC/USDT', 'ETH/USDT']).
        models: Dict or list of forecasters (default: TimesFM, Chronos, Granite, Persistence).
        output_dir: Destination directory (defaults to Google Drive in Colab, or ./accuracy_eval).
        initial_backfill_hours: Hours to evaluate on fresh uninitialized CSVs (default: 48).
        max_horizon: Lookback horizons count (default: 48).
        context_window: History context length (default: 512).
        force_fallback: If True, forces models to use instant local offline mathematical fallbacks.
        df_ohlcv_map: Optional pre-loaded market data map {pair: DataFrame} for fast offline/test use.
        verbose: If True, logs progress information.

    Returns:
        dict[str, Path]: Map of '{model}_{pair}' key to saved CSV Path.
    """
    eval_pairs = pairs if pairs is not None else list(DEFAULT_EVAL_PAIRS)
    resolved_models = resolve_eval_models(models=models, force_fallback=force_fallback)
    target_dir = get_accuracy_eval_dir(output_dir=output_dir)

    results: dict[str, Path] = {}

    if verbose:
        print(f"[Accuracy Eval] Target directory: {target_dir}")
        print(f"[Accuracy Eval] Configured pairs ({len(eval_pairs)}): {eval_pairs}")
        print(f"[Accuracy Eval] Models ({len(resolved_models)}): {list(resolved_models.keys())}")

    for pair in eval_pairs:
        clean_pair_name = sanitize_pair_symbol(pair)
        if verbose:
            print(f"\n--- Ingesting Market Data for {pair} ({clean_pair_name}) ---")

        # Ingest or fetch market data
        if df_ohlcv_map is not None and pair in df_ohlcv_map:
            df_raw = df_ohlcv_map[pair]
        else:
            try:
                df_raw = fetch_ohlcv(symbol=pair, interval="1h", limit=1000)
            except Exception as exc:
                logger.error("Failed to fetch market data for %s: %s", pair, exc)
                continue

        df_clean = normalize_datetime_index(df_raw)
        df_hourly = reindex_and_impute(df_clean, freq="1h")

        for model_name, model_instance in resolved_models.items():
            model_prefix = get_model_prefix(model_instance or model_name)
            key = f"{model_prefix}_{clean_pair_name}"
            filename = get_accuracy_csv_filename(model_prefix, clean_pair_name)
            csv_path = target_dir / filename

            latest_ts = inspect_latest_recorded_timestamp(csv_path)
            pending_ts = identify_pending_timestamps(
                df_hourly=df_hourly,
                latest_ts=latest_ts,
                min_context_window=context_window,
                max_lookback_horizon=max_horizon,
                initial_backfill_hours=initial_backfill_hours,
            )

            if not pending_ts:
                if verbose:
                    print(f"[{model_prefix} | {pair}] Up to date (latest: {latest_ts}). 0 pending hours.")
                # Ensure existing or empty file exists
                if not csv_path.exists():
                    persist_accuracy_dataset(pd.DataFrame(columns=ACCURACY_EVAL_COLUMNS), csv_path)
                results[key] = csv_path
                continue

            if verbose:
                print(
                    f"[{model_prefix} | {pair}] Evaluating {len(pending_ts)} pending hours "
                    f"({pending_ts[0].strftime('%Y-%m-%d %H:%M')} to {pending_ts[-1].strftime('%Y-%m-%d %H:%M')})..."
                )

            # Evaluate pending hours with trajectory caching
            df_eval = evaluate_historical_range_cached(
                model=model_instance,
                df_hourly=df_hourly,
                target_timestamps=pending_ts,
                max_horizon=max_horizon,
                context_window=context_window,
            )

            # Persist and merge
            saved_df = persist_accuracy_dataset(df_new=df_eval, csv_path=csv_path)
            results[key] = csv_path

            if verbose:
                print(f"[{model_prefix} | {pair}] Saved {len(saved_df)} total rows to {csv_path.name}")

    if verbose:
        print(f"\n[Accuracy Eval] Completed successfully. Total files updated/verified: {len(results)}")

    return results
