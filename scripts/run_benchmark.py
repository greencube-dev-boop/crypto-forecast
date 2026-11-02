"""Headless Benchmark CLI Runner for Cryptocurrency Forecasting Models.

Executes quantitative benchmarking across candidate time-series models
(TimesFM 2.5, Chronos-Bolt, Granite TTM, Persistence) and cryptocurrencies
(BTC, ETH, SOL, BNB, ADA, DOGE, XRP) across all 11 discrete forecasting horizons.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from crypto_forecast.benchmark.compatibility import run_asset_compatibility_analysis
from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HORIZON_TO_HOURS,
)
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import TimesFMForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_benchmark")


def build_models(selected_models: list[str]) -> list[Any]:
    models = []
    for name in selected_models:
        name_clean = name.strip().lower()
        if "timesfm" in name_clean:
            models.append(TimesFMForecaster())
        elif "chronos" in name_clean:
            models.append(ChronosForecaster())
        elif "granite" in name_clean or "ttm" in name_clean:
            models.append(GraniteTTMForecaster())
        elif "persistence" in name_clean or "baseline" in name_clean:
            models.append(PersistenceForecaster())
        else:
            logger.warning(f"Unknown model name '{name}', falling back to Persistence")
            models.append(PersistenceForecaster(name=name))
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Horizon Crypto Forecasting Benchmark CLI")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE", "XRP"],
        help="List of crypto symbols to evaluate",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Forecasting horizons (e.g. 1h 24h 7d 30d)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["timesfm", "chronos", "granite", "persistence"],
        help="Candidate model names",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    logger.info("Initializing candidate models...")
    models = build_models(args.models)

    all_results = {}
    summary_records = []

    for model in models:
        logger.info(f"Evaluating model: {model.model_id} on {len(args.symbols)} symbols...")
        df_compat = run_asset_compatibility_analysis([model], symbols=args.symbols, horizons=args.horizons)
        all_results[model.model_id] = df_compat.to_dict(orient="records")

        # Compute summary
        mean_hit_rate = float(df_compat["hit_rate_pct"].mean())
        mean_mae = float(df_compat["mae_model"].mean())
        mean_rmse = float(df_compat["rmse_model"].mean())
        mean_skill = float(df_compat["skill_mae_pct"].mean())

        summary_records.append(
            {
                "model": model.model_id,
                "mean_hit_rate_pct": round(mean_hit_rate, 2),
                "mean_mae": round(mean_mae, 4),
                "mean_rmse": round(mean_rmse, 4),
                "mean_skill_mae_pct": round(mean_skill, 2),
            }
        )

    summary_df = pd.DataFrame(summary_records).sort_values(by="mean_hit_rate_pct", ascending=False)
    best_model = summary_df.iloc[0]["model"]

    output_payload = {
        "best_candidate_model": best_model,
        "summary": summary_records,
        "details": all_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("=" * 60)
    logger.info("BENCHMARK EVALUATION COMPLETE")
    logger.info(f"Best Candidate Model: {best_model}")
    logger.info(f"Results saved to: {out_path.resolve()}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
