"""Headless Forecasting CLI Runner.

Generates real-time multi-horizon predictions for a given cryptocurrency
across all 11 discrete horizons using the selected forecasting model.
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

from crypto_forecast.config import DEFAULT_HORIZONS
from crypto_forecast.data.ingestion import fetch_ohlcv
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.sentiment import analyze_sentiment, fetch_crypto_news
from crypto_forecast.models.timesfm_model import TimesFMForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_forecast")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Horizon Crypto Forecast CLI")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Crypto symbol (e.g. BTC/USDT)")
    parser.add_argument("--model", type=str, default="timesfm", choices=["timesfm", "chronos", "granite", "persistence"])
    parser.add_argument("--with-sentiment", action="store_true", help="Apply sentiment conviction drift")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    logger.info(f"Fetching market data for {args.symbol}...")
    try:
        df = fetch_ohlcv(args.symbol, interval="1h", limit=500)
        prices = df["close"].values
    except Exception as e:
        logger.warning(f"Could not fetch live market data ({e}); using synthetic fallback")
        from crypto_forecast.benchmark.compatibility import _generate_synthetic_prices
        prices = _generate_synthetic_prices(args.symbol, n_bars=300)

    # Select model
    if args.model == "timesfm":
        forecaster = TimesFMForecaster()
    elif args.model == "chronos":
        forecaster = ChronosForecaster()
    elif args.model == "granite":
        forecaster = GraniteTTMForecaster()
    else:
        forecaster = PersistenceForecaster()

    polarity = None
    if args.with_sentiment:
        try:
            logger.info("Computing sentiment drift score...")
            articles = fetch_crypto_news()
            if articles:
                titles = [a["title"] for a in articles[:10] if "title" in a]
                if titles:
                    sentiments = analyze_sentiment(titles)
                    pols = [float(s["polarity"]) for s in sentiments if "polarity" in s]
                    polarity = float(np.mean(pols)) if pols else 0.0
            logger.info(f"Applied sentiment polarity: {polarity}")
        except Exception as e:
            logger.warning(f"Sentiment engine bypassed: {e}")

    logger.info(f"Generating 11-horizon forecast with {forecaster.model_id}...")
    preds = forecaster.forecast(prices, horizons=DEFAULT_HORIZONS, sentiment_polarity=polarity)

    out_records = [p.to_dict() for p in preds.values()]
    print("\n" + "=" * 80)
    print(f"MULTI-HORIZON FORECAST: {args.symbol} | Model: {forecaster.model_id}")
    print("=" * 80)
    print(f"{'Horizon':<8} {'Step (h)':<10} {'Predicted':<14} {'Return %':<12} {'Direction':<10} {'[P10, P90]'}")
    print("-" * 80)
    for p in preds.values():
        print(
            f"{p.horizon:<8} {p.step_hours:<10} ${p.predicted_price:<13.2f} "
            f"{p.predicted_return_pct:>+8.2f}%   {p.direction:<10} "
            f"[${p.confidence_p10:.2f}, ${p.confidence_p90:.2f}]"
        )
    print("=" * 80 + "\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_records, f, indent=2)
        logger.info(f"Forecast saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
