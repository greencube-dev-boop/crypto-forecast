"""Adversarial Invariant and Boundary Verification Test Suite.

Authored by challenger_enhancement_2 (Adversarial Invariants Challenger).
Empirically verifies all invariants across Requirements R1-R4:
- R1 Invariants: TimesFM device & dtype (float32) preservation on CUDA/CPU,
  and metric mathematical invariants (MAE >= 0, RMSE >= MAE, sMAPE in [0, 200], Hit-Rate in [0, 100]).
- R2 Invariants: HTTP 451 & 418 MUST NEVER trigger exponential backoff or retry loops;
  3-tier cascade fallback in fetch_ohlcv(source='auto') returns sanitized DataFrame on any tier success.
- R3 Invariants: 48 sequential hourly steps (1h..48h), Gradio table 49-row structure (Row 0: 0h anchor,
  Rows 1..48: timestamp progression), and quantile envelope monotonicity (P10 <= P50 <= P90).
- R4 Invariants: CSV export 49 rows matching table, Google Drive path format, and headless non-interactive
  exception suppression in mount_google_drive().
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import requests

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure yfinance mock exists for offline test resilience
if "yfinance" not in sys.modules:
    sys.modules["yfinance"] = MagicMock()

from crypto_forecast.benchmark.compatibility import run_asset_compatibility_analysis
from crypto_forecast.benchmark.metrics import (
    calculate_hit_rate,
    calculate_mae,
    calculate_rmse,
    calculate_skill_score,
    calculate_smape,
)
from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HOURLY_48_HORIZONS,
    HORIZON_TO_HOURS,
    OHLCV_COLUMNS,
    get_horizon_hours,
)
from crypto_forecast.data.ingestion import (
    BinanceAPIError,
    BinanceGeoBlockedError,
    DataIngestionError,
    fetch_binance_klines,
    fetch_binance_ticker,
    fetch_ohlcv,
)
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.chronos_model import ChronosForecaster
from crypto_forecast.models.granite_model import GraniteTTMForecaster
from crypto_forecast.models.timesfm_model import FallbackTimesFMForecaster, TimesFMForecaster
from crypto_forecast.web.components import build_forecast_table_rows
from crypto_forecast.web.export import (
    create_forecast_dataframe,
    export_forecast_to_csv,
    mount_google_drive,
)


# ==============================================================================
# Helper Mock Generators
# ==============================================================================

def make_mock_kline_batch(count: int, start_time: int = 1700000000000) -> list[list[Any]]:
    """Generates synthetic Binance kline payload."""
    interval_ms = 3600000
    return [
        [
            start_time + i * interval_ms,
            f"{50000.0 + i * 10:.2f}",
            f"{50100.0 + i * 10:.2f}",
            f"{49900.0 + i * 10:.2f}",
            f"{50050.0 + i * 10:.2f}",
            "100.0",
            start_time + (i + 1) * interval_ms - 1,
            "5000000.0",
            1000,
            "50.0",
            "2500000.0",
            "0",
        ]
        for i in range(count)
    ]


def make_mock_yahoo_df(periods: int = 50) -> pd.DataFrame:
    """Generates synthetic sanitized Yahoo Finance OHLCV DataFrame."""
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now, periods=periods, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": np.linspace(48000.0, 49000.0, periods),
            "high": np.linspace(48500.0, 49500.0, periods),
            "low": np.linspace(47500.0, 48500.0, periods),
            "close": np.linspace(48200.0, 49200.0, periods),
            "volume": np.full(periods, 150.0),
        },
        index=idx,
    )
    return df[OHLCV_COLUMNS]


# ==============================================================================
# 1. Requirement R1 Invariants: TimesFM Device/Dtype & Metric Boundaries
# ==============================================================================

class TestR1TimesFMAndMetricInvariants:
    """Adversarial stress-testing of R1 requirements."""

    def test_timesfm_device_delegation_cuda_cpu_invariants(self):
        """Invariant: TimesFM wrapper delegates device placement to underlying PyTorch module.

        TimesFM_2p5_200M_torch lacks .to(), so delegation to model.model is required.
        On CUDA, torch.set_float32_matmul_precision('high') must be invoked.
        """
        # Scenario A: Target device = CPU
        mock_torch = MagicMock()
        mock_cpu_device = MagicMock()
        mock_cpu_device.type = "cpu"
        mock_torch.device.return_value = mock_cpu_device

        mock_inner_module = MagicMock()
        mock_inner_module.to.return_value = mock_inner_module
        mock_outer_model = MagicMock(spec=["model", "compile", "forecast"])
        mock_outer_model.model = mock_inner_module

        mock_timesfm_pkg = MagicMock()
        mock_timesfm_pkg.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_outer_model

        with patch.dict(sys.modules, {"torch": mock_torch, "timesfm": mock_timesfm_pkg}):
            forecaster = TimesFMForecaster(device="cpu", force_fallback=False)
            forecaster.load()

            assert forecaster.is_loaded is True
            assert forecaster.is_fallback_active is False
            mock_inner_module.to.assert_called_once_with(mock_cpu_device)
            assert mock_inner_module.device == mock_cpu_device
            mock_torch.set_float32_matmul_precision.assert_not_called()

        # Scenario B: Target device = CUDA
        mock_torch_cuda = MagicMock()
        mock_cuda_device = MagicMock()
        mock_cuda_device.type = "cuda"
        mock_torch_cuda.device.return_value = mock_cuda_device

        mock_inner_module_cuda = MagicMock()
        mock_inner_module_cuda.to.return_value = mock_inner_module_cuda
        mock_outer_model_cuda = MagicMock(spec=["model", "compile", "forecast"])
        mock_outer_model_cuda.model = mock_inner_module_cuda

        mock_timesfm_pkg_cuda = MagicMock()
        mock_timesfm_pkg_cuda.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_outer_model_cuda

        with patch.dict(sys.modules, {"torch": mock_torch_cuda, "timesfm": mock_timesfm_pkg_cuda}):
            forecaster_cuda = TimesFMForecaster(device="cuda", force_fallback=False)
            forecaster_cuda.load()

            assert forecaster_cuda.is_loaded is True
            assert forecaster_cuda.is_fallback_active is False
            mock_inner_module_cuda.to.assert_called_once_with(mock_cuda_device)
            assert mock_inner_module_cuda.device == mock_cuda_device
            mock_torch_cuda.set_float32_matmul_precision.assert_called_once_with("high")

    def test_timesfm_input_dtype_float32_preservation(self):
        """Invariant: Inputs to TimesFM compile and forecast must be preserved as float32."""
        forecaster = TimesFMForecaster(force_fallback=False)

        # Test varying input dtypes and lengths
        test_inputs = [
            np.array([100.0, 101.0, 102.0], dtype=np.float64),
            np.array([50.0] * 50, dtype=np.float32),
            np.array([10, 20, 30, 40], dtype=np.int64),
            [65000.0, 65100.0, 65200.0],  # Python float list
            np.random.normal(50000, 1000, size=2000),  # Long sequence (>max_context)
            np.array([1e-5, 2e-5, 3e-5], dtype=np.float64),  # Micro-scale
        ]

        for inp in test_inputs:
            context = forecaster._prepare_timesfm_context(np.asarray(inp, dtype=np.float64))
            assert isinstance(context, np.ndarray)
            assert context.dtype == np.float32, f"Expected float32, got {context.dtype} for input {type(inp)}"
            assert len(context) >= 32, f"Expected length >= 32, got {len(context)}"
            assert len(context) <= forecaster.max_context
            assert np.all(np.isfinite(context)), "Context contains non-finite values"

    def test_metric_mathematical_invariants_property_based(self):
        """Invariant: Metric bounds hold across 500 randomized vector pairs:
        - MAE >= 0
        - RMSE >= MAE (Cauchy-Schwarz / Jensen's Inequality)
        - sMAPE in [0, 200]
        - Hit-Rate in [0, 100]
        """
        rng = np.random.default_rng(seed=20261102)

        for _ in range(500):
            n = rng.integers(1, 200)
            regime = rng.choice(["normal", "micro", "macro", "negative", "zeros", "identical"])

            if regime == "normal":
                y_true = rng.uniform(10.0, 100000.0, size=n)
                y_pred = y_true + rng.normal(0, 100.0, size=n)
            elif regime == "micro":
                y_true = rng.uniform(1e-6, 1e-4, size=n)
                y_pred = y_true + rng.normal(0, 1e-6, size=n)
            elif regime == "macro":
                y_true = rng.uniform(1e6, 1e8, size=n)
                y_pred = y_true + rng.normal(0, 1e5, size=n)
            elif regime == "negative":
                # For returns
                y_true = rng.normal(0, 0.05, size=n)
                y_pred = rng.normal(0, 0.05, size=n)
            elif regime == "zeros":
                y_true = np.zeros(n)
                y_pred = np.zeros(n)
            else:  # identical
                y_true = rng.uniform(100.0, 500.0, size=n)
                y_pred = y_true.copy()

            mae = calculate_mae(y_true, y_pred)
            rmse = calculate_rmse(y_true, y_pred)
            smape = calculate_smape(y_true, y_pred)
            hit_rate = calculate_hit_rate(y_true, y_pred)

            assert mae >= 0.0, f"MAE negative: {mae}"
            assert rmse >= mae - 1e-9, f"RMSE ({rmse}) < MAE ({mae})"
            assert 0.0 <= smape <= 200.0 + 1e-9, f"sMAPE out of [0, 200]: {smape}"
            assert 0.0 <= hit_rate <= 100.0 + 1e-9, f"Hit-Rate out of [0, 100]: {hit_rate}"

            if regime == "identical":
                assert mae == 0.0
                assert rmse == 0.0
                assert smape == 0.0

    def test_metric_empty_array_error_handling(self):
        """Invariant: Metrics must explicitly reject empty arrays with ValueError."""
        empty = np.array([])
        with pytest.raises(ValueError, match="Cannot calculate MAE on empty arrays"):
            calculate_mae(empty, empty)
        with pytest.raises(ValueError, match="Cannot calculate RMSE on empty arrays"):
            calculate_rmse(empty, empty)
        with pytest.raises(ValueError, match="Cannot calculate sMAPE on empty arrays"):
            calculate_smape(empty, empty)
        with pytest.raises(ValueError, match="Cannot calculate Hit-Rate on empty arrays"):
            calculate_hit_rate(empty, empty)

    def test_benchmark_compatibility_empty_slice_protection(self):
        """Invariant: Compatibility analysis must complete without raising empty array error."""
        # 1. Short series passed via df_map (120 bars, smaller than step + 100)
        short_series = np.linspace(100.0, 150.0, 120)
        df_short = pd.DataFrame({"close": short_series})
        records = run_asset_compatibility_analysis(
            models=[PersistenceForecaster()],
            symbols=["BTC/USDT"],
            horizons=["1h", "24h", "48h"],
            df_map={"BTC/USDT": df_short},
        )
        assert isinstance(records, pd.DataFrame)
        assert len(records) == 3
        assert "mae_model" in records.columns
        assert "rmse_model" in records.columns
        assert (records["mae_model"] >= 0.0).all()

        # 2. Dynamic scaling with _generate_synthetic_prices for long horizons (e.g. 14d / 336h)
        records_scaled = run_asset_compatibility_analysis(
            models=[TimesFMForecaster(force_fallback=True)],
            symbols=["BTC/USDT"],
            horizons=["14d", "48h"],
        )
        assert isinstance(records_scaled, pd.DataFrame)
        assert len(records_scaled) == 2
        assert "mae_model" in records_scaled.columns
        assert (records_scaled["mae_model"] >= 0.0).all()


# ==============================================================================
# 2. Requirement R2 Invariants: Zero-Backoff on 451/418 & 3-Tier Failover
# ==============================================================================

class TestR2GeoBlockAndFailoverInvariants:
    """Adversarial verification of HTTP 451/418 handling and failover cascade."""

    def test_http_451_never_triggers_exponential_backoff(self):
        """Invariant: HTTP 451 MUST abort immediately with 0 retries and 0 sleep calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Restricted Location"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("crypto_forecast.data.ingestion.time.sleep") as mock_sleep:
            # 1. Klines endpoint
            with pytest.raises(BinanceGeoBlockedError) as exc_info:
                fetch_binance_klines("BTC/USDT", max_retries=5, session=mock_session)
            assert exc_info.value.status_code == 451
            assert mock_session.get.call_count == 1, "HTTP 451 must abort on first attempt"
            mock_sleep.assert_not_called()

            # Reset call count
            mock_session.get.reset_mock()
            mock_sleep.reset_mock()

            # 2. Ticker endpoint
            with pytest.raises(BinanceGeoBlockedError) as exc_info_ticker:
                fetch_binance_ticker("BTC/USDT", max_retries=5, session=mock_session)
            assert exc_info_ticker.value.status_code == 451
            assert mock_session.get.call_count == 1, "HTTP 451 must abort on first attempt"
            mock_sleep.assert_not_called()

    def test_http_418_never_triggers_exponential_backoff(self):
        """Invariant: HTTP 418 MUST abort immediately with 0 retries and 0 sleep calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 418
        mock_resp.text = "IP auto-banned"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("crypto_forecast.data.ingestion.time.sleep") as mock_sleep:
            # 1. Klines endpoint
            with pytest.raises(BinanceAPIError) as exc_info:
                fetch_binance_klines("BTC/USDT", max_retries=5, session=mock_session)
            assert exc_info.value.status_code == 418
            assert mock_session.get.call_count == 1, "HTTP 418 must abort on first attempt"
            mock_sleep.assert_not_called()

            # Reset call count
            mock_session.get.reset_mock()
            mock_sleep.reset_mock()

            # 2. Ticker endpoint
            with pytest.raises(BinanceAPIError) as exc_info_ticker:
                fetch_binance_ticker("BTC/USDT", max_retries=5, session=mock_session)
            assert exc_info_ticker.value.status_code == 418
            assert mock_session.get.call_count == 1, "HTTP 418 must abort on first attempt"
            mock_sleep.assert_not_called()

    def test_http_429_retries_and_backs_off_contrast(self):
        """Contrast Invariant: HTTP 429 MUST invoke exponential backoff and retry up to max_retries.
        This provides empirical proof that backoff is active for rate-limiting, but specifically
        bypassed for 451 geo-blocking and 418 auto-bans.
        """
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_429.headers = {"Retry-After": "1.5"}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp_429

        with patch("crypto_forecast.data.ingestion.time.sleep") as mock_sleep:
            from crypto_forecast.data.ingestion import RateLimitExceededError
            with pytest.raises(RateLimitExceededError):
                fetch_binance_klines("BTC/USDT", max_retries=3, session=mock_session)

            assert mock_session.get.call_count == 3, "HTTP 429 must retry up to max_retries"
            assert mock_sleep.call_count == 3, "HTTP 429 must trigger sleep backoff"

    def test_fetch_ohlcv_auto_cascade_tier_matrix(self):
        """Invariant: fetch_ohlcv(source='auto') returns sanitized DataFrame on any tier success:
        - Tier 1: Binance Global
        - Tier 2: Binance US (if Tier 1 fails with 451, 418, 500, or empty)
        - Tier 3: Yahoo Finance (if Tier 1 and Tier 2 fail)
        """
        valid_candles = make_mock_kline_batch(50)
        valid_yahoo_df = make_mock_yahoo_df(50)

        # Case 1: Tier 1 succeeds directly
        with patch("crypto_forecast.data.ingestion.fetch_binance_klines") as mock_klines:
            mock_klines.return_value = pd.DataFrame(
                {"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC"),
                 "open": [100.0] * 10, "high": [105.0] * 10, "low": [95.0] * 10,
                 "close": [102.0] * 10, "volume": [50.0] * 10}
            )
            df = fetch_ohlcv("BTC/USDT", source="auto")
            assert not df.empty
            assert list(df.columns) == OHLCV_COLUMNS
            assert mock_klines.call_count == 1

        # Case 2: Tier 1 hits HTTP 451 -> Tier 2 (Binance US) succeeds
        def klines_tier2_side_effect(*args, **kwargs):
            base_url = kwargs.get("base_url", "")
            if "api.binance.com" in base_url:
                raise BinanceGeoBlockedError("HTTP 451 Geo-blocked", status_code=451)
            if "api.binance.us" in base_url:
                return pd.DataFrame(
                    {"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC"),
                     "open": [100.0] * 10, "high": [105.0] * 10, "low": [95.0] * 10,
                     "close": [102.0] * 10, "volume": [50.0] * 10}
                )
            raise ValueError(f"Unexpected base_url: {base_url}")

        with patch("crypto_forecast.data.ingestion.fetch_binance_klines", side_effect=klines_tier2_side_effect) as mock_klines:
            df = fetch_ohlcv("BTC/USDT", source="auto")
            assert not df.empty
            assert mock_klines.call_count == 2, "Expected Tier 1 attempt followed by Tier 2 attempt"

        # Case 3: Tier 1 & Tier 2 hit HTTP 451 -> Tier 3 (Yahoo Finance) succeeds
        def klines_both_451(*args, **kwargs):
            raise BinanceGeoBlockedError("HTTP 451 Geo-blocked", status_code=451)

        with patch("crypto_forecast.data.ingestion.fetch_binance_klines", side_effect=klines_both_451) as mock_klines, \
             patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv", return_value=valid_yahoo_df) as mock_yahoo:
            df = fetch_ohlcv("BTC/USDT", source="auto")
            assert not df.empty
            assert mock_klines.call_count == 2
            assert mock_yahoo.call_count == 1
            assert list(df.columns) == OHLCV_COLUMNS

        # Case 4: All 3 tiers fail -> raises DataIngestionError
        with patch("crypto_forecast.data.ingestion.fetch_binance_klines", side_effect=klines_both_451), \
             patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv", side_effect=Exception("Yahoo network error")):
            with pytest.raises(DataIngestionError, match="All data sources failed"):
                fetch_ohlcv("BTC/USDT", source="auto")


# ==============================================================================
# 3. Requirement R3 Invariants: 48 Continuous Horizons, Table & Quantile Monotonicity
# ==============================================================================

class TestR3HorizonTableAndQuantileInvariants:
    """Adversarial testing of 48-hour horizons, table formatting, and quantile envelope."""

    def test_hourly_48_horizons_sequential_and_strictly_increasing(self):
        """Invariant: HOURLY_48_HORIZONS has 48 steps ('1h'..'48h') with step_hours == i."""
        assert len(DEFAULT_HORIZONS) == 48
        assert len(HOURLY_48_HORIZONS) == 48

        for i in range(1, 49):
            expected_h = f"{i}h"
            assert DEFAULT_HORIZONS[i - 1] == expected_h
            assert HORIZON_TO_HOURS[expected_h] == i
            assert get_horizon_hours(expected_h) == i

        hours_sequence = [HORIZON_TO_HOURS[h] for h in DEFAULT_HORIZONS]
        assert hours_sequence == list(range(1, 49))

    @pytest.mark.parametrize(
        "forecaster_cls,kwargs",
        [
            (PersistenceForecaster, {}),
            (TimesFMForecaster, {"force_fallback": True}),
            (ChronosForecaster, {}),
            (GraniteTTMForecaster, {"use_fallback": True}),
        ],
    )
    def test_forecasters_produce_all_48_horizons_and_valid_predictions(self, forecaster_cls, kwargs):
        """Invariant: All forecasters return dict of 48 HorizonPrediction objects with step_hours == i."""
        forecaster = forecaster_cls(**kwargs)
        prices = np.linspace(60000.0, 65000.0, 150)
        preds = forecaster.forecast(prices, horizons=DEFAULT_HORIZONS)

        assert len(preds) == 48
        for i in range(1, 49):
            h = f"{i}h"
            assert h in preds
            p = preds[h]
            assert p.horizon == h
            assert p.step_hours == i
            assert p.predicted_price > 0.0
            assert p.confidence_p10 <= p.confidence_p50 <= p.confidence_p90

    def test_gradio_table_invariant_49_rows_and_timestamp_progression(self):
        """Invariant: Gradio table has exactly 49 rows.
        - Row 0: '0h' current spot price, timestamp == current_time
        - Rows 1..48: timestamp == current_time + timedelta(hours=i)
        """
        forecaster = PersistenceForecaster()
        curr_p = 68450.25
        cutoff = pd.Timestamp("2026-11-02 12:00:00", tz="UTC")
        prices = [curr_p - 100, curr_p]
        preds = forecaster.forecast(prices, horizons=DEFAULT_HORIZONS)

        rows = build_forecast_table_rows(preds, current_time=cutoff, current_price=curr_p)

        assert len(rows) == 49

        # Row 0 Verification
        row0 = rows[0]
        assert row0[0] == "0h"
        assert row0[1] == "2026-11-02 12:00:00 UTC"
        assert "$68,450.25" in row0[2]
        assert row0[3] == "0.00%"
        assert row0[4] == "CURRENT"
        assert "[$68,450.25, $68,450.25]" in row0[5]

        # Rows 1..48 Verification
        for i in range(1, 49):
            row_i = rows[i]
            expected_h = f"{i}h"
            expected_ts = (cutoff + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S UTC")

            assert row_i[0] == expected_h
            assert row_i[1] == expected_ts
            assert row_i[4] in ("BULLISH", "BEARISH", "NEUTRAL")
            assert row_i[5].startswith("[") and row_i[5].endswith("]")

    def test_quantile_envelope_monotonicity_across_price_scales(self):
        """Invariant: P10 <= P50 <= P90 across extreme price regimes ($0.0000085 to $150,000)."""
        regimes = [
            ("meme_pepe", np.linspace(0.0000080, 0.0000095, 100)),
            ("mega_btc", np.linspace(140000.0, 150000.0, 100)),
            ("flat_asset", np.full(100, 42.0)),
            ("volatile", 50000.0 * np.exp(np.cumsum(np.random.normal(0, 0.05, 100)))),
        ]

        forecaster = TimesFMForecaster(force_fallback=True)
        for name, prices in regimes:
            preds = forecaster.forecast(prices, horizons=DEFAULT_HORIZONS)
            for h, p in preds.items():
                assert p.confidence_p10 <= p.confidence_p50, f"Violation in {name} at {h}: P10 > P50"
                assert p.confidence_p50 <= p.confidence_p90, f"Violation in {name} at {h}: P50 > P90"
                assert p.confidence_p10 > 0.0, f"Violation in {name} at {h}: P10 <= 0"


# ==============================================================================
# 4. Requirement R4 Invariants: CSV Export 49 Rows, Drive Path & Headless Safety
# ==============================================================================

class TestR4CSVExportAndDriveInvariants:
    """Adversarial testing of CSV export, Google Drive path structure, and headless safety."""

    def test_dataframe_and_csv_export_49_rows_matching_table(self, tmp_path):
        """Invariant: CSV export contains exactly 49 rows (Step 0h + Steps 1h..48h)."""
        forecaster = PersistenceForecaster()
        curr_price = 72150.0
        cutoff = pd.Timestamp("2026-11-02 15:30:00", tz="UTC")
        preds = forecaster.forecast([curr_price], horizons=DEFAULT_HORIZONS)

        # 1. Direct DataFrame creation
        df = create_forecast_dataframe(preds, current_time=cutoff, current_price=curr_price)
        assert len(df) == 49
        assert df.iloc[0]["Step"] == "0h"
        assert list(df["Step"]) == ["0h"] + [f"{i}h" for i in range(1, 49)]

        # 2. Export to CSV and inspect contents
        table_rows = build_forecast_table_rows(preds, current_time=cutoff, current_price=curr_price)
        local_csv, drive_csv = export_forecast_to_csv(
            table_rows,
            symbol="BTC",
            output_dir=tmp_path,
            current_time=cutoff,
            current_price=curr_price,
        )

        assert Path(local_csv).exists()
        df_csv = pd.read_csv(local_csv)
        assert len(df_csv) == 49
        assert df_csv.iloc[0]["Step"] == "0h"
        assert list(df_csv["Step"]) == ["0h"] + [f"{i}h" for i in range(1, 49)]
        assert df_csv.iloc[0]["Price / Prediction"] == "$72,150.00"

    def test_google_drive_path_format_and_sync(self, tmp_path):
        """Invariant: Google Drive target path matches /content/drive/MyDrive/Trader/<timestamp>.csv."""
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast([50000.0], horizons=DEFAULT_HORIZONS)
        fixed_time = datetime(2026, 11, 2, 16, 45, 0, tzinfo=timezone.utc)

        # Mock Drive mounted at tmp_path / "content" / "drive"
        mock_drive_root = tmp_path / "content" / "drive"
        mock_trader_dir = mock_drive_root / "MyDrive" / "Trader"
        mock_trader_dir.mkdir(parents=True, exist_ok=True)

        def mock_path_constructor(p):
            p_str = str(p)
            if "/content/drive/MyDrive/Trader" in p_str or "\\content\\drive\\MyDrive\\Trader" in p_str:
                fname = Path(p_str).name
                return mock_trader_dir / fname
            return Path(p)

        with patch("crypto_forecast.web.export.mount_google_drive", return_value=True), \
             patch("crypto_forecast.web.export.Path", side_effect=mock_path_constructor):
            local_csv, drive_csv = export_forecast_to_csv(
                preds,
                symbol="ETH",
                timestamp=fixed_time,
                output_dir=tmp_path / "exports",
            )

            assert drive_csv is not None
            assert drive_csv.endswith("20261102_164500.csv")
            assert "MyDrive" in drive_csv and "Trader" in drive_csv
            assert Path(drive_csv).exists()

    def test_headless_non_interactive_drive_mounting_invariant(self):
        """Invariant: mount_google_drive() returns False without raising when google.colab is absent or fails."""
        # 1. Non-colab standard test environment (google.colab not installed)
        with patch.dict(sys.modules, {"google.colab": None}):
            res = mount_google_drive(mount_point="/nonexistent/mount")
            assert res is False

        # 2. google.colab exists but drive.mount raises arbitrary exception (e.g. auth cancel / timeout)
        mock_colab = MagicMock()
        mock_colab.drive.mount.side_effect = RuntimeError("Headless authentication timed out")
        with patch.dict(sys.modules, {"google.colab": mock_colab, "google.colab.drive": mock_colab.drive}):
            res = mount_google_drive(mount_point="/nonexistent/mount")
            assert res is False

    def test_google_drive_write_permission_error_graceful_fallback(self, tmp_path):
        """Invariant: If writing to Google Drive raises PermissionError or OSError, local CSV is preserved and no exception is raised."""
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast([50000.0], horizons=DEFAULT_HORIZONS)

        orig_to_csv = pd.DataFrame.to_csv

        def selective_to_csv(self_df, path, *args, **kwargs):
            if "Trader" in str(path) or "MyDrive" in str(path):
                raise PermissionError("Read-only filesystem on Colab Drive")
            return orig_to_csv(self_df, path, *args, **kwargs)

        mock_trader_dir = tmp_path / "mock_drive" / "Trader"
        mock_trader_dir.mkdir(parents=True, exist_ok=True)

        with patch("crypto_forecast.web.export.mount_google_drive", return_value=True), \
             patch("crypto_forecast.web.export.Path", side_effect=lambda p: mock_trader_dir / Path(p).name if "Trader" in str(p) else Path(p)), \
             patch.object(pd.DataFrame, "to_csv", new=selective_to_csv):
            local_csv, drive_csv = export_forecast_to_csv(
                preds,
                symbol="SOL",
                output_dir=tmp_path / "exports",
            )
            assert Path(local_csv).exists()
            assert drive_csv is None

    def test_export_csv_polymorphic_inputs_preserve_49_rows(self, tmp_path):
        """Invariant: export_forecast_to_csv accepts DataFrame, dict of HorizonPrediction, and list of rows."""
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast([65000.0], horizons=DEFAULT_HORIZONS)
        cutoff = pd.Timestamp("2026-11-02 14:00:00", tz="UTC")
        table_rows = build_forecast_table_rows(preds, current_time=cutoff, current_price=65000.0)
        df_direct = create_forecast_dataframe(preds, current_time=cutoff, current_price=65000.0)

        # 1. From dict of predictions
        csv1, _ = export_forecast_to_csv(preds, symbol="BTC1", output_dir=tmp_path, current_time=cutoff, current_price=65000.0)
        # 2. From list of table rows
        csv2, _ = export_forecast_to_csv(table_rows, symbol="BTC2", output_dir=tmp_path)
        # 3. From DataFrame directly
        csv3, _ = export_forecast_to_csv(df_direct, symbol="BTC3", output_dir=tmp_path)

        for p in (csv1, csv2, csv3):
            df_read = pd.read_csv(p)
            assert len(df_read) == 49
            assert df_read.iloc[0]["Step"] == "0h"
            assert list(df_read["Step"]) == ["0h"] + [f"{i}h" for i in range(1, 49)]

    def test_gradio_table_timezone_normalization_resilience(self):
        """Invariant: build_forecast_table_rows handles timezone-naive and non-UTC timestamps by converting to UTC."""
        forecaster = PersistenceForecaster()
        preds = forecaster.forecast([100.0], horizons=DEFAULT_HORIZONS)

        # Naive timestamp
        naive_ts = pd.Timestamp("2026-11-02 12:00:00")
        rows_naive = build_forecast_table_rows(preds, current_time=naive_ts, current_price=100.0)
        assert len(rows_naive) == 49
        assert "2026-11-02 12:00:00 UTC" in rows_naive[0][1]

        # Non-UTC timezone (e.g. US/Eastern = UTC-5 in Nov)
        eastern_ts = pd.Timestamp("2026-11-02 12:00:00", tz="US/Eastern")
        rows_eastern = build_forecast_table_rows(preds, current_time=eastern_ts, current_price=100.0)
        assert len(rows_eastern) == 49
        # 12:00 Eastern is 17:00 UTC
        assert "2026-11-02 17:00:00 UTC" in rows_eastern[0][1]
