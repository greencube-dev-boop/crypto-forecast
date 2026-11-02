"""Empirical Adversarial Stress Testing Suite for Requirements R1-R4.

Engineered by challenger_enhancement_1 for empirical verification of:
- R1: TimesFM 2.5 PyTorch model loading without .to() and benchmark stability
- R2: Binance HTTP 451 rapid failover, 0 sleep delays, and 3-tier cascade
- R3: 48 Continuous Hourly Horizons and Gradio 49-row table
- R4: CSV export and headless Google Drive mounting resilience
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import MagicMock, patch

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.benchmark.compatibility import run_asset_compatibility_analysis
from crypto_forecast.benchmark.metrics import calculate_mae
from crypto_forecast.config import (
    DEFAULT_HORIZONS,
    HOURLY_48_HORIZONS,
    HORIZON_TO_HOURS,
    get_horizon_hours,
)
from crypto_forecast.data.ingestion import (
    BINANCE_BASE_URL,
    BINANCE_US_BASE_URL,
    BinanceAPIError,
    BinanceGeoBlockedError,
    DataIngestionError,
    fetch_binance_klines,
    fetch_binance_ticker,
    fetch_ohlcv,
)
from crypto_forecast.models.base import HorizonPrediction
from crypto_forecast.models.baseline import PersistenceForecaster
from crypto_forecast.models.timesfm_model import TimesFMForecaster
from crypto_forecast.web.app import build_reference_gradio_app, run_forecast_pipeline
from crypto_forecast.web.components import build_forecast_table_rows
from crypto_forecast.web.export import (
    create_forecast_dataframe,
    export_forecast_to_csv,
    mount_google_drive,
)


# ==============================================================================
# R1 Adversarial Stress Tests: TimesFM Model Loading & Benchmark
# ==============================================================================


class TestR1TimesFMLoadingAndBenchmarkStress:
    """Adversarial stress testing for TimesFM PyTorch device loading and benchmark stability."""

    def test_timesfm_load_mock_without_to_delegates_to_model_model_cpu(self):
        """Asserts TimesFMForecaster.load() delegates to self.model.model.to() on CPU

        when outer wrapper object has NO .to() attribute.
        """
        forecaster = TimesFMForecaster(device="cpu", force_fallback=False)

        # Build mock where outer object lacks .to() entirely
        mock_inner_model = MagicMock()
        mock_inner_model.to.return_value = mock_inner_model

        class MockOuterWrapper:
            def __init__(self):
                self.model = mock_inner_model

            def compile(self, *args, **kwargs):
                pass

        mock_outer = MockOuterWrapper()
        # Verify outer has no .to
        assert not hasattr(mock_outer, "to")
        assert hasattr(mock_outer, "model")
        assert hasattr(mock_outer.model, "to")

        mock_timesfm_mod = MagicMock()
        mock_timesfm_mod.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_outer
        mock_torch_mod = MagicMock()
        mock_torch_mod.device.return_value = MagicMock(type="cpu")

        with patch.dict(sys.modules, {"timesfm": mock_timesfm_mod, "torch": mock_torch_mod}):
            forecaster.load()

        assert forecaster.is_loaded
        assert not forecaster.is_fallback_active
        # Assert model.model.to was called with CPU device
        mock_inner_model.to.assert_called_once()
        assert mock_inner_model.device == mock_torch_mod.device.return_value

    def test_timesfm_load_mock_without_to_delegates_to_model_model_cuda(self):
        """Asserts TimesFMForecaster.load() delegates on CUDA and sets float32 matmul precision."""
        forecaster = TimesFMForecaster(device="cuda", force_fallback=False)

        mock_inner_model = MagicMock()
        mock_inner_model.to.return_value = mock_inner_model

        class MockOuterWrapper:
            def __init__(self):
                self.model = mock_inner_model

            def compile(self, *args, **kwargs):
                pass

        mock_outer = MockOuterWrapper()
        assert not hasattr(mock_outer, "to")

        mock_timesfm_mod = MagicMock()
        mock_timesfm_mod.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_outer
        mock_torch_mod = MagicMock()
        cuda_device = MagicMock(type="cuda")
        mock_torch_mod.device.return_value = cuda_device
        mock_torch_mod.cuda.is_available.return_value = True

        with patch.dict(sys.modules, {"timesfm": mock_timesfm_mod, "torch": mock_torch_mod}):
            forecaster.load()

        assert forecaster.is_loaded
        assert not forecaster.is_fallback_active
        mock_torch_mod.set_float32_matmul_precision.assert_called_once_with("high")
        mock_inner_model.to.assert_called_once_with(cuda_device)
        assert mock_inner_model.device == cuda_device

    def test_asset_compatibility_short_series_never_calls_mae_on_empty_arrays(self):
        """Stress tests run_asset_compatibility_analysis with short series and extreme steps.

        Guarantees calculate_mae is NEVER invoked with empty arrays and never raises ValueError.
        """
        # Monitor calculate_mae calls
        original_mae = calculate_mae
        mae_calls: list[tuple[int, int]] = []

        def spy_calculate_mae(y_true, y_pred):
            y_t = np.asarray(y_true)
            y_p = np.asarray(y_pred)
            assert y_t.size > 0, "calculate_mae was called with an EMPTY y_true array!"
            assert y_p.size > 0, "calculate_mae was called with an EMPTY y_pred array!"
            mae_calls.append((y_t.size, y_p.size))
            return original_mae(y_t, y_p)

        # Provide very short series (50, 105, 150 bars) with extreme steps (336, 720)
        short_df_map = {
            "BTC/USDT": pd.DataFrame({"close": np.linspace(50000, 52000, 120)}),
            "ETH/USDT": pd.DataFrame({"close": np.linspace(3000, 3100, 50)}),
        }

        # Forecaster with fallback
        forecaster = TimesFMForecaster(force_fallback=True)

        with patch("crypto_forecast.benchmark.compatibility.calculate_mae", side_effect=spy_calculate_mae):
            df_compat = run_asset_compatibility_analysis(
                models=[forecaster],
                symbols=["BTC/USDT", "ETH/USDT"],
                horizons=["1h", "24h", "14d", "30d"],  # Steps: 1, 24, 336, 720
                df_map=short_df_map,
            )

        assert not df_compat.empty
        assert len(df_compat) == 8  # 2 symbols * 4 horizons
        # Verify all mae_model and mae_persist are valid finite numbers
        assert df_compat["mae_model"].notna().all()
        assert df_compat["mae_persist"].notna().all()
        # Ensure that every call to calculate_mae had size > 0
        for size_t, size_p in mae_calls:
            assert size_t > 0
            assert size_p > 0

    def test_asset_compatibility_boundary_series_exact_edge(self):
        """Tests exact boundary edge cases where len(prices) == step + 100 or step + 101."""
        # For step = 10, step + 100 = 110 bars. Test exactly 110 bars and 111 bars.
        edge_df_map = {
            "SOL/USDT": pd.DataFrame({"close": np.linspace(100, 150, 110)}),
            "DOGE/USDT": pd.DataFrame({"close": np.linspace(0.1, 0.2, 111)}),
        }
        forecaster = TimesFMForecaster(force_fallback=True)

        df_compat = run_asset_compatibility_analysis(
            models=[forecaster],
            symbols=["SOL/USDT", "DOGE/USDT"],
            horizons=["10h"],
            df_map=edge_df_map,
        )

        assert len(df_compat) == 2
        assert df_compat["mae_model"].notna().all()

    def test_timesfm_load_direct_to_fallback(self):
        """Asserts that if the model DOES have a direct .to() method, it is properly called."""
        forecaster = TimesFMForecaster(device="cpu", force_fallback=False)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        # Ensure it does NOT have .model attribute
        del mock_model.model

        mock_timesfm_mod = MagicMock()
        mock_timesfm_mod.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_model
        mock_torch_mod = MagicMock()
        cpu_device = MagicMock(type="cpu")
        mock_torch_mod.device.return_value = cpu_device

        with patch.dict(sys.modules, {"timesfm": mock_timesfm_mod, "torch": mock_torch_mod}):
            forecaster.load()

        assert forecaster.is_loaded
        mock_model.to.assert_called_once_with(cpu_device)

    def test_timesfm_load_mps_device_resolution(self):
        """Asserts MPS device is selected if available and device is not explicitly given."""
        forecaster = TimesFMForecaster(device=None, force_fallback=False)

        mock_inner_model = MagicMock()
        mock_inner_model.to.return_value = mock_inner_model

        class MockOuterWrapper:
            def __init__(self):
                self.model = mock_inner_model

            def compile(self, *args, **kwargs):
                pass

        mock_outer = MockOuterWrapper()
        mock_timesfm_mod = MagicMock()
        mock_timesfm_mod.TimesFM_2p5_200M_torch.from_pretrained.return_value = mock_outer

        mock_torch_mod = MagicMock()
        mock_torch_mod.cuda.is_available.return_value = False
        mock_torch_mod.backends.mps.is_available.return_value = True
        mps_device = MagicMock(type="mps")
        mock_torch_mod.device.return_value = mps_device

        with patch.dict(sys.modules, {"timesfm": mock_timesfm_mod, "torch": mock_torch_mod}):
            forecaster.load()

        mock_torch_mod.device.assert_called_with("mps")
        mock_inner_model.to.assert_called_once_with(mps_device)

    def test_asset_compatibility_zero_length_series(self):
        """Asserts run_asset_compatibility_analysis handles series with 0 bars without raising ValueError."""
        empty_df_map = {
            "BTC/USDT": pd.DataFrame({"close": np.array([], dtype=float)}),
        }
        forecaster = TimesFMForecaster(force_fallback=True)

        df_compat = run_asset_compatibility_analysis(
            models=[forecaster],
            symbols=["BTC/USDT"],
            horizons=["1h", "24h"],
            df_map=empty_df_map,
        )

        assert len(df_compat) == 2
        assert df_compat["mae_model"].notna().all()

    def test_asset_compatibility_extreme_step_720_exact_boundary(self):
        """Asserts step=720 (30d) works at exact boundaries len(prices) = 820 and 821."""
        boundary_df_map = {
            "BTC/USDT": pd.DataFrame({"close": np.linspace(50000, 60000, 820)}),  # exactly step + 100
            "ETH/USDT": pd.DataFrame({"close": np.linspace(3000, 4000, 821)}),    # exactly step + 100 + 1
        }
        forecaster = TimesFMForecaster(force_fallback=True)

        df_compat = run_asset_compatibility_analysis(
            models=[forecaster],
            symbols=["BTC/USDT", "ETH/USDT"],
            horizons=["30d"],
            df_map=boundary_df_map,
        )

        assert len(df_compat) == 2
        assert df_compat["mae_model"].notna().all()



# ==============================================================================
# R2 Adversarial Stress Tests: HTTP 451 Geo-Blocking & Cascade Fallback
# ==============================================================================


class TestR2GeoBlockingAndCascadeStress:
    """Adversarial stress testing for HTTP 451 rapid abort, zero delay, and 3-tier cascade."""

    def test_binance_klines_451_raises_binance_geoblocked_error_immediately_zero_sleep(self):
        """Asserts HTTP 451 raises BinanceGeoBlockedError immediately with zero time.sleep calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Restricted Location"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        start_time = time.perf_counter()
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(BinanceGeoBlockedError) as exc_info:
                fetch_binance_klines(
                    symbol="BTC/USDT",
                    interval="1h",
                    limit=10,
                    session=mock_session,
                    max_retries=5,
                )
            elapsed = time.perf_counter() - start_time

            # Assertions:
            assert exc_info.value.status_code == 451
            mock_sleep.assert_not_called()
            assert elapsed < 0.1, f"Execution took {elapsed:.4f}s, expected < 0.1s"

    def test_binance_ticker_451_raises_binance_geoblocked_error_immediately_zero_sleep(self):
        """Asserts HTTP 451 on ticker raises BinanceGeoBlockedError immediately with zero time.sleep."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Geo-blocked"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        start_time = time.perf_counter()
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(BinanceGeoBlockedError) as exc_info:
                fetch_binance_ticker(
                    symbol="BTC/USDT",
                    session=mock_session,
                    max_retries=5,
                )
            elapsed = time.perf_counter() - start_time

            assert exc_info.value.status_code == 451
            mock_sleep.assert_not_called()
            assert elapsed < 0.1

    def test_3_tier_cascade_exact_sequence_global_451_us_451_to_yahoo(self):
        """Asserts exact 3-tier cascade sequence:

        1. Binance Global (returns 451) ->
        2. Binance US (returns 451) ->
        3. Yahoo Finance (succeeds)
        All without retry sleep delays.
        """
        call_urls: list[str] = []

        def mock_get(url, *args, **kwargs):
            call_urls.append(url)
            resp = MagicMock()
            resp.headers = {}
            if "api.binance.com" in url:
                resp.status_code = 451
                resp.text = "Binance Global 451"
                return resp
            if "api.binance.us" in url:
                resp.status_code = 451
                resp.text = "Binance US 451"
                return resp
            # Yahoo Finance direct chart fallback
            resp.status_code = 200
            resp.json.return_value = {
                "chart": {
                    "result": [
                        {
                            "timestamp": [1700000000, 1700003600],
                            "indicators": {
                                "quote": [
                                    {
                                        "open": [100.0, 101.0],
                                        "high": [105.0, 106.0],
                                        "low": [99.0, 100.0],
                                        "close": [102.0, 103.0],
                                        "volume": [1000.0, 1500.0],
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get

        with patch("time.sleep") as mock_sleep:
            df = fetch_ohlcv(
                symbol="BTC/USDT",
                interval="1h",
                limit=2,
                source="auto",
                session=mock_session,
            )

            # Assert sleep was never called
            mock_sleep.assert_not_called()

        assert not df.empty
        assert len(df) == 2
        # Assert exact call sequence
        assert len(call_urls) >= 2
        assert "api.binance.com" in call_urls[0]
        assert "api.binance.us" in call_urls[1]

    def test_cascade_stops_at_binance_us_when_us_returns_200(self):
        """Asserts that if Binance Global returns 451 but Binance US returns 200,

        data is returned from Binance US and Yahoo Finance is NOT queried.
        """
        call_urls: list[str] = []

        raw_us_candles = [
            [
                1700000000000,
                "50000.0",
                "51000.0",
                "49500.0",
                "50500.0",
                "100.0",
                1700003599999,
                "5050000.0",
                10,
                "50.0",
                "2525000.0",
                "0",
            ],
            [
                1700003600000,
                "50500.0",
                "52000.0",
                "50400.0",
                "51500.0",
                "120.0",
                1700007199999,
                "6180000.0",
                12,
                "60.0",
                "3090000.0",
                "0",
            ],
        ]

        def mock_get(url, *args, **kwargs):
            call_urls.append(url)
            resp = MagicMock()
            resp.headers = {}
            if "api.binance.com" in url:
                resp.status_code = 451
                resp.text = "Geo-blocked Global"
                return resp
            if "api.binance.us" in url:
                resp.status_code = 200
                resp.json.return_value = raw_us_candles
                return resp
            # Yahoo or other
            resp.status_code = 500
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get

        with patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv") as mock_yahoo:
            with patch("time.sleep") as mock_sleep:
                df = fetch_ohlcv(
                    symbol="BTC/USDT",
                    interval="1h",
                    limit=2,
                    source="auto",
                    session=mock_session,
                )
                mock_sleep.assert_not_called()
                mock_yahoo.assert_not_called()

        assert not df.empty
        assert len(df) == 2
        assert float(df["close"].iloc[-1]) == 51500.0
        # Check URLs: Binance Global was called, then Binance US, and nothing else
        assert any("api.binance.com" in u for u in call_urls)
        assert any("api.binance.us" in u for u in call_urls)
        assert not any("yahoo" in u.lower() for u in call_urls)

    def test_source_binance_explicit_451_raises_without_cascade(self):
        """Asserts that when source='binance' is specified, HTTP 451 raises BinanceGeoBlockedError without fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Restricted"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with pytest.raises(BinanceGeoBlockedError):
            fetch_ohlcv(
                symbol="BTC/USDT",
                interval="1h",
                limit=10,
                source="binance",
                session=mock_session,
            )

    def test_source_binance_us_explicit_451_raises_without_cascade(self):
        """Asserts that when source='binance_us' is specified, HTTP 451 raises BinanceGeoBlockedError without fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Restricted US"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with pytest.raises(BinanceGeoBlockedError):
            fetch_ohlcv(
                symbol="BTC/USDT",
                interval="1h",
                limit=10,
                source="binance_us",
                session=mock_session,
            )

    def test_cascade_handles_binance_418_ip_ban_immediately(self):
        """Asserts that HTTP 418 (IP auto-ban) immediately fails over to Binance US without retrying."""
        call_urls: list[str] = []

        def mock_get(url, *args, **kwargs):
            call_urls.append(url)
            resp = MagicMock()
            resp.headers = {}
            if "api.binance.com" in url:
                resp.status_code = 418
                resp.text = "IP auto-ban"
                return resp
            if "api.binance.us" in url:
                resp.status_code = 200
                resp.json.return_value = [
                    [1700000000000, "100.0", "105.0", "99.0", "102.0", "10.0", 1700003599999, "1020.0", 5, "5.0", "510.0", "0"]
                ]
                return resp
            resp.status_code = 500
            return resp

        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get

        with patch("time.sleep") as mock_sleep:
            df = fetch_ohlcv("BTC/USDT", interval="1h", limit=1, source="auto", session=mock_session)
            mock_sleep.assert_not_called()

        assert not df.empty
        assert len(df) == 1
        assert any("api.binance.com" in u for u in call_urls)
        assert any("api.binance.us" in u for u in call_urls)

    def test_cascade_all_three_tiers_fail_raises_data_ingestion_error(self):
        """Asserts that when Binance Global, Binance US, and Yahoo Finance all fail, DataIngestionError is raised."""
        mock_resp = MagicMock()
        mock_resp.status_code = 451
        mock_resp.text = "Geo-blocked"
        mock_resp.headers = {}

        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("crypto_forecast.data.ingestion.fetch_yahoo_ohlcv", side_effect=Exception("Yahoo network timeout")):
            with pytest.raises(DataIngestionError) as exc_info:
                fetch_ohlcv("BTC/USDT", interval="1h", limit=10, source="auto", session=mock_session)
            assert "All data sources failed" in str(exc_info.value)



# ==============================================================================
# R3 Adversarial Stress Tests: 48 Continuous Hourly Horizons & Table Output
# ==============================================================================


class TestR3HourlyHorizonsAndTableStress:
    """Adversarial stress testing for 48 continuous hourly horizons and 49-row Gradio table."""

    def test_continuous_hourly_horizons_completeness_1_to_48(self):
        """Asserts HOURLY_48_HORIZONS contains exactly 48 hourly steps from 1h to 48h."""
        expected = [f"{i}h" for i in range(1, 49)]
        assert HOURLY_48_HORIZONS == expected
        assert len(HOURLY_48_HORIZONS) == 48
        assert DEFAULT_HORIZONS == HOURLY_48_HORIZONS

    def test_horizon_to_hours_all_48_and_legacy_mappings(self):
        """Asserts HORIZON_TO_HOURS contains all 48 steps and legacy multi-day horizons."""
        for i in range(1, 49):
            h_str = f"{i}h"
            assert h_str in HORIZON_TO_HOURS
            assert HORIZON_TO_HOURS[h_str] == i
            assert get_horizon_hours(h_str) == i

        # Legacy multi-day keys
        legacy_checks = {
            "1d": 24,
            "2d": 48,
            "3d": 72,
            "7d": 168,
            "14d": 336,
            "30d": 720,
        }
        for k, v in legacy_checks.items():
            assert HORIZON_TO_HOURS[k] == v
            assert get_horizon_hours(k) == v

    def test_gradio_forecast_table_rows_exact_49_rows_step_0h_anchor(self):
        """Asserts build_forecast_table_rows produces exactly 49 rows starting with Step 0h."""
        now = pd.Timestamp("2026-11-02 12:00:00", tz="UTC")
        spot_price = 68420.50

        # Construct predictions for all 48 horizons
        predictions: dict[str, HorizonPrediction] = {}
        for i in range(1, 49):
            h = f"{i}h"
            pred_price = spot_price * (1.0 + 0.001 * i)
            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=i,
                current_price=spot_price,
                predicted_price=pred_price,
                predicted_change=pred_price - spot_price,
                predicted_return_pct=0.1 * i,
                confidence_p10=pred_price * 0.98,
                confidence_p50=pred_price,
                confidence_p90=pred_price * 1.02,
                direction="BULLISH",
            )

        rows = build_forecast_table_rows(
            predictions=predictions,
            current_time=now,
            current_price=spot_price,
        )

        assert len(rows) == 49, f"Expected 49 rows, got {len(rows)}"

        # Verify Row 0 (Anchor Row)
        row_0 = rows[0]
        assert row_0[0] == "0h"
        assert row_0[1] == "2026-11-02 12:00:00 UTC"
        assert row_0[2] == "$68,420.50"
        assert row_0[3] == "0.00%"
        assert row_0[4] == "CURRENT"
        assert row_0[5] == "[$68,420.50, $68,420.50]"

        # Verify Rows 1..48
        for i in range(1, 49):
            row_i = rows[i]
            expected_h = f"{i}h"
            expected_time = (now + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S UTC")
            assert row_i[0] == expected_h
            assert row_i[1] == expected_time
            assert "$" in row_i[2]
            assert "%" in row_i[3]
            assert row_i[4] == "BULLISH"
            assert "[" in row_i[5] and "]" in row_i[5]

    def test_gradio_app_structure_contains_table_and_download_file(self):
        """Asserts build_reference_gradio_app builds properly and has file download."""
        demo = build_reference_gradio_app()
        assert demo is not None
        # Verify app blocks can be rendered
        import gradio as gr
        # Check components in demo
        has_file_component = any(isinstance(c, gr.File) for c in demo.blocks.values())
        has_dataframe_component = any(isinstance(c, gr.Dataframe) for c in demo.blocks.values())
        assert has_file_component, "Gradio app missing gr.File download component"
        assert has_dataframe_component, "Gradio app missing gr.Dataframe component"

    def test_run_forecast_pipeline_full_execution_generates_49_rows_and_csv(self):
        """Asserts run_forecast_pipeline returns 49 rows and writes a valid CSV file."""
        fig, table_rows, scoreboard_md, csv_file = run_forecast_pipeline(
            coin="BTC",
            model_choice="Persistence (Baseline)",
            apply_sentiment=False,
        )

        assert fig is not None
        assert len(table_rows) == 49
        assert table_rows[0][0] == "0h"
        assert table_rows[48][0] == "48h"
        assert "Accuracy Scoreboard" in scoreboard_md
        assert csv_file is not None
        assert os.path.exists(csv_file)

        df_csv = pd.read_csv(csv_file)
        assert len(df_csv) == 49
        assert df_csv.iloc[0]["Step"] == "0h"
        assert df_csv.iloc[48]["Step"] == "48h"

    def test_constants_module_reexport_integrity(self):
        """Asserts crypto_forecast.constants re-exports core constants."""
        import crypto_forecast.constants as constants

        assert hasattr(constants, "DEFAULT_HORIZONS")
        assert hasattr(constants, "HORIZON_TO_HOURS")
        assert constants.DEFAULT_HORIZONS == HOURLY_48_HORIZONS
        assert len(constants.DEFAULT_HORIZONS) == 48



# ==============================================================================
# R4 Adversarial Stress Tests: CSV Export & Google Drive Resilience
# ==============================================================================


class TestR4CSVExportAndDriveResilienceStress:
    """Adversarial stress testing for CSV export and Google Drive headless resilience."""

    def test_create_forecast_dataframe_contains_all_49_rows(self):
        """Asserts create_forecast_dataframe builds standardized 49-row DataFrame."""
        now = pd.Timestamp("2026-11-02 14:00:00", tz="UTC")
        spot_price = 100.0

        predictions: dict[str, HorizonPrediction] = {}
        for i in range(1, 49):
            h = f"{i}h"
            pred_p = 100.0 + i
            predictions[h] = HorizonPrediction(
                horizon=h,
                step_hours=i,
                current_price=spot_price,
                predicted_price=pred_p,
                predicted_change=pred_p - spot_price,
                predicted_return_pct=float(i),
                confidence_p10=95.0 + i,
                confidence_p50=pred_p,
                confidence_p90=105.0 + i,
                direction="BULLISH",
            )

        df = create_forecast_dataframe(
            predictions=predictions,
            current_time=now,
            current_price=spot_price,
        )

        assert len(df) == 49
        assert df.iloc[0]["Step"] == "0h"
        assert df.iloc[0]["Direction"] == "CURRENT"
        assert df.iloc[48]["Step"] == "48h"
        assert "Price / Prediction" in df.columns
        assert "Prediction Interval [P10, P90]" in df.columns

    def test_csv_export_with_drive_present(self, tmp_path):
        """Asserts export_forecast_to_csv writes to both local directory and Google Drive when mounted."""
        now = pd.Timestamp("2026-11-02 14:00:00", tz="UTC")
        spot_price = 2000.0

        predictions: dict[str, HorizonPrediction] = {
            f"{i}h": HorizonPrediction(
                horizon=f"{i}h",
                step_hours=i,
                current_price=spot_price,
                predicted_price=2000.0 + i,
                predicted_change=float(i),
                predicted_return_pct=0.05 * i,
                confidence_p10=1950.0 + i,
                confidence_p50=2000.0 + i,
                confidence_p90=2050.0 + i,
                direction="BULLISH",
            )
            for i in range(1, 49)
        }

        mock_drive_base = tmp_path / "mock_drive"
        mock_drive_trader = mock_drive_base / "MyDrive" / "Trader"
        mock_drive_trader.mkdir(parents=True)

        local_out = tmp_path / "local_exports"

        with patch("crypto_forecast.web.export.mount_google_drive", return_value=True):
            with patch("crypto_forecast.web.export.Path") as mock_path_cls:
                # Wrap Path to intercept /content/drive/MyDrive/Trader and redirect to mock_drive_trader
                def path_side_effect(*args, **kwargs):
                    p = Path(*args, **kwargs)
                    if str(p).startswith("/content/drive/MyDrive/Trader") or str(p).startswith("\\content\\drive\\MyDrive\\Trader"):
                        sub = str(p).replace("/content/drive/MyDrive/Trader", "").replace("\\content\\drive\\MyDrive\\Trader", "").lstrip("/\\")
                        return mock_drive_trader / sub if sub else mock_drive_trader
                    return p

                mock_path_cls.side_effect = path_side_effect

                local_path, drive_path = export_forecast_to_csv(
                    df_table=predictions,
                    symbol="ETH",
                    output_dir=local_out,
                    current_time=now,
                    current_price=spot_price,
                )

        assert os.path.exists(local_path)
        df_local = pd.read_csv(local_path)
        assert len(df_local) == 49
        assert drive_path is not None
        assert "Trader" in drive_path

    def test_csv_export_with_drive_absent(self, tmp_path):
        """Asserts export_forecast_to_csv succeeds locally and sets drive_csv_path to None without error."""
        local_out = tmp_path / "exports"
        predictions = {
            "1h": HorizonPrediction(
                horizon="1h",
                step_hours=1,
                current_price=100.0,
                predicted_price=101.0,
                predicted_change=1.0,
                predicted_return_pct=1.0,
                confidence_p10=99.0,
                confidence_p50=101.0,
                confidence_p90=103.0,
                direction="BULLISH",
            )
        }

        with patch("crypto_forecast.web.export.mount_google_drive", return_value=False):
            local_path, drive_path = export_forecast_to_csv(
                df_table=predictions,
                symbol="SOL",
                output_dir=local_out,
            )

        assert os.path.exists(local_path)
        assert drive_path is None
        df_local = pd.read_csv(local_path)
        assert len(df_local) == 2  # Step 0h + 1h

    def test_mount_google_drive_headless_and_uninstalled_resilience(self):
        """Asserts mount_google_drive catches ImportError and generic exceptions in headless envs."""
        # Case 1: google.colab not installed
        with patch.dict(sys.modules, {"google.colab": None, "google.colab.drive": None}):
            # Ensure /content/drive/MyDrive does not exist
            with patch("os.path.exists", return_value=False):
                res = mount_google_drive("/content/drive")
                assert res is False

        # Case 2: google.colab installed, but drive.mount raises Exception (headless auth failure)
        mock_colab = MagicMock()
        mock_colab.drive.mount.side_effect = RuntimeError("Headless environment cannot prompt for auth")
        with patch.dict(sys.modules, {"google.colab": mock_colab, "google.colab.drive": mock_colab.drive}):
            with patch("os.path.exists", return_value=False):
                res = mount_google_drive("/content/drive")
                assert res is False

        # Case 3: /content/drive/MyDrive already exists
        with patch("os.path.exists", return_value=True):
            res = mount_google_drive("/content/drive")
            assert res is True

    def test_export_forecast_to_csv_accepts_all_input_formats(self, tmp_path):
        """Asserts export_forecast_to_csv accepts DataFrame, row-lists, or dicts without error."""
        # 1. From DataFrame directly
        df_in = pd.DataFrame([{"Step": "0h", "Price": "$100.00"}, {"Step": "1h", "Price": "$101.00"}])
        p1, _ = export_forecast_to_csv(df_in, symbol="BTC", output_dir=tmp_path / "f1")
        assert os.path.exists(p1)

        # 2. From list of rows directly
        rows_in = [
            ["0h", "2026-11-02 00:00:00 UTC", "$100.00", "0.00%", "CURRENT", "[$100.00, $100.00]"],
            ["1h", "2026-11-02 01:00:00 UTC", "$102.00", "+2.00%", "BULLISH", "[$99.00, $105.00]"],
        ]
        p2, _ = export_forecast_to_csv(rows_in, symbol="ETH", output_dir=tmp_path / "f2")
        assert os.path.exists(p2)
        df_p2 = pd.read_csv(p2)
        assert "Step" in df_p2.columns
        assert len(df_p2) == 2

    def test_export_forecast_creates_nested_directories_automatically(self, tmp_path):
        """Asserts deeply nested output_dir paths are created automatically."""
        deep_dir = tmp_path / "alpha" / "beta" / "gamma" / "exports"
        assert not deep_dir.exists()
        p, _ = export_forecast_to_csv([], symbol="SOL", output_dir=deep_dir)
        assert os.path.exists(p)
        assert deep_dir.exists()

    def test_export_forecast_drive_write_failure_retains_local_csv(self, tmp_path):
        """Asserts that if Google Drive write raises an exception, local CSV is still saved and no error escapes."""
        local_dir = tmp_path / "local"
        mock_drive_trader = tmp_path / "mock_drive" / "Trader"
        mock_drive_trader.mkdir(parents=True)
        original_to_csv = pd.DataFrame.to_csv

        def side_effect_to_csv(self, target, *args, **kwargs):
            target_str = str(target).replace("\\", "/")
            if "Trader" in target_str or "/content/drive" in target_str:
                raise PermissionError("Permission denied on Google Drive")
            return original_to_csv(self, target, *args, **kwargs)

        with patch("crypto_forecast.web.export.mount_google_drive", return_value=True):
            with patch("crypto_forecast.web.export.Path") as mock_path_cls:
                def path_side_effect(*args, **kwargs):
                    p = Path(*args, **kwargs)
                    if "Trader" in str(p):
                        sub = str(p).split("Trader")[-1].lstrip("/\\")
                        return mock_drive_trader / sub if sub else mock_drive_trader
                    return p
                mock_path_cls.side_effect = path_side_effect

                with patch.object(pd.DataFrame, "to_csv", side_effect=side_effect_to_csv, autospec=True):
                    local_path, drive_path = export_forecast_to_csv(
                        [],
                        symbol="BTC",
                        output_dir=local_dir,
                    )

        assert os.path.exists(local_path)
        assert drive_path is None


