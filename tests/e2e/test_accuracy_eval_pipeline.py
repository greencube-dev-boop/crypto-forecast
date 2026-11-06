"""End-to-End integration tests for Retrospective Accuracy Evaluation Pipeline and Colab Notebook."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import pytest

from crypto_forecast.eval.pipeline import run_accuracy_evaluation
from crypto_forecast.eval.schemas import (
    ACCURACY_EVAL_COLUMNS,
    REAL_TIME_COL,
    VALUE_AT_TIME_COL,
    validate_accuracy_df_schema,
)


def make_test_ohlcv(n_bars: int = 100, start_price: float = 1000.0, step: float = 10.0) -> pd.DataFrame:
    """Generates synthetic hourly OHLCV DataFrame."""
    base_ts = pd.Timestamp("2026-11-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(base_ts, periods=n_bars, freq="1h")
    closes = [start_price + step * i for i in range(n_bars)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 2.0 for c in closes],
            "low": [c - 2.0 for c in closes],
            "close": closes,
            "volume": [100.0] * n_bars,
        },
        index=timestamps,
    )


# ==============================================================================
# 1. Notebook Architecture & Headless Integrity Tests
# ==============================================================================


class TestAccuracyEvalNotebookStructure:
    @pytest.fixture
    def notebook_json(self) -> dict:
        repo_root = Path(__file__).resolve().parents[2]
        nb_path = repo_root / "notebooks" / "accuracy_eval.ipynb"
        assert nb_path.exists(), f"Notebook file not found at {nb_path}"
        with open(nb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_notebook_schema_and_metadata(self, notebook_json: dict) -> None:
        """Validates that accuracy_eval.ipynb is valid nbformat v4."""
        assert notebook_json.get("nbformat") >= 4
        assert "cells" in notebook_json
        assert len(notebook_json["cells"]) == 8

        # First cell is markdown, remaining 7 are code
        assert notebook_json["cells"][0]["cell_type"] == "markdown"
        for i in range(1, 8):
            assert notebook_json["cells"][i]["cell_type"] == "code"

    def test_notebook_strictly_no_gui(self, notebook_json: dict) -> None:
        """Mandatory requirement: Colab accuracy evaluation notebook has 0 GUI dependencies."""
        prohibited_terms = [
            "gradio",
            "streamlit",
            "create_app",
            "demo.launch",
            "share=True",
            "import dash",
            "flask",
        ]
        all_sources = []
        for cell in notebook_json["cells"]:
            source = "".join(cell.get("source", []))
            all_sources.append(source)

        combined_text = "\n".join(all_sources).lower()
        for term in prohibited_terms:
            assert term.lower() not in combined_text, (
                f"Prohibited GUI term '{term}' found in accuracy_eval.ipynb! "
                f"Notebook must be strictly headless."
            )

    def test_notebook_drive_mount_path(self, notebook_json: dict) -> None:
        """Asserts Google Drive persistence path is configured."""
        all_sources = "".join("".join(c.get("source", [])) for c in notebook_json["cells"])
        assert "/content/drive/MyDrive/Trader/accuracy_eval" in all_sources

    def test_notebook_github_remote_sync(self, notebook_json: dict) -> None:
        """Asserts presence of greencube-dev-boop/crypto-forecast GitHub synchronization."""
        all_sources = "".join("".join(c.get("source", [])) for c in notebook_json["cells"])
        assert "greencube-dev-boop/crypto-forecast" in all_sources


# ==============================================================================
# 2. End-to-End Pipeline Multi-Pair Multi-Model Tests
# ==============================================================================


class TestFullAccuracyEvaluationPipeline:
    def test_multi_pair_multi_model_execution(self, tmp_path: Path) -> None:
        """Full end-to-end execution across 2 pairs and 4 models yielding 8 CSV files."""
        df_btc = make_test_ohlcv(n_bars=100, start_price=50000.0, step=50.0)
        df_eth = make_test_ohlcv(n_bars=100, start_price=3000.0, step=10.0)
        data_map = {"BTC/USDT": df_btc, "ETH/USDT": df_eth}

        pairs = ["BTC/USDT", "ETH/USDT"]
        models = ["timesfm", "chronos", "granite", "persistence"]

        # Run 1: Initial creation (4 hours backfill)
        results_1 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=4,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map=data_map,
            verbose=False,
        )

        assert len(results_1) == 8, f"Expected 8 CSV files (2 pairs * 4 models), got {len(results_1)}"

        # Validate each CSV file
        for key, csv_path in results_1.items():
            assert csv_path.exists(), f"CSV file {csv_path} does not exist"
            df = pd.read_csv(csv_path)

            # Check 50 columns
            is_valid, errors = validate_accuracy_df_schema(df, strict=True)
            assert is_valid, f"Schema invalid for {key}: {errors}"
            assert len(df) == 4

            # Check numeric integrity (no NaNs or Infs)
            numeric_cols = [c for c in df.columns if c != REAL_TIME_COL]
            assert not df[numeric_cols].isna().any().any(), f"Found NaNs in {key}"
            assert not np.isinf(df[numeric_cols].values).any(), f"Found Infs in {key}"

            # Check timestamp ordering and uniqueness
            ts_series = pd.to_datetime(df[REAL_TIME_COL], utc=True)
            assert ts_series.is_monotonic_increasing
            assert ts_series.is_unique

        # Run 2: Idempotent re-run with zero new hours
        results_2 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=4,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map=data_map,
            verbose=False,
        )

        for key, csv_path in results_2.items():
            df_rerun = pd.read_csv(csv_path)
            assert len(df_rerun) == 4, "Idempotent re-run added duplicate rows!"

        # Run 3: Advance data by 3 hours
        df_btc_advanced = make_test_ohlcv(n_bars=103, start_price=50000.0, step=50.0)
        df_eth_advanced = make_test_ohlcv(n_bars=103, start_price=3000.0, step=10.0)
        advanced_data_map = {"BTC/USDT": df_btc_advanced, "ETH/USDT": df_eth_advanced}

        results_3 = run_accuracy_evaluation(
            pairs=pairs,
            models=models,
            output_dir=tmp_path,
            initial_backfill_hours=4,
            max_horizon=48,
            context_window=32,
            force_fallback=True,
            df_ohlcv_map=advanced_data_map,
            verbose=False,
        )

        for key, csv_path in results_3.items():
            df_advanced = pd.read_csv(csv_path)
            assert len(df_advanced) == 7, f"Expected 7 rows (4 + 3) after advancing 3 hours, got {len(df_advanced)}"
            # Verify original 4 rows were not altered
            assert np.isclose(df_advanced.iloc[0]["value at time"], df_btc.iloc[-4]["close"] if "BTC" in key else df_eth.iloc[-4]["close"])
