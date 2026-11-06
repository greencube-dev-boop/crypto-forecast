# Project: Multi-Horizon Cryptocurrency Price Forecasting System & Colab Web GUI

## Architecture
Modular Python package (`crypto_forecast`) operating under a dual-resolution sampling regime (hourly for intraday $1\text{h}\dots 48\text{h}$; daily for macro $1\text{d}\dots 30\text{d}$), coupled with zero-shot time-series foundation models (TimesFM 2.5, Chronos-2, Chronos-Bolt, Granite TTM), RoBERTa-based crypto sentiment NLP (`ElKulako/cryptobert`), and an interactive Gradio Blocks + Plotly Web GUI deployed directly from Google Colab with zero local artifact footprint.

```
                           +---------------------------------------+
                           |     Free Public Market Data APIs      |
                           |  Binance Global -> Binance US (451)   |
                           |       -> Yahoo Finance Fallback       |
                           +-------------------+-------------------+
                                               |
                                               v
+------------------------+         +-------------------------------+
| Keyless Crypto RSS     |         |  Continuous 48h Resampling    |
| CoinTelegraph/CoinDesk |         |  1h (1h..48h) & 1d (1d..30d)  |
+-----------+------------+         +---------------+---------------+
            |                                      |
            v                                      v
+------------------------+         +-------------------------------+
| ElKulako/cryptobert    |         | Technical Indicators Pipeline |
| Polarity Score & Decay |         | RSI, MACD, BB, ATR, OBV       |
+-----------+------------+         +---------------+---------------+
            |                                      |
            +-------------------+------------------+
                                |
                                v
                   +---------------------------+
                   | BaseForecaster Interface  |
                   | 48 Continuous Horizons:   |
                   | 1h .. 48h (plus legacy    |
                   | 2d, 3d, 7d, 14d, 30d)     |
                   +-------------+-------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
+------------------+    +------------------+    +------------------+
| google/timesfm-  |    | amazon/chronos-2 |    | Baseline         |
| 2.5-200m-pytorch |    | & chronos-bolt   |    | Persistence      |
| (model.model.to) |    | (Co-Candidates)  |    | (Benchmark)      |
+--------+---------+    +--------+---------+    +--------+---------+
         |                       |                       |
         +-----------------------+-----------------------+
                                 |
                                 v
                   +---------------------------+
                   | Quantitative Benchmarking |
                   | MAE, RMSE, sMAPE, Hit-Rate|
                   | Asset-Model Compatibility |
                   +-------------+-------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
+----------------------------------+   +----------------------------------+
| Interactive Colab Web GUI        |   | Google Colab Runner Notebook     |
| Gradio Blocks + Plotly Cones     |   | colab_runner.ipynb               |
| 49-Row Table & CSV Drive Export  |   | Remote GitHub Pull & Colab CLI   |
+----------------------------------+   +----------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Binance Public Klines Ingestion | Keyless REST `/api/v3/klines` 1000-candle paginated retrieval with 6000 wt/min pacing | M1 | Survey 1 |
| 2 | Binance 24hr Ticker & Spot Price | Keyless `/api/v3/ticker/24hr` spot price, VWAP, and volume retrieval | M1 | Survey 1 |
| 3 | Yahoo Finance Fallback Fetcher | Keyless `yfinance` fallback data stream with mandatory `.ffill().bfill()` data sanitization | M1 | Survey 1 |
| 4 | Dual-Resolution Time Series Alignment | Decoupled 1h ($1\text{h}\dots 48\text{h}$) and 1d ($1\text{d}\dots 30\text{d}$) continuous resampling with strict UTC alignment | M1 | Survey 1 |
| 5 | Missing Bar Gap Imputation | Continuous datetime reindexing forward-filling price and zero-filling volume | M1 | Survey 1 |
| 6 | Technical Indicator Suite | Calculation of RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14), and OBV | M1 | Survey 1 |
| 7 | CryptoBERT Model Wrapper | RoBERTa-base classifier (`ElKulako/cryptobert`) generating 3-class logits (Bearish, Neutral, Bullish) | M2 | Survey 1 |
| 8 | Keyless News RSS Ingestion | Real-time news extraction from CoinTelegraph and CoinDesk RSS streams | M2 | Survey 1 |
| 9 | Sentiment Polarity & Exponential Decay | Polarity scoring $s = p_{\text{bullish}} - p_{\text{bearish}}$ with hourly decay $\alpha=0.95$ | M2 | Survey 1 |
| 10 | Sentiment-Adjusted Multi-Horizon Drift | Horizon-attenuated sentiment conviction adjustment $\hat{P}_{t+h}^{\text{adj}} = \hat{P}_{t+h}(1 + \gamma_h \bar{S}_t)$ | M2 | Survey 1 |
| 11 | BaseForecaster Unified Interface | Standardized ABC for multi-horizon forecast extraction, point predictions, and quantiles | M3 | Survey 2 |
| 12 | TimesFM 2.5 Model Integration | `google/timesfm-2.5-200m-pytorch` native 720-step single forward pass and 10-quantile envelope | M3 | Survey 2 |
| 13 | Chronos-2 & Chronos-Bolt Wrappers | `amazon/chronos-2` and `amazon/chronos-bolt-small` with dual-resolution adaptation | M3 | Survey 2 |
| 14 | Granite TTM Forecaster Wrapper | `ibm-granite/granite-timeseries-ttm-r2` multi-horizon predictor | M3 | Survey 2 |
| 15 | Naive Persistence Baseline Model | Reference benchmark $\hat{y}_{t_0+h} = y_{t_0}$ computing baseline error across 11 horizons | M3 | Survey 2 |
| 16 | Quantitative Metric Computations | Vectorized implementations of MAE, RMSE, sMAPE, and Directional Hit-Rate ($\text{DA}_h$) | M4 | Survey 2 |
| 17 | Rolling Walk-Forward Backtester | Systematic rolling-window backtesting engine evaluating models against persistence | M4 | Survey 2 |
| 18 | Asset-to-Model Compatibility Matrix | Quantitative evaluation matrix across 7 assets (BTC, ETH, SOL, BNB, ADA, DOGE, XRP) | M4 | Survey 2 |
| 19 | Headless Benchmark CLI | Standalone CLI script `scripts/run_benchmark.py` generating JSON benchmark artifacts | M4 | Survey 2 |
| 20 | Interactive Gradio Blocks Web GUI | Full-featured web interface supporting crypto dropdown, model selector, horizon controls | M5 | Survey 3 |
| 21 | Plotly Candlestick & Uncertainty Cones | Interactive charting with historical prices and shaded multi-horizon confidence intervals | M5 | Survey 3 |
| 22 | Multi-Horizon Forecast Breakdown Table | Dynamic table rendering target time, forecasted price, delta, directional tag, and hit-rate | M5 | Survey 3 |
| 23 | Accuracy & Skill Scoreboard | Side-by-side metric display comparing candidate model vs persistence baseline | M5 | Survey 3 |
| 24 | Gradio `share=True` Live Tunneling | Public 72-hour `gradio.live` reverse TLS tunnel with zero auth tokens or port forward configs | M5 | Survey 3 |
| 25 | Serialized GPU Inference Queue | `demo.queue(1)` GPU inference queue preventing CUDA OOM on Colab T4 | M5 | Survey 3 |
| 26 | Google Colab 1-Click Runner Notebook | `notebooks/colab_runner.ipynb` with environment check, remote repo sync, and GUI launch | M6 | Survey 3 |
| 27 | Remote GitHub Synchronization | Idempotent `git fetch && git reset --hard origin/main` remote code pull mechanism | M6 | Survey 3 |
| 28 | Colab CLI Execution Workflow | Documented headless execution patterns via `colab exec` and `colab run` | M6 | Survey 3 |
| 29 | Opaque-Box E2E Test Suite (Tiers 1-4) | Comprehensive requirement-driven test suite covering all 11 horizons and components | M7 | Dual Track |
| 30 | Adversarial Coverage Hardening (Tier 5) | White-box stress testing, failure mode probes, and edge-case boundary hardening | M7 | Dual Track |
| 31 | Retrospective Lookback Math Engine | Given target $t$ and horizon $k \in [1, 48]$, slices historical context up to $t - k\text{h}$, generates predictions up to step $k$, discards intermediate steps $1 \dots k-1$, and records step $k$ for time $t$. Pairs ground truth settled close price at time $t$. | M8 | Survey 1 |
| 32 | Causal Temporal Isolation & Zero-Lookahead | Slicing guarantee that context arrays never contain data after $t - k\text{h}$ for lookback $k$. | M8 | Survey 1 |
| 33 | Multi-Model Retrospective Adapter | Unified execution adapter for `TimesFMForecaster`, `ChronosForecaster`, `GraniteTTMForecaster`, and `PersistenceForecaster`. | M8 | Survey 1 |
| 34 | Trajectory Caching Optimization | Batched/cached trajectory generation across cutoff hours with diagonal step extraction to accelerate multi-hour backfills. | M8 | Survey 1, 2 |
| 35 | CSV Schema Definition & Validation | 50-column canonical schema: `real time`, `value at time`, `value predicted 1h before`, ..., `48h before` with shorthand reader compatibility. | M8 | Survey 2 |
| 36 | Multi-Pair File Naming Convention | Strict `{model}_{pair}.csv` naming convention (e.g. `timesfm_BTC_USDT.csv`), yielding `(pair count) * 4` CSV files. | M8 | Survey 2 |
| 37 | Google Drive Persistence & Mounting | Colab Google Drive mount at `/content/drive/MyDrive/Trader/accuracy_eval/` with graceful local directory fallback. | M8 | Survey 2, 3 |
| 38 | Incremental State Detection & Resume | Inspect existing CSV for `max(real time)`, detect newly completed hourly timestamps, skip already recorded timestamps. | M8 | Survey 2, 3 |
| 39 | Idempotent Row Append & Atomic Write | Appends completed rows, deduplicates by `real time`, sorts monotonically, and writes atomically via temporary file. | M8 | Survey 2 |
| 40 | Headless Colab Evaluation Notebook | `notebooks/accuracy_eval.ipynb` 8-cell linear execution flow without GUI (no Gradio/Streamlit). | M8 | Survey 3 |
| 41 | Headless Verification CLI Script | `scripts/verify_accuracy_eval.py` standalone script validating math, schema, and incremental resume headlessly. | M8 | Survey 3 |
| 42 | Automated Unit & Integration Tests | `tests/unit/test_accuracy_eval.py` & `tests/e2e/test_accuracy_eval_pipeline.py` verifying mathematical oracles (persistence oracle, step-aware mock), schema compliance, and incremental state preservation. | M8 | Survey 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Market Data Ingestion & Technical Features | Binance REST `/api/v3/klines`, Yahoo Finance fallback, dual-resolution resampling (1h, 1d), missing data imputation, technical indicators (RSI, MACD, BB, ATR, OBV) | none | DONE |
| M2 | CryptoBERT Sentiment Engine | `ElKulako/cryptobert` Hugging Face model integration, keyless RSS news scraping (CoinTelegraph, CoinDesk), polarity scoring, exponential decay aggregation | M1 | DONE |
| M3 | Multi-Horizon Forecasting Engine | `BaseForecaster` interface, `timesfm-2.5` primary wrapper, `chronos-2`/`chronos-bolt` wrappers, `granite-ttm` wrapper, `persistence` baseline forecaster, sentiment drift adjustment | M1, M2 | DONE |
| M4 | Quantitative Benchmarking & Compatibility | Vectorized metrics (MAE, RMSE, sMAPE, Hit-Rate), walk-forward backtester, multi-asset compatibility matrix across 7 coins (BTC, ETH, SOL, BNB, ADA, DOGE, XRP), `scripts/run_benchmark.py` | M1, M3 | DONE |
| M5 | Interactive Colab Web GUI & Plotly Cones | Gradio Blocks dashboard, Plotly candlestick + shaded forecast cones (`fill='tonexty'`), 11-horizon forecast breakdown table, model accuracy scoreboard, `share=True` tunnel, `demo.queue(1)` | M3, M4 | DONE |
| M6 | Colab Runner Notebook & GitHub Packaging | `notebooks/colab_runner.ipynb` 1-click execution, Colab CLI headless runner, remote repo sync from `greencube-dev-boop`, `pyproject.toml`, `requirements.txt`, README | M1, M2, M3, M4, M5 | DONE |
| M7 | Final Milestone: E2E Verification & Hardening | 100% pass of E2E Test Suite published in `TEST_READY.md` (Tiers 1-4), followed by Tier 5 adversarial coverage hardening | M6, TEST_READY.md | DONE |
| M8 | Retrospective Accuracy Evaluation Pipeline | Implement `src/crypto_forecast/eval/` (`schemas.py`, `retrospective.py`, `dataset.py`, `pipeline.py`), `notebooks/accuracy_eval.ipynb`, `scripts/verify_accuracy_eval.py`, unit and e2e test suites | M1, M3, M4 | DONE |

## Interface Contracts

### M1 ↔ M3 / M4 (Market Data & Features)
- **Functions**:
  - `fetch_ohlcv(symbol: str, interval: str = '1h', limit: int = 1000) -> pd.DataFrame`
    - Returns DataFrame with columns `['timestamp', 'open', 'high', 'low', 'close', 'volume']` with continuous DatetimeIndex.
  - `get_dual_resolution_data(symbol: str, hourly_limit: int = 1000, daily_limit: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]`
    - Returns `(df_hourly, df_daily)` properly sanitized and normalized.
  - `add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame`
    - Adds columns: `rsi_14`, `macd`, `macd_signal`, `macd_hist`, `bb_upper`, `bb_middle`, `bb_lower`, `atr_14`, `obv`.

### M2 ↔ M3 (CryptoBERT Sentiment)
- **Functions**:
  - `fetch_latest_crypto_news() -> list[dict]`
    - Returns list of `{'title': str, 'published': datetime, 'source': str}`.
  - `analyze_sentiment(texts: list[str]) -> list[dict]`
    - Returns list of `{'bearish': float, 'neutral': float, 'bullish': float, 'polarity': float}`.
  - `get_sentiment_series(news_items: list[dict], target_timestamps: pd.DatetimeIndex, decay_factor: float = 0.95) -> pd.Series`
    - Returns hourly continuous polarity series in $[-1.0, +1.0]$.

### M3 ↔ M4 / M5 (Multi-Horizon Forecasting Engine)
- **Classes & Methods**:
  - `class BaseForecaster(ABC)`:
    - `forecast(context_prices: list[float] | np.ndarray, horizons: list[str] = DEFAULT_HORIZONS) -> dict[str, HorizonPrediction]`
    - Where `HorizonPrediction` is a dataclass containing:
      - `horizon: str` (e.g. `'1h'`, `'24h'`, `'7d'`, `'30d'`)
      - `step_hours: int` (e.g. 1, 24, 168, 720)
      - `predicted_price: float`
      - `current_price: float`
      - `predicted_change: float`
      - `predicted_return_pct: float`
      - `direction: str` (`'BULLISH'` or `'BEARISH'`)
      - `confidence_p10: float`
      - `confidence_p50: float`
      - `confidence_p90: float`
    - `get_full_trajectory(context_prices: np.ndarray, max_horizon_steps: int = 720) -> tuple[np.ndarray, np.ndarray]`
      - Returns `(point_predictions, quantile_predictions)` for continuous chart plotting.

### M4 ↔ M5 / M6 (Benchmarking & Evaluation)
- **Functions & Schemas**:
  - `evaluate_model_performance(model: BaseForecaster, df_hourly: pd.DataFrame, horizons: list[str]) -> pd.DataFrame`
    - Returns summary table with columns: `['horizon', 'mae_model', 'mae_persist', 'rmse_model', 'rmse_persist', 'hit_rate_pct', 'skill_mae_pct', 'skill_rmse_pct']`.
  - `run_asset_compatibility_analysis(models: list[BaseForecaster], symbols: list[str]) -> pd.DataFrame`
    - Returns cross-asset performance matrix for all tested coins.

### M8 (Retrospective Accuracy Evaluation)
- **Schemas (`crypto_forecast.eval.schemas`)**:
  - `ACCURACY_EVAL_COLUMNS: Final[list[str]]` (50 columns: `real time`, `value at time`, `value predicted 1h before` .. `48h before`)
  - `get_accuracy_csv_filename(model: Any, pair: str) -> str` (`{model}_{pair}.csv`)
  - `canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `validate_accuracy_df_schema(df: pd.DataFrame, strict: bool = True) -> tuple[bool, list[str]]`
- **Retrospective Engine (`crypto_forecast.eval.retrospective`)**:
  - `evaluate_retrospective_horizon(model, df_hourly, target_timestamp, k, context_window, min_context) -> float`
  - `compute_retrospective_row(model, df_hourly, target_timestamp, max_horizon, context_window, min_context) -> dict[str, Any]`
  - `evaluate_historical_range_cached(model, df_hourly, target_timestamps, max_horizon, context_window, min_context) -> pd.DataFrame`
- **Dataset Persistence (`crypto_forecast.eval.dataset`)**:
  - `get_accuracy_eval_dir(output_dir) -> Path` (`/content/drive/MyDrive/Trader/accuracy_eval/` with local fallback)
  - `inspect_latest_recorded_timestamp(csv_path) -> pd.Timestamp | None`
  - `identify_pending_timestamps(df_hourly, latest_ts, min_context_window, max_lookback_horizon, initial_backfill_hours) -> list[pd.Timestamp]`
  - `persist_accuracy_dataset(df_new, csv_path) -> pd.DataFrame`
- **Pipeline Coordinator (`crypto_forecast.eval.pipeline`)**:
  - `run_accuracy_evaluation(pairs, models, output_dir, initial_backfill_hours, max_horizon, context_window, force_fallback, df_ohlcv_map, verbose) -> dict[str, Path]`

## Code Layout
```
d:\code\trade\
├── .github/
│   └── workflows/
│       └── ci.yml                 # Remote CI workflow
├── notebooks/
│   ├── colab_runner.ipynb         # 1-Click Colab notebook with "Open in Colab" badge
│   └── accuracy_eval.ipynb        # Clean headless Colab accuracy evaluation notebook
├── src/
│   └── crypto_forecast/
│       ├── __init__.py
│       ├── config.py              # Constants, 48 horizons, model names, default pairs
│       ├── constants.py           # Re-exported constants module
│       ├── ui/
│       │   └── __init__.py        # Re-exporting web interface alias
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py       # Binance REST (Global/US) & Yahoo Finance fetchers
│       │   ├── preprocessor.py    # Dual-resolution resampling & gap imputation
│       │   └── indicators.py      # RSI, MACD, Bollinger Bands, ATR, OBV
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py            # BaseForecaster ABC and HorizonPrediction dataclass
│       │   ├── timesfm_model.py   # google/timesfm-2.5-200m-pytorch wrapper (model.model.to)
│       │   ├── chronos_model.py   # amazon/chronos-2 & chronos-bolt-small wrappers
│       │   ├── granite_model.py   # ibm-granite/granite-timeseries-ttm-r2 wrapper
│       │   ├── baseline.py        # Persistence & naive drift baseline
│       │   └── sentiment.py       # ElKulako/cryptobert RoBERTa model & RSS scraper
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── metrics.py         # Vectorized MAE, RMSE, sMAPE, Hit-Rate
│       │   ├── backtester.py      # Rolling walk-forward backtesting engine
│       │   └── compatibility.py   # Asset-to-model compatibility evaluator
│       ├── eval/
│       │   ├── __init__.py        # Accuracy evaluation exports
│       │   ├── schemas.py         # 50-col schema, aliases, file naming
│       │   ├── retrospective.py   # 48h lookback math, step k extraction, caching
│       │   ├── dataset.py         # Drive persistence, incremental state resume
│       │   └── pipeline.py        # Multi-pair multi-model evaluation coordinator
│       └── web/
│           ├── __init__.py
│           ├── app.py             # Gradio Blocks UI entry point with CSV download
│           ├── components.py      # UI control components & 49-row tables
│           ├── charts.py          # Plotly candlestick & shaded forecast cone charts
│           └── export.py          # CSV generation and Google Drive mount/export
├── scripts/
│   ├── run_benchmark.py           # Headless benchmark CLI runner
│   ├── run_forecast.py            # Headless forecasting CLI runner
│   ├── launch_gui.py              # Web GUI launcher (Gradio share=True)
│   └── verify_accuracy_eval.py    # Standalone headless accuracy verification script
├── tests/
│   ├── unit/                      # Fast local/Colab unit tests
│   │   ├── test_ingestion.py
│   │   ├── test_indicators.py
│   │   ├── test_sentiment.py
│   │   ├── test_models.py
│   │   ├── test_metrics.py
│   │   ├── test_web.py
│   │   └── test_accuracy_eval.py  # Unit tests for M8 accuracy evaluation
│   └── e2e/                       # Full-system opaque-box integration tests
│       ├── test_multi_horizon.py
│       ├── test_compatibility.py
│       ├── test_colab_runner.py
│       └── test_accuracy_eval_pipeline.py  # E2E tests for notebook & pipeline
├── pyproject.toml
├── requirements.txt
├── README.md
├── PROJECT.md
├── TEST_INFRA.md
└── TEST_READY.md                  # Published when E2E test suite is complete
```
