# E2E Test Infra: Multi-Horizon Cryptocurrency Price Forecasting System

## Test Philosophy
- **Opaque-Box & Requirement-Driven**: Tests derive strictly from user requirements in `ORIGINAL_REQUEST.md`, treating the system as an external consumer (evaluating API calls, prediction schemas, metric bounds, and GUI accessibility) without relying on internal private implementations.
- **Methodology**: Systematic 4-tier testing hierarchy (Category-Partition, Boundary Value Analysis, Pairwise Combinatorial, and Real-World Application Scenarios), followed by Tier 5 Adversarial Coverage Hardening.

## Feature Inventory & Test Coverage Matrix
| # | Feature | Source (Requirement) | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Scenario) |
|---|---------|----------------------|:-----------------:|:-----------------:|:----------------------:|:-----------------:|
| 1 | Binance Public Klines Ingestion | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 2 | Binance 24hr Ticker & Spot Price | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 3 | Yahoo Finance Fallback Fetcher | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 4 | Dual-Resolution Alignment (1h, 1d) | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 5 | Missing Bar Gap Imputation | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 6 | Technical Indicators (RSI, MACD, BB, ATR) | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 7 | CryptoBERT Sentiment Classifier | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 8 | Keyless News RSS Stream Ingestion | ORIGINAL_REQUEST §R1/R2 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 9 | Sentiment Polarity & Exponential Decay | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 10 | Sentiment-Adjusted Drift Modification | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 11 | BaseForecaster Unified Interface | ORIGINAL_REQUEST §R1 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 12 | TimesFM 2.5 Model Engine (Primary) | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 13 | Chronos-2 & Chronos-Bolt Engines | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 14 | Granite TTM Forecaster Engine | ORIGINAL_REQUEST Context | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 15 | Naive Persistence Baseline Forecaster | ORIGINAL_REQUEST §R2 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 16 | Quantitative Metrics (MAE, RMSE, Hit-Rate)| ORIGINAL_REQUEST §R2 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 17 | Rolling Walk-Forward Backtester | ORIGINAL_REQUEST §R2 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 18 | Asset-to-Model Compatibility Matrix | ORIGINAL_REQUEST §R2 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 19 | Headless Benchmark CLI Execution | ORIGINAL_REQUEST §R2/R3 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 20 | Interactive Gradio Blocks Web GUI | ORIGINAL_REQUEST §R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 21 | Plotly Candlestick & Forecast Cones | ORIGINAL_REQUEST §R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 22 | Multi-Horizon Forecast Breakdown Table | ORIGINAL_REQUEST §R1/R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 23 | Accuracy & Skill Scoreboard Display | ORIGINAL_REQUEST §R2/R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 24 | Gradio `share=True` Tunneling | ORIGINAL_REQUEST §R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 25 | Serialized GPU Inference Queue | ORIGINAL_REQUEST §R4 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 26 | Google Colab 1-Click Runner Notebook | ORIGINAL_REQUEST §R3 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 27 | Remote GitHub Code Pull Workflow | ORIGINAL_REQUEST §R3 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |
| 28 | Colab CLI Headless Execution | ORIGINAL_REQUEST §R3 | ≥ 5 tests | ≥ 5 tests | ✓ | ✓ |

## Test Architecture
- **Test Runner**: Pytest (`pytest tests/e2e/ -v`)
- **Execution Environment**: Colab CPU/T4 runtime & zero-local host execution
- **Test Case Format**: Parameterized Pytest fixtures with deterministic test vectors and schema validators
- **Directory Layout**:
  - `tests/unit/`: Fast unit tests per module boundary
  - `tests/e2e/test_multi_horizon.py`: Complete validation of 11 horizons across models
  - `tests/e2e/test_compatibility.py`: Asset-to-model compatibility validation
  - `tests/e2e/test_colab_runner.py`: Colab notebook structure, dependencies, and web launch checks

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Coins | Complexity | Acceptance Signal |
|---|----------|-------------------|--------------|------------|-------------------|
| 1 | Intraday Momentum Breakout Forecast | F1, F4, F6, F11, F12, F15, F16 | BTC, ETH | High | 1h, 2h, 4h, 6h, 12h, 24h predictions generated; MAE < persistence |
| 2 | Macro Regime Shift & Swing Trend | F1, F3, F4, F6, F12, F13, F18 | SOL, BNB | High | 2d, 3d, 7d, 14d, 30d forecast cones rendered with confidence intervals |
| 3 | Event-Driven Social Shock with Sentiment | F1, F7, F8, F9, F10, F12, F18 | DOGE, XRP | High | News parsed, polarity score computed, drift modified, hit-rate evaluated |
| 4 | Offline / Exchange Fallback Transition | F1, F3, F4, F5, F11, F12 | BTC, ADA | Medium | Simulated Binance 429 triggers transparent Yahoo Finance fallback with 0 NaNs |
| 5 | Full Remote Colab Web GUI Deployment | F20, F21, F22, F23, F24, F25, F26 | All 7 coins | High | Gradio app instantiates, generates share link, processes queued request |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: ≥ 5 test cases per feature (Happy-path isolation)
- **Tier 2 (Boundary & Corner Cases)**: ≥ 5 test cases per feature (Extreme limits, empty responses, zero volume, flat price, NaN recovery)
- **Tier 3 (Cross-Feature Combinations)**: Pairwise coverage across data, sentiment, models, and UI
- **Tier 4 (Real-World Scenarios)**: ≥ 5 comprehensive end-to-end user workflows
- **Tier 5 (Adversarial Hardening)**: White-box stress testing and adversarial gap resolution
