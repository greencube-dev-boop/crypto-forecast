"""Crypto Forecast - Multi-Horizon Cryptocurrency Price Forecasting System.

A modular framework providing:
- Keyless public market data ingestion (Binance REST, Yahoo Finance)
- Dual-resolution resampling (1h intraday, 1d macro) & gap imputation
- Pure vectorized technical indicator pipeline (RSI, MACD, Bollinger Bands, ATR, OBV)
- Multi-horizon forecasting with foundation models & CryptoBERT sentiment
- Quantitative walk-forward benchmarking & asset-model compatibility analysis
- Interactive Colab Web GUI (Gradio Blocks + Plotly uncertainty cones)
"""

__version__ = "0.1.0"
