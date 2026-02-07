# Forex Bot — Comprehensive Documentation 📘

This document provides a detailed overview of the Forex Bot system, its architecture, components, and how to operate it.

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
    - [Market Data Service](#market-data-service)
    - [News & Sentiment Service](#news--sentiment-service)
    - [Strategy Module (Bias & Setups)](#strategy-module)
    - [Backtesting Engine](#backtesting-engine)
    - [Risk Management](#risk-management)
    - [Machine Learning Pipeline](#machine-learning-pipeline)
4. [How the System Works (The Trading Flow)](#how-the-system-works)
5. [User Interface (Streamlit)](#user-interface)
6. [API Reference](#api-reference)
7. [Getting Started](#getting-started)

---

## Introduction

The **Forex Bot** is a defensive, risk-first trading system designed for small-capital forex trading. It doesn't just look at price action; it incorporates multi-timeframe analysis, news sentiment, and machine learning to select high-probability trades while maintaining strict risk controls.

## Architecture Overview

The system follows a modular architecture:
- **Backend**: Core logic for strategy, risk, execution, and ML.
- **Services**: External data integration (Market data, News).
- **UI**: Streamlit-based control panel for monitoring and manual intervention.
- **API**: FastAPI-based service for programmatic interaction.
- **DB**: Local storage for market data cache, news, and ML models.

## Core Components

### Market Data Service
- **File**: `backend/services/market_data.py` (and `services/market_data.py`)
- **Function**: Fetches OHLC data from Yahoo Finance via `yfinance`.
- **Caching**: Data is cached in `db/market_cache` as CSV files to reduce API calls and improve performance.

### News & Sentiment Service
- **Files**: `services/news_service.py`, `services/news_signal.py`
- **Function**:
    - `NewsService`: Scrapes or fetches headlines from Finviz.
    - `NewsSignalService`: Parses headlines, maps them to relevant currencies, and computes a sentiment score (Positive/Negative/Neutral).
    - **News Lock**: A safety feature that can block new trades if significant news events are detected or if the user manually engages it.

### Strategy Module
Located in `backend/strategy/`, this is the "brain" of the bot.

#### 1. Bias Detection (`bias.py`)
Determines the "Market Bias" (LONG, SHORT, or NEUTRAL).
- **4H Timeframe**: Detects the primary trend using Moving Average (MA) crossovers and slope analysis.
- **1H Timeframe**: Confirms the 4H trend. If the 1H momentum doesn't match the 4H trend, the bias remains NEUTRAL.
- **Range Detection**: If the market is moving sideways (low volatility over a lookback period), it sets bias to NEUTRAL to avoid "choppy" trades.

#### 2. Setup Detection (`setups.py`)
Identifies specific entry signals within the biased direction.
- **EMA Pullback**: In a trending market, it waits for the price to "pull back" to the 50-period EMA and show a bounce.
- **Break & Retest**: Identifies a breakout of a previous support/resistance level followed by a successful retest and bounce.

### Backtesting Engine
- **File**: `backend/backtesting/engine.py`
- **Function**: Runs the strategy over historical data. It accounts for:
    - **Slippage**: Simulated spread and execution delay.
    - **Equity Curve**: Tracks the account balance over time.
    - **Metrics**: Computes Sharpe ratio, Win Rate, Drawdown, etc.

### Risk Management
- **File**: `backend/risk/risk_manager.py`
- **Function**: The `RiskEngine` is the final gatekeeper.
    - **Position Sizing**: Calculates the lot size based on a fixed percentage risk (e.g., 1% of equity) and the distance to the stop loss.
    - **Hard Stops**: Enforces max daily loss and max drawdown limits.

### Machine Learning Pipeline
- **File**: `backend/ml/`
- **Function**: Uses a lightweight Logistic Regression model (NumPy-based) to predict trade success.
    - **Training**: It extracts features from previous trades (setup type, direction, pullback depth, risk ratio) and trains a model to predict if a trade will be profitable.
    - **Inference**: During live/backtesting, the model provides a "confidence score". Trades with low confidence can be skipped automatically.

---

## How the System Works (The Trading Flow)

1.  **Sync Data**: The bot fetches the latest OHLC and news.
2.  **Analyze Sentiment**: `NewsSignalService` updates the sentiment dashboard.
3.  **Check Bias**: `compute_bias()` determines if we are looking for LONGs or SHORTs.
4.  **Scan Setups**: `detect_setup()` looks for a valid EMA Pullback or Break & Retest.
5.  **Risk Check**: If a setup is found, the `RiskEngine` verifies if we have enough "risk budget" left.
6.  **ML Filter**: (Optional) The ML model provides a confidence score for the specific setup.
7.  **Execution**: If all checks pass, a trade is "opened" with a defined Entry, Stop Loss, and Take Profit.

---

## User Interface (Streamlit)

Run it with: `streamlit run ui/streamlit/app.py`

Features:
- **System Controls**: Enable/Disable the bot, toggle between Paper/Live modes, and set risk parameters.
- **Sentiment Dashboard**: Real-time visualization of currency sentiment.
- **Gold (XAU) Panel**: Specific monitoring for Gold prices and news.
- **Trade Log**: View current and historical (simulated) trades.
- **Backtest Runner**: Generate and view backtest results directly in the UI.

## API Reference

Run it with: `uvicorn backend.api.app:app --reload`

Key Endpoints:
- `GET /`: Health check and system status.
- `GET /state`: Current UI settings (risk, mode, etc.).
- `GET /sentiment`: Latest news sentiment scores.

---

## Getting Started

1.  **Environment**:
    - Copy `.env.example` to `.env`.
    - Run `./scripts/setup_env.sh` to install dependencies.
2.  **Running a Simulation**:
    - Open the Streamlit UI.
    - Click "Generate example backtest" to see the system in action.
3.  **Training ML**:
    - Use the examples in `examples/` to run a backtest and save the results.
    - Use `backend.ml.trainer` to train a model on those results.

---
_Note: This system is designed for educational and paper-trading purposes. Always exercise caution when trading live markets._
