# Forex Bot — Defensive Trading System

Small-capital, risk-first forex trading system scaffold.

Structure:
- backend: core services, strategy, risk, execution, backtesting
- ui: Streamlit control & monitoring
- configs: YAML/config files
- logs: runtime logs
- db: historical data & results

Getting started:
1. Copy `.env.example` to `.env` and edit.
2. Run `./scripts/setup_env.sh` to create venv and install deps.
3. Start API: `uvicorn backend.api.app:app --reload` (dev)
4. Start UI: `streamlit run ui/streamlit/app.py`

Defaults: paper-mode, EURUSD initial universe, strict risk controls.

## Market & News Services (quick usage)

Example: fetch 15m EURUSD OHLC via `MarketDataService`:

```python
from backend.services import MarketDataService
svc = MarketDataService()
df15 = svc.fetch_ohlc('EURUSD=X', '15m', start='2024-01-01', end='2024-02-01')
print(df15.head())
```

Example: fetch headlines (finviz) via `NewsService`:

```python
from backend.services import NewsService
ns = NewsService()
news = ns.fetch_headlines(['EURUSD'])
print(news)
```

**Docs:** See `docs/index.md` for architecture, Mermaid diagrams, and run examples (API, UI, tests).

## Machine learning & model registry

This project includes a lightweight, explainable ML pipeline for improving setup selection. Key points:

- Train a simple logistic model on labeled backtest trades using `backend.ml.trainer`.
- Persist trained models via the registry under `db/models/<name>/` using `backend.ml.registry.save_model`.
- Load a model and plug it into the event-driven backtester via the `classifier` arg in `BacktestEngine` (sets `confidence_threshold` to skip low-confidence setups).

See `docs/ml.md` for details and example code.
