# Run examples & Quickstart ✅

## 1. Setup (macOS)

```bash
# create and activate venv
python3 -m venv .venv
. .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> Tip: the repository includes a convenience Makefile. You can run `make install` to run `scripts/setup_env.sh`.

---

## 2. Run the API (development)

```bash
# using Makefile
make run-api

# or directly
uvicorn backend.api.app:app --reload
```

Open http://127.0.0.1:8000/health to confirm service is up.

---

## 3. Run the UI (Streamlit)

```bash
# using Makefile
make run-ui

# or directly
streamlit run ui/streamlit/app.py
```

The Streamlit control panel will open in your browser.

---

## 4. Run tests

```bash
make test
# or
pytest -q
```

---

## 5. Example: use MarketDataService from Python

A quick one-liner to fetch OHLC data (will contact yfinance):

```bash
python -c "from services.market_data import MarketDataService; m=MarketDataService(); print(m.fetch_ohlc('GC=F','1h',start='2025-01-01',end='2025-01-03').head())"
```

Notes:
- Interacting with `MarketDataService` requires network access and can be cached to `db/market_cache`.
- `NewsService` optionally uses `finvizfinance`. If not installed, it returns an empty list and logs a warning.

---

## 6. Useful paths & commands

- API: `backend/api/app.py` (FastAPI)
- UI: `ui/streamlit/app.py` (Streamlit)
- Market data: `services/market_data.py` (yfinance)
- News: `services/news_service.py` (finviz optional)
- Execution / Paper trading: `backend/execution/trading.py` (engine), `backend/execution/broker.py` (simulator)
- Makefile:
  - `make run-api`
  - `make run-ui`
  - `make test`

### 7. Paper trading examples

Run a small example that detects a setup and simulates a paper trade:

```bash
python examples/paper_trade_example.py
```

Or run the Risk-managed example which integrates `RiskEngine` to enforce per-trade and daily limits:

```bash
python examples/paper_trade_with_risk.py
```

These demonstrate entry sizing, stop placement, risk engine enforcement, and stop execution.

---

### 8. Risk Engine (capital survival) ⚠️

The repo contains an explainable `RiskEngine` (`backend/risk/risk_manager.py`) that:
- Enforces per-trade risk as a percentage of current balance
- Tracks daily P&L and kills trading if max daily loss is reached
- Tracks peak-to-trough drawdown and triggers a kill-switch if breached
- Tracks consecutive losses (configurable) and triggers a kill-switch
- Provides `approve_trade(proposed_risk)` and `on_trade_open`/`on_trade_closed(pnl)` hooks for integration

Use `examples/paper_trade_with_risk.py` to see how `RiskEngine` integrates with the `PaperTrader` engine.


---

### 7. News → Signal (explainable pipeline) 📡

Goal: convert raw headlines into an explainable per-currency sentiment signal.

Key steps implemented in `services/news_signal.py`:
- **Clean** headline text (lowercase, remove punctuation)
- **Score** using a small, transparent lexicon (word weights)
- **Recency decay** (exponential, configurable half-life)
- **Impact weighting** (simple heuristics for strong verbs and macro words)
- **Aggregate** per currency (USD, EUR) and store daily snapshots under `db/news_signals/`

Run the example script:

```bash
python examples/news_signal_example.py
```

This reads recent `db/news/news_*.jsonl` files and appends a snapshot to `db/news_signals/news_signals_YYYY-MM-DD.jsonl`.

---

If you'd like, I can also:
- Add a runnable example script like `examples/fetch_market_data.py` ✅
- Add a Swagger/OpenAPI snippet describing endpoints 🔧
- Expand docs into mkdocs/site structure for a hosted docs site 🌐

Which would you like next?