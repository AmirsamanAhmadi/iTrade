# Architecture (Mermaid) 🔧

This page shows a simple component diagram of the project.

```mermaid
flowchart LR
  UI["Streamlit UI\n(ui/streamlit/app.py)"] --> API["FastAPI API\n(backend.api.app)"]
  API --> Services["Services (services/*, backend/*)"]
  Services --> MD["MarketDataService\n(services/market_data.py)\nuses yfinance, caches to db/market_cache"]
  Services --> NS["NewsService\n(services/news_service.py)\noptionally uses finviz"]
  Services --> RM["RiskManager\n(backtesting/risk_manager.py or risk/risk_manager.py)"]
  Services --> STR["Strategy modules\n(strategy/*)"]
  Services --> EXEC["Execution / Broker\n(execution/broker.py)"]

  MD --> DB["db/ (market_cache CSVs)"]
  NS --> NEWS["db/news/*.jsonl"]
  API --> Logs["logs/ (app logs)"]
  UI -->|user actions| Clients["Operator / Tester"]
  Clients --> API

  subgraph Tests
    TM["tests/*"]
  end
  TM --> Services
```

### Notes
- The `MarketDataService` caches OHLC CSVs under `db/market_cache` and normalizes timestamps to UTC. ⚡
- The `NewsService` stores raw JSON lines under `db/news` when finviz is available. 🗞️
- The repo includes a Streamlit UI for monitoring and a FastAPI backend for programmatic access. 🎛️
