# UI & Control API

This document describes the small FastAPI endpoints used by the Streamlit control panel. Endpoints are intentionally lightweight and safe — config updates are UI-only and the service will never send live trade orders.

## Authentication
- Config update endpoints (POST `/ui/state`) are protected via HTTP Basic Auth.
- Provide credentials via environment variables:
  - `UI_API_USER` (default: `admin`)
  - `UI_API_PASS` (default: `admin`)

> Note: this is a local/basic credential mechanism intended for development and local control. For production, use a hardened auth mechanism (OAuth, mTLS, etc.).

## Endpoints
- GET `/health` — health check
- GET `/ui/state` — returns last saved UI state (204 if none)
- POST `/ui/state` — update UI state (requires basic auth)
  - Payload schema (JSON):
    - `system_on` (bool)
    - `mode` ("Paper"|"Live")
    - `risk_per_trade_pct` (float, 0.0-5.0)
    - `max_daily_loss_pct` (float, 0.0-100.0)
    - `max_drawdown_pct` (float, 0.0-100.0)
    - `news_lock` (bool)
    - `kill_switch` (bool)

- GET `/ui/sentiment` — latest news signal snapshot (204 if none)
- GET `/ui/backtest/latest` — last backtest result (204 if none)

## Redis sync (optional)
- If you set `REDIS_URL`, the API will attempt to persist the UI state to Redis under key `UI_REDIS_KEY` (default `forexbot:ui_state`), falling back to file storage if Redis or the key is not available.

## Examples
Save UI state (with basic auth):

curl -u admin:admin -X POST http://localhost:8000/ui/state -H 'Content-Type: application/json' -d '{"system_on": true, "mode": "Paper", "risk_per_trade_pct": 1.0, "max_daily_loss_pct": 2.0, "max_drawdown_pct": 20.0, "news_lock": false, "kill_switch": false}'

Get sentiment:

curl http://localhost:8000/ui/sentiment
