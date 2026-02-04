from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime

from backend.api.schemas import UIState
from backend.api.auth import get_basic_credentials
from backend.api.state import get_ui_state, set_ui_state
from services.news_signal import NewsSignalService
from pathlib import Path
import json

app = FastAPI(title="Forex Bot API")

@app.get("/health")
async def health():
    return {"status": "ok"}

# --------------------
# Read-only endpoints for UI display
# --------------------
@app.get("/ui/state")
async def read_ui_state():
    state = await get_ui_state()
    if not state:
        return JSONResponse(status_code=204, content={})
    return state

@app.get("/ui/sentiment")
async def read_sentiment():
    ns = NewsSignalService()
    latest = ns.latest_snapshot()
    if not latest:
        return JSONResponse(status_code=204, content={})
    return latest

@app.get("/ui/backtest/latest")
async def read_last_backtest():
    out_path = Path(__file__).resolve().parents[2] / 'db' / 'backtest_results'
    if not out_path.exists():
        return JSONResponse(status_code=204, content={})
    files = sorted(out_path.glob('backtest_*.jsonl'))
    if not files:
        return JSONResponse(status_code=204, content={})
    last = files[-1]
    try:
        j = json.loads(last.read_text())
        return j
    except Exception:
        raise HTTPException(status_code=500, detail='Failed to read backtest result')

# --------------------
# Config update endpoints (requires basic auth)
# --------------------
@app.post("/ui/state")
async def update_ui_state(payload: UIState, ok: bool = Depends(get_basic_credentials)):
    # augment with timestamp
    data = payload.dict()
    data['updated_at'] = datetime.utcnow().isoformat()
    success = await set_ui_state(data)
    if not success:
        raise HTTPException(status_code=500, detail='Failed to persist ui state')
    return {"status": "ok", "saved": True}
