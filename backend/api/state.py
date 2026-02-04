import json
import os
from pathlib import Path
from typing import Optional

try:
    import redis.asyncio as redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:
    redis = None
    REDIS_AVAILABLE = False

UI_STATE_FILE = Path(__file__).resolve().parents[2] / 'db' / 'ui_state.json'
UI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
REDIS_KEY = os.getenv('UI_REDIS_KEY', 'forexbot:ui_state')


async def get_ui_state() -> Optional[dict]:
    # Try redis first
    if REDIS_AVAILABLE and os.getenv('REDIS_URL'):
        try:
            r = redis.from_url(os.getenv('REDIS_URL'))
            v = await r.get(REDIS_KEY)
            if v:
                return json.loads(v)
        except Exception:
            pass

    # Fallback to file
    if UI_STATE_FILE.exists():
        try:
            with open(UI_STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None


async def set_ui_state(state: dict) -> bool:
    # validate JSON serializable
    try:
        j = json.dumps(state)
    except Exception:
        return False

    # Try redis
    if REDIS_AVAILABLE and os.getenv('REDIS_URL'):
        try:
            r = redis.from_url(os.getenv('REDIS_URL'))
            await r.set(REDIS_KEY, j)
            return True
        except Exception:
            pass

    # Fallback to file
    try:
        with open(UI_STATE_FILE, 'w', encoding='utf-8') as f:
            f.write(j)
        return True
    except Exception:
        return False
