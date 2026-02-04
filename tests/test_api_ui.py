import os
import json
from fastapi.testclient import TestClient
from backend.api.app import app
from backend.api.state import UI_STATE_FILE

client = TestClient(app)


def test_ui_state_crud(monkeypatch, tmp_path):
    # set credentials
    monkeypatch.setenv('UI_API_USER', 'testuser')
    monkeypatch.setenv('UI_API_PASS', 'testpass')
    # isolate UI state file
    monkeypatch.setattr('backend.api.state.UI_STATE_FILE', tmp_path / 'ui_state.json')

    # ensure GET returns 204 when no state
    r = client.get('/ui/state')
    assert r.status_code == 204

    payload = {
        'system_on': True,
        'mode': 'Paper',
        'risk_per_trade_pct': 0.5,
        'max_daily_loss_pct': 2.0,
        'max_drawdown_pct': 10.0,
        'news_lock': False,
        'kill_switch': False
    }

    # unauthenticated POST should 401
    r = client.post('/ui/state', json=payload)
    assert r.status_code == 401

    # authenticated POST
    r = client.post('/ui/state', json=payload, auth=('testuser', 'testpass'))
    assert r.status_code == 200
    body = r.json()
    assert body.get('status') == 'ok'

    # GET should now return state
    r = client.get('/ui/state')
    assert r.status_code == 200
    s = r.json()
    assert s['system_on'] == True
    assert s['mode'] == 'Paper'
    assert 'updated_at' in s
