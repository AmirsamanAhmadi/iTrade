import pytest
from backend.execution.trading import PaperTrader, ExecutionConfig
from backend.execution.broker import Broker


def test_position_sizing():
    broker = Broker(balance=10000.0)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01))
    units = trader._compute_size(broker.balance, entry=100.0, stop=98.0)
    # risk per unit = 2, risk budget = 100, so units = 50
    assert pytest.approx(units, rel=1e-6) == 50.0


def test_risk_cap_enforcement():
    broker = Broker(balance=10000.0)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.02, max_total_risk_pct=0.03))
    # first trade would risk 2% * 10000 = 200
    payload = {'entry': 100.0, 'stop': 95.0}
    res1 = trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0)
    assert res1 == 'TRADE_EXECUTED'
    # second trade would push total risk to 4% > 3% cap and should be rejected
    res2 = trader.process_setup('GBPUSD', 'LONG', payload, current_price=100.0)
    assert res2 == 'TRADE_REJECTED_RISK'


def test_entry_and_stop_execution():
    broker = Broker(balance=10000.0)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01))
    payload = {'entry': 100.0, 'stop': 95.0}
    res = trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0)
    assert res == 'TRADE_EXECUTED'
    assert 'EURUSD' in trader.open_positions
    pos = trader.open_positions['EURUSD']
    # simulate some price hits stop
    pnl = trader.on_price('EURUSD', pos.stop)
    # closed and returned a pnl value (may be negative)
    assert isinstance(pnl, float) or pnl is None
    assert 'EURUSD' not in trader.open_positions
    # balance changed after close
    assert broker.balance != 10000.0
