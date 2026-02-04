from backend.risk.risk_manager import RiskEngine
from backend.execution.trading import PaperTrader, ExecutionConfig
from backend.execution.broker import Broker


def test_per_trade_rejection_by_risk_engine():
    broker = Broker(balance=10000.0)
    # trader uses 1% sizing, risk engine only allows 0.01% per trade -> should reject
    risk_engine = RiskEngine(start_balance=10000.0, risk_per_trade_pct=0.0001)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01), risk_engine=risk_engine)

    payload = {'entry': 100.0, 'stop': 90.0}
    res = trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0)
    assert res == 'TRADE_REJECTED_RISK'


def test_daily_loss_kill_switch_and_rejection():
    broker = Broker(balance=10000.0)
    # set daily loss cap small so a single losing trade kills
    # set daily loss cap so that a losing trade of size 100 will equal the cap and cause kill when closed
    risk_engine = RiskEngine(start_balance=10000.0, max_daily_loss_pct=0.01)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01), risk_engine=risk_engine)

    payload = {'entry': 100.0, 'stop': 90.0}
    res = trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0)
    assert res == 'TRADE_EXECUTED'

    # simulate price reaching stop
    pos = trader.open_positions['EURUSD']
    pnl = trader.on_price('EURUSD', pos.stop)
    assert risk_engine.killed is True

    # subsequent trades should be rejected due to kill
    res2 = trader.process_setup('GBPUSD', 'LONG', payload, current_price=100.0)
    assert res2 == 'TRADE_REJECTED_RISK'


def test_consecutive_losses_trigger_kill():
    broker = Broker(balance=10000.0)
    risk_engine = RiskEngine(start_balance=10000.0, max_consecutive_losses=2)
    trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01), risk_engine=risk_engine)

    payload = {'entry': 100.0, 'stop': 90.0}
    # first loss
    assert trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0) == 'TRADE_EXECUTED'
    pos = trader.open_positions['EURUSD']
    trader.on_price('EURUSD', pos.stop)
    assert risk_engine.consecutive_losses == 1
    assert risk_engine.killed is False

    # second loss
    assert trader.process_setup('GBPUSD', 'LONG', payload, current_price=100.0) == 'TRADE_EXECUTED'
    pos = trader.open_positions['GBPUSD']
    trader.on_price('GBPUSD', pos.stop)
    assert risk_engine.consecutive_losses == 2
    assert risk_engine.killed is True
