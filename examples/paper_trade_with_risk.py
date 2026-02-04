"""Paper trade example showing RiskEngine integration."""
from backend.execution.trading import PaperTrader, ExecutionConfig
from backend.execution.broker import Broker
from backend.risk.risk_manager import RiskEngine

broker = Broker(balance=10000.0)
risk_engine = RiskEngine(start_balance=10000.0, risk_per_trade_pct=0.01, max_daily_loss_pct=0.01)
trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01), risk_engine=risk_engine)

payload = {'entry': 100.0, 'stop': 90.0}
print('Attempting trade...')
res = trader.process_setup('EURUSD', 'LONG', payload, current_price=100.0)
print('Result:', res)
print('Risk status:', risk_engine.status())

if res == 'TRADE_EXECUTED':
    pos = trader.open_positions['EURUSD']
    print('Simulate stop hit')
    pnl = trader.on_price('EURUSD', pos.stop)
    print('P&L on stop:', pnl)
    print('Risk status after close:', risk_engine.status())
else:
    print('No trade executed')
