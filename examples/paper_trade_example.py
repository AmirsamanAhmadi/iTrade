"""Example: run a simple pipeline: detect setup -> execute paper trade -> simulate stop hit."""
from backend.execution.trading import PaperTrader, ExecutionConfig
from backend.execution.broker import Broker
from backend.strategy.setups import detect_setup, SetupConfig
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Generate a contrived series with an EMA pullback
now = datetime.utcnow()
idx = pd.date_range(start=now - timedelta(days=10), periods=200, freq='1H')
close = 100.0 + np.linspace(0, 5, 200)
open_ = close - 0.1
high = close + 0.2
low = close - 0.2
vol = np.ones_like(close) * 100

df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': vol}, index=idx)
# make a small pullback
df['Close'].iloc[-5:-2] = df['Close'].iloc[-6] * 0.99
df['Close'].iloc[-1] = df['Close'].iloc[-2] * 1.01

cfg = SetupConfig(ema_period=20, pullback_max_pct=0.05, pullback_retest_bars=6)
state, payload = detect_setup(df, 'LONG', cfg)
print('Detect setup:', state, payload)

broker = Broker(balance=10000.0)
trader = PaperTrader(broker, ExecutionConfig(risk_per_trade_pct=0.01, max_total_risk_pct=0.05))
current_price = float(df['Close'].iloc[-1])
res = trader.process_setup('EURUSD', 'LONG', payload or {}, current_price)
print('Process setup result:', res)
print('Open positions:', trader.open_positions)
# simulate price move to stop
if trader.open_positions:
    sym = list(trader.open_positions.keys())[0]
    stop = trader.open_positions[sym].stop
    print('Simulate stop hit at', stop)
    pnl = trader.on_price(sym, stop)
    print('Stop P&L:', pnl)
    print('Final broker balance:', broker.balance)
else:
    print('No position opened.')
