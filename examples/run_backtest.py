"""Example: run a backtest with contrived data to validate pipeline."""
from backend.backtesting.engine import BacktestEngine, BacktestConfig, SlippageModel
from backend.risk.risk_manager import RiskEngine
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# make data
now = datetime.utcnow()
idx_h = pd.date_range(start=now - timedelta(days=10), periods=240, freq='1H')
close_h = 100.0 + np.linspace(0, 5, len(idx_h))
open_h = close_h - 0.1
high_h = close_h + 0.2
low_h = close_h - 0.2
vol = np.ones_like(close_h) * 100

df1h = pd.DataFrame({'Open': open_h, 'High': high_h, 'Low': low_h, 'Close': close_h, 'Volume': vol}, index=idx_h)
# 4H resample
idx_4h = pd.date_range(start=now - timedelta(days=10), periods=60, freq='4H')
df4h = df1h['Close'].resample('4H').ohlc()
df4h['Open'] = df1h['Open'].resample('4H').first()
df4h['High'] = df1h['High'].resample('4H').max()
df4h['Low'] = df1h['Low'].resample('4H').min()
df4h['Volume'] = df1h['Volume'].resample('4H').sum()

df4h = df4h[['Open','High','Low','Close','Volume']]

risk = RiskEngine(start_balance=10000.0)
slip = SlippageModel(spread_pips=0.0001, slippage_pct=0.0001)
engine = BacktestEngine(df1h, df4h, symbol='EURUSD', cfg=BacktestConfig(start_balance=10000.0, slippage=slip, news_lock_minutes=30), risk_engine=risk)

# simulate a news event 5 minutes before the last bar (should lock last bar and prevent trading)
news = [{'timestamp': (idx_h[-1] - timedelta(minutes=5)).isoformat(), 'mapped_currencies': ['USD']}]

res = engine.run(news_events=news)
print('Metrics:', res.metrics)
print('Trades:', res.trades)
print('Equity tail:')
print(res.equity_curve.tail())
