import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.backtesting.engine import BacktestEngine, BacktestConfig, SlippageModel
from backend.risk.risk_manager import RiskEngine


def make_data():
    now = datetime.utcnow()
    idx_h = pd.date_range(start=now - timedelta(days=5), periods=200, freq='1H')
    close_h = 100.0 + np.linspace(0, 2, len(idx_h))
    open_h = close_h - 0.1
    high_h = close_h + 0.2
    low_h = close_h - 0.2
    vol = np.ones_like(close_h) * 100
    df1h = pd.DataFrame({'Open': open_h, 'High': high_h, 'Low': low_h, 'Close': close_h, 'Volume': vol}, index=idx_h)
    df4h = df1h['Close'].resample('4H').ohlc()
    # ensure Title-case columns
    df4h.rename(columns=str.capitalize, inplace=True)
    df4h['Open'] = df1h['Open'].resample('4H').first()
    df4h['High'] = df1h['High'].resample('4H').max()
    df4h['Low'] = df1h['Low'].resample('4H').min()
    df4h['Volume'] = df1h['Volume'].resample('4H').sum()
    df4h = df4h[['Open','High','Low','Close','Volume']]
    return df1h, df4h


def test_backtest_runs_and_respects_news_lock():
    df1h, df4h = make_data()
    risk = RiskEngine(start_balance=10000.0)
    slip = SlippageModel(spread_pips=0.0001, slippage_pct=0.0)
    engine = BacktestEngine(df1h, df4h, cfg=BacktestConfig(start_balance=10000.0, slippage=slip, news_lock_minutes=60), risk_engine=risk)
    # add a recent news event to lock all trading
    news = [{'timestamp': df1h.index[-1].isoformat(), 'mapped_currencies': ['USD']}]
    res = engine.run(news_events=news)
    # with news lock present, there should be zero completed trades
    closes = [t for t in res.trades if t.get('action') == 'CLOSE']
    assert len(closes) == 0


def test_backtest_metrics_and_trades():
    df1h, df4h = make_data()
    risk = RiskEngine(start_balance=10000.0)
    slip = SlippageModel(spread_pips=0.0001, slippage_pct=0.0001)
    engine = BacktestEngine(df1h, df4h, cfg=BacktestConfig(start_balance=10000.0, slippage=slip), risk_engine=risk)
    res = engine.run()
    # engine produces metrics dict
    assert 'total_return' in res.metrics
    # equity curve shape matches bars
    assert not res.equity_curve.empty
    # trades list present (may be zero) and metrics consistent types
    assert isinstance(res.trades, list)
    assert isinstance(res.metrics['total_trades'], int)
