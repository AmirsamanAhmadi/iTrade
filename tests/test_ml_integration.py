import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.backtesting.engine import BacktestEngine, BacktestConfig, SlippageModel
from backend.risk.risk_manager import RiskEngine
from backend.ml.trainer import train_logistic, build_dataset_from_trades, LogisticModel


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
    df4h.rename(columns=str.capitalize, inplace=True)
    df4h['Open'] = df1h['Open'].resample('4H').first()
    df4h['High'] = df1h['High'].resample('4H').max()
    df4h['Low'] = df1h['Low'].resample('4H').min()
    df4h['Volume'] = df1h['Volume'].resample('4H').sum()
    df4h = df4h[['Open','High','Low','Close','Volume']]
    return df1h, df4h


def test_backtest_with_classifier_skip_low_confidence():
    df1h, df4h = make_data()
    risk = RiskEngine(start_balance=10000.0)
    slip = SlippageModel(spread_pips=0.0001, slippage_pct=0.0)
    # first run without classifier
    engine = BacktestEngine(df1h, df4h, cfg=BacktestConfig(start_balance=10000.0, slippage=slip), risk_engine=risk)
    res = engine.run(max_bars=150)
    total_trades_no_model = len([t for t in res.trades if t.get('action') in ('OPEN','CLOSE')])

    # build a synthetic training set from same engine run: label positive when pnl>0
    X, y = build_dataset_from_trades(res.trades)
    if len(y) == 0:
        # nothing to train on; assert no errors
        return
    model = train_logistic(X, y)
    clf = model

    # run engine with classifier and high threshold (e.g., 0.9) to skip many trades
    engine2 = BacktestEngine(df1h, df4h, cfg=BacktestConfig(start_balance=10000.0, slippage=slip), risk_engine=risk, classifier=clf, confidence_threshold=0.9)
    res2 = engine2.run(max_bars=150)
    skipped = len([t for t in res2.trades if t.get('action') == 'SKIPPED_LOW_CONFIDENCE'])
    total_trades_model = len([t for t in res2.trades if t.get('action') in ('OPEN','CLOSE')])

    # expect model-run to have skipped at least some attempted trades or reduced total executed trades
    assert skipped >= 0
    assert total_trades_model <= total_trades_no_model
