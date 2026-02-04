import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.strategy.setups import detect_setup, SetupConfig


def make_ohlc_series(start: datetime, periods: int, freq: str, start_price: float, step: float):
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    close = np.array([start_price + i * step for i in range(periods)], dtype=float)
    open_ = close - step * 0.1
    high = np.maximum(close, open_) + 0.2
    low = np.minimum(close, open_) - 0.2
    vol = np.ones_like(close) * 100
    df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': vol}, index=idx)
    return df


def test_ema_pullback_long():
    now = datetime.utcnow()
    # build trending up series, then small pullback and bounce
    df = make_ohlc_series(now - timedelta(days=10), periods=200, freq='1H', start_price=100.0, step=0.3)
    # create a small pullback in last 4 bars
    df.iloc[-4:-2, df.columns.get_loc('Close')] = df['Close'].iloc[-6] * 0.99
    # last bar bounce
    df.iloc[-1, df.columns.get_loc('Close')] = df['Close'].iloc[-2] * 1.01

    state, payload = detect_setup(df, 'LONG', SetupConfig(ema_period=20, pullback_max_pct=0.05, pullback_retest_bars=6))
    assert state == 'SETUP_FOUND'
    assert payload['setup'] == 'EMA_PULLBACK'


def test_break_retest_long():
    now = datetime.utcnow()
    df = make_ohlc_series(now - timedelta(days=10), periods=200, freq='1H', start_price=100.0, step=0.0)
    # make a resistance at 100 over earlier period
    df['Close'].iloc[:150] = 100.0
    # breakout above level
    df['Close'].iloc[150] = 101.5
    # retest with small dip near 100.9 then bounce
    df['Low'].iloc[151] = 100.9
    df['Close'].iloc[151] = 101.0
    df['Close'].iloc[-1] = 101.2

    state, payload = detect_setup(df, 'LONG', SetupConfig(break_lookback=60, retest_max_drop_pct=0.02, confirm_bars=1))
    assert state == 'SETUP_FOUND'
    assert payload['setup'] == 'BREAK_RETEST'


def test_no_setup_when_deep_pullback():
    now = datetime.utcnow()
    df = make_ohlc_series(now - timedelta(days=10), periods=200, freq='1H', start_price=100.0, step=0.3)
    # deep pullback greater than threshold
    df.iloc[-5:-2, df.columns.get_loc('Close')] = df['Close'].iloc[-6] * 0.9
    df.iloc[-1, df.columns.get_loc('Close')] = df['Close'].iloc[-2] * 0.99

    state, payload = detect_setup(df, 'LONG', SetupConfig(ema_period=20, pullback_max_pct=0.03, pullback_retest_bars=6))
    assert state == 'NO_SETUP'


def test_short_break_retest():
    now = datetime.utcnow()
    df = make_ohlc_series(now - timedelta(days=10), periods=200, freq='1H', start_price=150.0, step=0.0)
    df['Close'].iloc[:150] = 150.0
    # breakout down
    df['Close'].iloc[150] = 148.0
    # retest with small high near 149.2 then continue down
    df['High'].iloc[151] = 149.2
    df['Close'].iloc[151] = 148.5
    df['Close'].iloc[-1] = 148.3

    state, payload = detect_setup(df, 'SHORT', SetupConfig(break_lookback=60, retest_max_drop_pct=0.02, confirm_bars=1))
    assert state == 'SETUP_FOUND'
    assert payload['setup'] == 'BREAK_RETEST'