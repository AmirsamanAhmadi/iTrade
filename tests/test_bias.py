import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.strategy.bias import compute_bias, BiasConfig


def make_ohlc_series(start: datetime, periods: int, freq: str, start_price: float, step: float):
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    close = np.array([start_price + i * step for i in range(periods)], dtype=float)
    open_ = close - step * 0.1
    high = close + 0.2
    low = close - 0.2
    vol = np.ones_like(close) * 100
    df = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': vol}, index=idx)
    return df


def test_4h_long_with_1h_confirmation():
    now = datetime.utcnow()
    # 4H: increasing close
    df4h = make_ohlc_series(now - timedelta(hours=400), periods=100, freq='4H', start_price=100.0, step=0.5)
    # 1H: recent momentum up
    df1h = make_ohlc_series(now - timedelta(hours=100), periods=200, freq='1H', start_price=100.0, step=0.2)

    bias = compute_bias(df4h, df1h, BiasConfig())
    assert bias == 'LONG'


def test_4h_long_without_1h_confirmation():
    now = datetime.utcnow()
    df4h = make_ohlc_series(now - timedelta(hours=400), periods=100, freq='4H', start_price=100.0, step=0.5)
    # 1H: flat / slightly down (no confirmation)
    df1h = make_ohlc_series(now - timedelta(hours=100), periods=200, freq='1H', start_price=100.0, step=-0.01)

    bias = compute_bias(df4h, df1h, BiasConfig())
    assert bias == 'NEUTRAL'


def test_4h_short_with_confirmation():
    now = datetime.utcnow()
    df4h = make_ohlc_series(now - timedelta(hours=400), periods=100, freq='4H', start_price=200.0, step=-0.7)
    df1h = make_ohlc_series(now - timedelta(hours=100), periods=200, freq='1H', start_price=200.0, step=-0.3)

    bias = compute_bias(df4h, df1h, BiasConfig())
    assert bias == 'SHORT'


def test_ranging_detected():
    now = datetime.utcnow()
    # 4H: small oscillations ±0.5 around 100 -> range
    idx = pd.date_range(start=now - timedelta(days=10), periods=50, freq='4H')
    close = 100.0 + 0.5 * np.sin(np.linspace(0, 10, 50))
    df4h = pd.DataFrame({'Open': close - 0.1, 'High': close + 0.2, 'Low': close - 0.2, 'Close': close, 'Volume': np.ones_like(close)}, index=idx)
    # 1H data arbitrary
    df1h = make_ohlc_series(now - timedelta(hours=100), periods=200, freq='1H', start_price=100.0, step=0.0)

    bias = compute_bias(df4h, df1h, BiasConfig(range_threshold=0.01))
    assert bias == 'NEUTRAL'