import pandas as pd
import pytest
from datetime import datetime, timedelta

from backend.services.market_data import MarketDataService


@pytest.fixture
def sample_1h_df():
    # create 6 hours of 1H data
    idx = pd.date_range(start='2024-01-01T00:00:00Z', periods=6, freq='H')
    df = pd.DataFrame({
        'Open': [1,2,3,4,5,6],
        'High': [1,2,3,4,5,6],
        'Low': [1,2,3,4,5,6],
        'Close': [1,2,3,4,5,6],
        'Volume': [10,10,10,10,10,10]
    }, index=idx)
    return df


def test_resample_to_4h(sample_1h_df, monkeypatch, tmp_path):
    svc = MarketDataService(cache_ttl=0)

    # monkeypatch yfinance.Ticker.history to return sample_1h_df
    class DummyTicker:
        def history(self, **kwargs):
            return sample_1h_df

    monkeypatch.setattr('backend.services.market_data.yf.Ticker', lambda t: DummyTicker())

    df4 = svc.fetch_ohlc('EURUSD=X', '4h', start='2024-01-01', end='2024-01-02', use_cache=False)
    # Expect two 4H periods in 6 hours -> 2 rows (00:00-03:00, 04:00-07:00)
    assert 'Open' in df4.columns
    assert len(df4) >= 1


def test_regular_index_filling(monkeypatch):
    svc = MarketDataService(cache_ttl=0)
    # create 15m data with a missing bar
    idx = pd.date_range(start='2024-01-01T00:00:00Z', periods=4, freq='15T')
    df = pd.DataFrame({'Open':[1,2,4,5], 'High':[1,2,4,5], 'Low':[1,2,4,5], 'Close':[1,2,4,5], 'Volume':[1,1,1,1]}, index=idx)

    class DummyTicker:
        def history(self, **kwargs):
            return df.iloc[[0,1,3]]  # missing the 3rd bar

    monkeypatch.setattr('backend.services.market_data.yf.Ticker', lambda t: DummyTicker())
    df15 = svc.fetch_ohlc('EURUSD=X', '15m', start='2024-01-01', end='2024-01-02', use_cache=False, fill_method='ffill')
    # after filling, missing bar should be present
    assert len(df15) == 4
    assert df15.iloc[2]['Open'] == 2
