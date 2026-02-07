"""Market data service wrapper around yfinance with simple caching and resampling.

Features:
- fetch OHLC for 1m, 15m, 1h, 4h
- handles missing data gracefully (reindex + optional fill)
- normalizes timestamps to UTC
- caches responses to disk (CSV) to avoid repeat downloads
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from backend.logging import configure_logging

configure_logging()
import logging
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / 'db' / 'market_cache'
CACHE_TTL = 60 * 60  # 1 hour
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INTERVAL_MAP = {
    '1m': '1m',
    '15m': '15m',
    '1h': '60m',
    # 4h will be produced by resampling 1h
}

# FxPro symbol mapping (FxPro uses different symbols)
FXPRO_SYMBOL_MAP = {
    'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD',
    'USDJPY=X': 'USDJPY',
    'AUDUSD=X': 'AUDUSD',
    'USDCAD=X': 'USDCAD',
    'USDCHF=X': 'USDCHF',
    'NZDUSD=X': 'NZDUSD',
    'GC=F': 'XAUUSD',  # Gold
    'SI=F': 'XAGUSD',  # Silver
}

@dataclass
class MarketDataService:
    cache_ttl: int = CACHE_TTL
    use_fxpro: bool = False  # Toggle for FxPro data source
    
    def _cache_path(self, ticker: str, interval: str, start: Optional[str], end: Optional[str]) -> Path:
        key = f"{ticker}|{interval}|{start}|{end}"
        h = hashlib.sha1(key.encode()).hexdigest()
        return CACHE_DIR / f"{h}.csv"

    def _is_cache_valid(self, p: Path) -> bool:
        if not p.exists():
            return False
        age = time.time() - p.stat().st_mtime
        return age <= self.cache_ttl

    def fetch_ohlc(self, ticker: str, interval: str, start: Optional[str] = None, end: Optional[str] = None, use_cache: bool = True, fill_method: Optional[str] = 'ffill') -> pd.DataFrame:
        """Fetch OHLC data resampled to the requested interval.

        interval: one of '1m', '15m', '1h', '4h'
        start/end: ISO date strings or None
        fill_method: None | 'ffill' | 'bfill' to handle missing bars after resample
        """
        assert interval in ('1m', '15m', '1h', '4h')

        cache_p = self._cache_path(ticker, interval, start, end)
        if use_cache and self._is_cache_valid(cache_p):
            logger.info(f"Loading market data from cache: {cache_p}")
            df = pd.read_csv(cache_p, index_col=0, parse_dates=True)
            return df

        # Determine fetch interval (yf) and whether we need resampling
        yf_interval = INTERVAL_MAP.get(interval, '60m')
        need_resample_4h = (interval == '4h')
        if need_resample_4h:
            yf_interval = INTERVAL_MAP['1h']

        logger.info(f"Fetching {ticker} {interval} from yfinance (yf_interval={yf_interval}) start={start} end={end}")
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(interval=yf_interval, start=start, end=end, actions=False, prepost=False)
        except Exception as exc:
            logger.exception("yfinance fetch failed")
            raise

        if df.empty:
            logger.warning("No market data returned from yfinance (empty DataFrame)")
            return df

        # Ensure columns are named Open, High, Low, Close, Volume
        df = df.rename(columns={c: c.capitalize() for c in df.columns})
        # Normalize timezone to UTC and ensure a DatetimeIndex
        if df.index.tzinfo is None and df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        if need_resample_4h:
            df = self._resample_to_4h(df, fill_method)

        # Reindex to expected period frequency and optionally fill missing
        df = self._ensure_regular_index(df, interval, fill_method)

        # Cache
        try:
            df.to_csv(cache_p)
        except Exception:
            logger.exception("Failed to write market cache")

        return df

    def _resample_to_4h(self, df: pd.DataFrame, fill_method: Optional[str]) -> pd.DataFrame:
        # Resample 1h -> 4h using OHLC aggregation
        o = df['Open'].resample('4H').first()
        h = df['High'].resample('4H').max()
        l = df['Low'].resample('4H').min()
        c = df['Close'].resample('4H').last()
        v = df['Volume'].resample('4H').sum()
        res = pd.concat([o, h, l, c, v], axis=1)
        res.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if fill_method in ('ffill', 'bfill'):
            if fill_method == 'ffill':
                res = res.ffill()
            else:
                res = res.bfill()
        return res

    def _ensure_regular_index(self, df: pd.DataFrame, interval: str, fill_method: Optional[str]) -> pd.DataFrame:
        freq_map = {'1m': 'min', '15m': '15min', '1h': 'h', '4h': '4h'}
        freq = freq_map[interval]
        start = df.index[0]
        end = df.index[-1]
        expected_index = pd.date_range(start=start, end=end, freq=freq, tz='UTC')
        if len(expected_index) == len(df.index):
            return df

        logger.warning(f"Missing bars detected: expected {len(expected_index)} but got {len(df.index)}; reindexing")
        df = df.reindex(expected_index)
        if fill_method in ('ffill', 'bfill'):
            if fill_method == 'ffill':
                df = df.ffill()
            else:
                df = df.bfill()
        return df
    
    def fetch_fxpro_ohlc(self, ticker: str, interval: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch OHLC data from FxPro public API.
        
        FxPro provides free access to forex and commodity data.
        """
        import requests
        
        # Map interval to FxPro format
        fxpro_interval_map = {
            '1m': 'm1',
            '15m': 'm15', 
            '1h': 'h1',
            '4h': 'h4',
        }
        
        fxpro_interval = fxpro_interval_map.get(interval, 'h1')
        
        # Map ticker to FxPro symbol
        fxpro_symbol = FXPRO_SYMBOL_MAP.get(ticker, ticker.replace('=X', ''))
        
        # FxPro public API endpoint
        url = f"https://api.fxpro.com/api/candles/{fxpro_symbol}/{fxpro_interval}"
        
        params = {
            'limit': 1000,
        }
        
        if start:
            params['from'] = start
        if end:
            params['to'] = end
        
        logger.info(f"Fetching {ticker} {interval} from FxPro")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') != 'success':
                logger.warning(f"FxPro API returned status: {data}")
                return pd.DataFrame()
            
            candles = data.get('data', {}).get('candles', [])
            
            if not candles:
                logger.warning("No data returned from FxPro")
                return pd.DataFrame()
            
            # Parse candles
            records = []
            for candle in candles:
                records.append({
                    'Open': float(candle.get('o', 0)),
                    'High': float(candle.get('h', 0)),
                    'Low': float(candle.get('l', 0)),
                    'Close': float(candle.get('c', 0)),
                    'Volume': float(candle.get('v', 0)),
                    'time': pd.to_datetime(candle.get('time', candle.get('timestamp', ''))),
                })
            
            df = pd.DataFrame(records)
            df.set_index('time', inplace=True)
            df.index = df.index.tz_localize('UTC')
            
            logger.info(f"FxPro returned {len(df)} candles")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FxPro API request failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.exception(f"FxPro data parsing failed: {e}")
            return pd.DataFrame()
