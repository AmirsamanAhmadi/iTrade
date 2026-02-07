"""News service with multiple API sources.

This module fetches headlines from multiple sources including:
- Finviz (market news)
- NewsAPI (general news)
- Alpha Vantage (financial news)
- MarketWatch (financial news)

Stores raw news to disk (JSON lines) with currency mapping.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import requests
import time

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from finvizfinance.news import News as FinvizNews  # type: ignore
    FINVIZ_AVAILABLE = True
except Exception:
    FINVIZ_AVAILABLE = False

NEWS_DB = Path(__file__).resolve().parents[2] / 'db' / 'news'
NEWS_DB.mkdir(parents=True, exist_ok=True)

@dataclass
class NewsService:
    source: str = 'multi'
    
    # API Keys (should be set in environment variables)
    NEWSAPI_KEY: Optional[str] = None
    ALPHA_VANTAGE_KEY: Optional[str] = None
    
    def __post_init__(self):
        # Load API keys from environment
        self.NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
        self.ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.last_request_time = {}
        self.rate_limit_delay = 1.0  # seconds between requests to same source
    
    def _rate_limit(self, source: str):
        """Implement basic rate limiting."""
        now = time.time()
        if source in self.last_request_time:
            elapsed = now - self.last_request_time[source]
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time[source] = now

    def fetch_headlines(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch headlines from multiple sources.
        
        Each dict contains: timestamp (ISO), headline, source, mapped_currencies.
        """
        all_results = []
        seen_headlines = set()
        
        # Source 1: Finviz (if available)
        if FINVIZ_AVAILABLE:
            try:
                finviz_results = self._fetch_finviz()
                for r in finviz_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Finviz fetch failed: {e}")
        
        # Source 2: NewsAPI (if API key available)
        if self.NEWSAPI_KEY:
            try:
                newsapi_results = self._fetch_newsapi(tickers)
                for r in newsapi_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"NewsAPI fetch failed: {e}")
        
        # Source 3: Alpha Vantage (if API key available)
        if self.ALPHA_VANTAGE_KEY and tickers:
            try:
                av_results = self._fetch_alpha_vantage(tickers)
                for r in av_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Alpha Vantage fetch failed: {e}")
        
        # Source 4: MarketWatch scraping (fallback)
        try:
            mw_results = self._fetch_marketwatch(tickers)
            for r in mw_results:
                h = r.get('headline', '').strip()
                if h and h not in seen_headlines:
                    seen_headlines.add(h)
                    all_results.append(r)
        except Exception as e:
            logger.error(f"MarketWatch fetch failed: {e}")
        
        # Store all unique results
        for r in all_results:
            self._store_raw(r)
        
        # Sort by timestamp (most recent first)
        all_results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return all_results
    
    def _fetch_finviz(self) -> List[Dict]:
        """Fetch from Finviz."""
        results = []
        self._rate_limit('finviz')
        
        try:
            logger.info("Fetching from Finviz")
            news = FinvizNews()
            news_data = news.get_news()
            
            if news_data and 'news' in news_data:
                df = news_data['news']
                if not df.empty:
                    for _, row in df.iterrows():
                        ts = str(row.get('Date', datetime.utcnow().isoformat()))
                        headline = str(row.get('Title', ''))
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': ts,
                            'headline': headline,
                            'source': 'finviz',
                            'mapped_currencies': mapped,
                            'url': row.get('Link', '')
                        })
        except Exception as e:
            logger.error(f"Finviz error: {e}")
        
        return results
    
    def _fetch_newsapi(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from NewsAPI."""
        results = []
        self._rate_limit('newsapi')
        
        try:
            logger.info("Fetching from NewsAPI")
            
            # Build query from tickers or use general financial terms
            if tickers:
                query = ' OR '.join([f'"{t}"' for t in tickers[:3]])  # Limit to 3 tickers
            else:
                query = 'stock market OR trading OR forex OR cryptocurrency'
            
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': self.NEWSAPI_KEY,
                'from': (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok':
                    for article in data.get('articles', []):
                        headline = article.get('title', '')
                        if headline:
                            mapped = self._map_to_currencies(headline)
                            results.append({
                                'timestamp': article.get('publishedAt', datetime.utcnow().isoformat()),
                                'headline': headline,
                                'source': f"newsapi:{article.get('source', {}).get('name', 'NewsAPI')}",
                                'mapped_currencies': mapped,
                                'url': article.get('url', ''),
                                'description': article.get('description', '')
                            })
            else:
                logger.warning(f"NewsAPI returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
        
        return results
    
    def _fetch_alpha_vantage(self, tickers: List[str]) -> List[Dict]:
        """Fetch from Alpha Vantage."""
        results = []
        
        try:
            logger.info("Fetching from Alpha Vantage")
            
            # Fetch news for each ticker (limit to avoid rate limits)
            for ticker in tickers[:2]:  # Limit to 2 tickers
                self._rate_limit('alpha_vantage')
                
                url = 'https://www.alphavantage.co/query'
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'apikey': self.ALPHA_VANTAGE_KEY,
                    'limit': 10
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'feed' in data:
                        for item in data['feed']:
                            headline = item.get('title', '')
                            if headline:
                                mapped = self._map_to_currencies(headline)
                                results.append({
                                    'timestamp': item.get('time_published', datetime.utcnow().isoformat()),
                                    'headline': headline,
                                    'source': f"alphavantage:{item.get('source', 'AlphaVantage')}",
                                    'mapped_currencies': mapped,
                                    'url': item.get('url', ''),
                                    'sentiment': item.get('overall_sentiment_label', 'neutral')
                                })
                
                time.sleep(1)  # Rate limiting between requests
                
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
        
        return results
    
    def _fetch_marketwatch(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from MarketWatch (basic RSS or scraping)."""
        results = []
        
        try:
            logger.info("Fetching from MarketWatch")
            self._rate_limit('marketwatch')
            
            # MarketWatch RSS feed for market news
            url = 'https://feeds.content.dowjones.io/public/rss/mw_topstories'
            
            try:
                import feedparser
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:15]:  # Limit to 15 entries
                    headline = entry.get('title', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        published = entry.get('published', datetime.utcnow().isoformat())
                        
                        results.append({
                            'timestamp': published,
                            'headline': headline,
                            'source': 'marketwatch',
                            'mapped_currencies': mapped,
                            'url': entry.get('link', ''),
                            'description': entry.get('summary', '')
                        })
                        
            except ImportError:
                logger.info("feedparser not installed, skipping MarketWatch RSS")
                
        except Exception as e:
            logger.error(f"MarketWatch error: {e}")
        
        return results

    def _map_to_currencies(self, headline: str) -> List[str]:
        s = headline.lower()
        # find positions to preserve order of mention in the headline
        def first_index(variants):
            indices = [s.find(v) for v in variants]
            indices = [i for i in indices if i >= 0]
            return min(indices) if indices else -1

        positions = []
        eur_idx = first_index(['eur', 'euro', 'euros'])
        usd_idx = first_index(['usd', 'dollar', 'dollars'])
        if eur_idx >= 0:
            positions.append(('EUR', eur_idx))
        if usd_idx >= 0:
            positions.append(('USD', usd_idx))
        positions = sorted(positions, key=lambda x: x[1])
        return [p[0] for p in positions]

    def _store_raw(self, rec: Dict):
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        path = NEWS_DB / f"news_{date_str}.jsonl"
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            logger.exception('Failed to store raw news')
