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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
    GNEWS_KEY: Optional[str] = None
    CURRENTS_KEY: Optional[str] = None
    NYT_KEY: Optional[str] = None
    GUARDIAN_KEY: Optional[str] = None
    BENZINGA_KEY: Optional[str] = None
    STOCKTWITS_KEY: Optional[str] = None
    MASSIVE_KEY: Optional[str] = None  # https://massive.com/
    
    def __post_init__(self):
        # Load API keys from environment
        self.NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
        self.ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
        self.GNEWS_KEY = os.getenv('GNEWS_KEY', '')
        self.CURRENTS_KEY = os.getenv('CURRENTS_KEY', '')
        self.NYT_KEY = os.getenv('NYT_KEY', '')
        self.GUARDIAN_KEY = os.getenv('GUARDIAN_KEY', '')
        self.BENZINGA_KEY = os.getenv('BENZINGA_KEY', '')
        self.STOCKTWITS_KEY = os.getenv('STOCKTWITS_KEY', '')
        self.MASSIVE_KEY = os.getenv('MASSIVE_KEY', '')
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
        
        # Source 4: Massive Finance News API (if API key available)
        if self.MASSIVE_KEY:
            try:
                massive_results = self._fetch_massive(tickers)
                for r in massive_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Massive fetch failed: {e}")
        
        # Source 5: GNews (if API key available)
        if self.GNEWS_KEY:
            try:
                gnews_results = self._fetch_gnews(tickers)
                for r in gnews_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"GNews fetch failed: {e}")
        
        # Source 6: Currents (if API key available)
        if self.CURRENTS_KEY:
            try:
                currents_results = self._fetch_currents(tickers)
                for r in currents_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Currents fetch failed: {e}")
        
        # Source 7: New York Times (if API key available)
        if self.NYT_KEY:
            try:
                nyt_results = self._fetch_nyt(tickers)
                for r in nyt_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"NYT fetch failed: {e}")
        
        # Source 8: The Guardian (if API key available)
        if self.GUARDIAN_KEY:
            try:
                guardian_results = self._fetch_guardian(tickers)
                for r in guardian_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Guardian fetch failed: {e}")
        
        # Source 9: Benzinga (if API key available)
        if self.BENZINGA_KEY:
            try:
                benzinga_results = self._fetch_benzinga(tickers)
                for r in benzinga_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"Benzinga fetch failed: {e}")
        
        # Source 10: StockTwits
        if tickers:
            try:
                st_results = self._fetch_stocktwits(tickers)
                for r in st_results:
                    h = r.get('headline', '').strip()
                    if h and h not in seen_headlines:
                        seen_headlines.add(h)
                        all_results.append(r)
            except Exception as e:
                logger.error(f"StockTwits fetch failed: {e}")
        
        # Source 11: Reddit
        try:
            reddit_results = self._fetch_reddit()
            for r in reddit_results:
                h = r.get('headline', '').strip()
                if h and h not in seen_headlines:
                    seen_headlines.add(h)
                    all_results.append(r)
        except Exception as e:
            logger.error(f"Reddit fetch failed: {e}")
        
        # Source 12: Yahoo Finance
        try:
            yf_results = self._fetch_yahoo_finance_news(tickers)
            for r in yf_results:
                h = r.get('headline', '').strip()
                if h and h not in seen_headlines:
                    seen_headlines.add(h)
                    all_results.append(r)
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
        
        # Source 13: Investing.com
        try:
            inv_results = self._fetch_investing_com(tickers)
            for r in inv_results:
                h = r.get('headline', '').strip()
                if h and h not in seen_headlines:
                    seen_headlines.add(h)
                    all_results.append(r)
        except Exception as e:
            logger.error(f"Investing.com fetch failed: {e}")
        
        # Source 14: MarketWatch scraping (fallback)
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
        
        logger.info(f"Total aggregated news items: {len(all_results)}")
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

    def _fetch_gnews(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from GNews API."""
        results = []
        if not self.GNEWS_KEY:
            return results
        
        try:
            logger.info("Fetching from GNews")
            self._rate_limit('gnews')
            
            query = ' OR '.join(tickers[:3]) if tickers else 'stock market trading'
            url = 'https://gnews.io/api/v4/search'
            params = {
                'q': query,
                'lang': 'en',
                'max': 20,
                'apikey': self.GNEWS_KEY,
                'from': (datetime.utcnow() - timedelta(days=3)).strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('articles', []):
                    headline = article.get('title', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': article.get('publishedAt', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': f"gnews:{article.get('source', {}).get('name', 'GNews')}",
                            'mapped_currencies': mapped,
                            'url': article.get('url', ''),
                            'description': article.get('description', '')
                        })
        except Exception as e:
            logger.error(f"GNews error: {e}")
        
        return results

    def _fetch_currents(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from Currents API."""
        results = []
        if not self.CURRENTS_KEY:
            return results
        
        try:
            logger.info("Fetching from Currents")
            self._rate_limit('currents')
            
            query = ' OR '.join(tickers[:3]) if tickers else 'finance trading'
            url = 'https://api.currentsapi.services/v1/search'
            params = {
                'keywords': query,
                'language': 'en',
                'apiKey': self.CURRENTS_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('news', []):
                    headline = article.get('title', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': article.get('published', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': f"currents:{article.get('author', 'Currents')}",
                            'mapped_currencies': mapped,
                            'url': article.get('url', ''),
                            'description': article.get('description', '')
                        })
        except Exception as e:
            logger.error(f"Currents error: {e}")
        
        return results

    def _fetch_nyt(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from New York Times API."""
        results = []
        if not self.NYT_KEY:
            return results
        
        try:
            logger.info("Fetching from NYT")
            self._rate_limit('nyt')
            
            query = ' '.join(tickers[:2]) if tickers else 'stock market'
            url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
            params = {
                'q': query,
                'api-key': self.NYT_KEY,
                'fq': 'news_desk:("Business" "Financial")',
                'sort': 'newest'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('response', {}).get('docs', []):
                    headline = article.get('headline', {}).get('main', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': article.get('pub_date', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': 'nytimes',
                            'mapped_currencies': mapped,
                            'url': article.get('web_url', ''),
                            'description': article.get('snippet', '')
                        })
        except Exception as e:
            logger.error(f"NYT error: {e}")
        
        return results

    def _fetch_guardian(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from The Guardian API."""
        results = []
        if not self.GUARDIAN_KEY:
            return results
        
        try:
            logger.info("Fetching from Guardian")
            self._rate_limit('guardian')
            
            query = ' '.join(tickers[:2]) if tickers else 'stock market'
            url = 'https://content.guardianapis.com/search'
            params = {
                'q': query,
                'api-key': self.GUARDIAN_KEY,
                'section': 'business',
                'show-fields': 'headline,trailText',
                'order-by': 'newest',
                'page-size': 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data.get('response', {}).get('results', []):
                    headline = article.get('webTitle', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        fields = article.get('fields', {})
                        results.append({
                            'timestamp': article.get('webPublicationDate', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': 'guardian',
                            'mapped_currencies': mapped,
                            'url': article.get('webUrl', ''),
                            'description': fields.get('trailText', '')
                        })
        except Exception as e:
            logger.error(f"Guardian error: {e}")
        
        return results

    def _fetch_benzinga(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from Benzinga API (financial news specialist)."""
        results = []
        if not self.BENZINGA_KEY:
            return results
        
        try:
            logger.info("Fetching from Benzinga")
            self._rate_limit('benzinga')
            
            url = 'https://api.benzinga.com/api/v2/news'
            headers = {'Accept': 'application/json'}
            params = {
                'token': self.BENZINGA_KEY,
                'pageSize': 20,
                'displayOutput': 'full'
            }
            
            if tickers:
                params['tickers'] = ','.join(tickers[:5])
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for article in data:
                    headline = article.get('title', '')
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': article.get('created', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': f"benzinga:{article.get('author', {}).get('name', 'Benzinga')}",
                            'mapped_currencies': mapped,
                            'url': article.get('url', ''),
                            'description': article.get('teaser', '')
                        })
        except Exception as e:
            logger.error(f"Benzinga error: {e}")
        
        return results

    def _fetch_stocktwits(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch trending symbols and ideas from StockTwits."""
        results = []
        
        try:
            logger.info("Fetching from StockTwits")
            self._rate_limit('stocktwits')
            
            # Get trending symbols first
            if tickers:
                for ticker in tickers[:2]:  # Limit to avoid rate limits
                    url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            for message in data.get('messages', [])[:10]:
                                body = message.get('body', '')
                                if body and len(body) > 50:  # Filter short messages
                                    mapped = self._map_to_currencies(body)
                                    results.append({
                                        'timestamp': message.get('created_at', datetime.utcnow().isoformat()),
                                        'headline': body[:150] + '...' if len(body) > 150 else body,
                                        'source': f"stocktwits:{message.get('user', {}).get('username', 'StockTwits')}",
                                        'mapped_currencies': mapped,
                                        'url': f"https://stocktwits.com/symbol/{ticker}",
                                        'sentiment': message.get('entities', {}).get('sentiment', {}).get('basic', 'neutral')
                                    })
                    except Exception as e:
                        logger.warning(f"StockTwits symbol {ticker} error: {e}")
                        
        except Exception as e:
            logger.error(f"StockTwits error: {e}")
        
        return results

    def _fetch_reddit(self, subreddits: List[str] = None) -> List[Dict]:
        """Fetch from Reddit finance communities."""
        results = []
        
        try:
            logger.info("Fetching from Reddit")
            self._rate_limit('reddit')
            
            # Default finance subreddits
            if not subreddits:
                subreddits = ['wallstreetbets', 'investing', 'stocks', 'StockMarket']
            
            for subreddit in subreddits[:2]:  # Limit to 2 subreddits
                try:
                    url = f'https://www.reddit.com/r/{subreddit}/hot.json'
                    headers = {'User-Agent': 'TradingBot/1.0'}
                    params = {'limit': 10}
                    
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            title = post_data.get('title', '')
                            if title and post_data.get('score', 0) > 50:  # Filter by popularity
                                mapped = self._map_to_currencies(title)
                                results.append({
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'headline': title,
                                    'source': f"reddit:r/{subreddit}",
                                    'mapped_currencies': mapped,
                                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                    'description': post_data.get('selftext', '')[:200],
                                    'score': post_data.get('score', 0)
                                })
                except Exception as e:
                    logger.warning(f"Reddit r/{subreddit} error: {e}")
                    
        except Exception as e:
            logger.error(f"Reddit error: {e}")
        
        return results

    def _fetch_massive(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from Massive Finance News API (massive.com)."""
        results = []
        if not self.MASSIVE_KEY:
            return results
        
        try:
            logger.info("Fetching from Massive Finance News API")
            self._rate_limit('massive')
            
            # Use official Massive Python library
            try:
                from massive import RESTClient
                from massive.rest.models import TickerNews
            except ImportError:
                logger.error("Massive library not installed. Run 'pip install massive'")
                return results
            
            client = RESTClient(self.MASSIVE_KEY)
            
            # Fetch news
            for item in client.list_ticker_news(
                order="asc",
                limit="50",
                sort="published_utc",
            ):
                if isinstance(item, TickerNews):
                    headline = item.title
                    if headline:
                        mapped = self._map_to_currencies(headline)
                        
                        # Extract sentiment if available
                        sentiment_label = getattr(item, 'sentiment', 'neutral')
                        sentiment_score = getattr(item, 'sentiment_score', 0)
                        
                        results.append({
                            'timestamp': getattr(item, 'published_utc', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': 'massive',
                            'mapped_currencies': mapped,
                            'url': getattr(item, 'article_url', ''),
                            'description': getattr(item, 'description', ''),
                            'sentiment': str(sentiment_label) if sentiment_label else 'neutral',
                            'sentiment_score': float(sentiment_score) if sentiment_score else 0,
                            'tickers': getattr(item, 'tickers', []),
                            'category': getattr(item, 'category', 'general')
                        })
            
            logger.info(f"Massive API returned {len(results)} articles")
                
        except Exception as e:
            logger.error(f"Massive API error: {e}")
        
        return results

    def _fetch_yahoo_finance_news(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch news from Yahoo Finance RSS."""
        results = []
        
        try:
            logger.info("Fetching from Yahoo Finance")
            self._rate_limit('yahoo_finance')
            
            try:
                import feedparser
                
                if tickers:
                    # Fetch for specific tickers
                    for ticker in tickers[:3]:
                        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
                        try:
                            feed = feedparser.parse(url)
                            for entry in feed.entries[:5]:
                                headline = entry.get('title', '')
                                if headline:
                                    mapped = self._map_to_currencies(headline)
                                    results.append({
                                        'timestamp': entry.get('published', datetime.utcnow().isoformat()),
                                        'headline': headline,
                                        'source': f"yahoo:{ticker}",
                                        'mapped_currencies': mapped,
                                        'url': entry.get('link', ''),
                                        'description': entry.get('summary', '')
                                    })
                        except Exception as e:
                            logger.warning(f"Yahoo Finance {ticker} error: {e}")
                else:
                    # Fetch general market news
                    url = 'https://feeds.finance.yahoo.com/rss/2.0/headline?region=US&lang=en-US'
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:10]:
                        headline = entry.get('title', '')
                        if headline:
                            mapped = self._map_to_currencies(headline)
                            results.append({
                                'timestamp': entry.get('published', datetime.utcnow().isoformat()),
                                'headline': headline,
                                'source': 'yahoo:market',
                                'mapped_currencies': mapped,
                                'url': entry.get('link', ''),
                                'description': entry.get('summary', '')
                            })
                            
            except ImportError:
                logger.info("feedparser not installed, skipping Yahoo Finance RSS")
                
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
        
        return results

    def _fetch_investing_com(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch from Investing.com news section."""
        results = []
        
        try:
            logger.info("Fetching from Investing.com")
            self._rate_limit('investing_com')
            
            # Investing.com news feed
            url = 'https://www.investing.com/rss/news.rss'
            
            try:
                import feedparser
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:15]:
                    headline = entry.get('title', '')
                    if headline:
                        # Filter if tickers provided
                        if tickers:
                            ticker_found = any(t.lower() in headline.lower() for t in tickers)
                            if not ticker_found:
                                continue
                        
                        mapped = self._map_to_currencies(headline)
                        results.append({
                            'timestamp': entry.get('published', datetime.utcnow().isoformat()),
                            'headline': headline,
                            'source': 'investing.com',
                            'mapped_currencies': mapped,
                            'url': entry.get('link', ''),
                            'description': entry.get('summary', '')
                        })
                        
            except ImportError:
                logger.info("feedparser not installed, skipping Investing.com")
                
        except Exception as e:
            logger.error(f"Investing.com error: {e}")
        
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
        except Exception as e:
            logger.error(f'Failed to store raw news to {path}: {e}')
            logger.exception('Exception details')
