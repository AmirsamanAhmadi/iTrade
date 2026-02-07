"""News service wrapper around finvizfinance (optional) and simple mapping to currencies.

This module fetches headlines and stores raw news to disk (JSON lines). It includes a simple currency-mapping helper
that tags headlines mentioning USD or EUR.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    from finvizfinance.news import News as FinvizNews  # type: ignore
    FINVIZ_AVAILABLE = True
except Exception:
    FINVIZ_AVAILABLE = False

NEWS_DB = Path(__file__).resolve().parents[2] / 'db' / 'news'
NEWS_DB.mkdir(parents=True, exist_ok=True)

@dataclass
class NewsService:
    source: str = 'finviz'

    def fetch_headlines(self, tickers: List[str] = None) -> List[Dict]:
        """Fetch headlines from finviz and return list of dicts.

        Each dict contains: timestamp (ISO), headline, source, mapped_currencies (subset of ['USD','EUR']).
        """
        if not FINVIZ_AVAILABLE:
            logger.warning("finvizfinance not available; returning empty list")
            return []

        results = []

        try:
            # Always fetch general market news from finviz
            logger.info("Fetching general market news from Finviz")
            try:
                news = FinvizNews() # General news
                news_data = news.get_news()
                if news_data and 'news' in news_data:
                    df = news_data['news']
                    if not df.empty:
                        for _, row in df.iterrows():
                            ts = str(row.get('Date', ''))
                            headline = str(row.get('Title', ''))
                            mapped = self._map_to_currencies(headline)
                            rec = {'timestamp': ts, 'headline': headline, 'source': 'finviz_general', 'mapped_currencies': mapped}
                            results.append(rec)
            except Exception as e:
                logger.error(f"Error fetching general news: {e}")

        except Exception:
            logger.exception('Failed to fetch headlines from finviz')

        # store raw and deduplicate
        seen_headlines = set()
        unique_results = []
        for r in results:
            h = r.get('headline')
            if h and h not in seen_headlines:
                seen_headlines.add(h)
                unique_results.append(r)
                self._store_raw(r)

        return unique_results

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
