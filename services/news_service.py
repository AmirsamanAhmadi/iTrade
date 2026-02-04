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
        """Fetch headlines from finviz for supplied tickers (if available) and return list of dicts.

        Each dict contains: timestamp (ISO), headline, source, mapped_currencies (subset of ['USD','EUR']).
        """
        if not FINVIZ_AVAILABLE:
            logger.warning("finvizfinance not available; returning empty list")
            return []

        results = []
        # If no tickers provided, fetch general news if available from finviz
        tickers = tickers or []
        try:
            if tickers:
                for t in tickers:
                    news = FinvizNews(ticker=t)
                    items = news.news_headlines
                    for item in items:
                        ts = item.get('date') or ''
                        headline = item.get('title') or item.get('text') or ''
                        mapped = self._map_to_currencies(headline)
                        rec = {'timestamp': ts, 'headline': headline, 'source': 'finviz', 'mapped_currencies': mapped}
                        results.append(rec)
            else:
                # finvizfinance does not provide global feed easily; return empty list
                logger.debug('No tickers provided, finviz fetch skipped')
        except Exception:
            logger.exception('Failed to fetch headlines from finviz')

        # store raw
        for r in results:
            self._store_raw(r)

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
