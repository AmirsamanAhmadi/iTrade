"""Convert raw news headlines into simple explainable currency signals.

Features:
- Clean headline text (lowercase, remove punctuation)
- Lexicon-based sentiment scoring (small built-in lexicon)
- Recency decay (half-life parameter)
- Impact weighting heuristics (strong verbs, macro keywords)
- Aggregate per currency (USD, EUR)
- Store snapshots as JSON lines under db/news_signals/

Keep it simple and explainable.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


NEWS_DB = Path(__file__).resolve().parents[2] / "db" / "news"
SIGNAL_DB = Path(__file__).resolve().parents[2] / "db" / "news_signals"
SIGNAL_DB.mkdir(parents=True, exist_ok=True)

# Small explainable lexicon. Values are intuitive (+1 positive, -1 negative with magnitudes).
LEXICON: Dict[str, float] = {
    "gain": 1.0,
    "gains": 1.0,
    "rise": 0.8,
    "rises": 0.8,
    "surge": 1.5,
    "surges": 1.5,
    "soar": 1.3,
    "soars": 1.3,
    "beat": 0.7,
    "beats": 0.7,
    "positive": 0.6,
    "up": 0.3,
    "strong": 0.8,

    "fall": -0.8,
    "falls": -0.8,
    "plunge": -1.5,
    "plunges": -1.5,
    "slide": -0.9,
    "slides": -0.9,
    "miss": -0.7,
    "misses": -0.7,
    "negative": -0.6,
    "down": -0.3,
    "weak": -0.8,

    # macro words that often carry sentiment when paired with context
    "inflation": -0.3,
    "deflation": 0.3,
    "growth": 0.4,
    "recession": -1.0,
    "rate": -0.2,
    "rates": -0.2,
}

IMPACT_KEYWORDS = {
    # strong verbs
    "surge": 2.0,
    "plunge": 2.0,
    "soar": 1.8,
    "crash": 2.2,
    # macro keywords that increase impact
    "fed": 1.5,
    "federal": 1.5,
    "ecb": 1.5,
    "inflation": 1.4,
    "rates": 1.3,
}

CURRENCY_LIST = ["USD", "EUR"]

CLEAN_RE = re.compile(r"[^a-z0-9\s]")


from dataclasses import field

@dataclass
class NewsSignalService:
    news_db: Path = NEWS_DB
    signal_db: Path = SIGNAL_DB
    lexicon: Dict[str, float] = field(default_factory=lambda: LEXICON.copy())
    impact_keywords: Dict[str, float] = field(default_factory=lambda: IMPACT_KEYWORDS.copy())
    half_life_hours: float = 6.0  # recency half-life for decay

    def clean_text(self, text: str) -> str:
        s = (text or "").lower()
        s = CLEAN_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def score_text(self, text: str) -> float:
        """Simple lexicon scoring: sum word weights, normalized by sqrt(word_count) to reduce length bias."""
        s = self.clean_text(text)
        if not s:
            return 0.0
        words = s.split()
        raw = 0.0
        for w in words:
            raw += self.lexicon.get(w, 0.0)
        norm = raw / math.sqrt(len(words)) if len(words) else 0.0
        # clamp to [-3,3] as a simple safety cap for extreme headlines
        return max(-3.0, min(3.0, norm))

    def impact_weight(self, text: str) -> float:
        s = self.clean_text(text)
        mult = 1.0
        for k, v in self.impact_keywords.items():
            if k in s:
                mult = max(mult, v)
        return mult

    def recency_decay(self, timestamp_iso: str) -> float:
        try:
            t = datetime.fromisoformat(timestamp_iso)
        except Exception:
            # if timestamp missing or bad, treat as old
            return 0.0
        now = datetime.utcnow()
        age_sec = (now - t).total_seconds()
        half_life = self.half_life_hours * 3600.0
        if age_sec < 0:
            age_sec = 0.0
        # exponential decay: weight = 2^(-age / half_life)
        return 2 ** (-age_sec / half_life)

    def process_recent(self, days: int = 1, max_items: Optional[int] = None) -> Dict[str, Dict]:
        """Process recent news JSONL files and produce aggregated signals per currency.

        Returns a dict keyed by currency with values: {count, sum_weighted_score, avg_score}
        Also appends a snapshot JSON line to the signal DB.
        """
        files = []
        for p in sorted(self.news_db.glob("news_*.jsonl")):
            files.append(p)
        # limit to files within `days`
        cutoff = datetime.utcnow().date() - timedelta(days=days - 1)
        selected = [p for p in files if self._file_date(p) >= cutoff]

        records = []
        for p in selected:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        records.append(rec)
                        if max_items and len(records) >= max_items:
                            break
            except Exception:
                continue
            if max_items and len(records) >= max_items:
                break

        # aggregate
        agg: Dict[str, Dict] = {c: {"count": 0, "sum_weighted": 0.0, "sum_raw_weights": 0.0} for c in CURRENCY_LIST}

        for rec in records:
            headline = rec.get("headline", "")
            ts = rec.get("timestamp") or datetime.utcnow().isoformat()
            mapped = rec.get("mapped_currencies") or []
            # if mapping absent, attempt to map heuristically
            if not mapped:
                mapped = self._map_heuristic(headline)

            base_score = self.score_text(headline)
            impact = self.impact_weight(headline)
            decay = self.recency_decay(ts)
            effective_weight = impact * decay
            weighted_score = base_score * effective_weight

            for cur in mapped:
                if cur not in CURRENCY_LIST:
                    continue
                agg[cur]["count"] += 1
                agg[cur]["sum_weighted"] += weighted_score
                agg[cur]["sum_raw_weights"] += effective_weight

        # compute averages
        snapshot = {"timestamp": datetime.utcnow().isoformat(), "by_currency": {}}
        for cur, v in agg.items():
            avg = 0.0
            if v["sum_raw_weights"]:
                avg = v["sum_weighted"] / v["sum_raw_weights"]
            snapshot["by_currency"][cur] = {
                "count": v["count"],
                "sum_weighted": v["sum_weighted"],
                "sum_raw_weights": v["sum_raw_weights"],
                "avg_score": avg,
            }

        # store snapshot
        fname = self.signal_db / f"news_signals_{datetime.utcnow().date().isoformat()}.jsonl"
        try:
            with open(fname, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return snapshot

    def latest_snapshot(self) -> Optional[Dict]:
        files = sorted(self.signal_db.glob("news_signals_*.jsonl"))
        if not files:
            return None
        last = files[-1]
        try:
            with open(last, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except Exception:
            return None

    def _file_date(self, p: Path) -> datetime.date:
        # file name like news_YYYY-MM-DD.jsonl
        try:
            name = p.stem
            parts = name.split("_")
            date_str = parts[-1]
            return datetime.fromisoformat(date_str).date()
        except Exception:
            return datetime(1970, 1, 1).date()

    def _map_heuristic(self, headline: str) -> List[str]:
        s = headline.lower()
        mapped: List[str] = []
        if any(x in s for x in ["usd", "dollar", "dollars"]):
            mapped.append("USD")
        if any(x in s for x in ["eur", "euro", "euros"]):
            mapped.append("EUR")
        return mapped
