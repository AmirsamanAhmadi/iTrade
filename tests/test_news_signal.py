import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from services.news_signal import NewsSignalService


def make_news_file(path: Path, entries: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_process_recent_basic(tmp_path):
    news_db = tmp_path / "news"
    signal_db = tmp_path / "news_signals"
    service = NewsSignalService(news_db=news_db, signal_db=signal_db, half_life_hours=24.0)

    now = datetime.utcnow()
    entries = [
        {"timestamp": (now - timedelta(hours=1)).isoformat(), "headline": "EURUSD surges after strong growth data", "mapped_currencies": ["EUR"]},
        {"timestamp": (now - timedelta(hours=2)).isoformat(), "headline": "Dollar weak; USD down on dovish fed", "mapped_currencies": ["USD"]},
        {"timestamp": (now - timedelta(days=2)).isoformat(), "headline": "Old news should be ignored", "mapped_currencies": ["USD"]},
    ]

    make_news_file(news_db / f"news_{now.date().isoformat()}.jsonl", entries[:2])
    # old file
    make_news_file(news_db / f"news_{(now - timedelta(days=2)).date().isoformat()}.jsonl", [entries[2]])

    snapshot = service.process_recent(days=1)
    assert snapshot is not None
    by = snapshot["by_currency"]
    # EUR should have 1 positive headline
    assert by["EUR"]["count"] == 1
    assert by["EUR"]["avg_score"] > 0
    # USD should have 1 headline in recent window
    assert by["USD"]["count"] == 1
    assert by["USD"]["avg_score"] < 0 or by["USD"]["avg_score"] <= 0.0


def test_score_and_decay():
    service = NewsSignalService(half_life_hours=1.0)
    now = datetime.utcnow()
    recent_ts = now.isoformat()
    old_ts = (now - timedelta(hours=10)).isoformat()

    s_recent = service.process_recent  # just to access methods
    score_recent = service.score_text("Stock surges after beat")
    score_old = service.score_text("Stock plunges on recession fears")
    assert score_recent != 0
    assert score_old != 0

    decay_recent = service.recency_decay(recent_ts)
    decay_old = service.recency_decay(old_ts)
    assert decay_recent > decay_old
