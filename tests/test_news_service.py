import pytest
from backend.services.news_service import NewsService


def test_map_to_currencies():
    ns = NewsService()
    assert ns._map_to_currencies('EUR/USD jumps after ECB press conference') == ['EUR', 'USD']
    assert ns._map_to_currencies('Dollar edges higher on US jobs') == ['USD']
    assert ns._map_to_currencies('No currency mentioned here') == []


def test_store_raw(tmp_path, monkeypatch):
    ns = NewsService()
    rec = {'timestamp':'2024-01-01T00:00:00Z', 'headline':'Euro dips', 'source':'finviz','mapped_currencies':['EUR']}
    # monkeypatch NEWS_DB to tmp_path
    monkeypatch.setattr('backend.services.news_service.NEWS_DB', tmp_path)
    ns._store_raw(rec)
    files = list(tmp_path.glob('news_*.jsonl'))
    assert len(files) == 1
    content = files[0].read_text()
    assert 'Euro dips' in content
