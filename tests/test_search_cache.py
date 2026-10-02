import importlib.util
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def cache(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'src/utils/db.py'
    spec = importlib.util.spec_from_file_location('search_cache_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CACHE_DIR = str(tmp_path / 'new-cache')
    return module


def test_first_lookup_creates_cache_and_supports_updates(cache):
    assert cache.get_cached_answer('question') is None
    cache.save_query('question', 'answer')
    first = cache.get_cached_answer('question')
    cache.save_query('question', 'new answer')
    updated = cache.get_cached_answer('question')
    assert updated[0] == 'new answer'
    assert updated[1] == first[1]
    assert cache.remove_query('question')
    assert cache.get_cached_answer('question') is None


def test_empty_database_repaired_without_dropping_existing_tables(cache):
    Path(cache.CACHE_DIR).mkdir()
    with sqlite3.connect(str(Path(cache.CACHE_DIR) / 'search.db')) as conn:
        conn.execute('CREATE TABLE retained (value TEXT)')
        conn.execute("INSERT INTO retained VALUES ('keep')")
    assert cache.get_cached_answer('question') is None
    cache.save_query('question', 'answer')
    cache.init_db()
    assert cache.get_cached_answer('question')[0] == 'answer'
    with sqlite3.connect(str(Path(cache.CACHE_DIR) / 'search.db')) as conn:
        assert conn.execute('SELECT value FROM retained').fetchone() == ('keep',)
