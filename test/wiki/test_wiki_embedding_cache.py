import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache


@pytest.fixture
def cache(tmp_path):
  return WikiEmbeddingCache(tmp_path / "wiki")


def test_load_empty(cache):
  assert cache.load() == {}


def test_save_and_load(cache):
  data = {"topic_a": [0.1, 0.2, 0.3]}
  cache.save(data)
  assert cache.load() == data


def test_remove_existing(cache):
  cache.save({"topic_a": [0.1], "topic_b": [0.2]})
  cache.remove("topic_a")
  assert "topic_a" not in cache.load()
  assert "topic_b" in cache.load()


def test_remove_missing_noop(cache):
  cache.save({"topic_a": [0.1]})
  cache.remove("nonexistent")
  assert cache.load() == {"topic_a": [0.1]}


def test_best_match_returns_closest():
  pool = [[1.0, 0.0], [0.0, 1.0]]
  i, sim = WikiEmbeddingCache.best_match([0.9, 0.1], pool)
  assert i == 0
  assert sim > 0.9


def test_best_match_empty_pool():
  i, sim = WikiEmbeddingCache.best_match([1.0, 0.0], [])
  assert i is None
  assert sim == 0.0


def test_load_or_compute_uses_cache(cache):
  mock_embed = MagicMock()
  mock_embed.embed_documents.return_value = [[0.5, 0.5]]
  cache.save({"topic_a": [0.1, 0.2]})
  result = cache.load_or_compute(mock_embed, ["topic_a"])
  mock_embed.embed_documents.assert_not_called()
  assert result == [[0.1, 0.2]]


def test_load_or_compute_embeds_missing(cache):
  mock_embed = MagicMock()
  mock_embed.embed_documents.return_value = [[0.3, 0.4]]
  result = cache.load_or_compute(mock_embed, ["new_topic"])
  mock_embed.embed_documents.assert_called_once_with(["new_topic"])
  assert result == [[0.3, 0.4]]


def test_load_or_compute_empty_pages(cache):
  mock_embed = MagicMock()
  result = cache.load_or_compute(mock_embed, [])
  assert result == []
  mock_embed.embed_documents.assert_not_called()
