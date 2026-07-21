import json
from pathlib import Path

import pytest

from vectoria_lib.wiki.wiki_storage import WikiStorage


@pytest.fixture
def storage(tmp_path):
  return WikiStorage(tmp_path / "wiki")


def test_storage_creates_dirs(tmp_path):
  s = WikiStorage(tmp_path / "wiki")
  assert (tmp_path / "wiki").exists()
  assert (tmp_path / "wiki" / "pages").exists()


def test_pending_round_trip(storage):
  entries = [{"source": "a.pdf", "snippet": "foo", "timestamp": "2026-01-01"}]
  storage.save_pending(entries)
  assert storage.load_pending() == entries


def test_sources_round_trip(storage):
  storage.save_sources({"topic": "a.pdf"})
  assert storage.load_sources() == {"topic": "a.pdf"}


def test_write_and_get_page(storage):
  storage.write_page("Quantum", "# Quantum\n\nbody")
  assert storage.get_page("Quantum") == "# Quantum\n\nbody"


def test_list_pages(storage):
  storage.write_page("Alpha", "content")
  storage.write_page("Beta", "content")
  assert storage.list_pages() == ["Alpha", "Beta"]


def test_delete_page(storage):
  storage.write_page("ToDelete", "content")
  storage.delete_page("ToDelete")
  assert storage.get_page("ToDelete") is None


def test_search_pages_found(storage):
  storage.write_page("Cats", "cats are fluffy animals")
  result = storage.search_pages("fluffy")
  assert "Cats" in result


def test_search_pages_not_found(storage):
  storage.write_page("Dogs", "dogs bark loudly")
  result = storage.search_pages("fluffy")
  assert "No wiki pages match" in result


def test_safe_stem_normalizes(storage):
  stem = storage.safe_stem("Quantum Mechanics!")
  assert len(stem) <= 60
  assert "!" not in stem


def test_page_path(storage):
  p = storage.page_path("Hello World")
  assert p.suffix == ".md"
