import threading
import time
from unittest.mock import MagicMock

from vectoria_lib.core.config_schema import VectorStoreConfig
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder


def test_concurrent_build_same_key_constructs_once(monkeypatch, tmp_path):
    """Regression test for the CACHE race: N threads racing VectorStoreBuilder.build()
    on the same key must not each construct their own vector store instance."""
    VectorStoreBuilder.CACHE.clear()

    build_calls = []

    def fake_build_vector_store(config):
        build_calls.append(1)
        time.sleep(0.05)  # widen the race window between the CACHE check and insert
        vs = MagicMock()
        vs.load_from_disk = MagicMock()
        return vs

    monkeypatch.setattr(VectorStoreBuilder, "_build_vector_store", staticmethod(fake_build_vector_store))

    cfg = VectorStoreConfig(name="chroma", model_name="m", device="cpu", collection_name="c")
    # detect_vector_store_type() identifies the backend from on-disk signatures,
    # so the fake index path needs a real chroma marker to match cfg's name.
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "chroma.sqlite3").touch()
    results = []
    errors = []

    def worker():
        try:
            results.append(VectorStoreBuilder.build(cfg, index_path=str(index_path)))
        except Exception as exc:  # pragma: no cover - only hit on real bugs
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(build_calls) == 1
    assert len({id(vs) for vs in results}) == 1

    VectorStoreBuilder.CACHE.clear()
