#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import threading
import time
from unittest.mock import MagicMock

from vectoria_lib.rag.vector_store.vector_store_base import _EMBEDDER_CACHE, _LockedEmbeddings, get_shared_embedder


def test_get_shared_embedder_concurrent_construction_builds_once(monkeypatch):
    """Regression test: tenants load concurrently at startup (real OS threads via
    asyncio.to_thread). An unprotected cache check let two threads both see a miss
    and both build a HuggingFaceEmbeddings for the same GPU at the same time --
    the same category of race that crashed the reranker's CrossEncoder under
    accelerate's dispatch_model() ("Cannot copy out of meta tensor; no data!")."""
    _EMBEDDER_CACHE.clear()
    build_calls = []

    def fake_embeddings(**kwargs):
        build_calls.append(1)
        time.sleep(0.05)  # widen the race window between the cache check and insert
        return MagicMock()

    monkeypatch.setattr("vectoria_lib.rag.vector_store.vector_store_base.HuggingFaceEmbeddings", fake_embeddings)

    results = []
    errors = []

    def worker():
        try:
            results.append(
                get_shared_embedder(
                    model_name="fake/model", device="cuda:0", normalize_embeddings=True, batch_size=64
                )
            )
        except Exception as exc:  # pragma: no cover - only hit on real bugs
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(build_calls) == 1
    assert len({id(r) for r in results}) == 1

    _EMBEDDER_CACHE.clear()


def test_locked_embeddings_serializes_concurrent_embed_query_calls():
    """Regression test for a production crash: the shared HF fast tokenizer
    (Rust/PyO3) is not safe for concurrent .encode() calls from multiple
    threads. A single agent turn routinely has several tool calls running
    concurrently (retriever, wiki_query, graph_search via LangGraph's
    ToolNode), and the retriever itself fans out multiple queries
    concurrently (MultiQueryRetriever, EnsembleRetriever) -- each hitting the
    SAME shared embedder via asyncio's run_in_executor thread pool. Observed
    as "RuntimeError: Already borrowed" from enable_padding(), which mutates
    the tokenizer's internal state on every call. _LockedEmbeddings must
    ensure only one thread is ever inside the wrapped embedder at a time."""
    active = []
    max_concurrent = []
    guard = threading.Lock()

    class _FakeInner:
        def embed_query(self, text):
            with guard:
                active.append(1)
                max_concurrent.append(len(active))
            time.sleep(0.02)  # widen the window so any overlap would show up
            with guard:
                active.pop()
            return [0.0]

        def embed_documents(self, texts):  # pragma: no cover - not exercised here
            raise NotImplementedError

    wrapped = _LockedEmbeddings(_FakeInner())

    def worker():
        wrapped.embed_query("hello")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(max_concurrent) == 1
