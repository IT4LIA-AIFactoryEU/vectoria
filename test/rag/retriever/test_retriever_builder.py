#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import pytest
from langchain_core.documents import Document

from vectoria_lib.rag.retriever.hybrid_retriever import HybridRetriever
from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder


def test_build_hybrid_retriever(config, chroma_index_test_folder):

    VectorStoreBuilder.CACHE.clear()

    # Empty vector store should raise
    with pytest.raises(ValueError):
        RetrieverBuilder.build(config.get("retriever"), vector_store=ChromaVectorStore(**config.get("vector_store")))

    VectorStoreBuilder.CACHE.clear()

    vector_store = VectorStoreBuilder.build(config.get("vector_store"), index_path=chroma_index_test_folder)

    retriever = RetrieverBuilder.build(config.get("retriever"), vector_store=vector_store)
    assert isinstance(retriever, RetrieverBase)
    assert isinstance(retriever, HybridRetriever)
    assert retriever.search_type == "hybrid"
    # Regression: /eval/chunks reads retriever.wrapped_vector_store to fetch stored
    # chunks -- from_vector_store() must actually set it, not leave the RetrieverBase default.
    assert retriever.wrapped_vector_store is vector_store

    docs = retriever.as_langchain_retriever().invoke("What is the Transformer model?")
    assert isinstance(docs, list)
    assert len(docs) == config.get("retriever", "top_k")
    assert isinstance(docs[0], Document)


def test_build_retriever_for_existing_index_on_different_backend_than_configured(
    config, chroma_index_test_folder
):
    """Regression test for the mismatch found testing agent.mode=test on Leonardo:
    an index written with one backend (chroma) must still load when
    vector_store.name now configures a different default (qdrant) --
    VectorStoreBuilder auto-detects the on-disk backend (see discovery.py),
    so RetrieverBuilder must trust the instance it's handed, not re-read Config()."""
    VectorStoreBuilder.CACHE.clear()

    config.set("vector_store", "name", value="qdrant")
    vector_store = VectorStoreBuilder.build(config.get("vector_store"), index_path=chroma_index_test_folder)
    assert isinstance(vector_store, ChromaVectorStore)

    retriever = RetrieverBuilder.build(config.get("retriever"), vector_store=vector_store)
    assert isinstance(retriever, HybridRetriever)
