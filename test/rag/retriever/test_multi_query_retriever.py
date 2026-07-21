from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from vectoria_lib.rag.retriever.multi_query_retriever import MultiQueryRetriever
from vectoria_lib.rag.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder


def test_multi_query_retriever_requires_llm(config, chroma_index_test_folder):
    VectorStoreBuilder.CACHE.clear()
    vector_store = VectorStoreBuilder.build(
        config.get("vector_store"),
        index_path=chroma_index_test_folder,
    )

    retriever_cfg = dict(config.get("retriever"), multi_query=True)
    with pytest.raises(ValueError, match="multi_query=True requires an LLM"):
        RetrieverBuilder.build(retriever_cfg, vector_store, llm=None)


def test_multi_query_retriever_wraps_base(config, chroma_index_test_folder):
    VectorStoreBuilder.CACHE.clear()
    vector_store = VectorStoreBuilder.build(
        config.get("vector_store"),
        index_path=chroma_index_test_folder,
    )

    # Use a mock LLM that returns 3 query variants
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="What is attention?\nHow does self-attention work?\nExplain transformer attention"
    )

    retriever_cfg = dict(config.get("retriever"), multi_query=True)
    retriever = RetrieverBuilder.build(retriever_cfg, vector_store, llm=mock_llm)

    assert isinstance(retriever, MultiQueryRetriever)
    # Regression: /eval/chunks reaches for retriever.wrapped_vector_store regardless
    # of whether multi_query wraps the base retriever -- this must not be None.
    assert retriever.wrapped_vector_store is vector_store
