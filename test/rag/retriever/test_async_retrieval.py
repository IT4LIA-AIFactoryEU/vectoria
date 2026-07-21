import asyncio
from unittest.mock import AsyncMock, MagicMock

from langchain_core.documents import Document

from vectoria_lib.llm.agents.tools.retriever_tool import RetrieverTool
from vectoria_lib.rag.retriever.hybrid_retriever import HybridRetriever
from vectoria_lib.rag.retriever.multi_query_retriever import _InvokeAdapter

# ponytail: no pytest-asyncio in this project's deps — asyncio.run() covers it in one line.


def test_hybrid_retriever_ainvoke_delegates_to_ensemble_and_truncates():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.top_k = 1
    retriever._ensemble = MagicMock()
    retriever._ensemble.ainvoke = AsyncMock(return_value=[Document("a"), Document("b")])

    docs = asyncio.run(retriever.ainvoke("query"))

    retriever._ensemble.ainvoke.assert_awaited_once_with("query")
    assert docs == [Document("a")]


def test_invoke_adapter_aget_relevant_documents_awaits_inner_ainvoke():
    inner = MagicMock()
    inner.ainvoke = AsyncMock(return_value=[Document("x")])
    adapter = _InvokeAdapter(inner)

    docs = asyncio.run(adapter._aget_relevant_documents("query", run_manager=None))

    inner.ainvoke.assert_awaited_once_with("query")
    assert docs == [Document("x")]


def test_retriever_tool_arun_matches_sync_run_output():
    docs = [Document("chunk one", metadata={"doc_file_name": "a.pdf"})]
    lc_retriever = MagicMock()
    lc_retriever.invoke.return_value = docs
    lc_retriever.ainvoke = AsyncMock(return_value=docs)

    base_retriever = MagicMock()
    base_retriever.as_langchain_retriever.return_value = lc_retriever

    tool = RetrieverTool(retriever=base_retriever)

    sync_result = tool._run("query")
    async_result = asyncio.run(tool._arun("query"))

    assert sync_result == async_result
    lc_retriever.ainvoke.assert_awaited_once_with("query")
