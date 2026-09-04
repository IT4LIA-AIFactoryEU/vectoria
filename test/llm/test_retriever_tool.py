#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from langchain_core.documents import Document

from vectoria_lib.llm.agents.tools.retriever_tool import RetrieverTool
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME


class _StubLangchainRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs

    async def ainvoke(self, query):
        return self._docs


class _StubRetriever:
    def __init__(self, docs):
        self._lc = _StubLangchainRetriever(docs)

    def as_langchain_retriever(self):
        return self._lc


def _make_tool(docs):
    return RetrieverTool(retriever=_StubRetriever(docs))


def test_run_without_filter_returns_all_docs():
    docs = [
        Document(page_content="a", metadata={DOC_FILE_NAME: "report.pdf"}),
        Document(page_content="b", metadata={DOC_FILE_NAME: "other.pdf"}),
    ]
    tool = _make_tool(docs)
    result = tool._run(query="q")
    assert "a" in result and "b" in result


def test_run_with_filter_restricts_to_named_document():
    docs = [
        Document(page_content="a", metadata={DOC_FILE_NAME: "report.pdf"}),
        Document(page_content="b", metadata={DOC_FILE_NAME: "other.pdf"}),
    ]
    tool = _make_tool(docs)
    result = tool._run(query="q", doc_file_names=["report.pdf"])
    assert "a" in result
    assert "b" not in result


def test_run_with_filter_matching_nothing_returns_helpful_message():
    docs = [Document(page_content="a", metadata={DOC_FILE_NAME: "report.pdf"})]
    tool = _make_tool(docs)
    result = tool._run(query="q", doc_file_names=["missing.pdf"])
    assert "missing.pdf" in result
    assert "list_documents" in result
