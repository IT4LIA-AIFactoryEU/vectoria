#
# VECTORIA
#
# @authors : Mattia Buzzoni
#

try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever as _LangchainMQR
except ImportError:
    raise ImportError("MultiQueryRetriever requires langchain_classic.")

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase

# Compact prompt: 2 variants instead of 3, no verbose preamble.
# LangChain's MultiQueryRetriever parses output by splitting on newlines.
_MULTI_QUERY_PROMPT = PromptTemplate.from_template(
    "Rewrite the following question in 2 different ways to improve document retrieval. "
    "Output one rewrite per line, nothing else.\n"
    "Question: {question}"
)


class _InvokeAdapter(BaseRetriever):
    """
    Adapts any object with an .invoke(query) method to LangChain's BaseRetriever
    interface. Needed for retrievers (e.g. HybridRetriever) that are not natively
    LangChain BaseRetriever instances.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, retriever):
        super().__init__()
        object.__setattr__(self, "_inner", retriever)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        del run_manager  # unused, required by interface
        return object.__getattribute__(self, "_inner").invoke(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        del run_manager  # unused, required by interface
        # ponytail: without this override, BaseRetriever's default ainvoke would just
        # thread-wrap the sync invoke() above, defeating MultiQueryRetriever's gather()
        # over query variants (see aretrieve_documents in langchain_classic).
        inner = object.__getattribute__(self, "_inner")
        return await inner.ainvoke(query)


class MultiQueryRetriever(RetrieverBase):
    """
    Wraps any RetrieverBase with LangChain's MultiQueryRetriever.

    The LLM generates N alternative phrasings of the original query;
    results from all variants are merged (deduplication by document id)
    before being returned. This improves recall when the user's wording
    does not closely match the indexed text.
    """

    def __init__(self, base_retriever: RetrieverBase, llm):
        super().__init__()
        self.logger.info("Wrapping retriever with MultiQueryRetriever")

        lc_retriever = base_retriever.as_langchain_retriever()
        if not isinstance(lc_retriever, BaseRetriever):
            lc_retriever = _InvokeAdapter(lc_retriever)

        self._wrapped = _LangchainMQR.from_llm(retriever=lc_retriever, llm=llm, prompt=_MULTI_QUERY_PROMPT)
        # base_retriever.from_vector_store() already ran (RetrieverBuilder wraps
        # after building), so its wrapped_vector_store is set by now -- expose it
        # here too, since callers (e.g. /eval/chunks) reach for it on whatever
        # retriever_tool.retriever ends up being, multi-query-wrapped or not.
        self.wrapped_vector_store = base_retriever.wrapped_vector_store

    def from_vector_store(self, vector_store: VectorStoreBase):
        return self  # already built in __init__

    def as_langchain_retriever(self):
        return self._wrapped
