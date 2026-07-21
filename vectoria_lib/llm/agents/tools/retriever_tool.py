#
# VECTORIA
#
# @authors : Mattia Buzzoni
#
import logging
from collections import defaultdict
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME

logger = logging.getLogger("llm")

_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}\n[Source: {doc_file_name}]")


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"{i + 1}. {format_document(doc, _DOCUMENT_PROMPT)}" for i, doc in enumerate(docs))


class RetrieverInput(BaseModel):
    query: str = Field(description="The search query to retrieve relevant documents")
    doc_file_names: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of exact filenames (e.g. 'report.pdf') to restrict the search to. "
            "Set this only when the user explicitly names one or more documents to search within; "
            "otherwise leave it unset to search across all indexed documents. "
            "Use list_documents first if you need the exact filename."
        ),
    )


class RetrieverTool(BaseTool):
    """
    LangChain Tool that wraps the Vectoria RAG pipeline:
    Retriever → (optional) Reranker → (optional) MMR → (optional) diversity enforcement → (optional) FullParagraphs.

    The LLM receives the formatted document text as the tool result.
    The raw Document objects are stored in `last_docs` so QAAgent can
    access them after graph invocation (e.g. for evaluation or inference output).
    """

    name: str = "retriever"
    description: str = (
        "Search the document knowledge base and retrieve relevant passages "
        "to answer the user's question. Use this tool whenever you need "
        "information from the documents. Optionally restrict the search to "
        "specific document(s) via doc_file_names when the user names them explicitly."
    )
    args_schema: type[BaseModel] = RetrieverInput

    retriever: Any = Field(description="Retriever instance")
    reranker: Any = Field(default=None, description="Reranker instance (optional)")
    full_paragraphs: Any = Field(default=None, description="FullParagraphs instance (optional)")
    min_per_source: int = Field(default=0, description="Minimum chunks per source document (0 = disabled)")
    mmr_enabled: bool = Field(default=False, description="Apply MMR after reranker to reduce semantic redundancy")
    mmr_lambda: float = Field(default=0.7, description="MMR relevance/diversity tradeoff, 0..1")
    mmr_top_k: int = Field(default=0, description="MMR output size (0 = keep all input docs)")
    embedder: Any = Field(default=None, description="Embedder for MMR cosine similarity")

    # Side-channel: raw docs from the last _run call, for QAAgent to read back
    last_docs: list[Document] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str, doc_file_names: Optional[list[str]] = None, run_manager=None) -> str:
        docs: list[Document] = self.retriever.as_langchain_retriever().invoke(query)
        logger.debug("Retrieved %d docs for query: %s", len(docs), query)
        docs, empty_msg = self._filter_by_filenames(docs, doc_file_names)
        if empty_msg:
            return empty_msg
        return _format_docs(self._post_process(query, docs))

    async def _arun(self, query: str, doc_file_names: Optional[list[str]] = None, run_manager=None) -> str:
        # ponytail: unlocks the gather()-based fan-out already implemented in
        # langchain_classic's EnsembleRetriever/MultiQueryRetriever — see
        # HybridRetriever.ainvoke and _InvokeAdapter._aget_relevant_documents.
        docs: list[Document] = await self.retriever.as_langchain_retriever().ainvoke(query)
        logger.debug("Retrieved %d docs for query: %s", len(docs), query)
        docs, empty_msg = self._filter_by_filenames(docs, doc_file_names)
        if empty_msg:
            return empty_msg
        return _format_docs(self._post_process(query, docs))

    def _filter_by_filenames(
        self, docs: list[Document], doc_file_names: Optional[list[str]]
    ) -> tuple[list[Document], Optional[str]]:
        """Post-filter retrieved docs to the requested filenames, before reranking.

        # ponytail: post-filter over the already-fetched top_k pool, not a native
        # pre-filter at the vector store level. Simple and backend-agnostic, but if
        # the target document's chunks are a small slice of a large corpus they can
        # fall outside the pool entirely. Upgrade path if that bites: push the filter
        # down into the vector store search (Qdrant Filter / Chroma where / FAISS filter).
        """
        if not doc_file_names:
            return docs, None
        wanted = set(doc_file_names)
        filtered = [d for d in docs if d.metadata.get(DOC_FILE_NAME) in wanted]
        if not filtered:
            return filtered, (
                f"No passages found in the specified document(s): {', '.join(sorted(wanted))}. "
                "Use list_documents to see available filenames."
            )
        return filtered, None

    def _post_process(self, query: str, docs: list[Document]) -> list[Document]:
        retrieval_pool = list(docs)  # keep pre-reranker pool for diversity enforcement

        # Optional reranking
        if self.reranker is not None:
            try:
                docs = self._rerank(query, docs)
                logger.debug("After reranking: %d docs", len(docs))
            except Exception as exc:
                logger.warning("Reranker failed (%s) — using unreranked results.", exc)

        # Optional MMR: reduce semantic redundancy (esp. intra-document)
        if self.mmr_enabled and self.embedder is not None:
            from vectoria_lib.rag.reranking.mmr import apply_mmr
            target_k = self.mmr_top_k if self.mmr_top_k > 0 else len(docs)
            docs = apply_mmr(query, docs, self.embedder, target_k, self.mmr_lambda)
            logger.debug("After MMR (lambda=%.2f): %d docs", self.mmr_lambda, len(docs))

        # Optional cross-document diversity enforcement
        if self.min_per_source > 0:
            docs = self._enforce_min_per_source(docs, retrieval_pool)
            logger.debug("After diversity enforcement: %d docs", len(docs))

        # Optional full-paragraph expansion
        if self.full_paragraphs is not None:
            try:
                docs = self.full_paragraphs.post_process(docs)
                logger.debug("After full-paragraph expansion: %d docs", len(docs))
            except Exception as exc:
                logger.warning("Full-paragraph expansion failed (%s) — skipping.", exc)

        # Store for QAAgent
        self.last_docs = docs
        return docs

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        return self.reranker.post_process(docs, query)

    def _enforce_min_per_source(self, ranked_docs: list[Document], retrieval_pool: list[Document]) -> list[Document]:
        """Ensure at least min_per_source chunks per unique source document.

        If ranked_docs lacks enough chunks from a source, pulls the best
        available candidates from retrieval_pool (ordered by retrieval rank).
        Sources with no candidates in the pool are silently skipped.
        """
        result = list(ranked_docs)
        covered: dict[str, int] = defaultdict(int)
        for doc in result:
            covered[doc.metadata.get(DOC_FILE_NAME, "")] += 1

        all_sources = {doc.metadata.get(DOC_FILE_NAME, "") for doc in retrieval_pool}
        result_object_ids = {id(d) for d in result}

        for source in sorted(all_sources):  # sorted for deterministic order
            deficit = self.min_per_source - covered[source]
            if deficit <= 0:
                continue
            candidates = [
                d
                for d in retrieval_pool
                if d.metadata.get(DOC_FILE_NAME, "") == source and id(d) not in result_object_ids
            ]
            for doc in candidates[:deficit]:
                result.append(doc)
                result_object_ids.add(id(doc))
                logger.debug(
                    "Diversity: added chunk from '%s' to meet min_per_source=%d",
                    source,
                    self.min_per_source,
                )

        return result
