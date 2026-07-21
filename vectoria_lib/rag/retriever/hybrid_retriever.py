#
# VECTORIA
#
# @authors : Mattia Buzzoni
#

import pickle
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase

_RRF_K = 60  # standard RRF constant
_BM25_FILENAME = "bm25_retriever.pkl"


class HybridRetriever(RetrieverBase):
    """
    Hybrid retriever combining BM25 (keyword) and semantic (vector) search
    via Reciprocal Rank Fusion (RRF).

    BM25 handles exact keyword matches; semantic search handles conceptual
    similarity. RRF merges the two ranked lists without requiring score
    normalization.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_type = "hybrid"
        self.top_k = kwargs["k"]
        self.bm25_weight = kwargs.get("bm25_weight", 0.5)

    def from_vector_store(self, vector_store: VectorStoreBase):
        self.logger.info(
            "Building hybrid retriever (top_k=%d, bm25_weight=%.2f)",
            self.top_k,
            self.bm25_weight,
        )

        # Determine where to persist/load the BM25 index.
        # index_path is set by load_from_disk() (e.g. .../tenant/model_chroma_index).
        # We save the BM25 file one level up, alongside the Chroma dir.
        bm25_path: Path | None = None
        if vector_store.index_path is not None:
            bm25_path = vector_store.index_path.parent / _BM25_FILENAME

        if bm25_path and bm25_path.exists():
            self.logger.info("Loading BM25 retriever from disk: %s", bm25_path)
            try:
                with bm25_path.open("rb") as f:
                    self._bm25 = pickle.load(f)
                if not isinstance(self._bm25, BM25Retriever):
                    raise ValueError(f"Unexpected object type in BM25 file: {type(self._bm25)}")
                self._bm25.k = self.top_k
            except Exception as exc:
                self.logger.warning("Failed to load BM25 from disk (%s) — rebuilding from documents.", exc)
                bm25_path.unlink(missing_ok=True)
                self._bm25 = None
        else:
            self._bm25 = None

        if self._bm25 is None:
            self.logger.info("Building BM25 retriever from documents...")
            docs = vector_store.get_all_documents()
            self._bm25 = BM25Retriever.from_documents(docs, k=self.top_k)
            if bm25_path:
                self.logger.info("Persisting BM25 retriever to disk: %s", bm25_path)
                with bm25_path.open("wb") as f:
                    pickle.dump(self._bm25, f)

        semantic = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        self._ensemble = EnsembleRetriever(
            retrievers=[self._bm25, semantic],
            weights=[self.bm25_weight, 1.0 - self.bm25_weight],
            c=_RRF_K,
        )
        self.wrapped_vector_store = vector_store
        return self

    def as_langchain_retriever(self):
        return self

    def invoke(self, query: str) -> list[Document]:
        return self._ensemble.invoke(query)[: self.top_k]

    async def ainvoke(self, query: str) -> list[Document]:
        # ponytail: EnsembleRetriever.ainvoke already gathers BM25+semantic concurrently
        docs = await self._ensemble.ainvoke(query)
        return docs[: self.top_k]
