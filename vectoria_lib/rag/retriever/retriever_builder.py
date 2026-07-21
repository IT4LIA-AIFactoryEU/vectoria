import logging

from vectoria_lib.core.config_schema import RetrieverConfig
from vectoria_lib.rag.retriever.hybrid_retriever import HybridRetriever
from vectoria_lib.rag.retriever.multi_query_retriever import MultiQueryRetriever
from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.retriever.simple_retriever import SimpleRetriever
from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.rag.vector_store.qdrant_vector_store import QdrantVectorStore
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase


class RetrieverBuilder:
    logger = logging.getLogger("rag")

    @staticmethod
    def build(
        config: dict | RetrieverConfig,
        vector_store: VectorStoreBase,
        llm=None,
    ) -> RetrieverBase:

        if isinstance(config, dict):
            config = RetrieverConfig(**config)

        if vector_store is None:
            raise ValueError("A vector store is required to build the retriever")

        if vector_store.is_empty():
            raise ValueError("Vector store is empty")

        # Derived from the actual instance, not Config() -- VectorStoreBuilder.build()
        # auto-detects the backend an *existing* index was written with (see
        # discovery.detect_vector_store_type) and may hand back a different class
        # than vector_store.name currently configures for new indexes. Comparing
        # against the instance we were actually given (rather than the static
        # config default) keeps that auto-detection meaningful here too.
        if isinstance(vector_store, FaissVectorStore):
            vs_name = "faiss"
        elif isinstance(vector_store, ChromaVectorStore):
            vs_name = "chroma"
        elif isinstance(vector_store, QdrantVectorStore):
            vs_name = "qdrant"
        else:
            raise ValueError(f"Unknown vector store type: {type(vector_store).__name__}")

        retriever_kwargs = dict(
            search_type=config.search_type,
            k=config.top_k,
            fetch_k=config.fetch_k,
            lambda_mult=config.lambda_mult,
            bm25_weight=config.bm25_weight,
        )

        if config.search_type == "hybrid":
            RetrieverBuilder.logger.info("Building hybrid retriever")
            retriever = HybridRetriever(**retriever_kwargs).from_vector_store(vector_store)
        else:
            RetrieverBuilder.logger.info("Building %s retriever (search_type=%s)", vs_name, config.search_type)
            retriever = SimpleRetriever(**retriever_kwargs).from_vector_store(vector_store)

        if config.multi_query:
            if llm is None:
                raise ValueError("multi_query=True requires an LLM to be passed to RetrieverBuilder.build()")
            retriever = MultiQueryRetriever(retriever, llm)

        return retriever
