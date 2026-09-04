#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging
import threading
from pathlib import Path

from vectoria_lib.core.config_schema import VectorStoreConfig
from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore
from vectoria_lib.rag.vector_store.discovery import detect_vector_store_type
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.rag.vector_store.qdrant_vector_store import QdrantVectorStore


class VectorStoreBuilder:
    CACHE = {}
    # Guards CACHE only — background index builds run concurrently per tenant
    # (asyncio.to_thread) and were racing on this class-level dict.
    _cache_lock = threading.Lock()
    logger = logging.getLogger("rag")

    @staticmethod
    def _cache_key(config: "VectorStoreConfig", index_path) -> str:
        return f"{config.name}::{config.model_name}::{config.device}::{config.normalize_embeddings}::{index_path}"

    @staticmethod
    def build(config: dict | VectorStoreConfig, index_path: str | Path = None):

        if isinstance(config, dict):
            config = VectorStoreConfig(**config)

        if index_path is None:
            # Building a fresh store (no existing index) — never cache, always fresh.
            # No index to detect a backend from yet, so vector_store.name governs.
            VectorStoreBuilder.logger.info("Vector store %s not in cache. Building new instance." % config.name)
            return VectorStoreBuilder._build_vector_store(config)

        # Loading an existing index: it was written by whatever backend created it,
        # which may not match the currently configured vector_store.name (e.g. the
        # default changed since this tenant was ingested). Detect and honor the
        # backend actually on disk so existing tenants stay readable without
        # re-ingestion — vector_store.name only decides the backend for new indexes.
        detected_name = detect_vector_store_type(index_path)
        if detected_name != config.name:
            VectorStoreBuilder.logger.info(
                "Index at %s was created with '%s'; loading with that backend "
                "(configured default '%s' still applies to new indexes).",
                index_path,
                detected_name,
                config.name,
            )
            config = config.model_copy(update={"name": detected_name})

        name = config.name
        key = VectorStoreBuilder._cache_key(config, index_path)

        with VectorStoreBuilder._cache_lock:
            vs = VectorStoreBuilder.CACHE.get(key)
            if vs is None:
                VectorStoreBuilder.logger.info("Vector store %s not in cache. Building new instance." % name)
                vs = VectorStoreBuilder._build_vector_store(config)
                VectorStoreBuilder.CACHE[key] = vs
            else:
                VectorStoreBuilder.logger.info("Vector store %s already in cache. Returning cached instance." % name)

        VectorStoreBuilder.logger.info("Loading index from disk: %s" % index_path)
        vs.load_from_disk(index_path)

        return vs

    @staticmethod
    def evict(index_path: str | Path) -> None:
        """Remove all cached entries associated with the given index_path and reset ChromaDB singleton cache."""
        with VectorStoreBuilder._cache_lock:
            keys_to_remove = [
                k
                for k in VectorStoreBuilder.CACHE
                if k.endswith(f"::{index_path}")  # last segment is always index_path
            ]
            for k in keys_to_remove:
                del VectorStoreBuilder.CACHE[k]
                VectorStoreBuilder.logger.debug("Evicted vector store cache key: %s", k)

        # ChromaDB keeps an internal singleton per persist_directory.
        # After replacing the index on disk, the old collection UUID becomes stale.
        # Clearing this cache forces a fresh client on next load.
        try:
            import chromadb.api

            chromadb.api.client.SharedSystemClient.clear_system_cache()
            VectorStoreBuilder.logger.debug("ChromaDB system cache cleared")
        except Exception as exc:
            VectorStoreBuilder.logger.debug("Could not clear ChromaDB system cache: %s", exc)

    @staticmethod
    def _build_vector_store(config: VectorStoreConfig):
        kwargs = dict(
            model_name=config.model_name,
            device=config.device,
            normalize_embeddings=config.normalize_embeddings,
            embedding_batch_size=config.embedding_batch_size,
        )
        if config.name == "faiss":
            VectorStoreBuilder.logger.info("Building FAISS vector store")
            return FaissVectorStore(**kwargs)
        elif config.name == "chroma":
            VectorStoreBuilder.logger.info("Building Chroma vector store")
            return ChromaVectorStore(**kwargs, collection_name=config.collection_name)
        elif config.name == "qdrant":
            VectorStoreBuilder.logger.info("Building Qdrant vector store")
            return QdrantVectorStore(
                **kwargs,
                collection_name=config.collection_name,
                qdrant_url=config.qdrant_url,
                qdrant_api_key=config.qdrant_api_key,
                qdrant_prefer_grpc=config.qdrant_prefer_grpc,
            )
        else:
            raise ValueError("Invalid vector store name: %s" % config.name)
