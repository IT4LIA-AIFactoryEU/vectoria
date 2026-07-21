#
# VECTORIA
#

import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME
from vectoria_lib.rag.vector_store.vector_store_base import (
    VectorStoreBase,
    get_shared_embedder,
    normalize_model_name,
)


class QdrantVectorStore(VectorStoreBase):
    """
    A wrapper around Qdrant to create and manage a Qdrant-based vector store.

    Supports both local (file-based) and remote (server-based) modes:
    - Local mode: QdrantClient(path=...) stores data in a local directory
    - Remote mode: QdrantClient(url=..., api_key=...) connects to a Qdrant server

    This class mirrors the FaissVectorStore and ChromaVectorStore interfaces, allowing
    transparent substitution.

    Local Qdrant does not support concurrent access, so we maintain a module-level cache
    of clients keyed by (mode, path/url) to reuse the same client across instances.

    Reference:
    * https://qdrant.tech/documentation/
    * https://python.langchain.com/docs/integrations/vectorstores/qdrant/
    """

    # Module-level cache of Qdrant clients to prevent concurrent access errors
    # Key: (path_or_url,) for local or (url, api_key) for remote
    _client_cache: dict[tuple, QdrantClient] = {}
    _cache_lock = threading.Lock()

    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = kwargs["model_name"]
        self.collection_name = kwargs.get("collection_name", "vectoria")
        self.qdrant_url = kwargs.get("qdrant_url")
        self.qdrant_api_key = kwargs.get("qdrant_api_key")
        self.qdrant_prefer_grpc = kwargs.get("qdrant_prefer_grpc", False)
        self.index = None
        self.client = None
        self._temp_persist_dir: Optional[Path] = None

        # Shared singleton: all tenants with the same model/device reuse the same
        # HuggingFaceEmbeddings object (and therefore the same model weights).
        self.hf_embedder = get_shared_embedder(
            model_name=self.model_name,
            device=kwargs["device"],
            normalize_embeddings=kwargs["normalize_embeddings"],
            batch_size=kwargs.get("embedding_batch_size", 64),
        )

    def _get_or_create_client(self, path: Optional[str] = None) -> QdrantClient:
        """Create or return the QdrantClient, using module-level cache to prevent
        concurrent access errors on local Qdrant instances.

        Thread-safe: uses a lock to prevent two threads from creating clients
        for the same path simultaneously.
        """
        if self.client is not None:
            return self.client

        with QdrantVectorStore._cache_lock:
            # Re-check after acquiring the lock (double-checked locking)
            if self.client is not None:
                return self.client

            if self.qdrant_url:
                cache_key = (self.qdrant_url, self.qdrant_api_key or "")
                if cache_key not in QdrantVectorStore._client_cache:
                    self.logger.info("Creating Qdrant client for remote server at %s", self.qdrant_url)
                    QdrantVectorStore._client_cache[cache_key] = QdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        prefer_grpc=self.qdrant_prefer_grpc,
                    )
                else:
                    self.logger.info("Reusing cached Qdrant client for %s", self.qdrant_url)
                self.client = QdrantVectorStore._client_cache[cache_key]
            else:
                if path is None:
                    raise ValueError("path must be provided for local Qdrant mode")
                # Normalize the path to avoid cache misses from formatting differences
                path = str(Path(path).resolve())
                cache_key = (path,)
                if cache_key not in QdrantVectorStore._client_cache:
                    self.logger.info("Creating local Qdrant client at %s", path)
                    QdrantVectorStore._client_cache[cache_key] = QdrantClient(path=path)
                else:
                    self.logger.info("Reusing cached local Qdrant client at %s", path)
                self.client = QdrantVectorStore._client_cache[cache_key]

        return self.client

    def _close_and_uncache_client(self, path: str):
        """Close a cached local client and remove it from the cache.

        Must be called before deleting/moving the directory a local client points to.
        """
        path = str(Path(path).resolve())
        cache_key = (path,)
        with QdrantVectorStore._cache_lock:
            client = QdrantVectorStore._client_cache.pop(cache_key, None)
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    self.logger.debug("Ignoring error closing Qdrant client for %s: %s", path, exc)
        # Also clear instance reference if it pointed to this client
        if self.client is client:
            self.client = None

    def make_index(self, docs: list[Document]):
        """
        Create a Qdrant index from a list of documents.
        In local mode, the index is persisted to a temporary directory until
        dump_to_disk() is called.
        """
        self.logger.info("Creating Qdrant index from %d documents..", len(docs))

        if self.qdrant_url:
            client = self._get_or_create_client()
        else:
            self._temp_persist_dir = Path(tempfile.mkdtemp())
            client = self._get_or_create_client(path=str(self._temp_persist_dir))

        # Create the collection before wrapping it: LangChainQdrant.__init__ validates
        # that the collection already exists and raises if it does not.
        # We avoid from_documents() because it creates its own QdrantClient internally,
        # bypassing our cache and causing file-lock conflicts on local mode.
        vector_size = len(self.hf_embedder.embed_query("test"))
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

        self.index = LangChainQdrant(
            client=client,
            collection_name=self.collection_name,
            embedding=self.hf_embedder,
        )
        self.index.add_documents(docs)

        return self

    def dump_to_disk(self, output_path: str | Path) -> Path:
        """
        For local mode: move the Qdrant index to output_path.
        For remote mode: no-op (data already on server).

        Returns the final index directory path (or None for remote mode).
        """
        if self.qdrant_url:
            self.logger.info("Remote Qdrant mode: data is already persisted on server")
            return Path(output_path)

        model_name = normalize_model_name(self.model_name)
        Path(output_path).mkdir(exist_ok=True, parents=True)
        target_path = Path(output_path) / f"{model_name}_qdrant_index"

        self.logger.info("Persisting Qdrant index to %s", target_path)

        # Close the temp client properly before moving files
        self.index = None
        self._close_and_uncache_client(str(self._temp_persist_dir))

        # Also close any existing client for the target path (e.g. from a previous index)
        self._close_and_uncache_client(str(target_path))

        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(str(self._temp_persist_dir), str(target_path))
        shutil.rmtree(str(self._temp_persist_dir))
        self._temp_persist_dir = None

        # Do NOT open a new client here. The caller (documents.py) will rename
        # this directory to the final location before calling load_from_disk().
        # Opening a client now would hold a lock on the tmp path's inode, which
        # moves with the rename and blocks the subsequent load_from_disk().
        return target_path

    def load_from_disk(self, input_path: str | Path):
        """Load a Qdrant index from disk (local mode only)."""
        if self.index:
            self.logger.info("Index already loaded. Skipping deserialization.")
            return self

        if self.qdrant_url:
            self.logger.warning("Remote mode: ignoring input_path and using configured qdrant_url")
            client = self._get_or_create_client()
        else:
            self.index_path = Path(input_path)

            # Check if the index path indicates a different vector store type
            path_str = str(input_path)
            if "_faiss_index" in path_str:
                raise ValueError(
                    f"Index at '{input_path}' is a FAISS index, but 'qdrant' is configured. "
                    f"Either: (1) Change vector_store.name to 'faiss' in config, or "
                    f"(2) Delete the old index to create a new Qdrant index."
                )
            if "_chroma_index" in path_str:
                raise ValueError(
                    f"Index at '{input_path}' is a ChromaDB index, but 'qdrant' is configured. "
                    f"Either: (1) Change vector_store.name to 'chroma' in config, or "
                    f"(2) Delete the old index to create a new Qdrant index."
                )

            self.logger.info("Loading Qdrant index from %s", input_path)
            client = self._get_or_create_client(path=str(input_path))

        self.index = LangChainQdrant(
            client=client,
            collection_name=self.collection_name,
            embedding=self.hf_embedder,
        )
        return self

    def close(self):
        """Close and uncache the local Qdrant client held by this instance.

        Must be called before the index directory is renamed/moved on disk so
        that the file lock is released before any subsequent client tries to
        open the same path.
        """
        self.index = None
        if self.client is not None and not self.qdrant_url:
            # Find and evict this client from the class cache
            with QdrantVectorStore._cache_lock:
                stale_keys = [k for k, v in QdrantVectorStore._client_cache.items() if v is self.client]
                for k in stale_keys:
                    QdrantVectorStore._client_cache.pop(k, None)
            try:
                self.client.close()
            except Exception as exc:
                self.logger.debug("Ignoring error closing Qdrant client: %s", exc)
            self.client = None

    def as_retriever(self, **kwargs):
        """Convert the Qdrant index into a retriever object."""
        self.logger.info("Converting Qdrant index to retriever with kwargs: %s", kwargs)
        if self.index is None:
            raise ValueError("Index is not created. Call make_index() first.")
        return self.index.as_retriever(**kwargs)

    def search(self, query: str, **kwargs) -> list[Document]:
        """Search the Qdrant index."""
        return self.index.similarity_search(query, **kwargs)

    def is_empty(self) -> bool:
        """Check if the Qdrant index is empty."""
        if self.index is None:
            return True
        client = self.index.client
        if client is None:
            return True
        count_result = client.count(collection_name=self.collection_name)
        return count_result.count == 0

    def get_all_documents(self) -> list[Document]:
        """Retrieve all documents from the Qdrant index, paginating via scroll cursor."""
        if self.index is None:
            return []

        client = self.index.client
        documents = []
        offset = None

        while True:
            points, next_offset = client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                page_content = payload.get("page_content", "")
                metadata = payload.get("metadata", {})
                documents.append(Document(page_content=page_content, metadata=metadata))

            if next_offset is None:
                break
            offset = next_offset

        return documents

    def add_documents(self, docs: list[Document]):
        """Add new documents to the Qdrant index."""
        if self.index is None:
            raise ValueError("Index is not initialized. Call make_index() first.")
        self.logger.info("Adding %d new documents to Qdrant index..", len(docs))
        self.index.add_documents(docs)

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks whose doc_file_name metadata equals filename.

        Returns the number of chunks deleted.
        """
        if self.index is None:
            return 0

        client = self.index.client
        if client is None:
            return 0

        # Build a filter for doc_file_name == filename
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key=f"metadata.{DOC_FILE_NAME}",
                    match=MatchValue(value=filename),
                )
            ]
        )

        # Count matching points before deletion
        matching_points, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        deleted_count = len(matching_points)

        if deleted_count > 0:
            client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition,
            )
            self.logger.info("Deleted %d chunks for document '%s'", deleted_count, filename)
        return deleted_count
