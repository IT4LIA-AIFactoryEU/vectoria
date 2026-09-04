#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase, get_shared_embedder, normalize_model_name


class ChromaVectorStore(VectorStoreBase):
    """
    A wrapper around ChromaDB to create and manage a Chroma-based vector store.

    This class mirrors the FaissVectorStore interface, allowing transparent
    substitution. The index is created in a temporary directory and moved to
    its final location by dump_to_disk().

    Reference:
    * https://python.langchain.com/docs/integrations/vectorstores/chroma/
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = kwargs["model_name"]
        self.collection_name = kwargs.get("collection_name", "vectoria")
        self.index = None
        self._temp_persist_dir = None

        # Shared singleton: all tenants with the same model/device reuse the same
        # HuggingFaceEmbeddings object (and therefore the same model weights).
        self.hf_embedder = get_shared_embedder(
            model_name=self.model_name,
            device=kwargs["device"],
            normalize_embeddings=kwargs["normalize_embeddings"],
            batch_size=kwargs.get("embedding_batch_size", 64),
        )

    def make_index(self, docs: list[Document]):
        """
        Create a Chroma index from a list of documents.
        The index is persisted to a temporary directory until dump_to_disk() is called.
        """
        self.logger.info("Creating Chroma index from %d documents.." % len(docs))
        self._temp_persist_dir = Path(tempfile.mkdtemp())
        self.index = Chroma.from_documents(
            docs,
            self.hf_embedder,
            persist_directory=str(self._temp_persist_dir),
            collection_name=self.collection_name,
        )
        return self

    def dump_to_disk(self, output_path: str | Path) -> Path:
        """
        Move the Chroma index from its temporary directory to output_path.

        Returns the final index directory path.
        """
        model_name = normalize_model_name(self.model_name)
        Path(output_path).mkdir(exist_ok=True, parents=True)
        target_path = Path(output_path) / f"{model_name}_chroma_index"

        self.logger.info("Persisting Chroma index to %s" % target_path)

        # Release SQLite file handles before moving the directory
        del self.index
        self.index = None

        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.copytree(str(self._temp_persist_dir), str(target_path))
        shutil.rmtree(str(self._temp_persist_dir))
        self._temp_persist_dir = None

        self.index = Chroma(
            persist_directory=str(target_path),
            embedding_function=self.hf_embedder,
            collection_name=self.collection_name,
        )
        return target_path

    def load_from_disk(self, input_path: str | Path):
        if self.index:
            self.logger.info("Index already loaded. Skipping deserialization.")
            return self

        # Check if the index path indicates a different vector store type
        path_str = str(input_path)
        if "_faiss_index" in path_str:
            raise ValueError(
                f"Index at '{input_path}' is a FAISS index, but 'chroma' is configured. "
                f"Either: (1) Change vector_store.name to 'faiss' in config, or "
                f"(2) Delete the old index to create a new ChromaDB index."
            )
        if "_qdrant_index" in path_str:
            raise ValueError(
                f"Index at '{input_path}' is a Qdrant index, but 'chroma' is configured. "
                f"Either: (1) Change vector_store.name to 'qdrant' in config, or "
                f"(2) Delete the old index to create a new ChromaDB index."
            )

        self.index_path = Path(input_path)
        self.logger.info("Loading Chroma index from %s" % input_path)
        self.index = Chroma(
            persist_directory=str(input_path),
            embedding_function=self.hf_embedder,
            collection_name=self.collection_name,
        )
        return self

    def as_retriever(self, **kwargs):
        self.logger.info("Converting Chroma index to retriever with kwargs: %s" % kwargs)
        if self.index is None:
            raise ValueError("Index is not created. Call make_index() first.")
        return self.index.as_retriever(**kwargs)

    def search(self, query: str, **kwargs) -> list[Document]:
        return self.index.similarity_search(query, **kwargs)

    def is_empty(self):
        if self.index is None:
            return True
        return self.index._collection.count() == 0

    def get_all_documents(self) -> list[Document]:
        result = self.index.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

    def add_documents(self, docs: list[Document]):
        if self.index is None:
            raise ValueError("Index is not initialized. Call make_index() first.")
        self.logger.info("Adding %d new documents to Chroma index.." % len(docs))
        self.index.add_documents(docs)

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks whose doc_file_name metadata equals filename.

        Returns the number of chunks deleted.
        """
        result = self.index._collection.get(
            where={DOC_FILE_NAME: filename},
            include=[],  # only fetch ids, no content needed
        )
        ids = result.get("ids", [])
        if ids:
            self.index._collection.delete(ids=ids)
            self.logger.info("Deleted %d chunks for document '%s'", len(ids), filename)
        return len(ids)
