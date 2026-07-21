#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document

from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase, get_shared_embedder, normalize_model_name


class FaissVectorStore(VectorStoreBase):
    """
    A wrapper around the FAISS library to create and manage a FAISS-based vector store.

    This class provides methods to create a FAISS index from documents, retrieve the index as a retriever,
    and serialize/deserialize the index using pickle.

    Reference to FAISS integration documentation:
    * https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
    * https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html
    """

    def __init__(self, **kwargs):
        """
        Initialize a FaissVectorStore object.
        """
        super().__init__()
        self.model_name = kwargs["model_name"]
        self.index = None

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
        Create a FAISS index from a list of documents.

        Parameters:
        - docs (list[str]): A list of documents to be indexed.

        Returns:
        - FaissVectorStore: The FaissVectorStore instance with the created index.
        """
        self.logger.info("Creating FAISS index from %d documents.." % len(docs))
        self.index = FAISS.from_documents(docs, self.hf_embedder)
        return self

    def dump_to_disk(self, output_path: str | Path) -> Path:
        """
        Serialize the FAISS index.

        Parameters:
        - output_path (str | Path): The path to the directory where the FAISS index should be saved.
        Returns:
        - Path: The path to the saved FAISS index.
        """
        model_name = normalize_model_name(self.model_name)
        Path(output_path).mkdir(exist_ok=True, parents=True)
        output_path = Path(output_path) / f"{model_name}_faiss_index"
        self.logger.info("Serializing FAISS index to pickle file %s" % output_path)
        self.index.save_local(output_path)
        return output_path

    def load_from_disk(self, input_path: str | Path):
        if self.index:
            self.logger.info("Index already loaded. Skipping deserialization.")
            return self

        # Check if the index path indicates a different vector store type
        path_str = str(input_path)
        if "_chroma_index" in path_str:
            raise ValueError(
                f"Index at '{input_path}' is a ChromaDB index, but 'faiss' is configured. "
                f"Either: (1) Change vector_store.name to 'chroma' in config, or "
                f"(2) Delete the old index to create a new FAISS index."
            )
        if "_qdrant_index" in path_str:
            raise ValueError(
                f"Index at '{input_path}' is a Qdrant index, but 'faiss' is configured. "
                f"Either: (1) Change vector_store.name to 'qdrant' in config, or "
                f"(2) Delete the old index to create a new FAISS index."
            )

        self.index_path = Path(input_path)
        self.logger.info("Deserializing FAISS index from pickle file %s" % input_path)
        self.index = FAISS.load_local(self.index_path, self.hf_embedder, allow_dangerous_deserialization=True)
        return self

    def as_retriever(self, **kwargs):
        """
        Convert the FAISS index into a retriever object.

        Parameters:
        - kwargs: Additional keyword arguments to configure the retriever.

        Returns:
        - Retriever: A retriever object based on the FAISS index.
        """
        self.logger.info("Converting FAISS index to retriever with kwargs: %s" % kwargs)
        if self.index is None:
            raise ValueError("Index is not created. Call make_index() first.")

        return self.index.as_retriever(**kwargs)

    def search(self, query: str, **kwargs) -> list[Document]:
        """
        Parameters: https://python.langchain.com/docs/integrations/vectorstores/faiss/#query-directly
        """
        return self.index.similarity_search(query, **kwargs)

    def is_empty(self):
        return self.index is None or len(self.index.docstore._dict) == 0

    def get_all_documents(self) -> list[Document]:
        if self.index is None:
            return []
        return list(self.index.docstore._dict.values())

    def add_documents(self, docs: list[Document]):
        """
        Add new documents to the FAISS index.

        Parameters:
        - docs (list[Document]): A list of new documents to be indexed.
        """
        if self.index is None:
            raise ValueError("Index is not initialized. Call make_index() first.")

        self.logger.info("Adding %d new documents to FAISS index.." % len(docs))
        self.index.add_documents(docs)
