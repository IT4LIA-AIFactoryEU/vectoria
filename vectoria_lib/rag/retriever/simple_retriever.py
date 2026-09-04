#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.

from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase


class SimpleRetriever(RetrieverBase):
    """Generic vector store retriever for similarity and MMR searches.

    Replaces the former ChromaRetriever and FaissRetriever, which were
    identical except for the type annotation on `from_vector_store`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_type = kwargs["search_type"]
        self.search_kwargs = {"k": kwargs["k"]}
        if self.search_type == "mmr":
            self.search_kwargs["fetch_k"] = kwargs["fetch_k"]
            self.search_kwargs["lambda_mult"] = kwargs["lambda_mult"]

    def from_vector_store(self, vector_store: VectorStoreBase) -> "SimpleRetriever":
        self.logger.info("Creating retriever from vector store with kwargs: %s", self.search_kwargs)
        self.wrapped_retriever = vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )
        self.wrapped_vector_store = vector_store
        return self

    def as_langchain_retriever(self):
        return self.wrapped_retriever
