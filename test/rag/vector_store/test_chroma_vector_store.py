#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

from vectoria_lib.rag.preprocessing.chunking import recursive_character_text_splitter
from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore


def test_chroma_vector_store(config):

    doc = Document(
        page_content=(
            "The Matrix is a 1999 science fiction action film written and directed by the Wachowskis. "
            "It is the first installment in the Matrix film series, starring Keanu Reeves, Laurence Fishburne, "
            "Carrie-Anne Moss, Hugo Weaving and Joe Pantoliano, and depicts a dystopian future in which "
            "humanity is unknowingly trapped inside the Matrix, a simulated reality that intelligent machines "
            "have created to distract humans while using their bodies as an energy source."
        ),
        metadata=dict(source="test", name="The Matrix", level=0, id=0),
    )

    docs = recursive_character_text_splitter(doc, chunk_size=50, chunk_overlap=20)

    vector_store = ChromaVectorStore(
        model_name=config.get("vector_store", "model_name"),
        device=config.get("vector_store", "device"),
        normalize_embeddings=config.get("vector_store", "normalize_embeddings"),
    ).make_index(docs)
    assert isinstance(vector_store.index, Chroma)

    index_path = vector_store.dump_to_disk("/tmp/test_chroma_vector_store")
    assert index_path.exists()

    vector_store = ChromaVectorStore(
        model_name=config.get("vector_store", "model_name"),
        device=config.get("vector_store", "device"),
        normalize_embeddings=config.get("vector_store", "normalize_embeddings"),
    ).load_from_disk(index_path)
    assert isinstance(vector_store.index, Chroma)

    results = vector_store.search("What is the Matrix?", k=1)
    assert len(results) == 1

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 2, "fetch_k": 2, "lambda_mult": 0.5},
    )
    assert isinstance(retriever, VectorStoreRetriever)

    results = retriever.invoke("What is the Matrix?")
    assert len(results) == 2
