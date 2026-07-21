import pytest

from vectoria_lib.rag.retriever.simple_retriever import SimpleRetriever
from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore


@pytest.mark.parametrize("k", [1, 2, 3])
def test_chroma_retriever(k, config, chroma_index_test_folder):

    vector_store = ChromaVectorStore(
        model_name=config.get("vector_store", "model_name"),
        device=config.get("vector_store", "device"),
        normalize_embeddings=config.get("vector_store", "normalize_embeddings"),
    ).load_from_disk(chroma_index_test_folder)

    retriever = SimpleRetriever(
        search_type="mmr",
        k=k,
        fetch_k=k,
        lambda_mult=0.5,
    ).from_vector_store(vector_store)

    docs = retriever.as_langchain_retriever().invoke("Who are the authors of the paper?")
    assert len(docs) == k
