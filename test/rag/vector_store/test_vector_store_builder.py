from vectoria_lib.rag.vector_store.chroma_vector_store import ChromaVectorStore
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder


def test_build_chroma_vector_store(config, chroma_index_test_folder):

    vector_store = VectorStoreBuilder.build(config.get("vector_store"), index_path=None)
    assert isinstance(vector_store, ChromaVectorStore)

    VectorStoreBuilder.CACHE.clear()

    vector_store = VectorStoreBuilder.build(config.get("vector_store"), index_path=chroma_index_test_folder)
    assert isinstance(vector_store, ChromaVectorStore)
    assert vector_store.index is not None
