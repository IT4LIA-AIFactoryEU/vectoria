from pathlib import Path

_VS_SUFFIXES = ("_qdrant_index", "_chroma_index", "_faiss_index")
_SUFFIX_TO_NAME = {"_qdrant_index": "qdrant", "_chroma_index": "chroma", "_faiss_index": "faiss"}


def find_index_path(tenant_dir):
    tenant_dir = Path(tenant_dir)
    if not tenant_dir.exists():
        return None
    for candidate in tenant_dir.iterdir():
        if not candidate.is_dir():
            continue
        if (candidate / "chroma.sqlite3").exists():
            return candidate
        if any(candidate.name.endswith(s) for s in _VS_SUFFIXES):
            return candidate
    return None


def has_index(tenant_dir):
    return find_index_path(tenant_dir) is not None


def detect_vector_store_type(index_path) -> str:
    """Identify which backend created the index at index_path, from on-disk
    signatures alone — lets callers load an existing index with the backend
    that actually wrote it, regardless of what vector_store.name is currently
    configured to (that config only governs newly created indexes)."""
    index_path = Path(index_path)
    if (index_path / "chroma.sqlite3").exists():
        return "chroma"
    for suffix, name in _SUFFIX_TO_NAME.items():
        if index_path.name.endswith(suffix):
            return name
    raise ValueError(f"Cannot determine vector store type for index at '{index_path}'")
