#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
import pickle
import shutil
import time
from pathlib import Path

from langchain_core.documents import Document

from vectoria_lib.core.config import Config
from vectoria_lib.core.io.folder_reader import get_files_in_folder
from vectoria_lib.rag.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from vectoria_lib.rag.vector_store.discovery import detect_vector_store_type, find_index_path
from vectoria_lib.rag.vector_store.vector_store_base import VectorStoreBase
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder

_BM25_FILENAME = "bm25_retriever.pkl"


def _build_and_persist_bm25(docs: list[Document], index_dir: Path, extra_docs=None):
    """Build a BM25 index from docs (+ extra_docs) and save it alongside the vector store."""
    try:
        config = Config()
        bm25_weight = config.get("retriever", "bm25_weight")
        if not bm25_weight:
            return
    except Exception:
        return
    try:
        from langchain_community.retrievers import BM25Retriever

        t0 = time.perf_counter()
        all_docs = list(extra_docs or []) + list(docs)
        bm25 = BM25Retriever.from_documents(all_docs, k=1)
        bm25_path = index_dir / _BM25_FILENAME
        with bm25_path.open("wb") as f:
            pickle.dump(bm25, f)
        logging.getLogger("tasks").info(
            "BM25 index built (%d docs) and persisted in %.2f sec", len(all_docs), time.perf_counter() - t0
        )
    except Exception as exc:
        logging.getLogger("tasks").warning("BM25 pre-build failed (will rebuild on load): %s", exc)


def build_index(**kwargs: dict) -> tuple[Path, VectorStoreBase, list[Document]]:
    logger = logging.getLogger("tasks")
    files = get_files_in_folder(Path(kwargs["input_docs_dir"]))
    for file in files:
        logger.debug("Found file: %s", file)
    if len(files) == 0:
        raise ValueError(f"No files found in the input folder {kwargs['input_docs_dir']}")
    return build_index_from_files(files, Path(kwargs["output_dir"]))


def build_index_from_files(
    files: list[Path],
    output_index_dir: Path,
    stop_event=None,
    progress_callback=None,
) -> tuple[Path, VectorStoreBase, list[Document]]:
    """Full rebuild: process all files and create a new index from scratch."""
    config = Config()
    logger = logging.getLogger("tasks")

    start_time = time.time()
    docs = PreprocessingPipeline.build_pipeline().run(files, stop_event=stop_event, progress_callback=progress_callback)
    logger.info("Created %d documents from %s in %.2f seconds", len(docs), files, time.time() - start_time)

    if not docs:
        raise ValueError(
            f"No text could be extracted from the provided file(s): {[f.name for f in files]}. "
            "The file may be a scanned image-only PDF. "
            "Try switching to backend: docling in the config for OCR support."
        )

    start_time = time.perf_counter()
    vector_store = VectorStoreBuilder().build(config.get("vector_store"), index_path=None)
    vector_store.make_index(docs)
    logger.debug("Creation of vector store index took %.2f seconds", time.perf_counter() - start_time)

    start_time = time.perf_counter()
    index_path = vector_store.dump_to_disk(output_index_dir)
    logger.info("Index dumped at %s in %.2f seconds", index_path, time.perf_counter() - start_time)

    _build_and_persist_bm25(docs, output_index_dir)

    # Release any file locks held by the vector store before returning.
    # documents.py will rename the tmp directory to its final location;
    # a held lock (e.g. Qdrant local) on the tmp path inode would block
    # the subsequent load_from_disk() after the rename.
    vector_store.close()

    return index_path, vector_store, docs


def add_files_to_existing_index(
    new_files: list[Path],
    existing_index_dir: Path,
    output_index_dir: Path,
    stop_event=None,
    progress_callback=None,
) -> tuple[Path, VectorStoreBase, list[Document]]:
    """
    Incremental add: process only new_files and add their chunks to a copy of
    the existing index, without re-embedding previously indexed documents.

    output_index_dir must not exist yet (will be created as a copy of existing_index_dir).
    """
    config = Config()
    logger = logging.getLogger("tasks")

    # Process only the new files
    start_time = time.time()
    new_docs = PreprocessingPipeline.build_pipeline().run(
        new_files, stop_event=stop_event, progress_callback=progress_callback
    )
    logger.info(
        "Incremental add: processed %d new docs from %d file(s) in %.2f sec",
        len(new_docs),
        len(new_files),
        time.time() - start_time,
    )

    if not new_docs:
        raise ValueError(
            f"No text could be extracted from the new file(s): {[f.name for f in new_files]}. "
            "The file may be a scanned image-only PDF."
        )

    # Load existing BM25 docs before copytree so we can merge them with new_docs.
    existing_bm25_docs: list[Document] = []
    existing_bm25_path = existing_index_dir / _BM25_FILENAME
    if existing_bm25_path.exists():
        try:
            from langchain_community.retrievers import BM25Retriever

            with existing_bm25_path.open("rb") as f:
                old_bm25 = pickle.load(f)
            if isinstance(old_bm25, BM25Retriever):
                existing_bm25_docs = old_bm25.docs
        except Exception as exc:
            logger.warning("Could not load existing BM25 for merge: %s", exc)

    # Reuse the existing tenant's own backend (not the currently configured
    # default, which may have changed since this tenant was ingested).
    existing_vs_path = find_index_path(existing_index_dir)
    if existing_vs_path is None:
        raise ValueError(f"No vector store index found in {existing_index_dir}")

    # Copy existing index to output (atomic swap happens in the caller)
    shutil.copytree(str(existing_index_dir), str(output_index_dir))

    index_path = output_index_dir / existing_vs_path.name
    start_time = time.perf_counter()
    vector_store = VectorStoreBuilder().build(config.get("vector_store"), index_path=index_path)
    vector_store.add_documents(new_docs)
    logger.info("Added %d docs to existing index in %.2f sec", len(new_docs), time.perf_counter() - start_time)

    _build_and_persist_bm25(new_docs, output_index_dir, extra_docs=existing_bm25_docs)

    # Release file locks before documents.py renames the tmp directory.
    vector_store.close()

    return index_path, vector_store, new_docs


def delete_document_from_index(
    filename: str,
    existing_index_dir: Path,
    output_index_dir: Path,
) -> tuple[Path, VectorStoreBase]:
    """
    Incremental delete: remove all chunks belonging to filename from a copy of
    the existing index, without re-embedding any document.

    Only supported for Chroma (FAISS does not support per-document deletion).
    Raises NotImplementedError for other vector stores.
    """
    config = Config()
    logger = logging.getLogger("tasks")

    # Reuse the existing tenant's own backend (not the currently configured
    # default, which may have changed since this tenant was ingested).
    existing_vs_path = find_index_path(existing_index_dir)
    if existing_vs_path is None:
        raise ValueError(f"No vector store index found in {existing_index_dir}")
    vs_name = detect_vector_store_type(existing_vs_path)
    if vs_name not in ("chroma", "qdrant"):
        raise NotImplementedError(f"Incremental delete is only supported for Chroma and Qdrant, not '{vs_name}'.")

    # Copy existing index to output
    shutil.copytree(str(existing_index_dir), str(output_index_dir))

    # Invalidate stale BM25
    (output_index_dir / _BM25_FILENAME).unlink(missing_ok=True)

    index_path = output_index_dir / existing_vs_path.name
    vector_store = VectorStoreBuilder().build(config.get("vector_store"), index_path=index_path)

    n_deleted = vector_store.delete_by_filename(filename)
    logger.info("Deleted %d chunks for document '%s'", n_deleted, filename)

    return index_path, vector_store
