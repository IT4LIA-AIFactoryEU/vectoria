#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli, Mattia Buzzoni
#

import importlib
import logging
import multiprocessing as mp
import time
from pathlib import Path

from langchain_core.runnables import RunnableLambda

from vectoria_lib.core.config import Config
from vectoria_lib.core.config_schema import DataIngestionConfig, ExtractionConfig
from vectoria_lib.rag.preprocessing.chunking import recursive_character_text_splitter, semantic_text_splitter

# ----------------------------------------------------------------------------------------------------------------------
# DO NOT REMOVE THESE IMPORTS, they are get from global namespace
from vectoria_lib.rag.preprocessing.extraction_pdf import extract_text_from_pdf_file
from vectoria_lib.rag.preprocessing.utils import seek_and_replace

_DEFAULT_MAX_WORKERS = 4

# Extensions handled natively by Docling (no native-extraction/python-docx fallback needed)
_DOCLING_NATIVE_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".adoc",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
}


def _pin_worker_to_gpu():
    """Assign this worker one GPU, round-robin, instead of every worker defaulting
    to cuda:0. Each spawned worker independently loads its own copy of any GPU model
    (Docling layout, semantic-chunking embedder) — with N workers all on cuda:0, GPU
    memory use multiplies by N on a single device instead of spreading across however
    many GPUs are actually visible to this process."""
    import os

    import torch

    if not torch.cuda.is_available():
        return
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return

    # ponytail: multiprocessing.Pool assigns each worker a 1-based slot via this
    # semi-private attribute — no public API exposes "which worker am I" otherwise.
    identity = mp.current_process()._identity
    slot = (identity[0] - 1) if identity else 0
    device_str = f"cuda:{slot % device_count}"

    os.environ["DOCLING_DEVICE"] = device_str  # docling reads this env var natively

    chunking_cfg = Config().get("data_ingestion", "chunking")
    if str(chunking_cfg.get("semantic_device", "")).startswith("cuda"):
        chunking_cfg["semantic_device"] = device_str
        Config().set("data_ingestion", "chunking", value=chunking_cfg)


def _warmup_worker(tokenizer_name, semantic_embedding_model=None):
    """Pool initializer: pin this worker to a GPU and pre-load the tokenizer and
    (if semantic chunking is configured) the sentence-transformer model into
    their lru_caches before any file arrives.

    Without this, semantic_text_splitter's .map() step fans a document's
    paragraphs out across a thread pool, and every thread calls the
    lru_cache'd _load_sentence_transformer before any of them has finished
    populating the cache — a cache stampede where N threads all redundantly
    load and transfer the same multi-GB model onto the GPU. Pre-loading here,
    before the pool ever starts processing files, guarantees the cache is
    already warm when .map() begins, so every thread gets an instant hit.
    """
    _pin_worker_to_gpu()
    if tokenizer_name:
        from vectoria_lib.rag.preprocessing.chunking import _load_tokenizer

        _load_tokenizer(tokenizer_name)
    if semantic_embedding_model:
        from vectoria_lib.rag.preprocessing.chunking import _load_sentence_transformer

        # Read the device *after* _pin_worker_to_gpu(), which may have just
        # rewritten it to this worker's pinned "cuda:N" in the process-local Config.
        device = Config().get("data_ingestion", "chunking").get("semantic_device", "cpu")
        _load_sentence_transformer(semantic_embedding_model, device)


def _worker_run_file(file_path: Path):
    """Module-level worker: safe to pickle, rebuilds the pipeline in the subprocess."""
    pipeline = PreprocessingPipeline.build_pipeline()
    return pipeline.run_on_file(file_path)


# Non-PDF extensions with a dedicated extractor, keyed by (module, function name).
# All extractors share the same (path, dump_doc_structure_on_file, regexes_for_metadata_extraction) signature.
_EXTENSION_EXTRACTORS = {
    ".docx": ("vectoria_lib.rag.preprocessing.extraction_docx", "extract_text_from_docx_file"),
    ".doc": ("vectoria_lib.rag.preprocessing.extraction_docx", "extract_text_from_docx_file"),
    ".xlsx": ("vectoria_lib.rag.preprocessing.extraction_xlsx", "extract_text_from_xlsx_file"),
    ".pptx": ("vectoria_lib.rag.preprocessing.extraction_pptx", "extract_text_from_pptx_file"),
    ".html": ("vectoria_lib.rag.preprocessing.extraction_html", "extract_text_from_html_file"),
    ".htm": ("vectoria_lib.rag.preprocessing.extraction_html", "extract_text_from_html_file"),
    ".md": ("vectoria_lib.rag.preprocessing.extraction_plaintext", "extract_text_from_plaintext_file"),
    ".markdown": ("vectoria_lib.rag.preprocessing.extraction_plaintext", "extract_text_from_plaintext_file"),
    ".txt": ("vectoria_lib.rag.preprocessing.extraction_plaintext", "extract_text_from_plaintext_file"),
    ".csv": ("vectoria_lib.rag.preprocessing.extraction_plaintext", "extract_text_from_plaintext_file"),
}


class _Extractor:
    """Picklable callable that routes files to the correct extraction backend."""

    def __init__(self, backend: str):
        self.backend = backend

    def _docling(self, p: Path, **kwargs):
        from vectoria_lib.rag.preprocessing.extraction_pdf_docling import extract_text_from_pdf_file_docling

        return extract_text_from_pdf_file_docling(p, **kwargs)

    def __call__(self, file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]):
        p = Path(file_path)
        ext = p.suffix.lower()
        kwargs = dict(
            dump_doc_structure_on_file=dump_doc_structure_on_file,
            regexes_for_metadata_extraction=regexes_for_metadata_extraction,
        )

        if self.backend == "docling" and ext in _DOCLING_NATIVE_EXTENSIONS:
            return self._docling(p, **kwargs)

        if ext in _EXTENSION_EXTRACTORS:
            module_path, func_name = _EXTENSION_EXTRACTORS[ext]
            extractor_fn = getattr(importlib.import_module(module_path), func_name)
            return extractor_fn(p, **kwargs)

        if self.backend == "auto":
            if ext == ".pdf":
                from vectoria_lib.rag.preprocessing.extraction_pdf_auto import extract_text_from_pdf_file_auto

                return extract_text_from_pdf_file_auto(p, **kwargs)
            # Non-PDF, non-natively-handled with auto → docling handles it
            if ext in _DOCLING_NATIVE_EXTENSIONS:
                return self._docling(p, **kwargs)

        if ext == ".pdf":
            return extract_text_from_pdf_file(p, **kwargs)

        raise ValueError(f"Unsupported file type '{ext}' for backend '{self.backend}'")


class PreprocessingPipeline:
    logger = logging.getLogger("rag")

    @staticmethod
    def get_extraction_fn(extraction_config: ExtractionConfig):
        return _Extractor(extraction_config.backend)

    @staticmethod
    def _build_extraction_step(cfg: DataIngestionConfig):
        extraction_fn = PreprocessingPipeline.get_extraction_fn(cfg.extraction)
        extract = RunnableLambda(extraction_fn).bind(
            dump_doc_structure_on_file=cfg.extraction.dump_doc_structure_on_file,
            regexes_for_metadata_extraction=cfg.extraction.regexes_for_metadata_extraction,
        )
        clean = RunnableLambda(seek_and_replace).bind(regex_configs=cfg.regexes_for_replacement).map()
        return extract | clean

    @staticmethod
    def _build_chunking_step(cfg: DataIngestionConfig):
        if cfg.chunking.method == "semantic":
            embedding_model = cfg.chunking.semantic_embedding_model or Config().get("vector_store", "model_name")
            return (
                RunnableLambda(semantic_text_splitter)
                .bind(
                    embedding_model_name=embedding_model,
                    breakpoint_threshold_type=cfg.chunking.semantic_breakpoint_threshold_type,
                    breakpoint_threshold_amount=cfg.chunking.semantic_breakpoint_threshold_amount,
                    device=cfg.chunking.semantic_device,
                    max_chunk_size=cfg.chunking.semantic_max_chunk_size,
                    chunk_overlap=cfg.chunking.chunk_overlap,
                    tokenizer_name=cfg.chunking.tokenizer_name,
                    dump_chunks_on_file=cfg.chunking.dump_chunks_on_file,
                )
                .map()
            )
        return (
            RunnableLambda(recursive_character_text_splitter)
            .bind(
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
                separators=cfg.chunking.separators,
                is_separator_regex=cfg.chunking.is_separator_regex,
                dump_chunks_on_file=cfg.chunking.dump_chunks_on_file,
                tokenizer_name=cfg.chunking.tokenizer_name,
            )
            .map()
        )

    @staticmethod
    def build_pipeline():
        cfg = DataIngestionConfig(**Config().get("data_ingestion"))
        chain = PreprocessingPipeline._build_extraction_step(cfg) | PreprocessingPipeline._build_chunking_step(cfg)

        try:
            PreprocessingPipeline.logger.info("%s", chain.get_graph().draw_ascii())
        except Exception as e:
            PreprocessingPipeline.logger.warning("Failed to print graph: %s", e)

        gpu_bound = cfg.extraction.backend in ("docling", "auto") or (
            cfg.chunking.method == "semantic" and str(cfg.chunking.semantic_device).startswith("cuda")
        )

        semantic_embedding_model = None
        if cfg.chunking.method == "semantic":
            semantic_embedding_model = cfg.chunking.semantic_embedding_model or Config().get(
                "vector_store", "model_name"
            )

        return PreprocessingPipeline(
            chain=chain,
            multiprocessing=cfg.multiprocessing,
            n_workers=cfg.multiprocessing_workers,
            tokenizer_name=cfg.chunking.tokenizer_name,
            gpu_bound=gpu_bound,
            semantic_embedding_model=semantic_embedding_model,
        )

    def __init__(
        self,
        chain,
        multiprocessing,
        n_workers=None,
        tokenizer_name=None,
        gpu_bound=False,
        semantic_embedding_model=None,
    ):
        self.logger = logging.getLogger("rag")
        self.chain = chain
        self.multiprocessing = multiprocessing
        self.n_workers = n_workers
        self.tokenizer_name = tokenizer_name
        self.gpu_bound = gpu_bound
        self.semantic_embedding_model = semantic_embedding_model

    def _effective_workers(self, n_files: int) -> int:
        cap = self.n_workers if self.n_workers is not None else min(mp.cpu_count(), _DEFAULT_MAX_WORKERS)
        if self.gpu_bound:
            import torch

            if torch.cuda.is_available():
                # ponytail: each worker loads its own copy of the GPU model(s) onto
                # its pinned device (see _pin_worker_to_gpu) — more workers than GPUs
                # means multiple full model copies stacked on the same device, OOM.
                cap = min(cap, max(torch.cuda.device_count(), 1))
        return min(cap, n_files)

    def run(self, input_docs: list[Path], stop_event=None, progress_callback=None):
        """Run the preprocessing pipeline on a list of files.

        Parameters
        ----------
        stop_event : threading.Event, optional
            When set, the pipeline terminates the worker pool and raises
            InterruptedError as soon as the current file finishes processing.
        progress_callback : callable(completed: int, total: int), optional
            Called after each file completes. Use to update progress tracking.
        """
        self.logger.info("Running preprocessing pipeline")

        if len(input_docs) == 0:
            raise ValueError("No files found")

        # Single file or multiprocessing disabled: skip pool entirely to avoid
        # subprocess spawn + tokenizer reload overhead.
        use_pool = self.multiprocessing and len(input_docs) > 1

        if use_pool:
            n_workers = self._effective_workers(len(input_docs))
            self.logger.info("Running preprocessing pipeline with multiprocessing (workers=%d)", n_workers)
            # spawn avoids both (a) the fork-in-thread deadlock that occurs when
            # os.fork() is called from within the asyncio thread pool (FastAPI), and
            # (b) "Cannot re-initialize CUDA in forked subprocess" — forkserver only
            # avoids that if its server process happens to start before CUDA is ever
            # touched in the main process, which isn't guaranteed here (embedders for
            # already-loaded tenants may have initialized CUDA before the first pool
            # is created). spawn never inherits process state, so it's safe regardless.
            # _worker_run_file is a module-level function (picklable), so the
            # chain closure never needs to cross the process boundary.
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(
                processes=n_workers,
                initializer=_warmup_worker,
                initargs=(self.tokenizer_name, self.semantic_embedding_model),
            )
            preprocessed_docs = []
            try:
                for i, result in enumerate(pool.imap_unordered(_worker_run_file, input_docs)):
                    preprocessed_docs.append(result)
                    if progress_callback:
                        progress_callback(i + 1, len(input_docs))
                    if stop_event and stop_event.is_set():
                        raise InterruptedError("Preprocessing cancelled")
                pool.close()
            except Exception:
                pool.terminate()
                raise
            finally:
                # Pool's own context-manager __exit__ only sends terminate() and
                # returns immediately, without waiting for worker processes to
                # actually exit — racing the next ingestion job's workers trying
                # to allocate GPU memory on the same device before this pool's
                # workers have released theirs. Blocking here on join() guarantees
                # the GPU memory is freed before this call returns.
                pool.join()
        else:
            self.logger.info("Running preprocessing pipeline without multiprocessing (files=%d)", len(input_docs))
            preprocessed_docs = []
            for i, file in enumerate(input_docs):
                if stop_event and stop_event.is_set():
                    raise InterruptedError("Preprocessing cancelled")
                preprocessed_docs.append(self.run_on_file(file))
                if progress_callback:
                    progress_callback(i + 1, len(input_docs))

        # Flatten: files → paragraphs → chunks
        preprocessed_docs = [
            doc for document_chunks in preprocessed_docs for document_chunk in document_chunks for doc in document_chunk
        ]
        self.logger.info("Total number of chunks: %d", len(preprocessed_docs))

        return preprocessed_docs

    def run_on_file(self, file_path: Path):
        start_time = time.perf_counter()
        output = self.chain.invoke(file_path)
        self.logger.debug("Processing file %s took %.2f seconds", file_path, time.perf_counter() - start_time)
        return output
