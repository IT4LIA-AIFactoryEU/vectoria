#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("rag")


class _LockedEmbeddings(Embeddings):
    """Serializes calls into a shared embedder.

    The HuggingFace fast tokenizer (Rust/PyO3) is not safe for concurrent
    .encode() calls from multiple OS threads on the same instance. A single
    agent turn routinely has several tool calls running concurrently
    (retriever, wiki_query, graph_search via LangGraph's ToolNode gather),
    and the retriever itself fans out multiple queries concurrently
    (MultiQueryRetriever, EnsembleRetriever's rank fusion) — each hitting
    this SAME shared embedder via asyncio's run_in_executor thread pool.
    Observed in production: "RuntimeError: Already borrowed" from
    tokenization_utils_fast.py's enable_padding(), which mutates the
    tokenizer's internal state on every call. Serializing with a lock trades
    a little parallelism for correctness — embedding a handful of short
    queries is not the bottleneck retrieval latency comes from.
    """

    def __init__(self, inner: Embeddings):
        self._inner = inner
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            return self._inner.embed_query(text)


# Module-level embedder cache — shared across ALL tenants and vector store instances
# that use the same (model_name, device, normalize_embeddings) combination.
# Prevents loading duplicate model weights into memory for each tenant.
_EMBEDDER_CACHE: dict[str, Embeddings] = {}
# Tenants load concurrently at startup (asyncio.gather over asyncio.to_thread, real OS
# threads) -- an unprotected "if key not in cache: build" check races: two threads can
# both see a cache miss and both build a HuggingFaceEmbeddings for the same GPU at the
# same time, corrupting accelerate's dispatch_model() (the same concurrent-load crash
# observed for the reranker's CrossEncoder despite its own device_map fix).
_CACHE_LOCK = threading.Lock()


def get_shared_embedder(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    batch_size: int = 64,
) -> Embeddings:
    """Return a cached, thread-safe embedder, creating it on first call."""
    key = f"{model_name}::{device}::{normalize_embeddings}::{batch_size}"
    with _CACHE_LOCK:
        if key not in _EMBEDDER_CACHE:
            logger.info(
                "Loading embedder '%s' (device=%s, batch_size=%d) — will be shared across all tenants",
                model_name,
                device,
                batch_size,
            )
            # ponytail: transformers 4.57+ always uses init_empty_weights() (meta tensors).
            # For safetensors checkpoints without device_map, some tensors (buffers, tied weights)
            # remain on meta after _load_state_dict_into_meta_model → self.to(device) raises
            # NotImplementedError. Passing device_map=target_device triggers dispatch_model() which
            # materialises ALL meta tensors directly on the right device in one shot.
            # low_cpu_mem_usage is silently discarded in 4.57+ (kwargs.pop → /dev/null).
            target_device = device or "cpu"
            _inner_kw: dict = {"device_map": target_device}
            if target_device != "cpu":
                # torch_dtype, not dtype: transformers 4.51 (our pinned version, see
                # pyproject.toml) only pops "torch_dtype" in from_pretrained() — a
                # "dtype" kwarg falls through unconsumed into the model's own
                # __init__, which rejects it (TypeError: unexpected keyword 'dtype').
                _inner_kw["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            embedder = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": target_device,
                    "model_kwargs": _inner_kw,
                },
                encode_kwargs={
                    "normalize_embeddings": normalize_embeddings,
                    "batch_size": batch_size,
                },
            )
            _EMBEDDER_CACHE[key] = _LockedEmbeddings(embedder)
        return _EMBEDDER_CACHE[key]


def normalize_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe format (replace '/' with '__').

    Example: "sentence-transformers/all-MiniLM-L6-v2"
             → "sentence-transformers__all-MiniLM-L6-v2"
    """
    return model_name.replace("/", "__")


class VectorStoreBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger("rag")
        # Populated by load_from_disk(); used by retrievers to locate sibling files
        # (e.g. persisted BM25 index).
        self.index_path: Optional[Path] = None

    @abstractmethod
    def make_index(self, docs: list[Document]):
        pass

    @abstractmethod
    def dump_to_disk(self, output_path: str | Path = ".", output_suffix: str = ""):
        pass

    @abstractmethod
    def load_from_disk(self, input_path: str | Path):
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        pass

    @abstractmethod
    def search(self, query: str, **kwargs):
        pass

    @abstractmethod
    def is_empty(self):
        pass

    @abstractmethod
    def get_all_documents(self) -> list:
        pass

    def close(self):
        """Release any held resources (e.g. file locks). No-op by default."""
        pass
