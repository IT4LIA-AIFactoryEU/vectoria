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
import warnings

import torch
from langchain_core.documents import Document

from vectoria_lib.rag.reranking.postretrieval_step_base import PostRetrievalStepBase

logger = logging.getLogger("llm")

# Module-level cache: CrossEncoder weights are shared across all tenants that use
# the same (model_name, device) pair.  Only top_n may differ per instance.
_MODEL_CACHE: dict = {}
# Tenants load concurrently at startup (asyncio.gather over asyncio.to_thread, real
# OS threads) -- an unprotected "if key not in cache: build" check races: two threads
# can both see a cache miss and both call CrossEncoder(...) for the same GPU at the
# same time, corrupting accelerate's dispatch_model() (observed: "Cannot copy out of
# meta tensor; no data!" even with the device_map fix below, only under concurrent load).
_CACHE_LOCK = threading.Lock()


class Reranker(PostRetrievalStepBase):
    """
    Post-retrieval reranker based on a HuggingFace cross-encoder model.

    Uses sentence_transformers.CrossEncoder to score (query, passage) pairs
    and return the top-n most relevant documents.
    """

    def __init__(self, model_name: str, top_n: int = 3, device: str = "cpu"):
        super().__init__()
        from sentence_transformers import CrossEncoder

        key = f"{model_name}::{device}"
        with _CACHE_LOCK:
            if key not in _MODEL_CACHE:
                logger.info(
                    "Loading CrossEncoder '%s' (device=%s) — will be shared across all tenants",
                    model_name,
                    device,
                )
                # ponytail: transformers 4.57+ always uses init_empty_weights() (meta tensors).
                # The old "load to cpu, then .to(cuda)" two-step left some buffers/tied
                # weights stranded on the meta device, crashing with "Cannot copy out of
                # meta tensor; no data!" (same root cause fixed for the shared embedder in
                # vector_store_base.py). Passing device_map materialises every tensor
                # directly on the target device via from_pretrained's dispatch_model(), so
                # CrossEncoder's own internal self.to(device) call (which always runs) has
                # nothing left on meta to trip over.
                target_device = device or "cpu"
                # Force safetensors — see vector_store_base.get_shared_embedder for why.
                _model_kwargs = {"device_map": target_device, "use_safetensors": True}
                if target_device != "cpu":
                    # torch_dtype, not dtype — see vector_store_base.get_shared_embedder
                    # for why (transformers 4.51 only pops "torch_dtype").
                    _model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                # Suppress the sentencepiece byte-fallback warning (unfixable in fast tokenizer).
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*byte fallback.*", category=UserWarning)
                    encoder = CrossEncoder(
                        model_name,
                        device=target_device,
                        model_kwargs=_model_kwargs,
                        tokenizer_kwargs={"fix_mistral_regex": True},
                    )
                _MODEL_CACHE[key] = encoder
            self.model = _MODEL_CACHE[key]
        self.top_n = top_n
        logger.info("Reranker ready: model=%s top_n=%d device=%s", model_name, top_n, device)

    def post_process(self, docs: list[Document], query: str = "") -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
