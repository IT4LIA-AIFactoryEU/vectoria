#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

import logging

from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLanguageModel

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

logger = logging.getLogger("llm")

# Fields that belong to Vectoria's generic config but are not accepted by ChatLlamaCpp.
_EXCLUDED_FIELDS = {
    # OpenAI-specific
    "url",
    "api_key",
    "base_url",
    "openai_api_base",
    "openai_api_key",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "request_timeout",
    "max_retries",
    "batch_size",
    "streaming",
    # HuggingFace-specific
    "device",
    "load_in_4bit",
    "load_in_8bit",
    "max_new_tokens",
    "trust_remote_code",
    "device_map",
    "do_sample",
}

# Vectoria generic → llama.cpp parameter name
_FIELD_ALIASES = {
    "model_name": "model_path",
}


class LlamaCppInferenceEngine(InferenceEngineBase):
    def __init__(self, args: dict):
        super().__init__(args)

    def _llamacpp_kwargs(self) -> dict:
        if self.args.get("request_timeout") is not None:
            logger.warning(
                "inference_engine.request_timeout is set but has no effect on the llama.cpp backend "
                "(ChatLlamaCpp runs generation in-process, with no timeout support) — it will be ignored."
            )
        kwargs = {k: v for k, v in self.args.items() if v is not None and k not in _EXCLUDED_FIELDS}
        for generic, llamacpp_key in _FIELD_ALIASES.items():
            if generic in kwargs:
                kwargs[llamacpp_key] = kwargs.pop(generic)
        return kwargs

    def as_langchain_chat_model(self) -> BaseLanguageModel:
        return ChatLlamaCpp(**self._llamacpp_kwargs())

    def as_langchain_completion_model(self) -> BaseLanguageModel:
        return self.as_langchain_chat_model()

    def as_langchain_embeddings_model(self) -> Embeddings:
        return LlamaCppEmbeddings(**self._llamacpp_kwargs())
