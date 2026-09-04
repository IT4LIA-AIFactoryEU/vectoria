#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

# Fields that ChatOpenAI/OpenAIEmbeddings do not accept.
_HUGGINGFACE_ONLY_FIELDS = {
    "device",
    "load_in_4bit",
    "load_in_8bit",
    "max_new_tokens",
    "trust_remote_code",
    "device_map",
    "do_sample",
}
_LLAMACPP_ONLY_FIELDS = {
    "n_ctx",
    "n_gpu_layers",
    "n_threads",
    "n_batch",
    "use_mlock",
    "use_mmap",
    "repeat_penalty",
}
_EXCLUDED_FIELDS = _HUGGINGFACE_ONLY_FIELDS | _LLAMACPP_ONLY_FIELDS

# Generic Vectoria fields that map to OpenAI-specific equivalents.
_FIELD_ALIASES = {
    "api_key": "openai_api_key",
    "url": "openai_api_base",
}

# Keys inside model_kwargs that are Ollama options (not valid OpenAI params).
_OLLAMA_OPTION_KEYS = {"num_ctx", "num_predict", "repeat_penalty", "top_k", "top_p", "seed"}

# Keys inside model_kwargs that must be promoted to direct ChatOpenAI kwargs.
# Passing extra_body via model_kwargs does NOT forward it as the HTTP extra_body field
# in langchain-openai >=0.2, causing parameters like enable_thinking to be silently dropped.
_PROMOTE_TO_DIRECT_KWARGS = {"extra_body"}


class OpenAIInferenceEngine(InferenceEngineBase):
    def __init__(self, args: dict):
        super().__init__(args)

    def _openai_kwargs(self) -> dict:
        kwargs = {
            k: v
            for k, v in self.args.items()
            if v is not None and k not in _EXCLUDED_FIELDS and k not in _FIELD_ALIASES and k != "model_kwargs"
        }
        # Map generic Vectoria fields to their OpenAI equivalents (only if not already set).
        for generic, openai_key in _FIELD_ALIASES.items():
            if self.args.get(generic) is not None and openai_key not in kwargs:
                kwargs[openai_key] = self.args[generic]
        # Convert model_kwargs: Ollama options go via extra_body; extra_body is promoted
        # to a direct kwarg so it reaches the HTTP request body correctly.
        raw_mkwargs = self.args.get("model_kwargs") or {}
        ollama_opts = {k: v for k, v in raw_mkwargs.items() if k in _OLLAMA_OPTION_KEYS}
        other_mkwargs = {
            k: v for k, v in raw_mkwargs.items()
            if k not in _OLLAMA_OPTION_KEYS and k not in _PROMOTE_TO_DIRECT_KWARGS
        }
        merged_mkwargs = dict(other_mkwargs)
        if ollama_opts:
            merged_mkwargs.setdefault("extra_body", {})
            merged_mkwargs["extra_body"].setdefault("options", {}).update(ollama_opts)
        if merged_mkwargs:
            kwargs["model_kwargs"] = merged_mkwargs
        # Promote extra_body directly so langchain-openai passes it as the HTTP extra_body.
        if "extra_body" in raw_mkwargs:
            existing = kwargs.get("extra_body") or {}
            existing.update(raw_mkwargs["extra_body"])
            kwargs["extra_body"] = existing
        return kwargs

    def as_langchain_chat_model(self) -> BaseLanguageModel:
        return ChatOpenAI(**self._openai_kwargs())

    def as_langchain_completion_model(self) -> BaseLanguageModel:
        return self.as_langchain_chat_model()

    def as_langchain_embeddings_model(self) -> Embeddings:
        return OpenAIEmbeddings(**self._openai_kwargs())
