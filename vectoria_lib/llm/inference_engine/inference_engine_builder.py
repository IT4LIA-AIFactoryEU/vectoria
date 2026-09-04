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
from copy import deepcopy

from vectoria_lib.core.config_schema import InferenceEngineConfig
from vectoria_lib.core.utils import Singleton
from vectoria_lib.llm.inference_engine.huggingface_inference_engine import HuggingFaceInferenceEngine
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from vectoria_lib.llm.inference_engine.llamacpp_inference_engine import LlamaCppInferenceEngine
from vectoria_lib.llm.inference_engine.anthropic_inference_engine import AnthropicInferenceEngine
from vectoria_lib.llm.inference_engine.openai_inference_engine import OpenAIInferenceEngine


class InferenceEngineBuilder(Singleton):
    """
    Builds an inference engine objects and caches them.
    """

    CACHE = {}
    _cache_lock = threading.Lock()
    logger = logging.getLogger("llm")

    @staticmethod
    def build_inference_engine(args: dict | InferenceEngineConfig) -> InferenceEngineBase:
        if isinstance(args, dict):
            args = InferenceEngineConfig(**args)

        name = args.name
        model_name = args.model_name
        # exclude_unset=True keeps only fields that were explicitly provided,
        # matching the original dict-based behaviour expected by the engine classes.
        args_dict = args.model_dump(exclude_unset=True)

        cache_key = f"{name}-{model_name}"
        with InferenceEngineBuilder._cache_lock:
            if cache_key in InferenceEngineBuilder.CACHE:
                InferenceEngineBuilder.logger.info("Returning cached inference engine for %s-%s", name, model_name)
                return InferenceEngineBuilder.CACHE[cache_key].update_args(args_dict)

            if name == "huggingface":
                inference_engine = HuggingFaceInferenceEngine(args_dict)
            elif name == "llamacpp":
                inference_engine = LlamaCppInferenceEngine(args_dict)
            elif name == "anthropic":
                inference_engine = AnthropicInferenceEngine(args_dict)
            else:  # "openai" (and any OpenAI-compatible server, e.g. vLLM served via /v1)
                inference_engine = OpenAIInferenceEngine(args_dict)

            InferenceEngineBuilder.CACHE[cache_key] = inference_engine
            InferenceEngineBuilder.logger.info(
                "Caching inference engine for %s-%s. Cache size: %d. Cache keys: %s",
                name,
                model_name,
                len(InferenceEngineBuilder.CACHE),
                list(InferenceEngineBuilder.CACHE.keys()),
            )

        return inference_engine

    @staticmethod
    def build_inference_engine_no_thinking(args: dict | InferenceEngineConfig) -> InferenceEngineBase:
        """Like build_inference_engine, but forces thinking mode off.

        For auxiliary calls that parse the model's raw output structurally — a
        leaked <think> block would break that parsing. Returns a distinct engine
        (never mutates the cached original), so this is safe regardless of the
        main chat model's own thinking setting.
        """
        if isinstance(args, dict):
            args = InferenceEngineConfig(**args)
        args_dict = deepcopy(args.model_dump(exclude_unset=True))
        extra_body = args_dict.setdefault("model_kwargs", {}).setdefault("extra_body", {})
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        return InferenceEngineBuilder.build_inference_engine(args_dict)

    @staticmethod
    def clear_cache():
        with InferenceEngineBuilder._cache_lock:
            InferenceEngineBuilder.CACHE = {}
        InferenceEngineBuilder.logger.info("Cleared inference engine cache")
