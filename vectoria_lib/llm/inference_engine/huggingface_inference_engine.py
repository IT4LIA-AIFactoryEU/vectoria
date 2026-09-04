#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

import time

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase


class HuggingFaceInferenceEngine(InferenceEngineBase):
    """
    Wrapper on HuggingFace.

    TextGenerationPipeline init parameters: https://huggingface.co/docs/transformers/v4.45.1/en/main_classes/pipelines#transformers.TextGenerationPipeline
    TextGenerationPipeline call parameters: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/pipelines/text_generation.py#L215
        https://huggingface.co/docs/transformers/en/generation_strategies

    How to
    https://python.langchain.com/docs/how_to/#chat-models
    """

    def __init__(self, args: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        super().__init__(args)

        start_time = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args["model_name"], trust_remote_code=self.args["trust_remote_code"]
        )
        self.logger.debug("Loading tokenizer took %.2f seconds", time.perf_counter() - start_time)

        quantization_config = None
        if self.args["load_in_8bit"]:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        elif self.args["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            self.args["device"] = None

        start_time = time.perf_counter()

        # TODO: device_map="auto" will not deallocate memory between tests
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args["model_name"],
            quantization_config=quantization_config,
            device_map=self.args["device_map"],
            trust_remote_code=self.args["trust_remote_code"],
            # device = self.args["device"]
        )
        self.pipe = None
        self.logger.debug(
            "Loading model %s took %.2f seconds", self.args["model_name"], time.perf_counter() - start_time
        )
        self.model.eval()

    def as_langchain_completion_model(self) -> BaseLanguageModel:
        from transformers import pipeline

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.args["max_new_tokens"],
            return_full_text=False,
            temperature=self.args["temperature"],
            # do_sample=self.args["do_sample"],
        )
        return HuggingFacePipeline(pipeline=self.pipe)

    def as_langchain_chat_model(self) -> BaseLanguageModel:
        from transformers import pipeline

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.args["max_new_tokens"],
            return_full_text=False,
            temperature=self.args["temperature"],
            do_sample=self.args["do_sample"],
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=self.pipe))

    def as_langchain_embeddings_model(self) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=self.args["model_name"])
