from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLanguageModel

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

_EXCLUDED_FIELDS = {
  "name", "provider", "url", "base_url", "openai_api_base", "openai_api_key",
  "device", "load_in_4bit", "load_in_8bit", "max_new_tokens", "trust_remote_code",
  "device_map", "do_sample", "n_ctx", "n_gpu_layers", "n_threads", "n_batch",
  "use_mlock", "use_mmap", "repeat_penalty", "streaming", "seed", "top_p",
  "frequency_penalty", "presence_penalty", "n", "request_timeout", "max_retries",
  "model_kwargs",
}

class AnthropicInferenceEngine(InferenceEngineBase):
  def __init__(self, args: dict):
    super().__init__(args)

  def _anthropic_kwargs(self) -> dict:
    return {
      k: v for k, v in self.args.items()
      if v is not None and k not in _EXCLUDED_FIELDS
    }

  def as_langchain_chat_model(self) -> BaseLanguageModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(**self._anthropic_kwargs())

  def as_langchain_completion_model(self) -> BaseLanguageModel:
    return self.as_langchain_chat_model()

  def as_langchain_embeddings_model(self) -> Embeddings:
    raise NotImplementedError("Anthropic does not provide an embeddings model")
