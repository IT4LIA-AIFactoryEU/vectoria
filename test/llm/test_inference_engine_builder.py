#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import pytest

from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder

try:
    import torch

    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False

pytestmark = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")


def test_inference_engine_builder(config):
    InferenceEngineBuilder.clear_cache()

    config.set("inference_engine", "name", "huggingface")

    config.set("inference_engine", "model_name", "HuggingFaceTB/SmolLM-135M")
    engine_1 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))

    config.set("inference_engine", "model_name", "HuggingFaceTB/SmolLM-135M")
    engine_2 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))

    config.set("inference_engine", "model_name", "BAAI/bge-reranker-base")
    engine_3 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))

    assert engine_1 == engine_2
    assert engine_1 != engine_3
    assert engine_2 != engine_3

    assert InferenceEngineBuilder.CACHE[f"{engine_1.name}-{engine_1.args.get('model_name')}"] == engine_1
    assert InferenceEngineBuilder.CACHE[f"{engine_2.name}-{engine_2.args.get('model_name')}"] == engine_2
    assert InferenceEngineBuilder.CACHE[f"{engine_3.name}-{engine_3.args.get('model_name')}"] == engine_3

    assert len(InferenceEngineBuilder.CACHE) == 2
