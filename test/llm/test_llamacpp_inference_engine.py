#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging

from vectoria_lib.llm.inference_engine.llamacpp_inference_engine import LlamaCppInferenceEngine


def test_request_timeout_is_dropped_and_warned(caplog):
    engine = LlamaCppInferenceEngine({"name": "llamacpp", "model_name": "model.gguf", "request_timeout": 30})

    with caplog.at_level(logging.WARNING, logger="llm"):
        kwargs = engine._llamacpp_kwargs()

    assert "request_timeout" not in kwargs
    assert any("request_timeout" in r.message and "ignored" in r.message for r in caplog.records)


def test_no_warning_when_request_timeout_unset(caplog):
    engine = LlamaCppInferenceEngine({"name": "llamacpp", "model_name": "model.gguf"})

    with caplog.at_level(logging.WARNING, logger="llm"):
        engine._llamacpp_kwargs()

    assert not caplog.records
