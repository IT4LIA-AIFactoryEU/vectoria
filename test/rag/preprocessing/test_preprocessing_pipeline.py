#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import os
from pathlib import Path

import langchain_core
import pytest

from vectoria_lib.rag.preprocessing.preprocessing_pipeline import PreprocessingPipeline, _warmup_worker


def test_build_preprocessing_pipeline(config):
    pipeline: PreprocessingPipeline = PreprocessingPipeline.build_pipeline()
    assert isinstance(pipeline.chain, langchain_core.runnables.base.RunnableSequence)
    assert pipeline.chain.first is not None
    assert len(pipeline.chain.middle) == 1
    assert pipeline.chain.last is not None


@pytest.mark.parametrize("multiproc", [False, True])
def test_run_preprocessing_pipeline(config, data_dir, multiproc):
    config.set("data_ingestion", "multiprocessing", multiproc)
    pipeline = PreprocessingPipeline.build_pipeline()
    docs = [Path(data_dir / "docx" / doc_name) for doc_name in os.listdir(data_dir / "docx")]
    processed_docs = pipeline.run(docs)
    assert len(processed_docs) == 95


def test_build_pipeline_computes_semantic_embedding_model_for_semantic_chunking(config):
    config.set("data_ingestion", "chunking", value={"method": "semantic"})
    pipeline = PreprocessingPipeline.build_pipeline()
    assert pipeline.semantic_embedding_model == "BAAI/bge-m3"  # falls back to vector_store.model_name


def test_build_pipeline_computes_semantic_embedding_model_override(config):
    config.set("data_ingestion", "chunking", value={"method": "semantic", "semantic_embedding_model": "other/model"})
    pipeline = PreprocessingPipeline.build_pipeline()
    assert pipeline.semantic_embedding_model == "other/model"


def test_build_pipeline_semantic_embedding_model_none_for_recursive_chunking(config):
    pipeline = PreprocessingPipeline.build_pipeline()
    assert pipeline.semantic_embedding_model is None


def test_warmup_worker_preloads_semantic_model_when_configured(config, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.preprocessing_pipeline._pin_worker_to_gpu",
        lambda: None,
    )
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.chunking._load_sentence_transformer",
        lambda model_name, device: calls.append((model_name, device)),
    )

    _warmup_worker(tokenizer_name=None, semantic_embedding_model="BAAI/bge-m3")

    assert calls == [("BAAI/bge-m3", "cpu")]


def test_warmup_worker_skips_semantic_preload_when_not_configured(config, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.preprocessing_pipeline._pin_worker_to_gpu",
        lambda: None,
    )
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.chunking._load_sentence_transformer",
        lambda model_name, device: calls.append((model_name, device)),
    )

    _warmup_worker(tokenizer_name=None, semantic_embedding_model=None)

    assert calls == []
