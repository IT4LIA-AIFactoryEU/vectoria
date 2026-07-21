from unittest.mock import MagicMock, patch

from vectoria_lib.core.config import Config
from vectoria_lib.rag.preprocessing.preprocessing_pipeline import _pin_worker_to_gpu


def test_no_op_without_cuda(config, monkeypatch):
    monkeypatch.setenv("DOCLING_DEVICE", "")
    monkeypatch.delenv("DOCLING_DEVICE", raising=False)
    with patch("torch.cuda.is_available", return_value=False):
        _pin_worker_to_gpu()
    assert "DOCLING_DEVICE" not in __import__("os").environ


def test_no_op_with_single_gpu(config, monkeypatch):
    monkeypatch.delenv("DOCLING_DEVICE", raising=False)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        _pin_worker_to_gpu()
    assert "DOCLING_DEVICE" not in __import__("os").environ


def test_round_robin_assigns_second_worker_to_second_gpu(config, monkeypatch):
    Config().set(
        "data_ingestion", "chunking",
        value={**Config().get("data_ingestion", "chunking"), "semantic_device": "cuda"},
    )
    fake_process = MagicMock()
    fake_process._identity = (2,)  # 1-based pool slot -> worker #2

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("multiprocessing.current_process", return_value=fake_process):
        _pin_worker_to_gpu()

    import os
    assert os.environ["DOCLING_DEVICE"] == "cuda:1"
    assert Config().get("data_ingestion", "chunking")["semantic_device"] == "cuda:1"


def test_does_not_override_cpu_semantic_device(config, monkeypatch):
    Config().set(
        "data_ingestion", "chunking",
        value={**Config().get("data_ingestion", "chunking"), "semantic_device": "cpu"},
    )
    fake_process = MagicMock()
    fake_process._identity = (1,)

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("multiprocessing.current_process", return_value=fake_process):
        _pin_worker_to_gpu()

    assert Config().get("data_ingestion", "chunking")["semantic_device"] == "cpu"
