#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import threading
import time

from vectoria_lib.rag.reranking.huggingface_reranker import Reranker, _MODEL_CACHE


def test_reranker_loads_cpu_model_without_device_map(monkeypatch):
    """CPU should still work with a plain device_map="cpu" (no dtype override)."""
    _MODEL_CACHE.clear()
    calls = []

    class _FakeCrossEncoder:
        def __init__(self, model_name, device=None, model_kwargs=None, tokenizer_kwargs=None):
            calls.append({"model_name": model_name, "device": device, "model_kwargs": model_kwargs})

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)

    Reranker(model_name="fake/model", top_n=3, device="cpu")

    assert len(calls) == 1
    assert calls[0]["device"] == "cpu"
    assert calls[0]["model_kwargs"] == {"device_map": "cpu", "use_safetensors": True}


def test_reranker_loads_cuda_model_with_device_map_and_dtype(monkeypatch):
    """The bug this guards against: the old code loaded to CPU then called
    encoder.model.to(cuda), which crashed with "Cannot copy out of meta tensor;
    no data!" on transformers 4.57+. The fix passes device_map so
    from_pretrained materialises tensors directly on the target device --
    verify device_map (not a later .to() call) is how the device is set."""
    _MODEL_CACHE.clear()
    calls = []

    class _FakeCrossEncoder:
        def __init__(self, model_name, device=None, model_kwargs=None, tokenizer_kwargs=None):
            calls.append({"model_name": model_name, "device": device, "model_kwargs": model_kwargs})

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: True)

    Reranker(model_name="fake/model", top_n=3, device="cuda:0")

    assert len(calls) == 1
    assert calls[0]["device"] == "cuda:0"
    assert calls[0]["model_kwargs"]["device_map"] == "cuda:0"
    import torch

    assert calls[0]["model_kwargs"]["torch_dtype"] == torch.bfloat16


def test_reranker_reuses_cached_model_for_same_key(monkeypatch):
    _MODEL_CACHE.clear()
    calls = []

    class _FakeCrossEncoder:
        def __init__(self, model_name, device=None, model_kwargs=None, tokenizer_kwargs=None):
            calls.append(1)

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)

    r1 = Reranker(model_name="fake/model", top_n=3, device="cpu")
    r2 = Reranker(model_name="fake/model", top_n=5, device="cpu")

    assert len(calls) == 1
    assert r1.model is r2.model
    assert r1.top_n == 3
    assert r2.top_n == 5


def test_reranker_concurrent_construction_builds_once(monkeypatch):
    """Regression test: tenants load concurrently at startup (real OS threads via
    asyncio.to_thread). An unprotected cache check let two threads both see a miss
    and both call CrossEncoder(...) for the same GPU at the same time, corrupting
    accelerate's dispatch_model() ("Cannot copy out of meta tensor; no data!" even
    with the device_map fix, only under concurrent load)."""
    _MODEL_CACHE.clear()
    build_calls = []

    class _FakeCrossEncoder:
        def __init__(self, model_name, device=None, model_kwargs=None, tokenizer_kwargs=None):
            build_calls.append(1)
            time.sleep(0.05)  # widen the race window between the cache check and insert

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)

    results = []
    errors = []

    def worker():
        try:
            results.append(Reranker(model_name="fake/model", top_n=3, device="cuda:0"))
        except Exception as exc:  # pragma: no cover - only hit on real bugs
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(build_calls) == 1
    assert len({id(r.model) for r in results}) == 1

    _MODEL_CACHE.clear()
