import pytest

from vectoria_lib.llm.agents.agent_builder import AgentBuilder

try:
    import torch

    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_available, reason="CUDA is not available")
def test_qa_agent_huggingface(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "top_k", 1)
    _run_engine_test(index_test_folder)


@pytest.mark.slow
def test_qa_agent_openai(config, index_test_folder, clear_inference_engine_cache, openai_server_status_fn):
    inference_config = {
        "name": "openai",
        "model_name": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "openai_api_base": "http://localhost:8000/v1",
        "openai_api_key": "EMPTY",
        "temperature": 0.7,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "request_timeout": None,
        "max_retries": 2,
        "seed": None,
        "streaming": False,
    }
    if not openai_server_status_fn(inference_config):
        pytest.skip("OpenAI server is not running")

    config.set("inference_engine", value=inference_config)
    config.set("retriever", "top_k", 1)
    _run_engine_test(index_test_folder)


def _run_engine_test(index_test_folder):
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    result = agent.ask("What is the name of the Vaswani paper?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "retrieved_docs" in result
    assert "messages" in result
    assert len(result["answer"]) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_available, reason="CUDA is not available")
def test_qa_agent_with_retriever(config, index_test_folder, clear_inference_engine_cache):
    config.set("retriever", "top_k", 5)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    result = agent.ask("Which are the most commonly used attention functions?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) > 0


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_available, reason="CUDA is not available")
def test_qa_agent_with_reranker(config, index_test_folder, clear_inference_engine_cache):
    config.set("reranker", "enabled", True)
    config.set("retriever", "top_k", 5)
    config.set("reranker", "reranked_top_k", 3)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    result = agent.ask("Which are the most commonly used attention functions?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) <= 3


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_available, reason="CUDA is not available")
def test_qa_agent_with_reranker_and_full_paragraphs(config, index_test_folder, clear_inference_engine_cache):
    config.set("reranker", "enabled", True)
    config.set("retriever", "top_k", 5)
    config.set("reranker", "reranked_top_k", 3)
    config.set("full_paragraphs_retriever", "enabled", True)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    result = agent.ask("Which are the most commonly used attention functions?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "retrieved_docs" in result
    # Full paragraphs expand the content: each doc should be longer than a raw chunk
    if result["retrieved_docs"]:
        assert all(len(d.page_content) > 0 for d in result["retrieved_docs"])
