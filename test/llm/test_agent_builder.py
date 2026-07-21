import pytest

from vectoria_lib.llm.agents.agent_builder import AgentBuilder

try:
    import torch

    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False

pytestmark = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")


def test_agent_builder(config, index_test_folder):
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    assert agent is not None
    assert agent.graph is not None
    assert agent.retriever_tool is not None
    assert agent.retriever_tool.reranker is None
    assert agent.retriever_tool.full_paragraphs is None


def test_agent_builder_no_retriever(config, index_test_folder):
    config.set("retriever", "enabled", False)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    assert agent is not None
    assert agent.graph is not None
    assert agent.retriever_tool.retriever is None


def test_agent_builder_with_reranker(config, index_test_folder):
    config.set("reranker", "enabled", True)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    assert agent is not None
    assert agent.graph is not None
    assert agent.retriever_tool.reranker is not None


def test_agent_builder_with_reranker_with_full_paragraphs_retriever(config, index_test_folder):
    config.set("reranker", "enabled", True)
    config.set("full_paragraphs_retriever", "enabled", True)
    agent = AgentBuilder.build_qa_agent(index_path=index_test_folder)
    assert agent is not None
    assert agent.graph is not None
    assert agent.retriever_tool.reranker is not None
    assert agent.retriever_tool.full_paragraphs is not None
