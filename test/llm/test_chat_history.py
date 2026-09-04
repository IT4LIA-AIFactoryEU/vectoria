#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
"""
Tests for multi-turn conversation management in make_react_graph:
  1. Messages accumulate across turns on the same thread_id
  2. SummaryNode is triggered after summarize_after_n_messages
  3. SQLite checkpointer persists state across separate graph instances
"""

import sqlite3
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from vectoria_lib.llm.agents.graphs import make_react_graph

# ── helpers ────────────────────────────────────────────────────────────────────


def _make_mock_llm(answer: str = "Mock answer"):
    """Returns a mock LLM that always returns a plain AIMessage (no tool calls)."""
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.bind.return_value = llm  # used by create_agent when tools=[]
    llm.invoke.return_value = AIMessage(content=answer)
    return llm


def _state(graph, thread_id: str) -> dict:
    """Reads the current LangGraph state for a given thread."""
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values


def _ask(graph, question: str, thread_id: str) -> dict:
    return graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ── tests ──────────────────────────────────────────────────────────────────────


def test_messages_accumulate_across_turns():
    """Same thread_id → messages grow with each turn."""
    llm = _make_mock_llm()
    graph = make_react_graph(llm, tools=[], sys_prompt="You are helpful.")
    tid = str(uuid.uuid4())

    _ask(graph, "First question", tid)
    _ask(graph, "Second question", tid)
    _ask(graph, "Third question", tid)

    state = _state(graph, tid)
    messages = state.get("messages", [])
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 3, f"Expected 3 human messages, got {len(human_msgs)}"


def test_different_threads_are_isolated():
    """Different thread_ids must not share state."""
    llm = _make_mock_llm()
    graph = make_react_graph(llm, tools=[], sys_prompt="You are helpful.")
    tid_a = str(uuid.uuid4())
    tid_b = str(uuid.uuid4())

    _ask(graph, "Question on thread A", tid_a)
    _ask(graph, "Question on thread A again", tid_a)
    _ask(graph, "Question on thread B", tid_b)

    state_a = _state(graph, tid_a)
    state_b = _state(graph, tid_b)

    human_a = [m for m in state_a.get("messages", []) if isinstance(m, HumanMessage)]
    human_b = [m for m in state_b.get("messages", []) if isinstance(m, HumanMessage)]

    assert len(human_a) == 2
    assert len(human_b) == 1


def test_middleware_parameter_accepted():
    """make_react_graph accepts middleware=None without errors."""
    llm = _make_mock_llm()
    graph = make_react_graph(llm, tools=[], sys_prompt="You are helpful.", middleware=None)
    tid = str(uuid.uuid4())

    _ask(graph, "Question", tid)
    state = _state(graph, tid)
    assert len(state.get("messages", [])) > 0


def test_llm_called_for_each_turn():
    """LLM is invoked once per conversation turn."""
    llm = _make_mock_llm()
    graph = make_react_graph(llm, tools=[], sys_prompt="You are helpful.")
    tid = str(uuid.uuid4())

    _ask(graph, "Turn 1", tid)
    _ask(graph, "Turn 2", tid)
    _ask(graph, "Turn 3", tid)

    assert llm.invoke.call_count == 3


def test_sqlite_persistence_across_graph_instances():
    """State survives creating a brand-new graph instance on the same DB."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_memory.db")
        tid = str(uuid.uuid4())

        llm = _make_mock_llm()

        # First instance: ask two questions
        graph1 = make_react_graph(llm, tools=[], sys_prompt="You are helpful.", checkpoint_db_path=db_path)
        _ask(graph1, "First question", tid)
        _ask(graph1, "Second question", tid)

        # Second instance on the same DB: state must already contain 2 human msgs
        graph2 = make_react_graph(llm, tools=[], sys_prompt="You are helpful.", checkpoint_db_path=db_path)
        state = _state(graph2, tid)
        human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        assert len(human_msgs) == 2, f"Expected 2 persisted messages, got {len(human_msgs)}"
