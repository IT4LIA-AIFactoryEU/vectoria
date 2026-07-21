#
# VECTORIA
#
# @authors : Mattia Buzzoni
#
import logging
from typing import Optional

from langchain.agents.factory import create_agent

logger = logging.getLogger("llm")


def make_react_graph(
    llm,
    tools: list,
    sys_prompt: str,
    checkpoint_db_path: Optional[str] = None,
    middleware=None,
):
    """
    Builds and compiles a ReAct agent graph using LangChain's create_agent factory.

    create_agent handles the full agent loop (model → tools → model → …),
    system prompt injection, middleware application, and checkpointing.

    Parameters
    ----------
    llm                : BaseChatModel for the agent
    tools              : list of BaseTool instances
    sys_prompt         : system prompt string
    checkpoint_db_path : path to SQLite file for persistent memory;
                         if None, uses in-memory checkpointer
    middleware         : list of AgentMiddleware instances (or None/empty to disable)
    """
    if checkpoint_db_path:
        from vectoria_lib.llm.agents.threaded_sqlite_saver import ThreadedSqliteSaver

        checkpointer = ThreadedSqliteSaver.from_path(checkpoint_db_path)
        logger.info("Using SQLite checkpointer at %s", checkpoint_db_path)
    else:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        logger.debug("Using in-memory checkpointer")

    compiled = create_agent(
        model=llm,
        tools=tools,
        system_prompt=sys_prompt,
        middleware=middleware or [],
        checkpointer=checkpointer,
    )
    logger.debug("ReAct graph compiled with %d tool(s)", len(tools))
    return compiled
