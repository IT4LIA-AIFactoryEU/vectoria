#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging

from vectoria_lib.core.config import Config
from vectoria_lib.llm.agents.graphs import make_react_graph
from vectoria_lib.llm.agents.qa import QAAgent
from vectoria_lib.llm.agents.tools.retriever_tool import RetrieverTool
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
from vectoria_lib.rag.reranking.full_paragraphs import FullParagraphs
from vectoria_lib.rag.reranking.huggingface_reranker import Reranker
from vectoria_lib.rag.retriever.retriever_builder import RetrieverBuilder
from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder


class AgentBuilder:
    logger = logging.getLogger("llm")

    @staticmethod
    def _create_chain_configuration(kwargs):
        config = Config()

        if "index_path" not in kwargs:
            raise ValueError("No index path provided")

        vector_store = VectorStoreBuilder().build(config.get("vector_store"), index_path=kwargs["index_path"])

        retriever_config = None
        if config.get("retriever", "enabled"):
            llm_for_mq = None
            if config.get("retriever", "multi_query"):
                llm_for_mq = InferenceEngineBuilder.build_inference_engine_no_thinking(
                    config.get("inference_engine")
                ).as_langchain_chat_model()
            retriever_config = {
                "retriever": RetrieverBuilder().build(config.get("retriever"), vector_store, llm=llm_for_mq)
            }

        reranker_config = None
        if config.get("reranker", "enabled"):
            reranker_config = {
                "reranker": Reranker(
                    model_name=config.get("reranker", "model_name"),
                    top_n=config.get("reranker", "reranked_top_k"),
                    device=config.get("reranker", "device"),
                ),
            }

        full_paragraphs_retriever_config = None
        if config.get("full_paragraphs_retriever", "enabled"):
            full_paragraphs_retriever_config = {"retriever": FullParagraphs(vector_store)}
        return retriever_config, reranker_config, full_paragraphs_retriever_config

    @staticmethod
    def _resolve_tenant_checkpoint_path(checkpoint_db_path, tenant_id: str):
        """Suffix checkpoint_db_path with the tenant id so each tenant gets an
        isolated SQLite checkpoint DB — always, with no exception for
        tenant_id == "default" (a real tenant literally named "default" must
        not silently share the unsuffixed base path with anyone).

        Migrates a pre-existing unsuffixed DB into place on first use, so a
        deployment that already has history under the old unsuffixed path
        doesn't lose it — only fires once (skipped if the suffixed path
        already exists).

        Returns checkpoint_db_path unchanged (including None) if it's falsy
        — chat_history.enabled: false means no checkpointer at all.
        """
        if not checkpoint_db_path:
            return checkpoint_db_path
        from pathlib import Path as _Path

        p = _Path(checkpoint_db_path)
        new_path = p.with_stem(f"{p.stem}_{tenant_id}")
        if not new_path.exists() and p.exists():
            p.rename(new_path)
            # SQLite WAL mode (used by the checkpointer) can leave recently
            # committed rows only in -wal (and sometimes -shm), not yet
            # folded into the main file. Move them too, or a clean rename of
            # just the .db file silently orphans the most recent history.
            for suffix in ("-wal", "-shm"):
                old_sidecar = _Path(str(p) + suffix)
                if old_sidecar.exists():
                    old_sidecar.rename(_Path(str(new_path) + suffix))
        return str(new_path)

    @staticmethod
    def build_qa_agent(**kwargs: dict) -> QAAgent:
        return AgentBuilder.build_react_agent(**kwargs)

    @staticmethod
    def build_react_agent(**kwargs) -> QAAgent:
        """
        Builds a QAAgent backed by a LangGraph ReAct graph.

        The RAG pipeline (Retriever + optional Reranker + optional FullParagraphs)
        is exposed to the LLM as a single BaseTool.  The LLM decides autonomously
        when to call the retriever and reasons over the results before answering.

        Accepts the same kwargs as build_qa_agent() (e.g. index_path=...).
        """
        config = Config()

        retriever_config, reranker_config, full_paragraphs_config = AgentBuilder._create_chain_configuration(kwargs)

        min_per_source = config.get("retriever", "min_per_source") or 0
        mmr_enabled = bool(config.get("retriever", "mmr_enabled") or False)
        mmr_lambda = float(config.get("retriever", "mmr_lambda") or 0.7)
        mmr_top_k = int(config.get("retriever", "mmr_top_k") or 0)
        embedder = None
        if mmr_enabled:
            from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache
            embedder = WikiEmbeddingCache.get_embedder()
        tool = RetrieverTool(
            retriever=retriever_config["retriever"] if retriever_config else None,
            reranker=reranker_config["reranker"] if reranker_config else None,
            full_paragraphs=full_paragraphs_config["retriever"] if full_paragraphs_config else None,
            min_per_source=min_per_source,
            mmr_enabled=mmr_enabled,
            mmr_lambda=mmr_lambda,
            mmr_top_k=mmr_top_k,
            embedder=embedder,
        )

        llm = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine")).as_langchain_chat_model()

        middleware: list = []
        if config.get("chat_history", "enabled"):
            from langchain.agents.middleware.summarization import SummarizationMiddleware

            summary_llm_cfg = config.get("chat_history", "summary_llm")
            if summary_llm_cfg:
                if isinstance(summary_llm_cfg, str):
                    base_cfg = dict(config.get("inference_engine"))
                    base_cfg["model_name"] = summary_llm_cfg
                    summary_llm_cfg = base_cfg
                summary_model = InferenceEngineBuilder.build_inference_engine(summary_llm_cfg).as_langchain_chat_model()
            else:
                summary_model = llm

            middleware.append(
                SummarizationMiddleware(
                    model=summary_model,
                    trigger=("messages", config.get("chat_history", "summarize_after_n_messages")),
                    keep=("messages", config.get("chat_history", "max_conv_messages")),
                )
            )
            AgentBuilder.logger.info(
                "SummarizationMiddleware enabled (trigger=%d msgs, keep=%d msgs)",
                config.get("chat_history", "summarize_after_n_messages"),
                config.get("chat_history", "max_conv_messages"),
            )

        tool_call_limit_cfg = config.get("agent_guardrails", "tool_call_limit")
        if tool_call_limit_cfg.get("enabled"):
            from langchain.agents.middleware import ToolCallLimitMiddleware

            middleware.append(
                ToolCallLimitMiddleware(
                    thread_limit=tool_call_limit_cfg.get("thread_limit"),
                    run_limit=tool_call_limit_cfg.get("run_limit"),
                    exit_behavior=tool_call_limit_cfg.get("exit_behavior"),
                )
            )
            AgentBuilder.logger.info(
                "ToolCallLimitMiddleware enabled (run_limit=%s, thread_limit=%s, exit_behavior=%s)",
                tool_call_limit_cfg.get("run_limit"),
                tool_call_limit_cfg.get("thread_limit"),
                tool_call_limit_cfg.get("exit_behavior"),
            )

        human_approval_cfg = config.get("agent_guardrails", "human_approval")
        if human_approval_cfg.get("enabled") and human_approval_cfg.get("tools"):
            from langchain.agents.middleware import HumanInTheLoopMiddleware

            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        tool_name: {"allowed_decisions": ["approve", "reject"]}
                        for tool_name in human_approval_cfg["tools"]
                    },
                    description_prefix="Conferma richiesta prima di eseguire questa azione",
                )
            )
            AgentBuilder.logger.info("HumanInTheLoopMiddleware enabled for tools: %s", human_approval_cfg["tools"])

        upload_url = kwargs.get("upload_url", "http://localhost:8080/upload")
        extra_tools = kwargs.get("extra_tools", [])
        _graph_enabled = any(getattr(t, "name", "") == "graph_search" for t in extra_tools)
        sys_prompt = PromptBuilder.get_prompt(
            "qa.txt",
            config.get("system_prompts_lang"),
            variables={
                "upload_url": upload_url,
                "graph_tools_list": ", graph_search, graph_neighbors" if _graph_enabled else "",
                "graph_step": (
                    "\n\n   For relationship queries (people, organizations, projects, products, regulations):\n"
                    "   - CALL graph_search(query=<topic>) BEFORE the retriever to find relevant entities and their source documents.\n"
                    "   - Optionally CALL graph_neighbors(entity=<name>) to explore direct connections of a specific entity."
                    if _graph_enabled else ""
                ),
                "graph_step_it": (
                    "\n\n   Per domande relazionali (persone, organizzazioni, progetti, prodotti, normative):\n"
                    "   - CHIAMA graph_search(query=<argomento>) PRIMA del retriever per trovare entità rilevanti e i loro documenti sorgente.\n"
                    "   - Usa CALL graph_neighbors(entity=<nome>) per esplorare le connessioni dirette di un'entità specifica."
                    if _graph_enabled else ""
                ),
            },
        )

        tenant_id = kwargs.get("tenant_id", "default")
        checkpoint_db_path = AgentBuilder._resolve_tenant_checkpoint_path(
            config.get("chat_history", "checkpoint_db_path"), tenant_id
        )

        graph = make_react_graph(
            llm,
            [tool] + extra_tools,
            sys_prompt,
            checkpoint_db_path=checkpoint_db_path,
            middleware=middleware,
        )

        AgentBuilder.logger.info("Created ReAct QA agent with graph backend")
        return QAAgent(graph=graph, retriever_tool=tool)
