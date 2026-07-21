import logging
import os
import time
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("llm")


# ── MLflow tracing (optional) ─────────────────────────────────────────────────

_mlflow_configured = False
_mlflow_experiment_id: Optional[str] = None


def get_mlflow() -> bool:
    """Configure MLflow once and return True if enabled."""
    global _mlflow_configured, _mlflow_experiment_id
    if _mlflow_configured:
        return True
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return False
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        exp = client.get_experiment_by_name("vectoria")
        if exp is None:
            _mlflow_experiment_id = client.create_experiment("vectoria")
        else:
            _mlflow_experiment_id = exp.experiment_id
        mlflow.set_experiment(experiment_id=_mlflow_experiment_id)
        _mlflow_configured = True
        logger.info("MLflow tracing enabled (uri=%s, experiment_id=%s)", tracking_uri, _mlflow_experiment_id)
    except ImportError:
        logger.debug("mlflow package not installed — tracing disabled")
    except Exception as exc:
        logger.warning("MLflow setup failed: %s", exc)
    return _mlflow_configured


# ── Local logging handler ─────────────────────────────────────────────────────


class VectoriaCallbackHandler(BaseCallbackHandler):
    """Lightweight LangChain callback for structured local logging."""

    def __init__(self):
        super().__init__()
        self._llm_start: dict[str, float] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        model = (serialized.get("kwargs") or {}).get("model_name", "unknown")
        self._llm_start[run_id] = time.perf_counter()
        n = sum(len(batch) for batch in messages)
        logger.info("LLM start | model=%s | messages=%d", model, n)

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        model = (serialized.get("kwargs") or {}).get("model_name", "unknown")
        self._llm_start[run_id] = time.perf_counter()
        logger.info("LLM start | model=%s | prompts=%d", model, len(prompts))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        start = self._llm_start.pop(run_id, None)
        if start is None:
            logger.warning("on_llm_end: no start time for run_id=%s", run_id)
            elapsed = 0.0
        else:
            elapsed = time.perf_counter() - start
        usage = {}
        try:
            usage = response.llm_output.get("token_usage", {}) or {}
        except Exception:
            pass
        logger.info(
            "LLM end | %.2fs | prompt_tokens=%s | completion_tokens=%s",
            elapsed,
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
        )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._llm_start.pop(run_id, None)  # avoid orphaned entry
        logger.error("LLM error: %s", error)

    def on_retriever_start(self, serialized: dict, query: str, **kwargs: Any) -> None:
        logger.info("Retriever query: %r", query)

    def on_retriever_end(self, documents: list, **kwargs: Any) -> None:
        logger.info("Retriever returned %d document(s)", len(documents))

    def on_retriever_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error("Retriever error: %s", error)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        logger.info("Tool start | name=%s | input=%r", serialized.get("name", "?"), input_str[:120])

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        logger.info("Tool end | output=%r...", str(output)[:200])

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        logger.error("Tool error: %s", error)


def build_callbacks() -> list:
    """Returns the list of active callback handlers for local logging."""
    return [VectoriaCallbackHandler()]


def _mlflow_pipeline_params() -> dict:
    """Read pipeline configuration to log as MLflow params."""
    try:
        from vectoria_lib.core.config import Config

        cfg = Config()
        return {
            "llm_model": cfg.get("inference_engine", "model_name"),
            "top_k": str(cfg.get("retriever", "top_k")),
            "multi_query": str(cfg.get("retriever", "multi_query")),
            "search_type": cfg.get("retriever", "search_type"),
            "reranker_enabled": str(cfg.get("reranker", "enabled")),
            "reranked_top_k": str(cfg.get("reranker", "reranked_top_k")),
            "prompt_lang": cfg.get("system_prompts_lang"),
        }
    except Exception:
        return {}


async def mlflow_tracked_invoke(
    graph,
    input_msgs,
    config: dict,
    tags: Optional[dict] = None,
) -> dict:
    """
    Run a LangGraph graph and log inputs/outputs/metrics to MLflow explicitly.

    Logged per run:
      params  — question, answer, llm_model, retriever/reranker config, session/tenant
      metrics — input_tokens, output_tokens, total_tokens, latency_s, answer_chars

    Falls back to plain ainvoke when MLflow is not configured.
    Returns the final graph state dict (same shape as ainvoke).

    input_msgs is normally {"messages": [...]}, but can also be a LangGraph
    `Command` (e.g. Command(resume=...) used to resume a HumanInTheLoopMiddleware
    interrupt) — in that case there is no "messages" key to log a question from.
    """
    if not get_mlflow():
        return await graph.ainvoke(input_msgs, config=config)

    import time

    from langchain_core.messages import AIMessage as _AIMessage
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = client.create_run(
        experiment_id=_mlflow_experiment_id,
        run_name="chat",
        tags={k: str(v) for k, v in (tags or {}).items()},
    )
    run_id = run.info.run_id

    messages = input_msgs.get("messages", []) if isinstance(input_msgs, dict) else []
    question = next(
        (m.get("content", "") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )

    params = {"question": question[:500], "streaming": "false"}
    params.update(_mlflow_pipeline_params())
    params.update({k: str(v)[:500] for k, v in (tags or {}).items()})
    for k, v in params.items():
        client.log_param(run_id, k, str(v))

    t0 = time.time()
    try:
        result = await graph.ainvoke(input_msgs, config=config)

        for m in reversed(result.get("messages", [])):
            if isinstance(m, _AIMessage) and not m.tool_calls:
                content = m.content
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                client.log_param(run_id, "answer", content[:500])
                usage = getattr(m, "usage_metadata", None) or {}
                metrics: dict = {
                    "latency_s": round(time.time() - t0, 2),
                    "answer_chars": len(content),
                }
                if usage:
                    metrics["input_tokens"] = usage.get("input_tokens", 0)
                    metrics["output_tokens"] = usage.get("output_tokens", 0)
                    metrics["total_tokens"] = usage.get("total_tokens", 0)
                for k, v in metrics.items():
                    client.log_metric(run_id, k, v)
                break

        client.set_terminated(run_id, "FINISHED")
    except Exception:
        client.log_metric(run_id, "latency_s", round(time.time() - t0, 2))
        client.set_terminated(run_id, "FAILED")
        raise

    return result
