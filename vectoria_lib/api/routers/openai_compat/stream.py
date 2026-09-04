#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import html
import json
import logging
import re
import traceback
from typing import Optional

from .helpers import strip_special_tokens
from .interrupts import format_confirmation_prompt

logger = logging.getLogger("api")

# OpenWebUI's rendered result size for a tool call detail block; longer results
# get truncated to avoid the ~65k HTML-attribute limit in OpenWebUI's markdown parser.
_TOOL_RESULT_MAX_CHARS = 4000


def _json_attr(value) -> str:
  """Encode a value as OpenWebUI expects for a <details> tag attribute:
  JSON-encoded, then HTML-escaped (its frontend does the reverse: html-unescape,
  then JSON.parse — see ToolCallDisplay.svelte)."""
  return html.escape(json.dumps(value, ensure_ascii=False, default=str))


def _tool_call_details(run_id: str, name: str, args, result) -> str:
  """Render a self-contained-agent tool call as OpenWebUI's native
  <details type="tool_calls"> block — the format OpenWebUI's frontend renders
  as a collapsible "tool executed" card, independent of the <think> block."""
  result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
  if len(result_str) > _TOOL_RESULT_MAX_CHARS:
    result_str = result_str[:_TOOL_RESULT_MAX_CHARS] + "… [troncato]"
  return (
    f'<details type="tool_calls" done="true" id="{run_id}" name="{name}" '
    f'arguments="{_json_attr(args)}" result="{_json_attr(result_str)}">\n'
    f"<summary>Tool eseguito: {name}</summary>\n</details>\n"
  )


async def stream_sse(
  *,
  agent,
  graph_input: dict,
  graph_config: dict,
  completion_id: str,
  created: int,
  model_name: str,
  question: str,
  thread_id: str,
  tenant_id: str,
  mlflow_enabled: bool,
):
  import time as _time
  from vectoria_lib.llm.callbacks import _mlflow_experiment_id, _mlflow_pipeline_params

  collected_tokens: list[str] = []
  mlf_client: Optional[object] = None
  mlf_run_id: Optional[str] = None
  _answer_done = False
  t0 = _time.time()

  if mlflow_enabled:
    try:
      from mlflow.tracking import MlflowClient as _MlflowClient
      mlf_client = _MlflowClient()
      mlf_run = mlf_client.create_run(
        experiment_id=_mlflow_experiment_id,
        run_name="chat",
        tags={"session_id": thread_id, "tenant_id": tenant_id, "model": model_name},
      )
      mlf_run_id = mlf_run.info.run_id
      params = {"question": question[:500], "session_id": thread_id, "tenant_id": tenant_id, "streaming": "true"}
      params.update(_mlflow_pipeline_params())
      for k, v in params.items():
        mlf_client.log_param(mlf_run_id, k, str(v))
      logger.debug("MLflow run created: %s", mlf_run_id)
    except Exception:
      logger.warning("MLflow: failed to create run", exc_info=True)

  num_tool_calls = 0
  _think_started = False   # True when think block opened (either path)
  _think_closed = False    # True when think block properly closed
  _think_via_rc = False    # True when think was opened via reasoning_content (Path 2)
  _pending_tool_calls: dict[str, dict] = {}  # run_id -> {"name":..., "args":...}

  def _sse(content: str) -> str:
    return f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

  try:
    async for event in agent.graph.astream_events(graph_input, config=graph_config, version="v2"):
      event_type = event["event"]

      if event_type == "on_tool_start":
        num_tool_calls += 1
        run_id = str(event.get("run_id", ""))
        _pending_tool_calls[run_id] = {
          "name": event.get("name", "tool"),
          "args": (event.get("data") or {}).get("input"),
        }
        # A tool call means the model finished this reasoning turn — close the
        # think block now so the tool card renders as a sibling, not nested inside.
        if _think_started and not _think_closed:
          yield _sse("</think>\n")
          _think_closed = True
        continue

      if event_type == "on_tool_end":
        run_id = str(event.get("run_id", ""))
        info = _pending_tool_calls.pop(run_id, None)
        if info:
          output = (event.get("data") or {}).get("output")
          result = getattr(output, "content", output)
          yield _sse(_tool_call_details(run_id, info["name"], info["args"], result))
        # Next model turn (post-tool) opens its own fresh think block.
        _think_started = False
        _think_closed = False
        _think_via_rc = False
        continue

      if event_type == "on_chat_model_end":
        node = (event.get("metadata") or {}).get("langgraph_node", "")
        if node and node != "model":
          continue
        output = (event.get("data") or {}).get("output")
        # Some models (e.g. Qwen3 via vLLM) put tool calls in additional_kwargs
        # instead of the top-level tool_calls field — check both.
        _ak = (getattr(output, "additional_kwargs", {}) or {}) if output else {}
        has_tool_calls = bool(
          (getattr(output, "tool_calls", None) or _ak.get("tool_calls"))
          if output else None
        )
        _out_content = getattr(output, "content", "") or "" if output else ""
        if isinstance(_out_content, list):
          _out_content = "".join(p.get("text", "") for p in _out_content if isinstance(p, dict))
        logger.info(
          "on_chat_model_end: node=%r has_tool_calls=%s think_started=%s think_closed=%s "
          "collected=%d content_len=%d reasoning_len=%d",
          node, has_tool_calls, _think_started, _think_closed, len(collected_tokens),
          len(_out_content), len(_ak.get("reasoning_content", "") or ""),
        )
        if not has_tool_calls:
          if _think_closed:
            _answer_done = True
          elif output and not collected_tokens:
            # Thinking models sometimes deliver the entire turn non-streamed:
            # emit reasoning first (inside <think>), then the final answer.
            if not _think_started:
              reasoning_out = _ak.get("reasoning_content", "")
              if reasoning_out:
                yield _sse("<think>\n")
                yield _sse(reasoning_out)
                _think_started = True
            content = getattr(output, "content", "") or ""
            if isinstance(content, list):
              content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            content = strip_special_tokens(content)
            if _think_started:
              # Reasoning already streamed; output.content may duplicate it inside
              # <think>...</think> — strip those blocks to avoid double output.
              content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
              if _think_started and not _think_closed:
                yield _sse("</think>\n")
                _think_closed = True
              collected_tokens.append(content)
              yield _sse(content)
              _answer_done = True
            elif _think_started and not _think_closed:
              # Model generated only thinking (content empty) — max_tokens likely exhausted
              # during the thinking phase. Close the block and surface a warning.
              logger.warning(
                "on_chat_model_end: thinking-only response (content empty after think). "
                "Check max_tokens config — model may have run out of tokens during thinking."
              )
              msg = "⚠️ Il modello ha esaurito i token durante la fase di ragionamento e non ha prodotto una risposta. Prova a riformulare la domanda in modo più conciso."
              yield _sse("</think>\n")
              _think_closed = True
              collected_tokens.append(msg)
              yield _sse(msg)
              _answer_done = True
        continue

      if event_type != "on_chat_model_stream":
        continue
      if _answer_done:
        continue

      node = (event.get("metadata") or {}).get("langgraph_node", "")
      if node and node != "model":
        continue
      chunk_msg = (event.get("data") or {}).get("chunk")
      if not chunk_msg:
        continue
      reasoning = (getattr(chunk_msg, "additional_kwargs", {}) or {}).get("reasoning_content", "")
      if reasoning:
        if not _think_started:
          logger.info("stream: first reasoning_content chunk, emitting <think>")
          _think_started = True
          _think_via_rc = True
          yield _sse("<think>\n")
        yield _sse(reasoning)
        continue
      raw = chunk_msg.content
      if isinstance(raw, list):
        raw = "".join(p.get("text", "") for p in raw if isinstance(p, dict))
      raw = strip_special_tokens(raw)
      if raw == "":
        continue
      # Path 1: model embeds <think> tags directly in content (not via reasoning_content).
      # Just track the open/close so finally and on_chat_model_end behave correctly.
      # Do NOT inject our own </think> — the model's inline tags go to SSE as-is.
      if not _think_started and "<think>" in raw:
        _think_started = True
      if _think_started and not _think_closed and "</think>" in raw:
        _think_closed = True
      # Path 2: reasoning_content was streamed (no inline <think> tag seen yet).
      # The first regular content chunk signals end of thinking → close the block.
      if _think_via_rc and not _think_closed:
        _think_closed = True
        yield _sse("</think>\n")
      collected_tokens.append(raw)
      yield _sse(raw)

    # HumanInTheLoopMiddleware pauses the graph via interrupt() before running
    # a guarded tool (e.g. delete_document) — astream_events simply stops with
    # no further model tokens. Surface the pending confirmation as the answer.
    if not collected_tokens:
      state = await agent.graph.aget_state(graph_config)
      if state.interrupts:
        confirmation = format_confirmation_prompt(state.interrupts[0])
        if _think_started and not _think_closed:
          _think_closed = True
          yield _sse("</think>\n")
        collected_tokens.append(confirmation)
        yield _sse(confirmation)

  except Exception as exc:
    logger.error("Streaming error in /v1/chat/completions: %s", traceback.format_exc())
    if "recursion_limit" in str(exc).lower() or "recursion limit" in str(exc).lower():
      msg = "Ho eseguito troppi passi senza trovare una risposta. Prova a riformulare la domanda in modo più specifico."
      if _think_started and not _think_closed:
        _think_closed = True
        yield _sse("</think>\n")
      collected_tokens.append(msg)
      yield _sse(msg)
    else:
      yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'stream_error'}})}\n\n"
  finally:
    if _think_started and not _think_closed:
      try:
        yield _sse("</think>\n")
      except Exception:
        pass
    answer = "".join(collected_tokens)
    if not answer:
      logger.warning(
        "Empty response after streaming (num_tool_calls=%d, check if model generated only special tokens or nothing)",
        num_tool_calls,
      )
    if mlf_run_id and mlf_client:
      try:
        mlf_client.log_param(mlf_run_id, "answer", answer[:500])
        mlf_client.log_metric(mlf_run_id, "latency_s", round(_time.time() - t0, 2))
        mlf_client.log_metric(mlf_run_id, "answer_chars", len(answer))
        mlf_client.log_metric(mlf_run_id, "num_tool_calls", num_tool_calls)
        mlf_client.set_terminated(mlf_run_id, "FINISHED")
        logger.debug("MLflow run finished: %s", mlf_run_id)
      except Exception:
        logger.warning("MLflow: failed to finalize run %s", mlf_run_id, exc_info=True)

  stop = {
    "id": completion_id,
    "object": "chat.completion.chunk",
    "created": created,
    "model": model_name,
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
  }
  yield f"data: {json.dumps(stop)}\n\n"
  yield "data: [DONE]\n\n"
