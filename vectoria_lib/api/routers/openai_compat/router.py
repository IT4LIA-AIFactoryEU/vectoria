#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
#
# Thread continuity: the thread_id is resolved in order of priority:
#   1. X-Thread-ID request header (explicit override)
#   2. chat_id field in the request body (sent by OpenWebUI per chat session)
#   3. X-Chat-Id request header (alternative OpenWebUI header)
#   4. hash of the first message only, combined with tenant_id (fallback).
#      Only the first message is hashed — not the whole history — because it's
#      the only message guaranteed to be at the same position in every turn's
#      payload for a growing-history conversation (turn 1 has just one
#      message; later turns prepend it plus everything since). Hashing more
#      would make the derived thread_id change every turn, since only the
#      last message is ever forwarded to the graph (next line) and continuity
#      depends entirely on later turns re-deriving the SAME thread_id turn
#      after turn. The trade-off: two different conversations on the same
#      tenant that happen to open with an identical first message, and both
#      skip session headers, will collide onto one thread. Accepted as a
#      narrow, low-stakes risk — the alternative (hashing more of the
#      history) breaks continuity for every affected client on every turn,
#      which is strictly worse.
# Only the latest message is forwarded to the graph; conversation history is
# maintained by LangGraph's SQLite checkpoint keyed on thread_id.
#
import asyncio
import hashlib
import json
import logging
import re
import time
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage

from vectoria_lib.api.deps import _check_token, _get_tenant_id
from vectoria_lib.core.constants import ALLOWED_EXTENSIONS as _ALLOWED_EXTENSIONS
from vectoria_lib.llm.callbacks import build_callbacks, get_mlflow, mlflow_tracked_invoke

from .helpers import strip_special_tokens
from .image import normalize_images
from .interrupts import (
  build_resume_command,
  format_confirmation_prompt,
  get_pending_interrupt,
  parse_decision,
)
from .models import ChatCompletionRequest, ChatMessage
from .stream import stream_sse

logger = logging.getLogger("api")

router = APIRouter(prefix="/v1", tags=["openai-compat"])

_TENANT_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")


def _derive_thread_id(tenant_id: str, messages: list[ChatMessage]) -> str:
  first = messages[0]
  key = tenant_id + "|" + json.dumps(
    {"role": first.role, "content": first.text()}, ensure_ascii=False, sort_keys=True
  )
  return hashlib.sha256(key.encode()).hexdigest()[:32]


def _extract_answer(result: dict) -> tuple[str, dict]:
  usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
  for m in reversed(result["messages"]):
    if isinstance(m, AIMessage) and not m.tool_calls:
      content = m.content
      if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
      content = strip_special_tokens(content).strip()
      meta = getattr(m, "usage_metadata", None) or {}
      if meta:
        usage["prompt_tokens"] = meta.get("input_tokens", 0)
        usage["completion_tokens"] = meta.get("output_tokens", 0)
        usage["total_tokens"] = meta.get("total_tokens", 0)
      return content, usage
  return "", usage


def _tenant_has_documents(upload_dir, tenant_id: str) -> bool:
  tenant_upload = upload_dir / tenant_id
  if not tenant_upload.exists():
    return False
  return any(f for f in tenant_upload.iterdir() if f.is_file() and f.suffix.lower() in _ALLOWED_EXTENSIONS)


def _tenant_from_model(model_name: str) -> Optional[str]:
  if model_name and model_name.startswith("vectoria-"):
    candidate = model_name[len("vectoria-"):]
    if candidate and _TENANT_ID_RE.match(candidate):
      return candidate
  return None


async def _locked_stream(lock, inner):
  """Wrap an SSE generator so the per-thread lock (already acquired by the
  caller) is released only once the stream actually finishes — not when
  chat_completions() returns the StreamingResponse object, which happens
  long before the client finishes consuming the stream."""
  try:
    async for chunk in inner:
      yield chunk
  finally:
    lock.release()


@router.get("/models", dependencies=[Depends(_check_token)])
async def list_models(request: Request):
  registry = request.app.state.tenant_registry
  upload_dir = request.app.state.upload_dir
  models = []
  for tenant_id, agent in registry.agents_snapshot.items():
    if agent is not None and _tenant_has_documents(upload_dir, tenant_id):
      models.append({"id": f"vectoria-{tenant_id}", "object": "model", "created": 0, "owned_by": "vectoria"})
  if not models:
    models.append({"id": "vectoria", "object": "model", "created": 0, "owned_by": "vectoria"})
  return {"object": "list", "data": models}


@router.post("/chat/completions", dependencies=[Depends(_check_token)])
async def chat_completions(
  body: ChatCompletionRequest,
  request: Request,
  tenant_id: str = Depends(_get_tenant_id),
):
  registry = request.app.state.tenant_registry

  model_tenant = _tenant_from_model(body.model)
  if model_tenant:
    tenant_id = model_tenant

  agent = registry.get_agent(tenant_id)

  if agent is None:
    status = registry.get_status(tenant_id)
    if status == "building":
      answer = "Indicizzazione in corso... attendere il completamento prima di inviare messaggi."
    else:
      upload_url = getattr(request.app.state, "upload_url", "http://localhost:8080/upload")
      answer = (
        "Nessun documento caricato. Per iniziare, apri la pagina di caricamento e trascina i tuoi file PDF o DOCX:\n\n"
        f"👉 **{upload_url}**\n\n"
        "Una volta completata l'indicizzazione, torna qui e fai le tue domande."
      )
    return {
      "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
      "object": "chat.completion",
      "created": int(time.time()),
      "model": body.model or "vectoria",
      "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

  if not body.messages:
    return JSONResponse(
      status_code=400,
      content={"error": {"message": "messages array is empty", "type": "invalid_request_error", "code": "invalid_request"}},
    )

  thread_id = (
    request.headers.get("X-Thread-ID")
    or body.chat_id
    or request.headers.get("X-Chat-Id")
    or _derive_thread_id(tenant_id, body.messages)
  )
  logger.debug("thread_id=%s (chat_id=%s)", thread_id, body.chat_id)

  content = body.messages[-1].content
  if isinstance(content, list):
    try:
      content = normalize_images(content)
    except ValueError as exc:
      return JSONResponse(
        status_code=400,
        content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "unsupported_image_format"}},
      )

  completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
  model_name = body.model or "vectoria"
  question = body.messages[-1].text()
  created = int(time.time())
  graph_input = {"messages": [{"role": body.messages[-1].role, "content": content}]}
  graph_config = {"configurable": {"thread_id": thread_id}, "callbacks": build_callbacks(), "recursion_limit": 50}

  # Serializes this thread's turns: two concurrent requests on the same
  # thread_id (double-click send, two browser tabs) must not both read the
  # same parent checkpoint and both write a new one. Held for the whole turn,
  # including the entire streaming response duration via _locked_stream —
  # NOT just until this function returns.
  lock = registry.get_thread_lock(tenant_id, thread_id)
  await lock.acquire()
  release_deferred = False
  try:
    # HumanInTheLoopMiddleware may have paused this thread waiting for approval
    # (e.g. before delete_document). If so, this message is the user's decision,
    # not a new question — resolve it instead of asking the agent anything.
    pending_interrupt = await get_pending_interrupt(agent.graph, graph_config)
    if pending_interrupt is not None:
      decision = parse_decision(question)
      if decision is None:
        # Fail-safe: an unrecognized reply implicitly rejects the pending
        # action (never execute a destructive tool on ambiguous input).
        # Resolve that rejection now, then fall through to answer the user's
        # actual message normally — graph_input already holds it (set above,
        # before this interrupt check), so no further action needed here.
        await asyncio.wait_for(
          mlflow_tracked_invoke(
            agent.graph,
            build_resume_command(pending_interrupt, "reject"),
            config=graph_config,
            tags={"session_id": thread_id, "tenant_id": tenant_id, "model": model_name},
          ),
          timeout=600,
        )
      else:
        graph_input = build_resume_command(pending_interrupt, decision)

    if body.stream:
      release_deferred = True
      return StreamingResponse(
        _locked_stream(
          lock,
          stream_sse(
            agent=agent,
            graph_input=graph_input,
            graph_config=graph_config,
            completion_id=completion_id,
            created=created,
            model_name=model_name,
            question=question,
            thread_id=thread_id,
            tenant_id=tenant_id,
            mlflow_enabled=get_mlflow(),
          ),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
      )

    try:
      result = await asyncio.wait_for(
        mlflow_tracked_invoke(
          agent.graph,
          graph_input,
          config=graph_config,
          tags={"session_id": thread_id, "tenant_id": tenant_id, "model": model_name},
        ),
        timeout=600,
      )
      new_interrupts = result.get("__interrupt__")
      if new_interrupts:
        answer = format_confirmation_prompt(new_interrupts[0])
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
      else:
        answer, usage = _extract_answer(result)
        if not answer:
          logger.warning("Empty response in non-streaming mode")
    except Exception as exc:
      logger.error("Error in /v1/chat/completions: %s", traceback.format_exc())
      return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_server_error", "code": "internal_error"}},
      )

    return {
      "id": completion_id,
      "object": "chat.completion",
      "created": created,
      "model": model_name,
      "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
      "usage": usage,
    }
  finally:
    if not release_deferred:
      lock.release()
