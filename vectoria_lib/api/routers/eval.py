import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from vectoria_lib.api.deps import _check_token

logger = logging.getLogger("api")

router = APIRouter(prefix="/eval", tags=["eval"])


class EvalQueryRequest(BaseModel):
    question: str
    tenant_id: str = "default"


class EvalChunksRequest(BaseModel):
    tenant_id: str = "default"
    limit: Optional[int] = None


@router.post("/query", dependencies=[Depends(_check_token)])
async def eval_query(body: EvalQueryRequest, request: Request):
    """
    Evaluation endpoint used by evaluationrag.
    Runs the full RAG pipeline and returns both the answer and the retrieved contexts.
    """
    registry = request.app.state.tenant_registry
    tenant_id = body.tenant_id.strip().lower() or "default"
    agent = registry.get_agent(tenant_id)

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not ready for tenant '{tenant_id}'. Upload a document first.",
        )

    import asyncio

    result = await asyncio.to_thread(agent.ask, body.question)

    answer = result.get("answer", "")
    raw_docs = result.get("retrieved_docs", [])
    contexts = [doc.page_content for doc in raw_docs]

    return {
        "question": body.question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": None,
    }


@router.post("/chunks", dependencies=[Depends(_check_token)])
async def eval_chunks(body: EvalChunksRequest, request: Request):
    registry = request.app.state.tenant_registry
    tenant_id = body.tenant_id.strip().lower() or "default"
    agent = registry.get_agent(tenant_id)

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not ready for tenant '{tenant_id}'. Upload a document first.",
        )

    retriever_tool = agent.retriever_tool
    if retriever_tool is None or retriever_tool.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not available for this tenant.")

    vector_store = retriever_tool.retriever.wrapped_vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not available for this tenant.")

    import asyncio

    all_docs = await asyncio.to_thread(vector_store.get_all_documents)

    if body.limit is not None:
        all_docs = all_docs[: body.limit]

    return {
        "tenant_id": tenant_id,
        "count": len(all_docs),
        "chunks": [{"page_content": d.page_content, "metadata": d.metadata} for d in all_docs],
    }
