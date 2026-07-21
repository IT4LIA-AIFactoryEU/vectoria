import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vectoria_lib.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from vectoria_lib.api.routers import agent as agent_router
from vectoria_lib.api.routers import documents as documents_router
from vectoria_lib.api.routers import eval as eval_router
from vectoria_lib.api.routers import openai_compat as openai_compat_router
from vectoria_lib.api.routers import upload_ui as upload_ui_router
from vectoria_lib.api.routers.documents import IndexStatus

logger = logging.getLogger("api")


def _cleanup_stale_index_dirs(index_dir: Path) -> None:
    """Remove tmp/old index directories left by crashes or cancelled builds.
    Only removes directories older than 5 minutes to avoid deleting active builds."""
    if not index_dir.exists():
        return
    cutoff = time.time() - 5 * 60
    # Use glob (non-recursive): tmp/old dirs are always direct children of index_dir.
    # rglob would descend into Qdrant's internal storage (thousands of files on Lustre)
    # causing very slow startup on HPC parallel filesystems.
    for stale in index_dir.glob("*_tmp_*"):
        if stale.is_dir() and stale.stat().st_mtime < cutoff:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("Removed stale tmp index dir at startup: %s", stale)
    for stale in index_dir.glob("*_old_*"):
        if stale.is_dir() and stale.stat().st_mtime < cutoff:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("Removed stale old index dir at startup: %s", stale)


async def _auto_load_indexes(app, index_dir: Path, explicit_index_path: Optional[str]):
    """
    Load agent(s) at startup.

    Priority:
    1. If INDEX_PATH is explicitly set, load it as the 'default' tenant.
    2. Otherwise, scan <index_dir>/<tenant_id>/ for existing Chroma or Qdrant indexes
       and load each one automatically.
    """
    from vectoria_lib.llm.agents.agent_builder import AgentBuilder

    registry = app.state.tenant_registry

    def _load(tenant_id: str, path: Path):
        from vectoria_lib.api.routers.documents import _make_document_tools

        try:
            upload_dir = app.state.upload_dir / tenant_id
            index_dir = app.state.index_dir / tenant_id
            extra_tools = _make_document_tools(app, upload_dir, index_dir, tenant_id)
            agent = AgentBuilder.build_qa_agent(
                index_path=str(path),
                tenant_id=tenant_id,
                extra_tools=extra_tools,
                upload_url=app.state.upload_url,
            )
            registry.set_agent(tenant_id, agent)
            registry.set_status(tenant_id, IndexStatus.READY)
            logger.info("Agent ready (tenant=%s, index=%s)", tenant_id, path)
        except Exception as exc:
            logger.error(
                "Could not load index for tenant '%s' at %s: %s",
                tenant_id,
                path,
                exc,
                exc_info=True,
            )
            registry.set_status(tenant_id, IndexStatus.ERROR)

    if explicit_index_path:
        import asyncio

        await asyncio.to_thread(_load, "default", Path(explicit_index_path))
        return

    if not index_dir.exists():
        logger.info("No index directory found — upload a document via POST /documents/ingest")
        return

    import asyncio

    from vectoria_lib.rag.vector_store.discovery import find_index_path

    tasks = [
        (tenant_dir.name, index_path)
        for tenant_dir in sorted(index_dir.iterdir())
        if tenant_dir.is_dir()
        for index_path in [find_index_path(tenant_dir)]
        if index_path
    ]

    if tasks:
        await asyncio.gather(*[asyncio.to_thread(_load, tenant_id, path) for tenant_id, path in tasks])
    else:
        logger.info("No existing indexes found — upload a document via POST /documents/ingest")


def create_app(
    index_path: Optional[str] = None,
    index_dir: str = "/data/index",
    upload_dir: str = "/data/uploads",
    upload_url: str = "http://localhost:8080/upload",
) -> FastAPI:
    """
    Factory that creates and configures the FastAPI application.

    Parameters
    ----------
    index_path : pre-built index to load at startup (optional).
                 When omitted the server starts without an agent;
                 use POST /documents/ingest to build one at runtime.
    index_dir  : base directory where runtime-built indexes are saved.
                 Each tenant gets a subdirectory: <index_dir>/<tenant_id>/
    upload_dir : base directory for uploaded documents.
                 Each tenant gets a subdirectory: <upload_dir>/<tenant_id>/
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import asyncio
        import threading

        from vectoria_lib.api.tenant_registry import TenantRegistry
        from vectoria_lib.core.config import Config

        app.state.index_dir = Path(index_dir)
        app.state.upload_dir = Path(upload_dir)
        app.state.upload_url = upload_url
        app.state.tenant_registry = TenantRegistry()

        _cleanup_stale_index_dirs(Path(index_dir))

        llama_manager = None
        try:
            llama_cfg = Config().config.get("llama_server")
            if llama_cfg and llama_cfg.get("enabled"):
                from vectoria_lib.llm.llama_server_manager import LlamaServerManager

                llama_manager = LlamaServerManager(llama_cfg)
                await asyncio.to_thread(llama_manager.start)
        except Exception as exc:
            logger.error("llama-server failed to start: %s", exc)
            logger.warning("Continuing without llama-server — requests will fail until an LLM is reachable")

        logger.info("Starting background index loader")

        def _bg_load():
            asyncio.run(_auto_load_indexes(app, Path(index_dir), explicit_index_path=index_path))

        _load_thread = threading.Thread(target=_bg_load, daemon=True, name="vectoria-index-loader")
        _load_thread.start()

        logger.info("Vectoria API ready")
        yield

        if llama_manager:
            llama_manager.stop()
        logger.info("Shutting down Vectoria API")

    app = FastAPI(
        title="Vectoria API",
        description="REST interface for the Vectoria RAG pipeline",
        version="1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # CORS: opt-in via VECTORIA_ALLOWED_ORIGINS (comma-separated list of origins).
    # Example: VECTORIA_ALLOWED_ORIGINS=http://localhost:8080,https://myapp.example.com
    # Leave unset to disable CORS headers (default — browsers block cross-origin requests).
    _allowed_origins = [o.strip() for o in os.environ.get("VECTORIA_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    if "*" in _allowed_origins:
        # allow_credentials=True below makes Starlette reflect the request's actual
        # Origin header instead of sending a literal "*" — i.e. "any origin, with credentials".
        raise RuntimeError(
            "VECTORIA_ALLOWED_ORIGINS cannot contain '*': combined with allow_credentials=True "
            "this allows any origin to make authenticated requests. List explicit origins instead."
        )
    if _allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["*"],
        )
        logger.info("CORS enabled for origins: %s", _allowed_origins)

    app.include_router(agent_router.router)
    app.include_router(documents_router.router)
    app.include_router(eval_router.router)
    app.include_router(openai_compat_router.router)
    app.include_router(upload_ui_router.router)

    return app
