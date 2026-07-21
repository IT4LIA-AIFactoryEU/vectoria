import asyncio
import json
import logging
import shutil
import threading
import uuid
from enum import Enum
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from vectoria_lib.api.deps import _check_token, _get_tenant_id
from vectoria_lib.core.constants import ALLOWED_EXTENSIONS
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME
from vectoria_lib.rag.vector_store.discovery import has_index as _has_vector_index

logger = logging.getLogger("api")

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # bytes
MAX_FILES_PER_REQUEST = 20
_UPLOAD_CHUNK = 1 * 1024 * 1024  # 1 MB read chunks


class IndexStatus(str, Enum):
    NOT_BUILT = "not_built"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"
    CANCELLED = "cancelled"


@router.get("/status", dependencies=[Depends(_check_token)])
async def status(
    request: Request,
    tenant_id: str = Depends(_get_tenant_id),
):
    """Returns the current indexing status and whether the agent is ready."""
    registry = request.app.state.tenant_registry
    progress = registry.get_progress(tenant_id)
    index_dir: Path = request.app.state.index_dir / tenant_id
    wiki_pending = 0
    try:
        from vectoria_lib.wiki.wiki_manager import WikiManager

        wiki_pending = WikiManager.from_config(index_dir / "wiki").pending_count()
    except Exception:
        pass
    return {
        "tenant_id": tenant_id,
        "status": registry.get_status(tenant_id),
        "agent_ready": registry.get_agent(tenant_id) is not None,
        "progress": progress,
        "wiki_pending_concepts": wiki_pending,
        "wiki": registry.get_wiki_state(tenant_id),
    }


@router.post("/cancel", dependencies=[Depends(_check_token)])
async def cancel_indexing(
    request: Request,
    tenant_id: str = Depends(_get_tenant_id),
):
    """Cancel an in-progress indexing task for the given tenant."""
    registry = request.app.state.tenant_registry
    task = registry.get_task(tenant_id)
    if task and not task.done():
        task.cancel()
        # Set status immediately — don't wait for the except block in the task
        registry.set_status(tenant_id, IndexStatus.CANCELLED)
        registry.set_progress(tenant_id, 0, "Annullamento in corso...")
        return {"detail": "Indicizzazione annullata.", "tenant_id": tenant_id}
    return {"detail": "Nessuna indicizzazione in corso.", "tenant_id": tenant_id}


@router.post("/ingest", dependencies=[Depends(_check_token)])
async def ingest(
    files: list[UploadFile],
    request: Request,
    tenant_id: str = Depends(_get_tenant_id),
):
    """
    Upload one or more PDF/DOCX files, build the index, and reload the agent.

    The operation runs in the background. Poll GET /documents/status to
    track progress. The /chat endpoint returns 503 while indexing is running.

    Pass X-Tenant-ID header to isolate documents per company.
    """
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: max {MAX_FILES_PER_REQUEST} per request.",
        )

    for file in files:
        safe_name = Path(file.filename).name  # strip any directory component
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
            )

    registry = request.app.state.tenant_registry

    if registry.get_status(tenant_id) == IndexStatus.BUILDING:
        raise HTTPException(
            status_code=409,
            detail=f"Indexing already in progress for tenant '{tenant_id}'.",
        )

    # After cancellation the asyncio task is gone but the underlying worker
    # thread may still be writing to index_dir.  Block until it finishes to
    # avoid concurrent writes to the same ChromaDB database.
    if registry.is_thread_running(tenant_id):
        raise HTTPException(
            status_code=409,
            detail="Annullamento ancora in corso. Riprova tra qualche secondo.",
        )

    # Save all uploaded files to the tenant-scoped staging directory
    upload_dir: Path = request.app.state.upload_dir / tenant_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for file in files:
        safe_name = Path(file.filename).name
        dest = upload_dir / safe_name
        size = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(_UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"'{safe_name}' exceeds maximum size of {MAX_FILE_SIZE_MB} MB.",
                        )
                    out.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            for already_saved in saved_files:
                already_saved.unlink(missing_ok=True)
            raise
        saved_files.append(dest)
        logger.info("Saved uploaded file to %s (tenant=%s)", dest, tenant_id)

    # Tenant-scoped index directory
    index_dir: Path = request.app.state.index_dir / tenant_id

    # Launch background indexing (non-blocking) — store task for cancellation
    task = asyncio.create_task(_build_and_reload(request.app, saved_files, index_dir, tenant_id))
    registry.set_task(tenant_id, task)

    filenames = [f.filename for f in files]
    return {
        "detail": f"Indexing started for {filenames}.",
        "tenant_id": tenant_id,
        "poll": "/documents/status",
    }


def _update_documents_meta(
    upload_dir: Path,
    new_docs=None,
    deleted_filename: str = None,
) -> None:
    """Incrementally update documents_meta.json — add summaries for new docs and/or
    remove the entry for a deleted document. No vector store query needed."""
    meta_path = upload_dir / "documents_meta.json"

    meta = {}
    if meta_path.exists():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    if new_docs:
        for doc in new_docs:
            filename = doc.metadata.get(DOC_FILE_NAME) or doc.metadata.get("source", "")
            if not filename or filename in meta:
                continue
            snippet = doc.page_content.replace("\n", " ").strip()
            meta[filename] = snippet[:200] + ("..." if len(snippet) > 200 else "")

    if deleted_filename and deleted_filename in meta:
        del meta[deleted_filename]

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.debug("Updated documents metadata at %s", meta_path)


def _atomic_swap_index(index_dir: Path, tmp_index_dir: Path) -> Path | None:
    """Rename tmp_index_dir into index_dir's place. Restores the previous
    index_dir on failure. Returns the renamed-away previous index_dir (still
    on disk, not yet deleted) so the caller can carry files over from it
    before cleaning it up, or None if there was no previous index."""
    old_index_dir = index_dir.parent / f"{index_dir.name}_old_{uuid.uuid4().hex[:8]}"
    moved_old = False
    if index_dir.exists():
        index_dir.rename(old_index_dir)
        moved_old = True
    try:
        tmp_index_dir.rename(index_dir)
    except Exception:
        if moved_old:
            old_index_dir.rename(index_dir)  # restore previous index
        raise
    return old_index_dir if moved_old else None


def _carry_over_subdirs(src_root: Path, dst_root: Path, names=("wiki", "graph")):
    """Copy wiki/graph dirs across the atomic swap when the destination doesn't already have them."""
    for name in names:
        src, dst = src_root / name, dst_root / name
        if src.exists() and not dst.exists():
            shutil.copytree(str(src), str(dst))


def _rebuild_sync(
    app,
    upload_dir: Path,
    index_dir: Path,
    tenant_id: str,
    deleted_filename: str,
    files: list[Path],
):
    """Synchronous rebuild used by DeleteDocumentTool (runs inside a thread).

    Uses incremental delete for Chroma; falls back to full rebuild for FAISS.
    Always performs an atomic swap so index_dir is never left in a partial state.
    """
    from vectoria_lib.cli.build_index import (
        build_index_from_files,
        delete_document_from_index,
    )
    from vectoria_lib.llm.agents.agent_builder import AgentBuilder
    from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder

    registry = app.state.tenant_registry
    registry.set_status(tenant_id, IndexStatus.BUILDING)
    registry.set_agent(tenant_id, None)

    if not files:
        for d in (index_dir, upload_dir):
            if d and d.exists():
                shutil.rmtree(d, ignore_errors=True)
                logger.info("Removed directory for tenant '%s': %s", tenant_id, d)
        registry.remove_tenant(tenant_id)
        return

    tmp_index_dir = index_dir.parent / f"{index_dir.name}_tmp_{uuid.uuid4().hex[:8]}"

    try:
        from vectoria_lib.core.config import Config

        vs_name = Config().get("vector_store", "name")
    except Exception:
        vs_name = None

    if deleted_filename and vs_name in ("chroma", "qdrant") and index_dir.exists():
        tmp_index_path, _ = delete_document_from_index(deleted_filename, index_dir, tmp_index_dir)
    else:
        tmp_index_path, _, _ = build_index_from_files(files, tmp_index_dir)

    _update_documents_meta(upload_dir, deleted_filename=deleted_filename)

    # Carry wiki/graph into tmp before the swap: delete_document_from_index
    # already copies the whole index_dir, build_index_from_files doesn't.
    _carry_over_subdirs(index_dir, tmp_index_dir)

    old_index_dir = _atomic_swap_index(index_dir, tmp_index_dir)
    if old_index_dir and old_index_dir.exists():
        shutil.rmtree(old_index_dir, ignore_errors=True)

    # Remove wiki pages and pending concepts that came from the deleted document.
    if deleted_filename:
        try:
            from vectoria_lib.wiki.wiki_manager import WikiManager

            wm = WikiManager.from_config(index_dir / "wiki")
            n = wm.delete_by_source(deleted_filename)
            if n:
                logger.info("Deleted %d wiki page(s) for document '%s'", n, deleted_filename)
        except Exception as exc:
            logger.warning("Wiki cleanup after delete failed (non-fatal): %s", exc)

        try:
            from vectoria_lib.rag.graph.graph_manager import GraphManager

            gm = GraphManager.from_config(index_dir / "graph")
            n = gm.delete_by_source(deleted_filename)
            if n:
                logger.info("Graph: removed %d node(s) for document '%s'", n, deleted_filename)
        except Exception as exc:
            logger.warning("Graph cleanup after delete failed (non-fatal): %s", exc)

    try:
        index_path = index_dir / tmp_index_path.relative_to(tmp_index_dir)
    except ValueError:
        logger.error("Index path %s is outside tmp dir %s", tmp_index_path, tmp_index_dir)
        raise

    VectorStoreBuilder.evict(index_path)

    extra_tools = _make_document_tools(app, upload_dir, index_dir, tenant_id)
    agent = AgentBuilder.build_qa_agent(
        index_path=index_path,
        tenant_id=tenant_id,
        extra_tools=extra_tools,
        upload_url=getattr(app.state, "upload_url", "http://localhost:8080/upload"),
    )
    registry.set_agent(tenant_id, agent)
    registry.set_status(tenant_id, IndexStatus.READY)
    logger.info("Index rebuilt after deletion — agent reloaded (tenant=%s)", tenant_id)


def _make_document_tools(app, upload_dir: Path, index_dir: Path, tenant_id: str):
    """Create document tools, wiki tools, and graph tools for the given tenant."""
    from vectoria_lib.llm.agents.tools.document_tools import DeleteDocumentTool, ListDocumentsTool
    from vectoria_lib.llm.agents.tools.graph_tools import make_graph_tools
    from vectoria_lib.llm.agents.tools.wiki_tools import make_wiki_tools

    def rebuild_after_delete(deleted_filename: str, remaining_files: list[Path]):
        _rebuild_sync(app, upload_dir, index_dir, tenant_id, deleted_filename, remaining_files)

    upload_url = getattr(app.state, "upload_url", "http://localhost:8080/upload")
    return [
        ListDocumentsTool(upload_dir=upload_dir, upload_url=upload_url),
        DeleteDocumentTool(
            upload_dir=upload_dir,
            index_dir=index_dir,
            rebuild_fn=rebuild_after_delete,
            upload_url=upload_url,
        ),
        *make_wiki_tools(index_dir),
        *make_graph_tools(index_dir),
    ]


def _run_wiki_synthesis_background(index_dir: Path, tenant_id: str, registry) -> None:
    from vectoria_lib.wiki.wiki_synthesis_runner import run_wiki_synthesis

    def _on_complete(status: str, n: int) -> None:
        registry.set_wiki_state(tenant_id, status, n)
        if n and status == "idle":
            logger.info("Background wiki synthesis: %d page(s) created for tenant=%s", n, tenant_id)
        elif status == "failed":
            logger.warning("Background wiki synthesis failed for tenant=%s", tenant_id)

    run_wiki_synthesis(index_dir, on_complete=_on_complete)


async def _build_and_reload(app, uploaded_files: list[Path], index_dir: Path, tenant_id: str):
    """Background task: build the index then reload the agent for the given tenant.

    Uses incremental indexing (add only the new files) when an index already
    exists; falls back to a full rebuild otherwise.
    """
    from vectoria_lib.cli.build_index import add_files_to_existing_index, build_index_from_files
    from vectoria_lib.llm.agents.agent_builder import AgentBuilder
    from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder

    registry = app.state.tenant_registry
    previous_agent = registry.get_agent(tenant_id)  # keep for rollback on cancel/error
    registry.set_status(tenant_id, IndexStatus.BUILDING)
    registry.set_agent(tenant_id, None)
    registry.set_progress(tenant_id, 0, "Avvio indicizzazione...")

    upload_dir = uploaded_files[0].parent
    incremental = _has_vector_index(index_dir)
    if incremental:
        logger.info(
            "Incremental add of %d file(s) to existing index for tenant=%s: %s",
            len(uploaded_files),
            tenant_id,
            [f.name for f in uploaded_files],
        )
    else:
        all_files = [f for f in upload_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS]
        logger.info(
            "Full index build of %d file(s) for tenant=%s: %s", len(all_files), tenant_id, [f.name for f in all_files]
        )

    # Clean up any orphaned tmp dirs left by previous cancelled builds.
    for stale in index_dir.parent.glob(f"{index_dir.name}_tmp_*"):
        shutil.rmtree(stale, ignore_errors=True)
        logger.debug("Removed stale tmp index dir: %s", stale)

    tmp_index_dir = index_dir.parent / f"{index_dir.name}_tmp_{uuid.uuid4().hex[:8]}"

    # threading.Event set when the worker thread finishes (even if asyncio task
    # was cancelled — the thread runs until completion regardless).
    thread_done = registry.new_thread_done_event(tenant_id)

    # threading.Event used to signal the preprocessing pool to stop when the
    # asyncio task is cancelled (see CancelledError handler below).
    stop_event = threading.Event()

    n_files = len(uploaded_files) if incremental else len(all_files)

    def _on_file_done(completed: int, total: int):
        pct = 10 + int((completed / total) * 55)  # 10% → 65% during preprocessing
        registry.set_progress(tenant_id, pct, f"Elaborazione {completed}/{total} documenti...")

    def _build_index_and_signal():
        try:
            if incremental:
                return add_files_to_existing_index(
                    uploaded_files,
                    index_dir,
                    tmp_index_dir,
                    stop_event=stop_event,
                    progress_callback=_on_file_done,
                )
            else:
                return build_index_from_files(
                    all_files,
                    tmp_index_dir,
                    stop_event=stop_event,
                    progress_callback=_on_file_done,
                )
        except InterruptedError:
            return None, None, None  # preprocessing cancelled — no swap needed
        finally:
            thread_done.set()

    try:
        registry.set_progress(tenant_id, 10, "Pre-elaborazione documenti...")
        tmp_index_path, vector_store, new_docs = await asyncio.to_thread(_build_index_and_signal)

        if tmp_index_path is None:
            # Preprocessing was stopped mid-way (cancel arrived during pool.imap_unordered).
            # The CancelledError handler already took care of rollback and status.
            return

        # Atomic swap: replace index_dir with the successfully built tmp dir.
        registry.set_progress(tenant_id, 75, "Salvataggio indice...")
        _update_documents_meta(upload_dir, new_docs=new_docs)

        old_index_dir = _atomic_swap_index(index_dir, tmp_index_dir)
        if old_index_dir:
            # Skip subdirs already present (incremental add copies the whole index_dir).
            _carry_over_subdirs(old_index_dir, index_dir)
            if old_index_dir.exists():
                shutil.rmtree(old_index_dir, ignore_errors=True)

        # Extract concepts from new docs → pending wiki updates (no LLM, free)
        try:
            from vectoria_lib.wiki.wiki_manager import WikiManager

            wm = WikiManager.from_config(index_dir / "wiki")
            n_new = wm.add_from_docs(new_docs)
            logger.info(
                "Wiki: %d new concept(s) pending (total=%d, tenant=%s)",
                n_new,
                wm.pending_count(),
                tenant_id,
            )
        except Exception as wiki_exc:
            logger.warning("Wiki concept extraction failed (non-fatal): %s", wiki_exc)

        # Extract entities from new docs → graph (no LLM for cooccurrence, free).
        # GLiNER/spaCy inference is real CPU/GPU work — run off the event loop
        # thread, same as the vector index build above, so it doesn't stall
        # other tenants' requests while this one's graph is being built.
        try:
            from vectoria_lib.rag.graph.graph_manager import GraphManager

            gm = GraphManager.from_config(index_dir / "graph")
            n_graph = await asyncio.to_thread(gm.add_chunks, new_docs)
            if n_graph:
                logger.info("Graph: %d entity mentions added (tenant=%s)", n_graph, tenant_id)
        except Exception as graph_exc:
            logger.warning("Graph entity extraction failed (non-fatal): %s", graph_exc)

        try:
            index_path = index_dir / tmp_index_path.relative_to(tmp_index_dir)
        except ValueError:
            logger.error("Index path %s is outside tmp dir %s", tmp_index_path, tmp_index_dir)
            raise
        VectorStoreBuilder.evict(index_path)

        registry.set_progress(tenant_id, 90, "Caricamento agente...")
        extra_tools = _make_document_tools(app, upload_dir, index_dir, tenant_id)
        agent = await asyncio.to_thread(
            AgentBuilder.build_qa_agent,
            index_path=index_path,
            tenant_id=tenant_id,
            extra_tools=extra_tools,
            upload_url=getattr(app.state, "upload_url", "http://localhost:8080/upload"),
        )
        registry.set_agent(tenant_id, agent)

        # Kick off wiki synthesis in the background — the ingest task is already done
        # and the agent is ready, so the user can chat immediately.
        step = "Pronto!"
        try:
            from vectoria_lib.wiki.wiki_manager import WikiManager

            wm = WikiManager.from_config(index_dir / "wiki")
            if wm.should_synthesize():
                if registry.try_acquire_wiki_lock(tenant_id):
                    threading.Thread(
                        target=_run_wiki_synthesis_background,
                        args=(index_dir, tenant_id, registry),
                        name=f"wiki-synth-{tenant_id}",
                        daemon=True,
                    ).start()
                    step = f"Pronto! Sintesi wiki in corso ({wm.pending_count()} concetti)."
                    logger.info("Background wiki synthesis started for tenant=%s", tenant_id)
                else:
                    step = "Pronto! (sintesi wiki già in corso)"
            elif wm.pending_count():
                step = f"Pronto! ({wm.pending_count()} concetti wiki in attesa)"
        except Exception as wiki_exc:
            logger.warning("Could not schedule wiki synthesis (non-fatal): %s", wiki_exc)

        registry.set_progress(tenant_id, 100, step)
        registry.set_status(tenant_id, IndexStatus.READY)
        registry.clear_task(tenant_id)
        registry.clear_thread_done_event(tenant_id)
        logger.info("Index ready at %s — agent reloaded (tenant=%s)", index_path, tenant_id)

    except asyncio.CancelledError:
        # Signal the preprocessing pool to stop as soon as the current file finishes.
        stop_event.set()

        # Roll back: remove the files uploaded in this (cancelled) request so the
        # upload dir stays consistent with the previous agent's index.
        for f in uploaded_files:
            try:
                f.unlink(missing_ok=True)
                logger.info("Rolled back cancelled upload — removed %s", f.name)
            except Exception as rm_err:
                logger.warning("Could not remove %s after cancel: %s", f, rm_err)

        # Restore the previous agent so the model stays visible in OpenWebUI.
        if previous_agent is not None:
            registry.set_agent(tenant_id, previous_agent)
            registry.set_progress(tenant_id, 0, "Annullato — agente precedente disponibile.")
        else:
            registry.set_progress(tenant_id, 0, "Indicizzazione annullata.")
        # Always CANCELLED — status=READY would make the frontend show "success".
        registry.set_status(tenant_id, IndexStatus.CANCELLED)
        registry.clear_task(tenant_id)
        # Do NOT clear thread_done here: ingest needs it to know when the thread exits.
        logger.info(
            "Indexing cancelled for tenant '%s' — previous agent restored: %s", tenant_id, previous_agent is not None
        )
        raise  # propagate so asyncio marks the task as cancelled

    except Exception as exc:
        # Roll back uploaded files so they don't pollute the next ingest attempt.
        for f in uploaded_files:
            try:
                f.unlink(missing_ok=True)
                logger.info("Rolled back failed upload — removed %s (tenant=%s)", f.name, tenant_id)
            except Exception as rm_err:
                logger.warning("Could not remove %s after error: %s", f, rm_err)

        # Restore previous agent on unexpected errors.
        if previous_agent is not None:
            registry.set_agent(tenant_id, previous_agent)
            registry.set_status(tenant_id, IndexStatus.READY)
            registry.set_progress(tenant_id, 100, "Ripristinato agente precedente.")
            logger.warning("Indexing failed for tenant '%s', previous agent restored: %s", tenant_id, exc)
        else:
            registry.set_progress(tenant_id, 0, "Errore durante l'indicizzazione.")
            registry.set_status(tenant_id, IndexStatus.ERROR)
            logger.exception("Indexing failed for tenant '%s': %s", tenant_id, exc)
        registry.clear_task(tenant_id)
        registry.clear_thread_done_event(tenant_id)
