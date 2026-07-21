import asyncio
import logging
import threading
from typing import Optional

from vectoria_lib.api.routers.documents import IndexStatus

logger = logging.getLogger("api")


class TenantRegistry:
    """Per-tenant state container: agent, index status, progress, async task, and thread-done event."""

    def __init__(self):
        self._lock = threading.RLock()
        self._agents: dict = {}
        self._statuses: dict[str, IndexStatus] = {}
        self._progress: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # threading.Event set when the underlying worker thread finishes,
        # regardless of whether the asyncio task was cancelled or completed.
        self._thread_done: dict[str, threading.Event] = {}
        # Wiki synthesis: state ("idle"|"synthesizing"|"failed") + last counts.
        self._wiki: dict[str, dict] = {}
        # Per-(tenant_id, thread_id) lock serializing chat turns on the same
        # conversation, so two concurrent requests on one thread_id (double
        # click, two browser tabs) can't both read the same parent checkpoint
        # and both write a new one — see get_thread_lock().
        self._thread_locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ── Agent ─────────────────────────────────────────────────────────────────

    def get_agent(self, tenant_id: str):
        with self._lock:
            return self._agents.get(tenant_id)

    def set_agent(self, tenant_id: str, agent) -> None:
        with self._lock:
            self._agents[tenant_id] = agent

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self, tenant_id: str) -> IndexStatus:
        with self._lock:
            return self._statuses.get(tenant_id, IndexStatus.NOT_BUILT)

    def set_status(self, tenant_id: str, status: IndexStatus) -> None:
        with self._lock:
            self._statuses[tenant_id] = status

    # ── Progress ──────────────────────────────────────────────────────────────

    def get_progress(self, tenant_id: str) -> dict:
        with self._lock:
            return self._progress.get(tenant_id, {"pct": 0, "step": ""})

    def set_progress(self, tenant_id: str, pct: int, step: str) -> None:
        with self._lock:
            self._progress[tenant_id] = {"pct": pct, "step": step}

    # ── Task ──────────────────────────────────────────────────────────────────

    def set_task(self, tenant_id: str, task: asyncio.Task) -> None:
        with self._lock:
            self._tasks[tenant_id] = task

    def get_task(self, tenant_id: str) -> Optional[asyncio.Task]:
        with self._lock:
            return self._tasks.get(tenant_id)

    def clear_task(self, tenant_id: str) -> None:
        with self._lock:
            self._tasks.pop(tenant_id, None)

    # ── Thread-done event ─────────────────────────────────────────────────────

    def new_thread_done_event(self, tenant_id: str) -> threading.Event:
        """Create and store a fresh threading.Event for tracking the worker thread."""
        event = threading.Event()
        with self._lock:
            self._thread_done[tenant_id] = event
        return event

    def is_thread_running(self, tenant_id: str) -> bool:
        """Return True if the worker thread for this tenant is still running."""
        with self._lock:
            event = self._thread_done.get(tenant_id)
        return event is not None and not event.is_set()

    def clear_thread_done_event(self, tenant_id: str) -> None:
        with self._lock:
            self._thread_done.pop(tenant_id, None)

    # ── Wiki synthesis state ──────────────────────────────────────────────────

    def get_wiki_state(self, tenant_id: str) -> dict:
        with self._lock:
            return dict(self._wiki.get(tenant_id, {"state": "idle", "created": 0}))

    def set_wiki_state(self, tenant_id: str, state: str, created: int = 0) -> None:
        with self._lock:
            self._wiki[tenant_id] = {"state": state, "created": created}

    def try_acquire_wiki_lock(self, tenant_id: str) -> bool:
        """Atomically transition wiki state to 'synthesizing' if currently idle/failed."""
        with self._lock:
            current = self._wiki.get(tenant_id, {}).get("state", "idle")
            if current == "synthesizing":
                return False
            self._wiki[tenant_id] = {"state": "synthesizing", "created": 0}
            return True

    # ── Per-thread lock ───────────────────────────────────────────────────────

    def get_thread_lock(self, tenant_id: str, thread_id: str) -> asyncio.Lock:
        """Return the asyncio.Lock for this (tenant_id, thread_id) pair,
        creating it on first use.

        Safe without extra guarding despite not using self._lock: asyncio has
        no preemption between the dict lookup and insert inside setdefault()
        (no `await` occurs in between), so two coroutines racing to create the
        same key can't both win — one always sees the other's insert first.

        Note: entries are never evicted, so long-running deployments with many
        distinct thread_ids accumulate lock objects (small, but unbounded).
        Not addressed here — out of scope for this correctness fix.
        """
        key = (tenant_id, thread_id)
        return self._thread_locks.setdefault(key, asyncio.Lock())

    # ── Agents snapshot (read-only iteration) ─────────────────────────────────

    @property
    def agents_snapshot(self) -> dict:
        """Return a shallow copy of the agents dict for safe iteration."""
        with self._lock:
            return dict(self._agents)

    # ── Remove ────────────────────────────────────────────────────────────────

    def remove_tenant(self, tenant_id: str) -> None:
        """Remove all state for a tenant (called when its last document is deleted)."""
        with self._lock:
            self._agents.pop(tenant_id, None)
            self._statuses.pop(tenant_id, None)
            self._progress.pop(tenant_id, None)
            self._tasks.pop(tenant_id, None)
            self._thread_done.pop(tenant_id, None)
            self._wiki.pop(tenant_id, None)
        logger.info("Tenant '%s' removed from registry", tenant_id)
