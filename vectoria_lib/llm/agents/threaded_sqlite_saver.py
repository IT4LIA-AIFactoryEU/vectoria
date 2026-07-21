import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.sqlite import SqliteSaver


class ThreadedSqliteSaver(SqliteSaver):
    """SqliteSaver with async methods delegated to a thread executor.

    Can be instantiated synchronously (from threads or sync startup code)
    but works correctly with async LangGraph by running SQLite operations
    in a thread pool executor — avoiding event-loop binding issues that
    plague AsyncSqliteSaver when the connection is opened outside the loop.
    """

    @classmethod
    def from_path(cls, db_path: str) -> "ThreadedSqliteSaver":
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        saver = cls(conn)
        saver.setup()
        return saver

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        loop = asyncio.get_running_loop()
        items = await loop.run_in_executor(
            None,
            lambda: list(self.list(config, filter=filter, before=before, limit=limit)),
        )
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.put_writes, config, writes, task_id, task_path)
