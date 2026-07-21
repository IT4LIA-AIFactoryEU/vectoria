from vectoria_lib.llm.agents.agent_builder import AgentBuilder


def test_resolve_tenant_checkpoint_path_suffixes_default_tenant(tmp_path):
    """Regression test: a real tenant literally named 'default' must not
    silently share the unsuffixed base checkpoint path."""
    base = tmp_path / "chat_history.db"
    result = AgentBuilder._resolve_tenant_checkpoint_path(str(base), "default")
    assert result == str(tmp_path / "chat_history_default.db")


def test_resolve_tenant_checkpoint_path_suffixes_other_tenants_too():
    result = AgentBuilder._resolve_tenant_checkpoint_path("/data/chat_history.db", "company-a")
    assert result == "/data/chat_history_company-a.db"


def test_resolve_tenant_checkpoint_path_returns_none_when_chat_history_disabled():
    assert AgentBuilder._resolve_tenant_checkpoint_path(None, "default") is None


def test_resolve_tenant_checkpoint_path_migrates_existing_unsuffixed_db(tmp_path):
    """A pre-existing unsuffixed DB (from before this fix) must be renamed
    into place, not silently orphaned."""
    base = tmp_path / "chat_history.db"
    base.write_text("old-history-marker")

    result = AgentBuilder._resolve_tenant_checkpoint_path(str(base), "default")

    new_path = tmp_path / "chat_history_default.db"
    assert result == str(new_path)
    assert new_path.read_text() == "old-history-marker"
    assert not base.exists()


def test_resolve_tenant_checkpoint_path_does_not_overwrite_already_migrated_db(tmp_path):
    """Migration must only fire once: if the suffixed DB already exists,
    leave both files alone rather than clobbering real history."""
    base = tmp_path / "chat_history.db"
    base.write_text("stale-file-should-be-ignored")
    new_path = tmp_path / "chat_history_default.db"
    new_path.write_text("already-migrated-history")

    result = AgentBuilder._resolve_tenant_checkpoint_path(str(base), "default")

    assert result == str(new_path)
    assert new_path.read_text() == "already-migrated-history"
    assert base.exists()  # left alone, not deleted


def test_resolve_tenant_checkpoint_path_migrates_wal_sidecar_files(tmp_path):
    """Regression test: SQLite WAL mode can leave recent, committed
    conversation turns only in the -wal sidecar, not yet folded into the
    main .db file. A migration that renames only the .db file silently
    orphans that history under the old path."""
    base = tmp_path / "chat_history.db"
    base.write_text("main-db-content")
    wal = tmp_path / "chat_history.db-wal"
    wal.write_text("uncheckpointed-recent-turns")
    shm = tmp_path / "chat_history.db-shm"
    shm.write_text("shm-index")

    AgentBuilder._resolve_tenant_checkpoint_path(str(base), "default")

    new_db = tmp_path / "chat_history_default.db"
    new_wal = tmp_path / "chat_history_default.db-wal"
    new_shm = tmp_path / "chat_history_default.db-shm"
    assert new_db.read_text() == "main-db-content"
    assert new_wal.read_text() == "uncheckpointed-recent-turns"
    assert new_shm.read_text() == "shm-index"
    assert not wal.exists()
    assert not shm.exists()


def test_resolve_tenant_checkpoint_path_handles_missing_wal_sidecar(tmp_path):
    """Most deployments won't have a -wal file at migration time (SQLite
    auto-checkpoints periodically) — the migration must not fail when it's
    absent."""
    base = tmp_path / "chat_history.db"
    base.write_text("main-db-content")

    result = AgentBuilder._resolve_tenant_checkpoint_path(str(base), "default")

    assert result == str(tmp_path / "chat_history_default.db")
