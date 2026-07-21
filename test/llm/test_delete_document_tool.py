import logging

from vectoria_lib.llm.agents.tools.document_tools import DeleteDocumentTool


def _make_tool(tmp_path, rebuild_fn):
    doc = tmp_path / "only.pdf"
    doc.write_bytes(b"%PDF-1.4")
    return DeleteDocumentTool(upload_dir=tmp_path, index_dir=tmp_path, rebuild_fn=rebuild_fn)


def test_delete_last_document_rebuild_failure_is_reported_and_logged(tmp_path, caplog):
    def failing_rebuild(clean, remaining):
        raise RuntimeError("index cleanup boom")

    tool = _make_tool(tmp_path, failing_rebuild)

    with caplog.at_level(logging.ERROR, logger="llm"):
        result = tool._run("only.pdf")

    assert "cleanup boom" in result
    assert "inconsistente" in result
    assert any("cleanup" in r.message.lower() for r in caplog.records)


def test_delete_last_document_success_still_reports_ok(tmp_path):
    calls = []
    tool = _make_tool(tmp_path, lambda clean, remaining: calls.append((clean, remaining)))

    result = tool._run("only.pdf")

    assert "eliminato" in result
    assert calls == [("only.pdf", [])]
