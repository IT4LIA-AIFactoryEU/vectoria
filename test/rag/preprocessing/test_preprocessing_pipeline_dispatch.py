from pathlib import Path

import pytest

from vectoria_lib.rag.preprocessing.preprocessing_pipeline import _Extractor


def test_extractor_routes_xlsx_to_native_for_native_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_xlsx.extract_text_from_xlsx_file",
        lambda *a, **kw: calls.append("xlsx") or [],
    )
    _Extractor("native")(Path("doc.xlsx"))
    assert calls == ["xlsx"]


def test_extractor_routes_pptx_to_native_for_auto_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_pptx.extract_text_from_pptx_file",
        lambda *a, **kw: calls.append("pptx") or [],
    )
    _Extractor("auto")(Path("doc.pptx"))
    assert calls == ["pptx"]


def test_extractor_routes_html_to_native_for_native_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_html.extract_text_from_html_file",
        lambda *a, **kw: calls.append("html") or [],
    )
    _Extractor("native")(Path("doc.html"))
    assert calls == ["html"]
    _Extractor("native")(Path("doc.htm"))
    assert calls == ["html", "html"]


def test_extractor_routes_txt_md_csv_to_plaintext(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_plaintext.extract_text_from_plaintext_file",
        lambda *a, **kw: calls.append("plaintext") or [],
    )
    for ext in (".txt", ".md", ".markdown", ".csv"):
        _Extractor("native")(Path(f"doc{ext}"))
    assert calls == ["plaintext"] * 4


def test_extractor_routes_xlsx_to_docling_when_backend_is_docling(monkeypatch):
    """Explicit backend: docling still wins for these formats, exactly like .docx today."""
    calls = []
    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_pdf_docling.extract_text_from_pdf_file_docling",
        lambda *a, **kw: calls.append("docling") or [],
    )
    _Extractor("docling")(Path("doc.xlsx"))
    assert calls == ["docling"]


def test_extractor_raises_for_unhandled_extension_under_native_backend():
    """Regression test: backend == "native" must not act as a catch-all that silently
    misroutes unhandled extensions (e.g. images, .xls, .ppt, .adoc) to the PDF extractor.
    Before this fix, `if ext == ".pdf" or self.backend == "native"` matched ANY extension
    under backend: native, and PyMuPDF would "successfully" open a non-PDF file with zero
    extracted characters instead of raising a clear error -- silent data loss, not a crash."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        _Extractor("native")(Path("doc.png"))
