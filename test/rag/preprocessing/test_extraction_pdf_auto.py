from langchain_core.documents import Document

from vectoria_lib.core.paths import TEST_DIR
from vectoria_lib.rag.preprocessing.extraction_pdf_auto import extract_text_from_pdf_file_auto


def test_extract_text_from_pdf_file_auto_uses_pypdf_for_digital_text(config):
    docs = extract_text_from_pdf_file_auto(
        TEST_DIR / "data/pdf/1.pdf", dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]
    )

    assert isinstance(docs, list)
    assert isinstance(docs[0], Document)
    assert "arXiv:1906.02243v1" in docs[0].page_content


def test_extract_text_from_pdf_file_auto_falls_back_to_docling_on_single_weak_page(config, monkeypatch):
    """A document where every page but one is text-rich must still route through
    Docling — a whole-document average would let that one weak page hide."""
    import vectoria_lib.rag.preprocessing.extraction_pdf_auto as extraction_pdf_auto_module

    fake_pages = [
        Document(page_content="a" * 2000),  # healthy digital page
        Document(page_content="a" * 2000),  # healthy digital page
        Document(page_content=""),  # scanned/image page: pypdf yields nothing
    ]

    class _FakeLoader:
        def __init__(self, path):
            pass

        def load(self):
            return fake_pages

    monkeypatch.setattr(
        "langchain_community.document_loaders.PyMuPDFLoader",
        _FakeLoader,
    )

    docling_calls = []

    def _fake_docling(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]):
        docling_calls.append(file_path)
        return [Document(page_content="docling result")]

    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_pdf_docling.extract_text_from_pdf_file_docling",
        _fake_docling,
    )

    pypdf_calls = []

    def _fake_pypdf(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]):
        pypdf_calls.append(file_path)
        return [Document(page_content="pypdf result")]

    monkeypatch.setattr(
        "vectoria_lib.rag.preprocessing.extraction_pdf.extract_text_from_pdf_file",
        _fake_pypdf,
    )

    docs = extraction_pdf_auto_module.extract_text_from_pdf_file_auto(
        TEST_DIR / "data/pdf/1.pdf", dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]
    )

    assert docs[0].page_content == "docling result"
    assert len(docling_calls) == 1
    assert len(pypdf_calls) == 0
