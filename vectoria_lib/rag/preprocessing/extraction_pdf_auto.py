#
# VECTORIA
#
# @authors : Mattia Buzzoni
#

import logging
from pathlib import Path

from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig

logger = logging.getLogger("rag")

# Pages with fewer chars than this threshold are considered image/scan pages.
# A typical digital-text page has 1000–4000 chars; scanned pages return near 0.
_MIN_CHARS_PER_PAGE = 100


def extract_text_from_pdf_file_auto(
    file_path: Path,
    dump_doc_structure_on_file: bool = False,
    regexes_for_metadata_extraction: list[RegexConfig] = [],
) -> list[Document]:
    """
    Smart PDF extractor: tries native extraction (PyMuPDF) first for speed; falls back to Docling+OCR
    only when the text yield indicates image-based or scanned pages.

    Decision rule: checked per page, not as a whole-document average. If every
    page yields at least _MIN_CHARS_PER_PAGE characters via native extraction, the document
    is treated as fully digital-text and the (fast) native extraction result is used as-is.
    If even a single page falls short, the whole document is re-extracted via
    Docling+OCR instead. A whole-document average would let a handful of scanned
    pages (e.g. a signature page or a scanned appendix) hide inside an otherwise
    digital-text document and silently lose that page's content — checking the
    minimum instead means any single weak page routes the entire file through
    the slower but lossless Docling path.
    """
    from langchain_community.document_loaders import PyMuPDFLoader

    logger.debug("Auto extraction: trying native extraction on %s", file_path.name)
    pages = PyMuPDFLoader(str(file_path)).load()
    min_chars_per_page = min((len(p.page_content) for p in pages), default=0)

    if min_chars_per_page >= _MIN_CHARS_PER_PAGE:
        logger.info(
            "Auto extraction: text-based PDF detected (min %d chars/page) — using native extraction result for %s",
            min_chars_per_page,
            file_path.name,
        )
        from vectoria_lib.rag.preprocessing.extraction_pdf import extract_text_from_pdf_file

        return extract_text_from_pdf_file(
            file_path,
            dump_doc_structure_on_file=dump_doc_structure_on_file,
            regexes_for_metadata_extraction=regexes_for_metadata_extraction,
        )

    logger.info(
        "Auto extraction: at least one low-yield page (min %d chars/page) — switching to Docling+OCR for %s",
        min_chars_per_page,
        file_path.name,
    )
    from vectoria_lib.rag.preprocessing.extraction_pdf_docling import extract_text_from_pdf_file_docling

    return extract_text_from_pdf_file_docling(
        file_path,
        dump_doc_structure_on_file=dump_doc_structure_on_file,
        regexes_for_metadata_extraction=regexes_for_metadata_extraction,
    )
