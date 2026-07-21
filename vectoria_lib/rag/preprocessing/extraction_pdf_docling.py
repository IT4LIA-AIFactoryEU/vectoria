#
# VECTORIA
#
# @authors : Mattia Buzzoni
#

import logging
from pathlib import Path

from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER

logger = logging.getLogger("rag")

# Module-level singleton: Docling loads RT-DETR v2 + Tesseract once per process
_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        logger.info("Initialising Docling DocumentConverter (first call)...")
        _converter = DocumentConverter()
        logger.info("Docling DocumentConverter ready.")
    return _converter


# Labels treated as section boundaries
_HEADER_LABELS = {"section_header", "title"}

# Labels whose text content is included in a section
_TEXT_LABELS = {"text", "paragraph", "list_item", "footnote", "caption", "code"}

# Labels rendered as markdown table
_TABLE_LABELS = {"table"}

# Labels to skip entirely (layout noise)
_SKIP_LABELS = {"page_header", "page_footer", "picture", "chart", "form", "key_value_region", "document_index"}


def extract_text_from_pdf_file_docling(
    file_path: Path,
    dump_doc_structure_on_file: bool = False,
    regexes_for_metadata_extraction: list[RegexConfig] = [],
) -> list[Document]:
    """
    Extracts text from a PDF file using Docling and returns one Document per section.

    Docling performs layout analysis and structure detection (headings, paragraphs,
    tables, lists) on both digital and scanned PDFs.  The output format mirrors the
    DOCX extractor: one Document per logical section with metadata keys
    doc_file_name, paragraph_name, and paragraph_number.
    """
    logger.debug("Extracting text from %s via Docling", file_path)

    result = _get_converter().convert(str(file_path))
    dl_doc = result.document

    sections: list[Document] = []
    section_counter = 0
    current_name = "document_start"
    current_lines: list[str] = []

    def _flush(name: str, number: int, lines: list[str]) -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(
                Document(
                    page_content=text,
                    metadata={
                        DOC_FILE_NAME: file_path.stem,
                        PARAGRAPH_NAME: name,
                        PARAGRAPH_NUMBER: str(number),
                    },
                )
            )

    for item, _ in dl_doc.iterate_items():
        label = item.label.lower() if hasattr(item.label, "lower") else str(item.label).lower()

        if label in _SKIP_LABELS:
            continue

        if label in _HEADER_LABELS:
            _flush(current_name, section_counter, current_lines)
            section_counter += 1
            current_name = getattr(item, "text", "") or current_name
            current_lines = []

        elif label in _TEXT_LABELS:
            text = getattr(item, "text", "") or ""
            if text.strip():
                current_lines.append(text)

        elif label in _TABLE_LABELS:
            try:
                table_md = item.export_to_markdown(doc=dl_doc)
            except Exception:
                table_md = getattr(item, "text", "") or ""
            if table_md.strip():
                current_lines.append(table_md)

    # Flush last section
    _flush(current_name, section_counter, current_lines)

    # Fallback: Docling found no structure → export whole doc as markdown
    if not sections:
        logger.warning("Docling found no structured content in %s, falling back to full markdown", file_path.name)
        sections.append(
            Document(
                page_content=dl_doc.export_to_markdown(),
                metadata={
                    DOC_FILE_NAME: file_path.stem,
                    PARAGRAPH_NAME: "document",
                    PARAGRAPH_NUMBER: "0",
                },
            )
        )

    logger.debug("Docling extracted %d section(s) from %s", len(sections), file_path.name)
    return sections
