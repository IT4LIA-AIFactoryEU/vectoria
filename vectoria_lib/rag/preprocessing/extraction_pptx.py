#
# VECTORIA
#

import logging
from pathlib import Path

from langchain_core.documents import Document
from pptx import Presentation

from vectoria_lib.core.config_schema import RegexConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER

logger = logging.getLogger("rag")


def extract_text_from_pptx_file(
    file_path: Path, dump_doc_structure_on_file: bool = False, regexes_for_metadata_extraction: list[RegexConfig] = []
) -> list[Document]:
    """
    Extracts text from a PPTX file, one Document per slide that has extractable text.

    All text-frame text on a slide is joined by newlines. Slides with no
    extractable text are skipped.
    """
    logger.debug("Extracting text from %s", file_path)

    presentation = Presentation(file_path)

    documents = []
    for slide_index, slide in enumerate(presentation.slides):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                lines.append(shape.text_frame.text.strip())

        if not lines:
            continue

        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    DOC_FILE_NAME: file_path.stem,
                    PARAGRAPH_NAME: f"Slide {slide_index + 1}",
                    PARAGRAPH_NUMBER: slide_index,
                },
            )
        )

    return documents
