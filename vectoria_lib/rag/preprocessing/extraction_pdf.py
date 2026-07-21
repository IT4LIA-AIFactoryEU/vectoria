#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER
from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig

logger = logging.getLogger("rag")


def extract_text_from_pdf_file(
    file_path: Path, dump_doc_structure_on_file: bool = False, regexes_for_metadata_extraction: list[RegexConfig] = []
) -> list[Document]:
    """
    Extracts text from a PDF file and returns a list of Document objects.

    Each Document object contains the page content and metadata.
    The metadata must include the keys: name, level, and id.
    """
    logger.debug("Extracting text from %s", file_path)

    pages = PyMuPDFLoader(file_path).load()

    pages_list = []
    for page in pages:
        pages_list.append(page.page_content)

    pages_str = "".join(pages_list)

    logger.debug("Loaded %d characters", len(pages_str))

    return [
        Document(
            page_content=pages_str, metadata={DOC_FILE_NAME: file_path.stem, PARAGRAPH_NAME: "", PARAGRAPH_NUMBER: 0}
        )
    ]
