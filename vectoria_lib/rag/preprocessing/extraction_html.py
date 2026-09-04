#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER

logger = logging.getLogger("rag")


def extract_text_from_html_file(
    file_path: Path, dump_doc_structure_on_file: bool = False, regexes_for_metadata_extraction: list[RegexConfig] = []
) -> list[Document]:
    """
    Extracts text from an HTML file, stripping all tags/scripts/styles, as a single Document.
    """
    logger.debug("Extracting text from %s", file_path)

    html = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n").strip()

    return [
        Document(
            page_content=text,
            metadata={DOC_FILE_NAME: file_path.stem, PARAGRAPH_NAME: "", PARAGRAPH_NUMBER: 0},
        )
    ]
