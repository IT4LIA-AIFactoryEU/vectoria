#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.

import csv
import io
import logging
from pathlib import Path

from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER

logger = logging.getLogger("rag")


def extract_text_from_plaintext_file(
    file_path: Path, dump_doc_structure_on_file: bool = False, regexes_for_metadata_extraction: list[RegexConfig] = []
) -> list[Document]:
    """
    Extracts text from .txt/.md/.markdown files (read as-is, no transformation)
    and .csv files (each row rendered as "col1: val1 | col2: val2" lines).

    Always returns a single Document — chunking downstream handles splitting
    long content, so there's no benefit to finer-grained Documents here.
    """
    logger.debug("Extracting text from %s", file_path)

    raw = file_path.read_text(encoding="utf-8", errors="replace")

    if file_path.suffix.lower() == ".csv":
        reader = csv.DictReader(io.StringIO(raw))
        lines = [" | ".join(f"{key}: {value}" for key, value in row.items()) for row in reader]
        text = "\n".join(lines)
    else:
        text = raw

    return [
        Document(
            page_content=text,
            metadata={DOC_FILE_NAME: file_path.stem, PARAGRAPH_NAME: "", PARAGRAPH_NUMBER: 0},
        )
    ]
