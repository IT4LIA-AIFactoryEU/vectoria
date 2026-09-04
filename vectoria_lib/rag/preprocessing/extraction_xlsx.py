#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.

import logging
from pathlib import Path

import openpyxl
from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME, PARAGRAPH_NAME, PARAGRAPH_NUMBER

logger = logging.getLogger("rag")


def extract_text_from_xlsx_file(
    file_path: Path, dump_doc_structure_on_file: bool = False, regexes_for_metadata_extraction: list[RegexConfig] = []
) -> list[Document]:
    """
    Extracts text from an XLSX file, one Document per non-empty sheet.

    Each sheet's rows are rendered as pipe-separated text lines (one line per row).
    """
    logger.debug("Extracting text from %s", file_path)

    workbook = openpyxl.load_workbook(file_path, data_only=True, read_only=True)

    documents = []
    for sheet_index, sheet_name in enumerate(workbook.sheetnames):
        sheet = workbook[sheet_name]
        lines = []
        for row in sheet.iter_rows(values_only=True):
            if all(cell is None for cell in row):
                continue
            lines.append(" | ".join("" if cell is None else str(cell) for cell in row))

        if not lines:
            continue

        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    DOC_FILE_NAME: file_path.stem,
                    PARAGRAPH_NAME: sheet_name,
                    PARAGRAPH_NUMBER: sheet_index,
                },
            )
        )

    workbook.close()
    return documents
