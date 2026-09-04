#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import openpyxl

from vectoria_lib.rag.preprocessing.extraction_xlsx import extract_text_from_xlsx_file


def test_extract_text_from_xlsx_file(tmp_path):
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Sheet1"
    ws1.append(["Name", "Age"])
    ws1.append(["Alice", 30])
    ws2 = wb.create_sheet("Sheet2")
    ws2.append(["Only one row"])
    file_path = tmp_path / "test.xlsx"
    wb.save(file_path)

    docs = extract_text_from_xlsx_file(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[])

    assert len(docs) == 2
    assert docs[0].metadata["paragraph_name"] == "Sheet1"
    assert docs[0].metadata["paragraph_number"] == 0
    assert "Name | Age" in docs[0].page_content
    assert "Alice | 30" in docs[0].page_content
    assert docs[1].metadata["paragraph_name"] == "Sheet2"
    assert "Only one row" in docs[1].page_content


def test_extract_text_from_xlsx_file_skips_empty_sheets(tmp_path):
    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Empty"
    ws2 = wb.create_sheet("Data")
    ws2.append(["value"])
    file_path = tmp_path / "test_empty.xlsx"
    wb.save(file_path)

    docs = extract_text_from_xlsx_file(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[])

    assert len(docs) == 1
    assert docs[0].metadata["paragraph_name"] == "Data"
