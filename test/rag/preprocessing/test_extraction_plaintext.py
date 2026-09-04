#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.rag.preprocessing.extraction_plaintext import extract_text_from_plaintext_file


def test_extract_text_from_plaintext_file_txt(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Plain text content here.", encoding="utf-8")

    docs = extract_text_from_plaintext_file(
        file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]
    )

    assert len(docs) == 1
    assert docs[0].page_content == "Plain text content here."
    assert docs[0].metadata["doc_file_name"] == "test"
    assert docs[0].metadata["paragraph_number"] == 0


def test_extract_text_from_plaintext_file_markdown(tmp_path):
    file_path = tmp_path / "test.md"
    file_path.write_text("# Heading\n\nSome **bold** text.", encoding="utf-8")

    docs = extract_text_from_plaintext_file(
        file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]
    )

    assert len(docs) == 1
    assert docs[0].page_content == "# Heading\n\nSome **bold** text."


def test_extract_text_from_plaintext_file_csv(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")

    docs = extract_text_from_plaintext_file(
        file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[]
    )

    assert len(docs) == 1
    assert "name: Alice | age: 30" in docs[0].page_content
    assert "name: Bob | age: 25" in docs[0].page_content
