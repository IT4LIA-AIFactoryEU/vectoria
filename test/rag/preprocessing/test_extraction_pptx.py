#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from pptx import Presentation

from vectoria_lib.rag.preprocessing.extraction_pptx import extract_text_from_pptx_file


def test_extract_text_from_pptx_file(tmp_path):
    prs = Presentation()
    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Title Slide"
    slide1.placeholders[1].text = "Body content here"

    prs.slides.add_slide(prs.slide_layouts[6])  # blank layout, no text

    file_path = tmp_path / "test.pptx"
    prs.save(file_path)

    docs = extract_text_from_pptx_file(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[])

    assert len(docs) == 1
    assert docs[0].metadata["paragraph_name"] == "Slide 1"
    assert docs[0].metadata["paragraph_number"] == 0
    assert "Title Slide" in docs[0].page_content
    assert "Body content here" in docs[0].page_content
