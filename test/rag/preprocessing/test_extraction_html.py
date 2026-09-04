#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.rag.preprocessing.extraction_html import extract_text_from_html_file


def test_extract_text_from_html_file(tmp_path):
    html_content = (
        "<html><head><title>Test</title></head><body>"
        "<h1>Hello</h1><p>World content</p>"
        "<script>ignored_js_call();</script>"
        "</body></html>"
    )
    file_path = tmp_path / "test.html"
    file_path.write_text(html_content, encoding="utf-8")

    docs = extract_text_from_html_file(file_path, dump_doc_structure_on_file=False, regexes_for_metadata_extraction=[])

    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert "World content" in docs[0].page_content
    assert "<h1>" not in docs[0].page_content
    assert "ignored_js_call" not in docs[0].page_content
