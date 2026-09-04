#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.api.routers.documents import ALLOWED_EXTENSIONS as documents_extensions
from vectoria_lib.api.routers.openai_compat.router import _ALLOWED_EXTENSIONS as openai_compat_extensions
from vectoria_lib.core.constants import ALLOWED_EXTENSIONS as core_extensions
from vectoria_lib.llm.agents.tools.document_tools import ALLOWED_EXTENSIONS as tools_extensions


def test_allowed_extensions_consistent_across_modules():
    assert documents_extensions is core_extensions
    assert tools_extensions is core_extensions
    assert openai_compat_extensions is core_extensions


def test_allowed_extensions_includes_new_formats():
    for ext in (".xlsx", ".pptx", ".html", ".htm", ".md", ".markdown", ".txt", ".csv"):
        assert ext in core_extensions


def test_allowed_extensions_still_includes_existing_formats():
    for ext in (".pdf", ".docx", ".doc"):
        assert ext in core_extensions
