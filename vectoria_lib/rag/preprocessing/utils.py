#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

import logging
import re

from langchain_core.documents import Document

from vectoria_lib.core.config_schema import RegexConfig

logger = logging.getLogger("rag")

regex_cache = {}


def seek_and_replace(doc: Document, regex_configs: list[RegexConfig] = []) -> str:
    for rc in regex_configs:
        logger.debug("Seeking and replacing %s with %s regex %s", rc.name, rc.replace_with, rc.pattern)
        if rc.pattern not in regex_cache:
            regex_cache[rc.pattern] = re.compile(rc.pattern, re.IGNORECASE)
        doc.page_content = regex_cache[rc.pattern].sub(rc.replace_with, doc.page_content)
    return doc


def extract_metadata_from_text(text: str, regex_configs: list[RegexConfig] = []) -> dict:
    metadata = {}
    for rc in regex_configs:
        logger.debug("Extracting %s from text using regex %s", rc.name, rc.pattern)
        if rc.pattern not in regex_cache:
            regex_cache[rc.pattern] = re.compile(rc.pattern, re.IGNORECASE)
        result = regex_cache[rc.pattern].search(text)
        if result:
            metadata[rc.name] = result.group(0).strip()
        else:
            logger.debug("No match found for regex %s in text", rc.pattern)
    return metadata
