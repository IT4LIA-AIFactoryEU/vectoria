#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#

from vectoria_lib.core.constants import ALLOWED_LANGS
from vectoria_lib.core.paths import ETC_DIR


class PromptBuilder:
    @staticmethod
    def get_prompt(file_name: str, lang, variables: dict = None):
        if lang not in ALLOWED_LANGS:
            raise ValueError(f"lang '{lang}' is not valid. Supported: 'it', 'eng'.")
        custom_prompt = ETC_DIR / "custom" / "prompts" / lang / file_name
        if custom_prompt.exists():
            text = custom_prompt.read_text(encoding="utf-8")
        else:
            text = (ETC_DIR / "shared" / "prompts" / lang / file_name).read_text(encoding="utf-8")
        if variables:
            text = text.format(**variables)
        return text
