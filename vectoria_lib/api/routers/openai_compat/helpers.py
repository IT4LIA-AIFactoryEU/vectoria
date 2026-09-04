#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import re

_SPECIAL_TOKEN_RE = re.compile(
  r"<\|?(?:eos|bos|endoftext|eot_id|end_of_turn|start_of_turn)\|?>|</s>"
)


def strip_special_tokens(text: str) -> str:
  return _SPECIAL_TOKEN_RE.sub("", text)
