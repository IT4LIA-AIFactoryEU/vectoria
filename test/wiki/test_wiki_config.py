#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
from vectoria_lib.core.config_schema import WikiConfig


def test_defaults():
  cfg = WikiConfig()
  assert cfg.pending_threshold == 5
  assert cfg.enabled is True
  assert cfg.max_topics_per_doc == 3
  assert cfg.max_topics_per_batch == 8
  assert cfg.language_normalization is True
  assert cfg.similarity_threshold == 0.80
  assert cfg.canonical_topic_language == "it"
  assert cfg.synthesis_context_chunks == 4
  assert cfg.max_page_words == 400


def test_from_dict_full():
  raw = {
    "pending_threshold": 10,
    "enabled": False,
    "max_topics_per_doc": 5,
    "max_topics_per_batch": 12,
    "language_normalization": False,
    "similarity_threshold": 0.90,
    "canonical_topic_language": "en",
    "synthesis_context_chunks": 6,
    "max_page_words": 600,
  }
  cfg = WikiConfig(**raw)
  assert cfg.pending_threshold == 10
  assert cfg.enabled is False
  assert cfg.canonical_topic_language == "en"
  assert cfg.max_page_words == 600


def test_from_dict_partial_uses_defaults():
  cfg = WikiConfig(pending_threshold=7)
  assert cfg.pending_threshold == 7
  assert cfg.max_page_words == 400


def test_from_dict_ignores_unknown_keys():
  cfg = WikiConfig(pending_threshold=3, unknown_key="ignored")
  assert cfg.pending_threshold == 3


def test_override_single_field():
  cfg = WikiConfig(max_page_words=800)
  assert cfg.max_page_words == 800
  assert cfg.pending_threshold == 5
