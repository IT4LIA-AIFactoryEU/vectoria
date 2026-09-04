#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging
import os

logger = logging.getLogger("graph")

_GLINER_CACHE = {}
_SPACY_CACHE = {}


def load_gliner(model_path, device="cpu"):
    cache_key = f"{model_path}:{device}"
    if cache_key in _GLINER_CACHE:
        return _GLINER_CACHE[cache_key]
    try:
        from gliner import GLiNER

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        model = GLiNER.from_pretrained(model_path, local_files_only=offline)
        if device and device != "cpu":
            model = model.to(device)
        _GLINER_CACHE[cache_key] = model
        return model
    except ImportError:
        logger.warning("GLiNER not installed — install with: uv sync --extra graph")
        return None
    except Exception as exc:
        logger.warning("GLiNER failed to load: %s", exc)
        return None


def batch_predict(gliner, texts, labels, threshold=0.5):
    """Run GLiNER over a batch of texts in a single call instead of looping
    predict_entities() once per text — lets the model batch the forward pass
    (real parallelism on GPU, less per-call overhead on CPU).

    Returns a list aligned 1:1 with `texts` (empty list for any text GLiNER
    couldn't process). Prefers the current `inference()` API, falling back to
    the older `batch_predict_entities()` name for older gliner versions.
    """
    if hasattr(gliner, "inference"):
        return gliner.inference(texts, labels, threshold=threshold)
    return gliner.batch_predict_entities(texts, labels, threshold=threshold)


def load_spacy(model_name):
    if not model_name:
        return None
    if model_name in _SPACY_CACHE:
        return _SPACY_CACHE[model_name]
    try:
        import spacy
        nlp = spacy.load(model_name, disable=["ner"])
        _SPACY_CACHE[model_name] = nlp
        return nlp
    except ImportError:
        logger.warning(
            "spaCy not installed — relation labels disabled (edges will have no 'relation'). "
            "Install with: uv sync --extra graph, then: python -m spacy download %s",
            model_name,
        )
    except OSError:
        logger.warning("spaCy model '%s' not found — run: python -m spacy download %s",
                       model_name, model_name)
    _SPACY_CACHE[model_name] = None
    return None


def _find_token(doc, name):
    target = name.split()[0].lower()
    for tok in doc:
        if tok.text.lower() == target:
            return tok
    return None


def _common_verb(a_tok, b_tok):
    a_ancestors = list(a_tok.ancestors) + [a_tok]
    b_ancestors = set(t.i for t in list(b_tok.ancestors) + [b_tok])
    for t in a_ancestors:
        if t.i in b_ancestors and t.pos_ == "VERB":
            return t
    return None


def spacy_relation_from_doc(doc, ent_a, ent_b):
    """Extract a relation label between two entities from an already-parsed
    spaCy Doc. Callers that need to check multiple entity pairs from the same
    sentence should parse it once with nlp() and reuse the Doc via this
    function, instead of re-parsing per pair (see spacy_relation)."""
    try:
        a_tok = _find_token(doc, ent_a["name"])
        b_tok = _find_token(doc, ent_b["name"])
        if a_tok is None or b_tok is None:
            return ""
        verb = _common_verb(a_tok, b_tok)
        if verb is None:
            return ""
        return verb.lemma_.lower() or verb.text.lower()
    except Exception as exc:
        logger.debug("spaCy relation failed: %s", exc)
        return ""


def spacy_relation(nlp, sentence, ent_a, ent_b):
    """Parse `sentence` and extract a relation label between two entities.

    Convenience wrapper around spacy_relation_from_doc() for callers that only
    need a single pair from a sentence. Prefer parsing once with nlp() and
    calling spacy_relation_from_doc() directly when checking multiple pairs
    from the same sentence, to avoid redundant parses.
    """
    try:
        doc = nlp(sentence)
    except Exception as exc:
        logger.debug("spaCy parse failed: %s", exc)
        return ""
    return spacy_relation_from_doc(doc, ent_a, ent_b)
