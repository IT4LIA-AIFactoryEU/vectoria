#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import json
import logging
import re

logger = logging.getLogger("graph")

DEFAULT_ENTITY_TYPES = [
    "persona",
    "organizzazione",
    "prodotto",
    "progetto",
    "luogo",
    "normativa",
    "concetto",
    "processo",
]


def build_llm():
    from vectoria_lib.core.config import Config
    from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
    return InferenceEngineBuilder.build_inference_engine(
        Config().get("inference_engine")
    ).as_langchain_chat_model()


def extract_relations_batch(items, llm_factory=build_llm):
    if not items:
        return {}
    try:
        block = "\n\n".join(
            f'{i + 1}. Entità: "{a}" — "{b}"\n   Testo: {snippet[:400]}'
            for i, (a, b, snippet) in enumerate(items)
        )
        prompt = (
            "Per ognuna delle coppie di entità qui sotto, deduci la relazione "
            "(2-4 parole italiane) dal testo fornito.\n\n"
            f"{block}\n\n"
            'Rispondi SOLO con un JSON oggetto con chiavi numeriche: '
            '{"1": "relazione", "2": "relazione", ...}'
        )
        llm = llm_factory()
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        result = json.loads(match.group())
        if not isinstance(result, dict):
            return {}
        labels = {}
        for k, v in result.items():
            try:
                idx = int(k) - 1
            except (ValueError, TypeError):
                continue
            if 0 <= idx < len(items):
                a, b, _ = items[idx]
                labels[f"{a}|{b}"] = str(v)
        return labels
    except Exception as exc:
        logger.warning("Relation extraction failed: %s", exc)
        return {}


def apply_relation_labels(g, labels):
    for key, relation in labels.items():
        parts = key.split("|", 1)
        if len(parts) == 2 and g.has_edge(parts[0], parts[1]):
            g[parts[0]][parts[1]]["relation"] = relation


def label_pairs(g, pair_snippets, relation_min_count, batch_size, llm_factory=build_llm):
    items = [
        (a, b, snippet)
        for (a, b), snippet in pair_snippets.items()
        if g.has_edge(a, b)
        and g[a][b].get("count", 1) >= relation_min_count
        and not g[a][b].get("relation")
    ]
    if not items:
        return 0
    batch_size = max(1, batch_size)
    n_calls = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        labels = extract_relations_batch(batch, llm_factory=llm_factory)
        apply_relation_labels(g, labels)
        n_calls += 1
    logger.info("Graph: labeled %d pairs in %d LLM calls", len(items), n_calls)
    return n_calls


def infer_entity_types(chunks, llm_factory=build_llm):
    try:
        sample = "\n\n".join(c.page_content[:300] for c in chunks[:15])
        prompt = (
            "Analizza questi estratti di documenti aziendali e identifica i tipi di entità "
            "più rilevanti per costruire un grafo della conoscenza.\n\n"
            f"{sample}\n\n"
            "Restituisci SOLO una lista JSON di 6-10 stringhe con i tipi di entità in italiano "
            '(minuscolo, 1-2 parole). Esempio: ["persona", "progetto", "prodotto"]\n'
            "Rispondi SOLO con il JSON, niente altro."
        )
        llm = llm_factory()
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            types = json.loads(match.group())
            if isinstance(types, list) and all(isinstance(t, str) for t in types):
                return [t.strip().lower() for t in types if t.strip()]
    except Exception as exc:
        logger.warning("Entity type inference failed: %s", exc)
    return []
