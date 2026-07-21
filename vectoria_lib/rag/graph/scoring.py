import math
import re

SENTENCE_SPLIT = re.compile(r"(?<=[\.!?])\s+")
PROXIMITY_CLOSE = 50
PROXIMITY_MEDIUM = 200


def split_sentences(text):
    out = []
    pos = 0
    parts = SENTENCE_SPLIT.split(text)
    for part in parts:
        if not part:
            continue
        start = text.find(part, pos)
        if start == -1:
            start = pos
        end = start + len(part)
        out.append((part, start, end))
        pos = end
    return out


def proximity_weight(distance):
    if distance < PROXIMITY_CLOSE:
        return 1.0
    if distance < PROXIMITY_MEDIUM:
        return 0.5
    return 0.25


def compute_pmi(g, edges=None):
    """Recompute PMI for graph edges.

    edges=None recomputes every edge in the graph (O(total edges) — used for a
    full/cold rebuild). Passing an iterable of (a, b) pairs restricts the
    recompute to just those edges (O(touched edges)) — the mentions/pair_mentions
    counts used in the formula are already exact running totals, so this gives
    the same PMI value for touched edges as a full recompute would.

    Note: `total` (total_sentences) is a global count that grows with every
    ingestion, so strictly every edge's PMI drifts by a tiny amount whenever it
    changes. Restricting the recompute to touched edges leaves untouched edges'
    PMI slightly stale until they're next touched or a full recompute is done —
    an intentional trade-off for avoiding an O(total edges) scan on every
    incremental ingestion.
    """
    total = max(1, g.graph.get("total_sentences", 0))
    if edges is None:
        edge_view = g.edges(data=True)
    else:
        edge_view = ((a, b, g[a][b]) for a, b in edges if g.has_edge(a, b))
    for a, b, data in edge_view:
        pair = data.get("pair_mentions", 0)
        ma = g.nodes[a].get("mentions", 0)
        mb = g.nodes[b].get("mentions", 0)
        if pair == 0 or ma == 0 or mb == 0:
            data["pmi"] = 0.0
            continue
        data["pmi"] = math.log((pair * total) / (ma * mb))


def edge_passes(g, a, b, min_count, min_pmi):
    data = g[a][b]
    return (
        data.get("count", 1) >= min_count
        and (min_pmi <= 0 or data.get("pmi", 0.0) >= min_pmi)
    )
