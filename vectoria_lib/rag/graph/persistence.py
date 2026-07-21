import json
import logging
from pathlib import Path

logger = logging.getLogger("graph")

GRAPH_FILE = "graph.json"
ENTITY_TYPES_FILE = "entity_types.json"


def graph_to_dict(g):
    nodes = {
        n: {
            "label": d.get("label", ""),
            "sources": sorted(d.get("sources", set())),
            "mentions": d.get("mentions", 0),
        }
        for n, d in g.nodes(data=True)
    }
    edges = [
        {
            "a": u, "b": v,
            "count": d.get("count", 1),
            "pair_mentions": d.get("pair_mentions", 0),
            "pmi": d.get("pmi", 0.0),
            "sources": sorted(d.get("sources", set())),
            "relation": d.get("relation", ""),
        }
        for u, v, d in g.edges(data=True)
    ]
    meta = {"total_sentences": g.graph.get("total_sentences", 0)}
    return {"nodes": nodes, "edges": edges, "meta": meta}


def dict_to_graph(data):
    import networkx as nx

    g = nx.Graph()
    for name, attrs in data.get("nodes", {}).items():
        g.add_node(
            name,
            label=attrs.get("label", ""),
            sources=set(attrs.get("sources", [])),
            mentions=attrs.get("mentions", 0),
        )
    for edge in data.get("edges", []):
        g.add_edge(
            edge["a"], edge["b"],
            count=edge.get("count", 1),
            pair_mentions=edge.get("pair_mentions", 0),
            pmi=edge.get("pmi", 0.0),
            sources=set(edge.get("sources", [])),
            relation=edge.get("relation", ""),
        )
    g.graph["total_sentences"] = data.get("meta", {}).get("total_sentences", 0)
    return g


def save_graph(graph_dir, g):
    path = Path(graph_dir) / GRAPH_FILE
    path.write_text(
        json.dumps(graph_to_dict(g), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path.stat().st_mtime


def load_graph_data(graph_dir):
    path = Path(graph_dir) / GRAPH_FILE
    if not path.exists():
        return None, 0
    mtime = path.stat().st_mtime
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict_to_graph(data), mtime
    except Exception as exc:
        logger.warning("Could not load graph (%s) — starting fresh", exc)
        return None, mtime


def load_entity_types(graph_dir):
    path = Path(graph_dir) / ENTITY_TYPES_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_entity_types(graph_dir, types):
    (Path(graph_dir) / ENTITY_TYPES_FILE).write_text(
        json.dumps(types, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class EmptyGraph:
    def number_of_nodes(self):
        return 0
