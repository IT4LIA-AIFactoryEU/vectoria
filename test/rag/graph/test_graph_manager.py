import pytest
from unittest.mock import MagicMock, patch

from vectoria_lib.rag.graph.graph_manager import GraphManager


# ── min_count tests ──────────────────────────────────────────────────────────

def test_search_respects_min_count(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, min_count=2)
  g = nx.Graph()
  g.add_node("mario", label="persona", sources={"doc.pdf"})
  g.add_node("progetto", label="progetto", sources={"doc.pdf"})
  g.add_node("banca", label="organizzazione", sources={"doc.pdf"})
  g.add_edge("mario", "progetto", count=3, sources={"doc.pdf"})
  g.add_edge("mario", "banca", count=1, sources={"doc.pdf"})
  gm._save_graph(g)

  result = gm.neighbors("mario")
  assert "progetto" in result
  assert "banca" not in result


def test_neighbors_min_count_1_shows_all(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, min_count=1)
  g = nx.Graph()
  g.add_node("mario", label="persona", sources={"doc.pdf"})
  g.add_node("banca", label="organizzazione", sources={"doc.pdf"})
  g.add_edge("mario", "banca", count=1, sources={"doc.pdf"})
  gm._save_graph(g)

  result = gm.neighbors("mario")
  assert "banca" in result


# ── relation_extraction tests ────────────────────────────────────────────────

def test_extract_relations_batch_returns_labels(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True)
  mock_llm = MagicMock()
  mock_llm.invoke.return_value.content = '{"1": "gestisce"}'
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    labels = gm._extract_relations_batch([("mario", "progetto alpha", "mario lavora su progetto alpha")])
  assert labels == {"mario|progetto alpha": "gestisce"}


def test_extract_relations_batch_maps_indices(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True)
  mock_llm = MagicMock()
  mock_llm.invoke.return_value.content = '{"1": "gestisce", "2": "fa parte di"}'
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    labels = gm._extract_relations_batch([
      ("mario", "alpha", "t1"),
      ("alpha", "banca", "t2"),
    ])
  assert labels == {"mario|alpha": "gestisce", "alpha|banca": "fa parte di"}


def test_extract_relations_batch_bad_json_returns_empty(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True)
  mock_llm = MagicMock()
  mock_llm.invoke.return_value.content = "non è json"
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    labels = gm._extract_relations_batch([("a", "b", "testo")])
  assert labels == {}


def test_label_pairs_skips_already_labeled(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True,
                    relation_min_count=1)
  g = nx.Graph()
  g.add_node("a", label="x", sources=set())
  g.add_node("b", label="x", sources=set())
  g.add_edge("a", "b", count=2, sources=set(), relation="già etichettata")
  mock_llm = MagicMock()
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    gm._label_pairs(g, {("a", "b"): "qualche testo"})
  mock_llm.invoke.assert_not_called()
  assert g["a"]["b"]["relation"] == "già etichettata"


def test_label_pairs_skips_below_min_count(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True,
                    relation_min_count=2)
  g = nx.Graph()
  g.add_node("a", label="x", sources=set())
  g.add_node("b", label="x", sources=set())
  g.add_edge("a", "b", count=1, sources=set())
  mock_llm = MagicMock()
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    gm._label_pairs(g, {("a", "b"): "testo"})
  mock_llm.invoke.assert_not_called()


def test_label_pairs_batches_calls(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True,
                    relation_min_count=1, relation_batch_size=2)
  g = nx.Graph()
  for n in ("a", "b", "c", "d", "e"):
    g.add_node(n, label="x", sources=set())
  pair_snippets = {}
  for a, b in [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e"), ("b", "c")]:
    g.add_edge(a, b, count=1, sources=set())
    pair_snippets[(a, b)] = "t"
  mock_llm = MagicMock()
  mock_llm.invoke.return_value.content = "{}"
  with patch.object(gm, "_build_llm", return_value=mock_llm):
    gm._label_pairs(g, pair_snippets)
  assert mock_llm.invoke.call_count == 3


def test_relation_stored_on_edge(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, relation_extraction=True)
  g = nx.Graph()
  g.add_node("mario", label="persona", sources=set())
  g.add_node("progetto alpha", label="progetto", sources=set())
  g.add_edge("mario", "progetto alpha", count=1, sources=set())
  gm._apply_relation_labels(g, {"mario|progetto alpha": "gestisce"})
  assert g["mario"]["progetto alpha"].get("relation") == "gestisce"


def test_neighbors_shows_relation(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True)
  g = nx.Graph()
  g.add_node("mario", label="persona", sources={"doc.pdf"})
  g.add_node("progetto alpha", label="progetto", sources={"doc.pdf"})
  g.add_edge("mario", "progetto alpha", count=2, sources={"doc.pdf"}, relation="gestisce")
  gm._save_graph(g)
  result = gm.neighbors("mario")
  assert "gestisce" in result


# ── entity_resolution tests ──────────────────────────────────────────────────

def test_resolve_entity_returns_canonical(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True,
                    entity_resolution=True, resolution_threshold=0.90)
  name_embs = {"mario rossi": [1.0, 0.0, 0.0]}
  mock_embed = MagicMock()
  mock_embed.embed_documents.return_value = [[0.99, 0.1, 0.0]]
  canonical = gm._resolve_entity("mario r.", name_embs, mock_embed)
  assert canonical == "mario rossi"


def test_resolve_entity_new_name_added(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True,
                    entity_resolution=True, resolution_threshold=0.90)
  name_embs = {"mario rossi": [1.0, 0.0, 0.0]}
  mock_embed = MagicMock()
  mock_embed.embed_documents.return_value = [[0.0, 1.0, 0.0]]
  canonical = gm._resolve_entity("banca centrale", name_embs, mock_embed)
  assert canonical == "banca centrale"
  assert "banca centrale" in name_embs


def test_resolve_entity_empty_cache(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True,
                    entity_resolution=True, resolution_threshold=0.90)
  name_embs = {}
  mock_embed = MagicMock()
  mock_embed.embed_documents.return_value = [[0.5, 0.5, 0.0]]
  canonical = gm._resolve_entity("primo ente", name_embs, mock_embed)
  assert canonical == "primo ente"
  assert "primo ente" in name_embs


# ── sentence-level / proximity / PMI tests ───────────────────────────────────

def test_split_sentences_preserves_offsets(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True)
  text = "Mario gestisce Alpha. Luigi gestisce Beta."
  sents = gm._split_sentences(text)
  assert len(sents) == 2
  for sent, start, end in sents:
    assert text[start:end] == sent


def test_proximity_weight_buckets(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True)
  assert gm._proximity_weight(10) == 1.0
  assert gm._proximity_weight(100) == 0.5
  assert gm._proximity_weight(500) == 0.25


def test_compute_pmi_high_for_exclusive_pair(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True)
  g = nx.Graph()
  g.graph["total_sentences"] = 100
  g.add_node("a", mentions=2)
  g.add_node("b", mentions=2)
  g.add_node("hub", mentions=100)
  g.add_edge("a", "b", pair_mentions=2, count=2.0)
  g.add_edge("a", "hub", pair_mentions=2, count=2.0)
  gm._compute_pmi(g)
  assert g["a"]["b"]["pmi"] > g["a"]["hub"]["pmi"]


def test_min_pmi_filters_low_pmi_edges(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True, min_pmi=1.0)
  g = nx.Graph()
  g.add_node("a", label="x", sources={"d.pdf"})
  g.add_node("strong", label="x", sources={"d.pdf"})
  g.add_node("weak", label="x", sources={"d.pdf"})
  g.add_edge("a", "strong", count=2.0, pmi=2.5, sources={"d.pdf"})
  g.add_edge("a", "weak", count=2.0, pmi=0.1, sources={"d.pdf"})
  gm._save_graph(g)
  out = gm.neighbors("a")
  assert "strong" in out
  assert "weak" not in out


def test_graph_round_trip_persists_pmi_and_mentions(tmp_path):
  import networkx as nx
  gm = GraphManager(tmp_path / "graph", enabled=True)
  g = nx.Graph()
  g.graph["total_sentences"] = 42
  g.add_node("a", label="x", sources={"d.pdf"}, mentions=5)
  g.add_node("b", label="x", sources={"d.pdf"}, mentions=3)
  g.add_edge("a", "b", count=1.5, pair_mentions=2, pmi=1.2, sources={"d.pdf"})
  gm._save_graph(g)
  gm._cached_graph = None
  g2 = gm._load_graph()
  assert g2.graph["total_sentences"] == 42
  assert g2.nodes["a"]["mentions"] == 5
  assert g2["a"]["b"]["pmi"] == 1.2
  assert g2["a"]["b"]["pair_mentions"] == 2


# ── spaCy relation extractor tests ────────────────────────────────────────────

def test_spacy_fallback_when_model_missing(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True, spacy_model="zz_nonexistent_model")
  assert gm._load_spacy() is None


def test_spacy_relation_returns_empty_without_nlp(tmp_path):
  gm = GraphManager(tmp_path / "graph", enabled=True, spacy_model="")
  assert gm._load_spacy() is None
