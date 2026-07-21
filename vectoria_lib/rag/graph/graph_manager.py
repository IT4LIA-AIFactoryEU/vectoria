import logging
from collections import OrderedDict
from pathlib import Path

from vectoria_lib.rag.graph import extractors, persistence, relations, resolution, scoring
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME

logger = logging.getLogger("graph")

# Bounds the per-GraphManager query-entity cache (see _extract_query_entities).
# A GraphManager instance lives for the lifetime of one tenant's agent, so this
# also bounds memory for long-lived conversations with many distinct queries.
_QUERY_CACHE_SIZE = 256


class GraphManager:
    def __init__(
        self,
        graph_dir,
        enabled=True,
        extraction_device="cpu",
        model_path="urchade/gliner_multi-v2.1",
        entity_types=None,
        min_entity_confidence=0.5,
        relation_method="cooccurrence",
        min_count=1,
        min_pmi=0.0,
        relation_extraction=False,
        relation_min_count=2,
        relation_batch_size=15,
        spacy_model="it_core_news_sm",
        entity_resolution=False,
        resolution_threshold=0.92,
    ):
        self.enabled = enabled
        self.graph_dir = Path(graph_dir)
        self.extraction_device = extraction_device
        self.model_path = model_path
        self._static_entity_types = entity_types
        self.min_entity_confidence = min_entity_confidence
        self.relation_method = relation_method
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.relation_extraction = relation_extraction
        self.relation_min_count = relation_min_count
        self.relation_batch_size = relation_batch_size
        self.spacy_model = spacy_model
        self.entity_resolution = entity_resolution
        self.resolution_threshold = resolution_threshold
        self._cached_graph = None
        self._cached_mtime = -1
        # LRU cache: normalized query text -> extracted entity list, so repeated
        # or near-repeated graph_search() calls within the same agent lifetime
        # (e.g. a multi-turn conversation) skip the GLiNER forward pass.
        self._query_entity_cache = OrderedDict()
        if self.enabled:
            self.graph_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, graph_dir):
        from vectoria_lib.core.config import Config

        cfg = Config()
        raw = (cfg.config.get("graph_store") if hasattr(cfg, "config") else None) or {}
        return cls(
            graph_dir,
            enabled=raw.get("enabled", False),
            extraction_device=raw.get("extraction_device", "cpu"),
            model_path=raw.get("model_path") or "urchade/gliner_multi-v2.1",
            entity_types=raw.get("entity_types", None),
            min_entity_confidence=raw.get("min_entity_confidence", 0.5),
            relation_method=raw.get("relation_method", "cooccurrence"),
            min_count=raw.get("min_count", 1),
            min_pmi=raw.get("min_pmi", 0.0),
            relation_extraction=raw.get("relation_extraction", False),
            relation_min_count=raw.get("relation_min_count", 2),
            relation_batch_size=raw.get("relation_batch_size", 15),
            spacy_model=raw.get("spacy_model", "it_core_news_sm"),
            entity_resolution=raw.get("entity_resolution", False),
            resolution_threshold=raw.get("resolution_threshold", 0.92),
        )

    def add_chunks(self, chunks):
        if not self.enabled:
            return 0

        entity_types = self._resolve_entity_types(chunks)
        gliner = extractors.load_gliner(self.model_path, self.extraction_device)
        if gliner is None:
            return 0

        try:
            import networkx as nx  # noqa: F401
        except ImportError:
            logger.warning("networkx not installed — install with: uv sync --extra graph")
            return 0

        # Prepare (source, text) pairs up front so GLiNER can run over the whole
        # batch in one call instead of one forward pass per chunk.
        prepared = []
        for chunk in chunks:
            source = chunk.metadata.get(DOC_FILE_NAME, "")
            text = chunk.page_content[:1000].strip()
            if text:
                prepared.append((source, text))
        if not prepared:
            return 0

        try:
            batch_entities = extractors.batch_predict(
                gliner, [t for _, t in prepared], entity_types, threshold=self.min_entity_confidence
            )
        except Exception as exc:
            logger.warning("GLiNER batch extraction failed: %s", exc)
            return 0

        g = self._load_graph()
        n_added = 0
        touched_edges = set()

        embedder = None
        resolution_cache = {}
        if self.entity_resolution:
            from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache
            embedder = WikiEmbeddingCache.get_embedder()
            resolution_cache = resolution.load_resolution_cache(g, embedder)

        pair_snippets = {}
        nlp = self._load_spacy() if self.relation_extraction is False else None

        for (source, text), entities in zip(prepared, batch_entities, strict=True):
            normalized = self._normalize_entities(entities, embedder, resolution_cache)
            n_added += self._ingest_sentences(g, text, source, normalized, pair_snippets, nlp, touched_edges)

        relations_labeled = False
        if self.relation_extraction and pair_snippets:
            self._label_pairs(g, pair_snippets)
            relations_labeled = True

        if n_added or touched_edges or relations_labeled:
            # Recompute PMI only for edges touched by this batch, not the whole
            # graph — see scoring.compute_pmi() for the incremental trade-off.
            self._compute_pmi(g, touched_edges or None)
            self._save_graph(g)
        else:
            logger.debug("Graph: no changes from this batch — skipping PMI recompute and save")

        logger.info("Graph: added %d entity mentions from %d chunks", n_added, len(chunks))
        return n_added

    def _normalize_entities(self, entities, embedder, resolution_cache):
        out = []
        for ent in entities:
            name = ent["text"].strip().lower()
            if len(name) < 2:
                continue
            if self.entity_resolution and embedder:
                name = self._resolve_entity(name, resolution_cache, embedder)
            out.append({
                "name": name,
                "label": ent["label"],
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
            })
        return out

    def _ingest_sentences(self, g, text, source, normalized, pair_snippets, nlp, touched_edges):
        n_added = 0
        for sent_text, sent_start, sent_end in self._split_sentences(text):
            sent_ents = [e for e in normalized if sent_start <= e["start"] < sent_end]
            if not sent_ents:
                continue
            g.graph["total_sentences"] = g.graph.get("total_sentences", 0) + 1

            seen = set()
            for ent in sent_ents:
                name = ent["name"]
                if name not in g.nodes:
                    g.add_node(name, label=ent["label"], sources=set(), mentions=0)
                g.nodes[name]["sources"].add(source)
                if name not in seen:
                    g.nodes[name]["mentions"] = g.nodes[name].get("mentions", 0) + 1
                    seen.add(name)
                    n_added += 1

            # Parse the sentence once and reuse the Doc for every entity pair in
            # it, instead of re-parsing per pair (spacy_relation used to call
            # nlp(sentence) again for every new edge from the same sentence).
            sent_doc = None
            if nlp is not None and len(sent_ents) > 1:
                try:
                    sent_doc = nlp(sent_text)
                except Exception as exc:
                    logger.debug("spaCy parse failed: %s", exc)

            for i, ea in enumerate(sent_ents):
                for eb in sent_ents[i + 1:]:
                    self._add_or_update_edge(g, ea, eb, source, sent_text, pair_snippets, sent_doc, touched_edges)
        return n_added

    def _add_or_update_edge(self, g, ea, eb, source, sent_text, pair_snippets, sent_doc, touched_edges):
        a, b = ea["name"], eb["name"]
        if a == b:
            return
        weight = self._proximity_weight(abs(eb["start"] - ea["end"]))
        if g.has_edge(a, b):
            g[a][b]["count"] = g[a][b].get("count", 1) + weight
            g[a][b]["pair_mentions"] = g[a][b].get("pair_mentions", 0) + 1
            g[a][b]["sources"].add(source)
        else:
            g.add_edge(a, b, count=weight, pair_mentions=1, sources={source}, pmi=0.0)
        touched_edges.add((a, b) if a < b else (b, a))
        if self.relation_extraction:
            key = (a, b) if a < b else (b, a)
            if key not in pair_snippets:
                pair_snippets[key] = sent_text
        elif sent_doc is not None and not g[a][b].get("relation"):
            label = extractors.spacy_relation_from_doc(sent_doc, ea, eb)
            if label:
                g[a][b]["relation"] = label

    def search(self, query, hops=1):
        if not self.enabled:
            return "Graph store disabled."

        g = self._load_graph()
        if g.number_of_nodes() == 0:
            return "Graph store is empty — no entities indexed yet."

        entity_types = self._resolve_entity_types([])
        gliner = extractors.load_gliner(self.model_path, self.extraction_device)

        query_entities = self._extract_query_entities(query, entity_types, gliner) if gliner else []

        if not query_entities:
            query_lower = query.lower()
            query_entities = [n for n in g.nodes if n in query_lower][:5]

        if not query_entities:
            return f"Nessuna entità trovata nella query: '{query}'"

        found_nodes = []
        for qe in query_entities:
            matches = [n for n in g.nodes if qe in n or n in qe]
            found_nodes.extend(matches[:3])
        found_nodes = list(dict.fromkeys(found_nodes))

        if not found_nodes:
            return f"Entità '{', '.join(query_entities)}' non trovate nel grafo."

        neighbor_nodes = self._bfs(g, found_nodes, hops)
        return self._format_search(g, found_nodes, neighbor_nodes)

    def _extract_query_entities(self, query, entity_types, gliner):
        """Run GLiNER over a search query to find entity mentions, cached per
        (normalized query, entity_types) so repeated queries in the same agent
        lifetime (e.g. a multi-turn conversation) don't re-run the model."""
        cache_key = (query[:500].strip().lower(), tuple(entity_types))
        cached = self._query_entity_cache.get(cache_key)
        if cached is not None:
            self._query_entity_cache.move_to_end(cache_key)
            return cached

        try:
            found = gliner.predict_entities(query[:500], entity_types, threshold=self.min_entity_confidence)
            result = [e["text"].strip().lower() for e in found if len(e["text"].strip()) >= 2]
        except Exception as exc:
            logger.warning("GLiNER query extraction failed: %s", exc)
            result = []

        self._query_entity_cache[cache_key] = result
        if len(self._query_entity_cache) > _QUERY_CACHE_SIZE:
            self._query_entity_cache.popitem(last=False)
        return result

    def _bfs(self, g, found_nodes, hops):
        visited = set(found_nodes)
        frontier = list(found_nodes)
        for _ in range(hops):
            new_frontier = []
            for node in frontier:
                new_nbrs = set(
                    n for n in g.neighbors(node)
                    if scoring.edge_passes(g, node, n, self.min_count, self.min_pmi)
                ) - visited
                visited.update(new_nbrs)
                new_frontier.extend(new_nbrs)
            frontier = new_frontier
        return visited - set(found_nodes)

    def _format_search(self, g, found_nodes, neighbor_nodes):
        lines = ["Entità trovate:"]
        for node in found_nodes[:10]:
            data = g.nodes[node]
            sources = ", ".join(sorted(data.get("sources", set())))
            lines.append(f"- **{node}** ({data.get('label', '?')}) — fonti: {sources or 'sconosciuta'}")

        if neighbor_nodes:
            related = []
            for n in sorted(neighbor_nodes)[:15]:
                data = g.nodes.get(n, {})
                edge_data = g.get_edge_data(found_nodes[0], n, default={})
                related.append(f"{n} ({data.get('label', '?')}, peso: {edge_data.get('count', 1)})")
            lines.append(f"\nEntità correlate: {', '.join(related)}")

        all_sources = set()
        for node in list(found_nodes) + list(neighbor_nodes):
            all_sources.update(g.nodes.get(node, {}).get("sources", set()))
        if all_sources:
            lines.append(f"\nDocumenti rilevanti: {', '.join(sorted(all_sources))}")

        return "\n".join(lines)

    def neighbors(self, entity):
        if not self.enabled:
            return "Graph store disabled."

        g = self._load_graph()
        entity_lower = entity.strip().lower()

        if entity_lower not in g.nodes:
            matches = [n for n in g.nodes if entity_lower in n]
            if not matches:
                return f"Entità '{entity}' non trovata nel grafo."
            entity_lower = matches[0]

        nbrs = [n for n in g.neighbors(entity_lower)
                if scoring.edge_passes(g, entity_lower, n, self.min_count, self.min_pmi)]
        if not nbrs:
            return f"'{entity_lower}' non ha connessioni nel grafo."

        result = []
        for n in sorted(nbrs)[:20]:
            data = g.nodes.get(n, {})
            edge = g.get_edge_data(entity_lower, n, default={})
            rel = edge.get("relation", "")
            rel_str = f", relazione: {rel}" if rel else ""
            count = edge.get("count", 1)
            count_str = f"{count:.1f}" if isinstance(count, float) else str(count)
            result.append(f"- {n} ({data.get('label', '?')}, peso: {count_str}{rel_str})")

        return f"Entità collegate a '{entity_lower}':\n" + "\n".join(result)

    def delete_by_source(self, filename):
        if not self.enabled:
            return 0
        g = self._load_graph()
        to_remove = []
        for node, data in g.nodes(data=True):
            sources = data.get("sources", set())
            sources.discard(filename)
            if not sources:
                to_remove.append(node)
        g.remove_nodes_from(to_remove)
        self._save_graph(g)
        logger.info("Graph: removed %d nodes sourced only from '%s'", len(to_remove), filename)
        return len(to_remove)

    def node_count(self):
        if not self.enabled:
            return 0
        return self._load_graph().number_of_nodes()

    # ── Test-facing wrappers around module helpers ──────────────────────────

    def _split_sentences(self, text):
        return scoring.split_sentences(text)

    def _proximity_weight(self, distance):
        return scoring.proximity_weight(distance)

    def _compute_pmi(self, g, edges=None):
        scoring.compute_pmi(g, edges)

    def _load_spacy(self):
        return extractors.load_spacy(self.spacy_model)

    def _build_llm(self):
        return relations.build_llm()

    def _extract_relations_batch(self, items):
        return relations.extract_relations_batch(items, llm_factory=self._build_llm)

    def _apply_relation_labels(self, g, labels):
        relations.apply_relation_labels(g, labels)

    def _label_pairs(self, g, pair_snippets):
        return relations.label_pairs(
            g, pair_snippets,
            self.relation_min_count, self.relation_batch_size,
            llm_factory=self._build_llm,
        )

    def _resolve_entity(self, name, name_embs, embedder):
        return resolution.resolve_entity(name, name_embs, embedder, self.resolution_threshold)

    def _resolve_entity_types(self, chunks):
        if self._static_entity_types:
            return self._static_entity_types
        saved = persistence.load_entity_types(self.graph_dir)
        if saved:
            return saved
        if not chunks:
            return relations.DEFAULT_ENTITY_TYPES
        inferred = relations.infer_entity_types(chunks, llm_factory=self._build_llm)
        if inferred:
            persistence.save_entity_types(self.graph_dir, inferred)
            logger.info("Graph: inferred entity types: %s", inferred)
            return inferred
        return relations.DEFAULT_ENTITY_TYPES

    # ── Graph IO with caching ───────────────────────────────────────────────

    def _load_graph(self):
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx not installed — install with: uv sync --extra graph")
            return persistence.EmptyGraph()
        g, mtime = persistence.load_graph_data(self.graph_dir)
        if g is None:
            if self._cached_graph is None:
                self._cached_graph = nx.Graph()
                self._cached_mtime = 0
            return self._cached_graph
        if self._cached_graph is not None and mtime == self._cached_mtime:
            return self._cached_graph
        self._cached_graph = g
        self._cached_mtime = mtime
        return self._cached_graph

    def _save_graph(self, g):
        self._cached_mtime = persistence.save_graph(self.graph_dir, g)
        self._cached_graph = g
