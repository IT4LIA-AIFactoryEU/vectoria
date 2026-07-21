def resolve_entity(name, name_embs, embedder, threshold):
    from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache

    vec = embedder.embed_documents([name])[0]
    if name_embs:
        existing = list(name_embs.keys())
        pool = [name_embs[k] for k in existing]
        i, sim = WikiEmbeddingCache.best_match(vec, pool)
        if i is not None and sim >= threshold:
            return existing[i]
    name_embs[name] = vec
    return name


def load_resolution_cache(g, embedder):
    if not embedder or g.number_of_nodes() == 0:
        return {}
    existing = list(g.nodes)
    vecs = embedder.embed_documents(existing)
    return dict(zip(existing, vecs))
