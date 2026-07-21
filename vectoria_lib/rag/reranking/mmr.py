"""MMR (Maximal Marginal Relevance) post-retrieval diversification step.

Reduces semantic redundancy among ranked docs by trading off relevance to the
query against dissimilarity to already-selected docs.

  score(d) = lambda * sim(q, d) - (1 - lambda) * max sim(d, chosen)

Lambda 1.0 = pure relevance. Lambda 0.0 = pure diversity. Default 0.7.
"""

import logging

logger = logging.getLogger("rag")


def mmr_select(query_emb, doc_embs, docs, top_k, lambda_):
  """Greedy MMR selection.

  query_emb: 1D vector (numpy or list)
  doc_embs: list of 1D vectors, same order as docs
  docs: list of Document objects
  top_k: target output size
  lambda_: relevance/diversity tradeoff, 0..1
  """
  try:
    import numpy as np
  except ImportError:
    logger.warning("numpy not available — MMR disabled")
    return docs[:top_k]

  if not docs or top_k <= 0:
    return []
  if len(docs) <= top_k:
    return list(docs)

  q = np.asarray(query_emb, dtype=np.float32)
  m = np.asarray(doc_embs, dtype=np.float32)
  qn = np.linalg.norm(q) + 1e-9
  mn = np.linalg.norm(m, axis=1) + 1e-9
  q_sims = (m @ q) / (mn * qn)

  selected_idx = []
  remaining = set(range(len(docs)))
  for _ in range(min(top_k, len(docs))):
    best_score = -1e9
    best_i = -1
    for i in remaining:
      if not selected_idx:
        diversity_penalty = 0.0
      else:
        chosen = m[selected_idx]
        sims = (chosen @ m[i]) / (np.linalg.norm(chosen, axis=1) * mn[i] + 1e-9)
        diversity_penalty = float(sims.max())
      score = lambda_ * float(q_sims[i]) - (1.0 - lambda_) * diversity_penalty
      if score > best_score:
        best_score = score
        best_i = i
    selected_idx.append(best_i)
    remaining.discard(best_i)

  return [docs[i] for i in selected_idx]


def apply_mmr(query, docs, embedder, top_k, lambda_):
  """Embed query + docs on-the-fly and apply MMR. Returns reordered/trimmed docs."""
  if not docs or embedder is None:
    return docs[:top_k] if top_k > 0 else docs
  try:
    texts = [d.page_content for d in docs]
    doc_embs = embedder.embed_documents(texts)
    query_emb = embedder.embed_query(query)
    return mmr_select(query_emb, doc_embs, docs, top_k, lambda_)
  except Exception as exc:
    logger.warning("MMR failed (%s) — returning input docs unchanged", exc)
    return docs[:top_k] if top_k > 0 else docs
