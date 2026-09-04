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
from pathlib import Path

logger = logging.getLogger("wiki")

_PAGE_EMB_FILE = "page_embeddings.json"


class WikiEmbeddingCache:
  def __init__(self, wiki_dir: Path):
    self._path = Path(wiki_dir) / _PAGE_EMB_FILE

  def load(self) -> dict[str, list[float]]:
    if not self._path.exists():
      return {}
    try:
      return json.loads(self._path.read_text(encoding="utf-8"))
    except Exception:
      return {}

  def save(self, cache: dict[str, list[float]]):
    self._path.parent.mkdir(parents=True, exist_ok=True)
    self._path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

  def remove(self, page_name: str):
    cache = self.load()
    if cache.pop(page_name, None) is not None:
      self.save(cache)

  def load_or_compute(self, embed, pages: list[str]) -> list[list[float]]:
    if not pages:
      return []
    cache = self.load()
    missing = [p for p in pages if p not in cache]
    stale = set(cache) - set(pages)
    if missing:
      new_embs = embed.embed_documents(missing)
      cache.update(zip(missing, new_embs, strict=True))
    for s in stale:
      cache.pop(s, None)
    if missing or stale:
      self.save(cache)
    return [cache[p] for p in pages]

  @staticmethod
  def best_match(
    query_emb: list[float],
    pool: list[list[float]],
  ) -> tuple[int | None, float]:
    if not pool:
      return None, 0.0
    try:
      import numpy as np

      q = np.asarray(query_emb, dtype=np.float32)
      m = np.asarray(pool, dtype=np.float32)
      qn = np.linalg.norm(q) + 1e-9
      mn = np.linalg.norm(m, axis=1) + 1e-9
      sims = (m @ q) / (mn * qn)
      i = int(sims.argmax())
      return i, float(sims[i])
    except Exception as exc:
      logger.debug("Similarity computation failed: %s", exc)
      return None, 0.0

  @staticmethod
  def get_embedder():
    try:
      from vectoria_lib.core.config import Config
      from vectoria_lib.rag.vector_store.vector_store_base import (
        get_shared_embedder,
        normalize_model_name,
      )

      vs_cfg = Config().get("vector_store")
      return get_shared_embedder(
        vs_cfg["model_name"],
        vs_cfg.get("device", "cpu"),
        vs_cfg.get("normalize_embeddings", True),
        batch_size=vs_cfg.get("embedding_batch_size", 64),
      )
    except Exception as exc:
      logger.debug("Could not load shared embedder for wiki dedup: %s", exc)
      return None
