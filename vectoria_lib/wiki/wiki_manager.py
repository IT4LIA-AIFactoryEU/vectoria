import logging
from datetime import datetime
from pathlib import Path

from vectoria_lib.core.config_schema import WikiConfig
from vectoria_lib.rag.metadata_keys import DOC_FILE_NAME
from vectoria_lib.wiki.wiki_embedding_cache import WikiEmbeddingCache
from vectoria_lib.wiki.wiki_storage import WikiStorage
from vectoria_lib.wiki.wiki_synthesizer import WikiSynthesizer

logger = logging.getLogger("wiki")


class WikiManager:
    def __init__(self, wiki_dir: Path, config: WikiConfig | None = None):
        self._config = config or WikiConfig()
        self.enabled = self._config.enabled
        self.pending_threshold = self._config.pending_threshold
        if not self.enabled:
            return
        self._storage = WikiStorage(Path(wiki_dir))
        self._emb_cache = WikiEmbeddingCache(Path(wiki_dir))
        self._synthesizer = WikiSynthesizer(
            storage=self._storage,
            emb_cache=self._emb_cache,
            config=self._config,
        )

    @classmethod
    def from_config(cls, wiki_dir: Path) -> "WikiManager":
        from vectoria_lib.core.config import Config

        cfg = Config()
        raw = (cfg.config.get("wiki") if hasattr(cfg, "config") else None) or {}
        return cls(wiki_dir, config=WikiConfig(**(raw or {})))

    def add_from_docs(self, docs: list) -> int:
        if not self.enabled:
            return 0
        existing = self._storage.load_pending()
        existing_snippets = {p["snippet"] for p in existing}

        new_entries: list[dict] = []
        seen: set[str] = set()
        for doc in docs:
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            if len(snippet) < 20:
                continue
            source = doc.metadata.get(DOC_FILE_NAME, "")
            key = snippet[:80]
            if key in seen or snippet in existing_snippets:
                continue
            seen.add(key)
            new_entries.append(
                {
                    "source": source,
                    "snippet": snippet,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        if new_entries:
            existing.extend(new_entries)
            self._storage.save_pending(existing)
            logger.info("Added %d pending snippets for wiki synthesis", len(new_entries))

        return len(new_entries)

    def pending_count(self) -> int:
        if not self.enabled:
            return 0
        return len(self._storage.load_pending())

    def should_synthesize(self) -> bool:
        return self.enabled and self.pending_count() >= self.pending_threshold

    def pending_summary(self) -> str:
        if not self.enabled:
            return "Wiki disabled."
        pending = self._storage.load_pending()
        if not pending:
            return "0 pending snippets."
        sources = list({p.get("source", "?") for p in pending})
        tail = "..." if len(sources) > 3 else ""
        return f"{len(pending)} snippet(s) da: {', '.join(sources[:3])}{tail}"

    def list_pages(self) -> list[str]:
        if not self.enabled:
            return []
        return self._storage.list_pages()

    def get_page(self, name: str) -> str | None:
        if not self.enabled:
            return None
        return self._storage.get_page(name)

    def search_pages(self, query: str) -> str:
        if not self.enabled:
            return "Wiki disabled."
        return self._storage.search_pages(query)

    def synthesize(self, llm, retriever, max_per_doc: int | None = None) -> list[str]:
        if not self.enabled:
            return []
        pending = self._storage.load_pending()
        created = self._synthesizer.synthesize(llm, retriever, pending, max_per_doc)
        self._storage.save_pending([])
        logger.info("Synthesized %d wiki pages", len(created))
        return created

    def delete_by_source(self, filename: str) -> int:
        if not self.enabled:
            return 0
        pending = [p for p in self._storage.load_pending() if p.get("source") != filename]
        self._storage.save_pending(pending)

        sources = self._storage.load_sources()
        deleted = 0
        for stem, src in list(sources.items()):
            parts = [s.strip() for s in src.split(",") if s.strip()]
            if filename in parts:
                parts.remove(filename)
                if parts:
                    sources[stem] = ", ".join(parts)
                    continue
                self._storage.delete_page(stem)
                self._emb_cache.remove(stem)
                del sources[stem]
                deleted += 1
                logger.info("Deleted wiki page '%s' (source: %s)", stem, filename)
        if deleted:
            self._storage.save_sources(sources)
        return deleted
