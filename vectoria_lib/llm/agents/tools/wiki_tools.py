import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger("wiki")


class _QueryInput(BaseModel):
    query: str


class WikiStatusTool(BaseTool):
    name: str = "wiki_status"
    description: str = (
        "Check the Wiki knowledge base: existing pages and how many new concepts "
        "are pending synthesis. Call proactively when the user asks about the wiki, "
        "after a document is uploaded, or at the start of a session to surface "
        "a synthesis suggestion."
    )
    wiki_dir: Any  # Path

    def _run(self, query: str = "") -> str:
        from vectoria_lib.wiki.wiki_manager import WikiManager

        wm = WikiManager.from_config(self.wiki_dir)
        pages = wm.list_pages()
        pending_summary = wm.pending_summary()
        pages_str = (
            f"{len(pages)} page(s): {', '.join(pages[:10])}{'...' if len(pages) > 10 else ''}"
            if pages
            else "No wiki pages yet."
        )
        suggestion = (
            "\n⚠️ Synthesis threshold reached — consider calling wiki_synthesize." if wm.should_synthesize() else ""
        )
        return f"Wiki — {pages_str}\nPending: {pending_summary}{suggestion}"


class WikiQueryTool(BaseTool):
    name: str = "wiki_query"
    description: str = (
        "Search the Wiki knowledge base for a topic or concept. "
        "Call this BEFORE the retriever when the user asks about a concept "
        "that may already be synthesised as a structured wiki page."
    )
    args_schema: Any = _QueryInput
    wiki_dir: Any  # Path

    def _run(self, query: str) -> str:
        from vectoria_lib.wiki.wiki_manager import WikiManager

        wm = WikiManager.from_config(self.wiki_dir)
        page = wm.get_page(query)
        if page:
            return f"Wiki page found:\n\n{page}"
        return wm.search_pages(query)


class WikiSynthesisTool(BaseTool):
    name: str = "wiki_synthesize"
    description: str = (
        "Synthesize new Wiki pages from pending concepts using the LLM. "
        "Call when the user confirms they want to update the wiki, "
        "or when wiki_status reports the threshold has been reached."
    )
    wiki_dir: Any  # Path
    index_dir: Any  # Path — tenant index root used to locate the vector store

    def _run(self, query: str = "") -> str:
        from vectoria_lib.core.config import Config
        from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
        from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder
        from vectoria_lib.wiki.wiki_manager import WikiManager

        config = Config()
        wm = WikiManager.from_config(self.wiki_dir)
        if not wm.enabled:
            return "Wiki is disabled in configuration."

        # Locate vector store (needed for both re-seeding and page synthesis)
        index_path = _find_index_path(Path(self.index_dir))
        vs = None
        retriever = _NullRetriever()
        if index_path:
            vs = VectorStoreBuilder().build(config.get("vector_store"), index_path=index_path)
            retriever = vs.as_retriever(search_kwargs={"k": 4})
        else:
            logger.warning("WikiSynthesisTool: no index found at %s", self.index_dir)

        # If pending is empty, re-seed from all documents in the vector store
        if wm.pending_count() == 0:
            if vs is None:
                pages = wm.list_pages()
                return f"Nothing to synthesize — no index found and wiki already has {len(pages)} page(s)."
            try:
                all_docs = vs.get_all_documents()
                if not all_docs:
                    pages = wm.list_pages()
                    return f"Nothing to synthesize — index is empty, wiki has {len(pages)} page(s)."
                n = wm.add_from_docs(all_docs)
                logger.info("WikiSynthesisTool: re-seeded %d snippets from vector store", n)
            except Exception as exc:
                logger.warning("WikiSynthesisTool: re-seed failed: %s", exc)
                pages = wm.list_pages()
                return f"Could not re-seed wiki (no pending data): {exc}"

        summary_before = wm.pending_summary()
        llm = InferenceEngineBuilder.build_inference_engine_no_thinking(config.get("inference_engine")).as_langchain_chat_model()

        created = wm.synthesize(llm, retriever)

        if not created:
            return f"Synthesis attempted but no pages were generated (from: {summary_before})."

        return (
            f"Wiki updated — {len(created)} new page(s) created: {', '.join(created[:10])}.\n"
            f"Processed from: {summary_before}"
        )


from vectoria_lib.rag.vector_store.discovery import find_index_path as _find_index_path


class _NullRetriever:
    """Fallback retriever used when no vector index exists yet."""

    def invoke(self, query: str) -> list:
        return []


def make_wiki_tools(index_dir: Path) -> list[BaseTool]:
    """Return the three wiki tools for a given tenant index directory.

    Returns an empty list when the wiki feature is disabled in config, so the
    agent never even sees the tools.
    """
    from vectoria_lib.wiki.wiki_manager import WikiManager

    wiki_dir = Path(index_dir) / "wiki"
    if not WikiManager.from_config(wiki_dir).enabled:
        logger.debug("Wiki disabled by config — skipping wiki tools for %s", index_dir)
        return []
    return [
        WikiStatusTool(wiki_dir=wiki_dir),
        WikiQueryTool(wiki_dir=wiki_dir),
        WikiSynthesisTool(wiki_dir=wiki_dir, index_dir=index_dir),
    ]
