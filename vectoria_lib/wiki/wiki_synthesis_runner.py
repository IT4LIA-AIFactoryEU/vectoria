#
# VECTORIA
#
# @authors : Luca Babetto, Leonardo Baroncelli, Carolina Berucci, Chiara Malizia, Eric Pascolo, Andrea Proia, Mattia Buzzoni, Michele Visciarelli
#
# This file is distributed under the MIT License.
# See the LICENSE file in the repository root for the full license text.
#
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger("wiki")


def run_wiki_synthesis(
    index_dir: Path,
    on_complete: Callable[[str, int], None],
) -> None:
    """Run wiki synthesis synchronously (designed to run in a background thread).

    Calls on_complete("idle", n_created) on success, on_complete("failed", 0) on error.
    """
    from vectoria_lib.core.config import Config
    from vectoria_lib.llm.agents.tools.wiki_tools import _NullRetriever, _find_index_path
    from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
    from vectoria_lib.rag.vector_store.vector_store_builder import VectorStoreBuilder
    from vectoria_lib.wiki.wiki_manager import WikiManager

    try:
        wm = WikiManager.from_config(index_dir / "wiki")
        if not wm.should_synthesize():
            on_complete("idle", 0)
            return

        cfg = Config()
        llm = InferenceEngineBuilder.build_inference_engine_no_thinking(cfg.get("inference_engine")).as_langchain_chat_model()
        ip = _find_index_path(index_dir)
        if ip:
            vs = VectorStoreBuilder().build(cfg.get("vector_store"), index_path=ip)
            retriever = vs.as_retriever(search_kwargs={"k": wm._config.synthesis_context_chunks})
        else:
            retriever = _NullRetriever()

        n_created = len(wm.synthesize(llm, retriever))
        on_complete("idle", n_created)
        if n_created:
            logger.info("Wiki synthesis: %d page(s) created", n_created)
    except Exception as exc:
        on_complete("failed", 0)
        logger.warning("Wiki synthesis failed: %s", exc)
