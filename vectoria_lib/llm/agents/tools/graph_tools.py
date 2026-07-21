import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

logger = logging.getLogger("graph")


class _QueryInput(BaseModel):
    query: str


class _EntityInput(BaseModel):
    entity: str


class GraphSearchTool(BaseTool):
    name: str = "graph_search"
    description: str = (
        "Search the knowledge graph for entities related to the query. "
        "Returns entities, their type, relationships, and the documents they appear in. "
        "Call this BEFORE the retriever when the query involves relationships between "
        "people, projects, products, organizations, or regulations. "
        "Use the returned document names to guide a follow-up retriever call."
    )
    args_schema: Any = _QueryInput
    manager: Any

    def _run(self, query: str) -> str:
        return self.manager.search(query)


class GraphNeighborsTool(BaseTool):
    name: str = "graph_neighbors"
    description: str = (
        "Find all entities directly connected to a specific entity in the knowledge graph. "
        "Call this to explore the immediate relationships of a known entity "
        "(e.g. all projects linked to a person, all regulations linked to a product)."
    )
    args_schema: Any = _EntityInput
    manager: Any

    def _run(self, entity: str) -> str:
        return self.manager.neighbors(entity)


def make_graph_tools(index_dir: Path) -> list[BaseTool]:
    from vectoria_lib.rag.graph.graph_manager import GraphManager

    graph_dir = Path(index_dir) / "graph"
    manager = GraphManager.from_config(graph_dir)
    if not manager.enabled:
        logger.debug("Graph store disabled by config — skipping graph tools for %s", index_dir)
        return []
    return [
        GraphSearchTool(manager=manager),
        GraphNeighborsTool(manager=manager),
    ]
