"""LangGraph graph definition and public ``run_graph`` entry-point.

Graph topology (Phase 1 — linear)
-----------------------------------
    START -> ingest_and_validate -> execute_vqa -> END

Conditional routing will be introduced in a later phase when task_type
resolution and spatial-analysis nodes are added.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from orchestrator.graph.nodes.ingestion import ingest_and_validate
from orchestrator.graph.nodes.vqa import execute_vqa
from orchestrator.graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

_builder: StateGraph = StateGraph(AgentState)

_builder.add_node("ingest_and_validate", ingest_and_validate)
_builder.add_node("execute_vqa", execute_vqa)

_builder.add_edge(START, "ingest_and_validate")
_builder.add_edge("ingest_and_validate", "execute_vqa")
_builder.add_edge("execute_vqa", END)

# Compile once at import time so ``run_graph`` has zero setup overhead.
_graph = _builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_graph(image_paths: list[str], query: str) -> dict[str, Any]:
    """Execute the SatQuery-AI agent graph and return the final state.

    Parameters
    ----------
    image_paths:
        Absolute paths to the uploaded raster files saved by the API layer.
    query:
        The natural-language question supplied by the user.

    Returns
    -------
    dict
        The complete ``AgentState`` after graph execution, serialisable to JSON.
        Key fields: ``messages``, ``execution_trace``, ``final_response``,
        ``input_config``, ``image_metadata``.
    """
    initial_state: AgentState = {
        "messages": [query],
        "raw_images": image_paths,
        "image_metadata": {},
        "input_config": "unknown",
        "task_type": "",
        "spatial_outputs": {},
        "execution_trace": [],
        "final_response": {},
    }

    logger.info(
        "Running graph: %d image(s), query='%s'", len(image_paths), query[:80]
    )

    final_state: dict[str, Any] = _graph.invoke(initial_state)
    return final_state
