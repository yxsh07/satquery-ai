"""LangGraph shared state schema for the SatQuery-AI agent."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Immutable-style state bag passed between LangGraph nodes.

    Fields
    ------
    messages:
        Conversation-turn messages (user + assistant).  Each entry is a plain
        string for now; later phases can switch to ``langchain_core.messages``
        objects.  The ``operator.add`` reducer means each node appends rather
        than replaces.
    raw_images:
        Absolute file-system paths to the uploaded raster images.
    image_metadata:
        Keyed by path; value is the dict returned by the ingestion node
        (crs, bbox, band_count, acquisition_date, …).
    input_config:
        Classifier set by the ingestion node:
        ``"single"`` | ``"cross_modal"`` | ``"bi_temporal"`` | ``"unknown"``.
    task_type:
        High-level task label resolved by a routing node (future phase).
        Starts empty.
    spatial_outputs:
        Keyed, JSON-serialisable spatial results (GeoJSON, stats, …).
        Populated by future spatial-analysis nodes.
    execution_trace:
        Append-only log of ``{"node": str, "result": …}`` dicts.
        The ``operator.add`` reducer means each node extends the list.
    final_response:
        The assembled JSON response returned to the caller.
    """

    messages: Annotated[list[str], operator.add]
    raw_images: list[str]
    image_metadata: dict[str, dict]
    input_config: str  # "single" | "cross_modal" | "bi_temporal" | "unknown"
    task_type: str
    spatial_outputs: dict[str, object]
    execution_trace: Annotated[list[dict], operator.add]
    final_response: dict[str, object]
