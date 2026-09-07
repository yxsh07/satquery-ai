"""VQA (Visual Question Answering) node for the SatQuery-AI LangGraph pipeline.

Calls the inference service and appends the answer to the conversation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from orchestrator.graph.state import AgentState

logger = logging.getLogger(__name__)

# Base URL of the inference microservice.  Overridden by the INFERENCE_URL
# environment variable so the same code works locally and in Docker.
INFERENCE_URL: str = os.getenv("INFERENCE_URL", "http://localhost:8001").rstrip("/")

# Seconds to wait for the inference service to respond.
_REQUEST_TIMEOUT: int = int(os.getenv("INFERENCE_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Inference client
# ---------------------------------------------------------------------------


def call_inference_api(
    image_paths: list[str],
    query: str,
    task: str = "vqa",
) -> dict[str, Any]:
    """POST to the inference service and return the parsed JSON response.

    Parameters
    ----------
    image_paths:
        Absolute paths to the raster files that were already validated by
        the ingestion node.
    query:
        The natural-language question to answer.
    task:
        Task identifier forwarded to the inference service.  Currently only
        ``"vqa"`` is supported.

    Returns
    -------
    dict
        Parsed JSON body; must contain at minimum ``"answer"`` (str) and
        ``"confidence"`` (float).

    Raises
    ------
    RuntimeError
        If the inference service is unreachable or returns a non-200 status.
    """
    endpoint = f"{INFERENCE_URL}/{task}"
    payload: dict[str, Any] = {
        "image_paths": image_paths,
        "query": query,
        "task": task,
    }

    logger.info("Calling inference API: POST %s  paths=%s", endpoint, image_paths)

    try:
        response = requests.post(endpoint, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach inference service at '{endpoint}'. "
            "Is the inference container running?"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Inference service returned an error: {exc.response.status_code} "
            f"{exc.response.text}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            f"Inference service timed out after {_REQUEST_TIMEOUT}s."
        ) from exc

    data: dict[str, Any] = response.json()

    # Validate minimum contract.
    if "answer" not in data or "confidence" not in data:
        raise RuntimeError(
            f"Inference service response missing required fields: {data}"
        )

    return data


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------


def execute_vqa(state: AgentState) -> dict[str, Any]:
    """LangGraph node: run VQA against the inference service.

    Reads ``raw_images`` and the last element of ``messages`` as the query.
    Appends the model answer to ``messages`` and records a trace entry.

    Parameters
    ----------
    state:
        Current agent state.

    Returns
    -------
    dict
        Partial state update with appended ``messages`` and ``execution_trace``.
    """
    image_paths: list[str] = state["raw_images"]
    messages: list[str] = state["messages"]

    if not messages:
        raise ValueError("No query found in state['messages'].")

    # The user query is always the first message.
    query: str = messages[0]

    result = call_inference_api(image_paths=image_paths, query=query, task="vqa")

    answer: str = result["answer"]
    confidence: float = float(result["confidence"])

    assistant_message = f"{answer}  (confidence: {confidence:.2f})"

    trace_entry: dict[str, Any] = {
        "node": "vqa",
        "result": {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "image_paths": image_paths,
        },
    }

    return {
        "messages": [assistant_message],
        "execution_trace": [trace_entry],
        "final_response": {
            "answer": answer,
            "confidence": confidence,
        },
    }
