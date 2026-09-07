"""Inference service — FastAPI stub.

Phase 1: Returns a hard-coded stub response so the orchestrator''s end-to-end
pipeline can be exercised without a GPU or a real model checkpoint.

# STUB: Replace the body of ``_run_vqa`` with a real Qwen2.5-VL call in the
# next phase once the model weights are available.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SatQuery-AI Inference Service",
    description=(
        "Hosts the vision-language model (Qwen2.5-VL) for satellite-imagery VQA. "
        "Phase-1 stub — real model not wired yet."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class VQARequest(BaseModel):
    """Payload sent by the orchestrator for a VQA task."""

    image_paths: list[str] = Field(
        ..., description="Absolute paths to the raster files to analyse."
    )
    query: str = Field(..., description="Natural-language question.")
    task: str = Field(default="vqa", description="Task identifier.")


class VQAResponse(BaseModel):
    """Response returned to the orchestrator."""

    answer: str = Field(..., description="Model answer.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1].")


# ---------------------------------------------------------------------------
# Internal model runner (stub)
# ---------------------------------------------------------------------------


def _run_vqa(image_paths: list[str], query: str) -> dict[str, Any]:
    """Run VQA inference.

    # STUB: Replace with real Qwen2.5-VL inference in the next phase.
    #       Expected implementation outline:
    #           1. Load each raster with rasterio / PIL and convert to RGB.
    #           2. Tokenise with Qwen2_5_VLProcessor.
    #           3. Run model.generate() with the query and image tensors.
    #           4. Decode the output tokens to a string.
    #           5. Compute or estimate a confidence score.
    """
    logger.info(
        "VQA stub called: %d image(s), query='%s'", len(image_paths), query[:80]
    )
    return {
        "answer": "stub response - real model not wired yet",
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "inference"}


@app.post("/vqa", response_model=VQAResponse, tags=["inference"])
def vqa(request: VQARequest) -> VQAResponse:
    """Run Visual Question Answering on the supplied satellite images.

    Parameters
    ----------
    request:
        Contains ``image_paths``, ``query``, and ``task``.

    Returns
    -------
    VQAResponse
        ``answer`` (str) and ``confidence`` (float in [0, 1]).
    """
    if not request.image_paths:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_paths must not be empty.",
        )
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query must not be empty.",
        )

    result = _run_vqa(image_paths=request.image_paths, query=request.query)
    return VQAResponse(**result)
