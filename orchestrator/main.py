"""Orchestrator FastAPI application.

Endpoints
---------
POST /query
    Accept one or more raster file uploads + a ``query`` form field.
    Save the files to a per-request temp directory, run the LangGraph
    pipeline, and return the agent''s messages and execution_trace as JSON.

GET /health
    Simple liveness probe used by Docker health checks and load-balancers.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orchestrator.graph.router import run_graph

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
    title="SatQuery-AI Orchestrator",
    description=(
        "Agentic satellite-imagery question-answering system. "
        "Upload one or more GeoTIFF/raster images and ask a question."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Allowed raster extensions
# ---------------------------------------------------------------------------

_ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {".tif", ".tiff", ".img", ".vrt", ".nc", ".hdf", ".h5", ".png", ".jpg", ".jpeg"}
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
def health_check() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "orchestrator"}


@app.post("/query", tags=["query"])
async def query(
    files: list[UploadFile] = File(..., description="One or more raster image files"),
    query: str = Form(..., description="Natural-language question about the image(s)"),
) -> JSONResponse:
    """Run the SatQuery-AI pipeline on uploaded raster(s) and return the result.

    Returns a JSON object with:
    - ``messages``: list of conversation turns (user query + model answer)
    - ``execution_trace``: per-node audit log
    - ``input_config``: detected image configuration
    - ``final_response``: answer + confidence dict from the VQA node
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one image file must be uploaded.",
        )

    query_str = query.strip()
    if not query_str:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query string must not be empty.",
        )

    # Create a unique temp directory for this request so concurrent requests
    # never collide on file paths.
    request_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.gettempdir()) / "satquery" / request_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    try:
        for upload in files:
            filename = upload.filename or f"image_{uuid.uuid4().hex}"
            suffix = Path(filename).suffix.lower()

            if suffix not in _ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=(
                        f"File '{filename}' has unsupported extension '{suffix}'. "
                        f"Allowed: {sorted(_ALLOWED_SUFFIXES)}"
                    ),
                )

            dest = tmp_dir / filename
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)

            saved_paths.append(str(dest))
            logger.info("Saved upload '%s' → %s", filename, dest)

        # -------------------------------------------------------------------
        # Run the LangGraph pipeline
        # -------------------------------------------------------------------
        try:
            final_state: dict[str, Any] = run_graph(
                image_paths=saved_paths, query=query_str
            )
        except ValueError as exc:
            # User-facing validation errors from ingestion node.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except RuntimeError as exc:
            # Inference service unavailable / returned error.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        response_body: dict[str, Any] = {
            "request_id": request_id,
            "messages": final_state.get("messages", []),
            "execution_trace": final_state.get("execution_trace", []),
            "input_config": final_state.get("input_config", "unknown"),
            "final_response": final_state.get("final_response", {}),
        }

        return JSONResponse(content=response_body, status_code=status.HTTP_200_OK)

    finally:
        # Clean up temp files regardless of success or failure.
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temp dir: %s", tmp_dir)
