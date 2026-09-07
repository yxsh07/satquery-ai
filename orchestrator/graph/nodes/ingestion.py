"""Ingestion and validation node for the SatQuery-AI LangGraph pipeline.

Responsibilities
----------------
* Open each raster in ``state["raw_images"]`` with *rasterio*.
* Extract CRS, bounding box (in EPSG:4326), band count, and optional
  ``acquisition_date`` tag.
* Classify ``input_config`` based on overlap / band / date relationships.
* Append a structured entry to ``execution_trace``.
* Raise ``ValueError`` with user-facing messages for unsupported inputs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from orchestrator.graph.state import AgentState

logger = logging.getLogger(__name__)

_WGS84 = "EPSG:4326"

# Tolerance for comparing bounding-box corners (degrees).
_BBOX_TOL: float = 1e-4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_wgs84_bbox(dataset: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS-84."""
    src_crs: CRS = dataset.crs
    if src_crs is None:
        raise ValueError(
            f"Image '{dataset.name}' has no CRS defined. "
            "Please provide a georeferenced raster."
        )
    bounds = dataset.bounds
    west, south, east, north = transform_bounds(
        src_crs, _WGS84, bounds.left, bounds.bottom, bounds.right, bounds.top
    )
    return (west, south, east, north)


def _bboxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return True if two WGS-84 bboxes have a non-trivial intersection."""
    w_a, s_a, e_a, n_a = a
    w_b, s_b, e_b, n_b = b
    return e_a > w_b and e_b > w_a and n_a > s_b and n_b > s_a


def _bboxes_equal(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return True if two bboxes are effectively identical (within tolerance)."""
    return all(abs(ai - bi) < _BBOX_TOL for ai, bi in zip(a, b))


def _parse_acquisition_date(tags: dict[str, str]) -> str | None:
    """Try common tag keys for an acquisition date; return ISO string or None."""
    candidate_keys = [
        "acquisition_date",
        "TIFFTAG_DATETIME",
        "date",
        "DATE_ACQUIRED",
        "sensing_time",
    ]
    for key in candidate_keys:
        value = tags.get(key)
        if value:
            return value.strip()
    return None


def _classify_config(
    meta: dict[str, dict],
) -> str:
    """Classify input_config from the extracted metadata dict.

    Rules (pairwise, first match wins):
    ┌──────────────────────────────────────────────────────────────────┐
    │ single image          → "single"                                 │
    │ same bbox + same date + different bands → "cross_modal"          │
    │ same bbox + different date              → "bi_temporal"          │
    │ otherwise (no clear pairing found)      → "unknown"              │
    └──────────────────────────────────────────────────────────────────┘
    """
    paths = list(meta.keys())
    if len(paths) == 1:
        return "single"

    # Compare the first image against all others.
    ref = meta[paths[0]]
    ref_bbox = tuple(ref["bbox"])
    ref_date = ref.get("acquisition_date")
    ref_bands = ref["band_count"]

    for path in paths[1:]:
        other = meta[path]
        other_bbox = tuple(other["bbox"])
        other_date = other.get("acquisition_date")
        other_bands = other["band_count"]

        if not _bboxes_overlap(ref_bbox, other_bbox):  # type: ignore[arg-type]
            raise ValueError(
                f"Images '{paths[0]}' and '{path}' have non-overlapping "
                "spatial extents and cannot be jointly analysed. "
                "Please upload images covering the same geographic region."
            )

        same_bbox = _bboxes_equal(ref_bbox, other_bbox)  # type: ignore[arg-type]

        if same_bbox and ref_date and other_date and ref_date != other_date:
            return "bi_temporal"

        if same_bbox and ref_bands != other_bands:
            return "cross_modal"

    return "unknown"


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def ingest_and_validate(state: AgentState) -> dict[str, Any]:
    """LangGraph node: open rasters, extract metadata, classify input_config.

    Parameters
    ----------
    state:
        Current agent state; reads ``raw_images``.

    Returns
    -------
    dict
        Partial state update with ``image_metadata``, ``input_config``,
        and an appended ``execution_trace`` entry.

    Raises
    ------
    ValueError
        If an image cannot be opened, lacks a CRS, or the pair has
        non-overlapping bounding boxes.
    """
    raw_images: list[str] = state["raw_images"]
    if not raw_images:
        raise ValueError("No images provided. Please upload at least one raster file.")

    metadata: dict[str, dict] = {}

    for path in raw_images:
        try:
            with rasterio.open(path) as ds:
                bbox_wgs84 = _to_wgs84_bbox(ds)
                tags: dict[str, str] = ds.tags() or {}
                acquisition_date = _parse_acquisition_date(tags)

                metadata[path] = {
                    "crs": str(ds.crs),
                    "bbox": list(bbox_wgs84),  # [west, south, east, north]
                    "band_count": ds.count,
                    "width": ds.width,
                    "height": ds.height,
                    "dtype": str(ds.dtypes[0]),
                    "acquisition_date": acquisition_date,
                    "driver": ds.driver,
                }
                logger.info("Ingested '%s': crs=%s, bands=%d", path, ds.crs, ds.count)

        except rasterio.errors.RasterioIOError as exc:
            raise ValueError(
                f"Cannot open image '{path}': {exc}. "
                "Ensure the file is a valid raster (GeoTIFF, etc.)."
            ) from exc

    input_config = _classify_config(metadata)

    summary: dict[str, Any] = {
        "image_count": len(raw_images),
        "input_config": input_config,
        "images": {
            p: {
                "crs": m["crs"],
                "bbox": m["bbox"],
                "band_count": m["band_count"],
                "acquisition_date": m["acquisition_date"],
            }
            for p, m in metadata.items()
        },
    }

    trace_entry: dict[str, Any] = {"node": "ingestion", "result": summary}

    return {
        "image_metadata": metadata,
        "input_config": input_config,
        "execution_trace": [trace_entry],
    }
