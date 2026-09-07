"""Dataset preparation: convert BigEarthNet.txt + VRSBench to ChatML JSONL.

Produces a single JSONL file where each line is a training sample in the
Qwen2.5-VL ChatML conversational format expected by ``train_lora.py``.

Supported source datasets
--------------------------
1.  **BigEarthNet.txt** -- scene description / semantic tagging examples
    derived from the BigEarthNet multi-label classification archive.
    Expected input: a directory of patch folders, each containing GeoTIFF
    band files and a JSON label file, or a pre-extracted metadata CSV/JSON.

2.  **VRSBench** -- captions, referring expressions, and VQA pairs for
    remote-sensing images.  Bounding boxes are normalised to a 0-1000
    integer scale in the assistant response for grounding examples.

Usage
-----
    python training/data_prep/build_dataset.py \
        --bigearthnet_dir /data/BigEarthNet-v1.0 \
        --vrsbench_dir /data/VRSBench \
        --output_path data/train.jsonl \
        --num_samples_ben 500 \
        --num_samples_vrs 500

The ``--num_samples_*`` flags let you cap each source independently so a
first run stays small (500-2000 total) and trains in < 1 h on a free GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ---------------------------------------------------------------------------
# System prompt shared across all samples
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a remote-sensing analysis assistant. You answer questions "
    "about satellite and aerial imagery, describe scenes, identify objects, "
    "and provide spatial coordinates when asked."
)

# ---------------------------------------------------------------------------
# BigEarthNet label mapping (19-class simplified nomenclature)
# ---------------------------------------------------------------------------

# Mapping from the 43-class CLC Level-3 labels to the simplified 19-class
# nomenclature recommended by the BigEarthNet authors.
BEN_19_CLASS_MAP: dict[str, str] = {
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coastal lagoons": "Coastal lagoons",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Coniferous forest": "Coniferous forest",
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Estuaries": "Marine waters",
    "Fruit trees and berry plantations": "Permanent crops",
    "Green urban areas": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Inland marshes": "Inland wetlands",
    "Intertidal flats": "Marine waters",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture",
    "Mineral extraction sites": "Mine, dump and construction sites",
    "Mixed forest": "Mixed forest",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Natural grasslands": "Natural grasslands",
    "Non-irrigated arable land": "Arable land",
    "Olive groves": "Permanent crops",
    "Pastures": "Pastures",
    "Peat bogs": "Inland wetlands",
    "Permanently irrigated land": "Arable land",
    "Port areas": "Industrial or commercial units",
    "Rice fields": "Arable land",
    "Road and rail networks and associated land": "Industrial or commercial units",
    "Salines": "Coastal wetlands",
    "Salt marshes": "Coastal wetlands",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Sea and ocean": "Marine waters",
    "Sport and leisure facilities": "Urban fabric",
    "Transitional woodland-shrub": "Transitional woodland/shrub",
    "Vineyards": "Permanent crops",
    "Water bodies": "Inland waters",
    "Water courses": "Inland waters",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Bare rocks": "Bare rock",
    "Burnt areas": "Burnt areas",
    "Dump sites": "Mine, dump and construction sites",
    "Construction sites": "Mine, dump and construction sites",
    "Glaciers and perpetual snow": "Glaciers and perpetual snow",
    "Sparsely vegetated areas": "Sparsely vegetated areas",
    "Airports": "Industrial or commercial units",
}


# Question templates for BigEarthNet scene description / tagging.
_BEN_QUESTION_TEMPLATES: list[str] = [
    "What land-cover classes are visible in this satellite image?",
    "Describe the land-cover types present in this Sentinel-2 scene.",
    "Identify and list all land-cover categories shown in this image.",
    "What types of terrain and surface cover can you see in this satellite patch?",
    "Provide a scene description for this remote-sensing image.",
]


def _build_ben_answer(labels: list[str]) -> str:
    """Create a natural-language answer from a list of land-cover labels."""
    if not labels:
        return "No land-cover classes could be identified in this image."
    unique = sorted(set(labels))
    if len(unique) == 1:
        return f"This satellite image shows {unique[0].lower()}."
    label_str = ", ".join(unique[:-1]) + f", and {unique[-1]}"
    return (
        f"The land-cover classes visible in this image are: {label_str.lower()}."
    )

# ---------------------------------------------------------------------------
# BigEarthNet converter
# ---------------------------------------------------------------------------


def _find_ben_rgb_image(patch_dir: Path) -> Path | None:
    """Find a suitable RGB-composite or TCI image inside a BEN patch folder.

    BigEarthNet v1.0 stores each band as a separate GeoTIFF.  We look for
    the True Colour Image (TCI) or fall back to the B04 (red) band file.
    """
    for pattern in ["*_TCI.tif", "*_B04.tif", "*.tif"]:
        matches = list(patch_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_ben_labels(patch_dir: Path) -> list[str]:
    """Load labels from the BigEarthNet patch JSON metadata file.

    The label file is named ``<patch_name>_labels_metadata.json`` and
    contains a ``labels`` key with a list of CLC Level-3 class names.
    """
    json_files = list(patch_dir.glob("*_labels_metadata.json"))
    if not json_files:
        return []
    with json_files[0].open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    raw_labels: list[str] = meta.get("labels", [])

    # Map to the simplified 19-class nomenclature.
    mapped = [BEN_19_CLASS_MAP.get(lbl, lbl) for lbl in raw_labels]
    return mapped


def convert_bigearthnet(
    bigearthnet_dir: str | Path,
    num_samples: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Convert BigEarthNet patches to ChatML training records.

    Parameters
    ----------
    bigearthnet_dir:
        Root directory of BigEarthNet-v1.0 (contains S2 patch folders).
    num_samples:
        Maximum number of samples to produce.  Patches without label files
        or images are silently skipped.
    seed:
        Random seed for reproducible sub-sampling.

    Returns
    -------
    list[dict]
        Each dict has a ``messages`` key in ChatML format.
    """
    bigearthnet_dir = Path(bigearthnet_dir)
    if not bigearthnet_dir.is_dir():
        logger.warning("BigEarthNet dir '%s' not found, skipping.", bigearthnet_dir)
        return []

    # Gather patch directories that have both an image and labels.
    candidates: list[Path] = []
    for entry in sorted(bigearthnet_dir.iterdir()):
        if not entry.is_dir():
            continue
        if _find_ben_rgb_image(entry) and _load_ben_labels(entry):
            candidates.append(entry)

    if not candidates:
        logger.warning("No valid BigEarthNet patches found in '%s'.", bigearthnet_dir)
        return []

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_samples]
    logger.info(
        "BigEarthNet: %d candidate patches, selecting %d.",
        len(candidates),
        len(selected),
    )

    records: list[dict[str, Any]] = []
    for patch_dir in selected:
        img_path = _find_ben_rgb_image(patch_dir)
        labels = _load_ben_labels(patch_dir)
        if img_path is None or not labels:
            continue

        question = rng.choice(_BEN_QUESTION_TEMPLATES)
        answer = _build_ben_answer(labels)

        record: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file:///{img_path.as_posix()}"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]
        }
        records.append(record)

    logger.info("BigEarthNet: produced %d training samples.", len(records))
    return records

# ---------------------------------------------------------------------------
# VRSBench converter
# ---------------------------------------------------------------------------

# VRSBench stores bounding boxes normalised to 0-100.  Qwen2.5-VL's
# grounding convention uses a 0-1000 integer scale.
_VRS_COORD_SCALE: int = 10  # multiply VRS coords by 10 to get 0-1000


def _format_bbox_1000(bbox_100: list[float]) -> str:
    """Convert a VRSBench [x1,y1,x2,y2] bbox (0-100) to a 0-1000 string.

    Example: [12.5, 34.2, 56.7, 78.9] -> "[125,342,567,789]"
    """
    scaled = [int(round(c * _VRS_COORD_SCALE)) for c in bbox_100]
    scaled = [max(0, min(1000, c)) for c in scaled]
    return f"[{scaled[0]},{scaled[1]},{scaled[2]},{scaled[3]}]"


def _load_vrs_json(path: Path) -> list[dict[str, Any]]:
    """Load a VRSBench JSON annotation file (list of records)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    return []


def convert_vrsbench(
    vrsbench_dir: str | Path,
    num_samples: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Convert VRSBench annotations to ChatML training records.

    Handles three annotation types:
    1.  **Captions** -- scene description from ``caption.json``
    2.  **Referring expressions** -- object grounding from ``ref_exp.json``
    3.  **VQA** -- question-answer pairs from ``vqa.json``

    Parameters
    ----------
    vrsbench_dir:
        Root of the VRSBench dataset.  Expected structure::

            vrsbench_dir/
              images/           # JPEG/PNG images
              caption.json      # [{image, caption}, ...]
              ref_exp.json      # [{image, expression, bbox}, ...]
              vqa.json          # [{image, question, answer}, ...]

    num_samples:
        Maximum total samples across all three annotation types.
        Allocated roughly evenly (caption: 30%, ref_exp: 35%, VQA: 35%).
    seed:
        Random seed for reproducible sub-sampling.

    Returns
    -------
    list[dict]
        Each dict has a ``messages`` key in ChatML format.
    """
    vrsbench_dir = Path(vrsbench_dir)
    if not vrsbench_dir.is_dir():
        logger.warning("VRSBench dir '%s' not found, skipping.", vrsbench_dir)
        return []

    images_dir = vrsbench_dir / "images"
    if not images_dir.is_dir():
        # Try alternate layout: images at root level.
        images_dir = vrsbench_dir

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    # Budget allocation across task types.
    n_caption = max(1, int(num_samples * 0.30))
    n_refexp = max(1, int(num_samples * 0.35))
    n_vqa = num_samples - n_caption - n_refexp

    # ---- 1. Captions ----
    captions_raw = _load_vrs_json(vrsbench_dir / "caption.json")
    rng.shuffle(captions_raw)
    for entry in captions_raw[:n_caption]:
        img_name = entry.get("image") or entry.get("img") or ""
        caption_text = entry.get("caption") or entry.get("description") or ""
        if not img_name or not caption_text:
            continue
        img_path = images_dir / img_name
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file:///{img_path.as_posix()}"},
                        {"type": "text", "text": "Provide a detailed description of this remote-sensing image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": caption_text.strip()}],
                },
            ]
        })

    # ---- 2. Referring expressions (grounding) ----
    refexp_raw = _load_vrs_json(vrsbench_dir / "ref_exp.json")
    rng.shuffle(refexp_raw)
    for entry in refexp_raw[:n_refexp]:
        img_name = entry.get("image") or entry.get("img") or ""
        expression = entry.get("expression") or entry.get("ref") or ""
        # VRSBench bbox: [x1, y1, x2, y2] in 0-100 scale
        bbox_raw = entry.get("bbox") or entry.get("obj_coord") or []
        if not img_name or not expression or len(bbox_raw) != 4:
            continue
        img_path = images_dir / img_name
        bbox_str = _format_bbox_1000(bbox_raw)

        question = f"Locate the following object in the image: {expression.strip()}"
        answer = (
            f"The object described as \"{expression.strip()}\" is located at "
            f"bounding box {bbox_str} (coordinates on a 0-1000 scale)."
        )

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file:///{img_path.as_posix()}"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]
        })

    # ---- 3. VQA ----
    vqa_raw = _load_vrs_json(vrsbench_dir / "vqa.json")
    rng.shuffle(vqa_raw)
    for entry in vqa_raw[:n_vqa]:
        img_name = entry.get("image") or entry.get("img") or ""
        question = entry.get("question") or ""
        answer_text = entry.get("answer") or ""
        if not img_name or not question or not answer_text:
            continue
        img_path = images_dir / img_name

        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file:///{img_path.as_posix()}"},
                        {"type": "text", "text": question.strip()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text.strip()}],
                },
            ]
        })

    logger.info(
        "VRSBench: produced %d training samples "
        "(captions: %d target, refexp: %d target, VQA: %d target).",
        len(records), n_caption, n_refexp, n_vqa,
    )
    return records

# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: str | Path,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Write training records to a JSONL file.

    Parameters
    ----------
    records:
        List of ChatML dicts (each with a ``messages`` key).
    output_path:
        Destination JSONL file path.
    shuffle:
        Whether to shuffle the combined records before writing.
    seed:
        Random seed for reproducible shuffling.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to '%s'", len(records), output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a ChatML JSONL dataset from BigEarthNet + VRSBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bigearthnet_dir",
        type=str,
        default="",
        help="Root directory of BigEarthNet-v1.0 (Sentinel-2 patches). "
             "Leave empty to skip.",
    )
    parser.add_argument(
        "--vrsbench_dir",
        type=str,
        default="",
        help="Root directory of VRSBench. Leave empty to skip.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/train.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--num_samples_ben",
        type=int,
        default=500,
        help="Max samples to produce from BigEarthNet.",
    )
    parser.add_argument(
        "--num_samples_vrs",
        type=int,
        default=500,
        help="Max samples to produce from VRSBench.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sub-sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point: load sources, convert, merge, write."""
    args = parse_args()
    all_records: list[dict[str, Any]] = []

    if args.bigearthnet_dir:
        ben_records = convert_bigearthnet(
            bigearthnet_dir=args.bigearthnet_dir,
            num_samples=args.num_samples_ben,
            seed=args.seed,
        )
        all_records.extend(ben_records)

    if args.vrsbench_dir:
        vrs_records = convert_vrsbench(
            vrsbench_dir=args.vrsbench_dir,
            num_samples=args.num_samples_vrs,
            seed=args.seed,
        )
        all_records.extend(vrs_records)

    if not all_records:
        logger.error(
            "No records produced. Provide at least one of "
            "--bigearthnet_dir or --vrsbench_dir."
        )
        return

    write_jsonl(all_records, output_path=args.output_path, seed=args.seed)
    logger.info("Dataset build complete: %d total samples.", len(all_records))


if __name__ == "__main__":
    main()