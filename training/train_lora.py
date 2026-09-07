"""LoRA fine-tuning script for Qwen/Qwen2.5-VL-3B-Instruct.

Designed to run on a single free-tier 16 GB GPU (Colab T4 / Kaggle T4).
Uses 4-bit NF4 quantisation (QLoRA) + gradient checkpointing to stay within
the VRAM budget.

Usage (script)
--------------
    python training/train_lora.py \
        --dataset_path data/train.jsonl \
        --output_dir ./checkpoints/satquery-lora-r64 \
        --epochs 3 --batch_size 1 --grad_accum_steps 8

Usage (Colab cell)
------------------
    !python training/train_lora.py --dataset_path /content/train.jsonl --epochs 1

Dataset contract  (JSONL, one JSON object per line)
---------------------------------------------------
Each line must contain a ``messages`` key whose value is a list of ChatML
turns:

    {"messages": [
        {"role": "system", "content": "You are ..."},
        {"role": "user",
         "content": [
             {"type": "image", "image": "file:///abs/path.jpg"},
             {"type": "text",  "text": "Describe this image."}
         ]},
        {"role": "assistant",
         "content": [{"type": "text", "text": "The image shows..."}]}
    ]}

The ``<image>`` placeholder syntax is also accepted in a flat-string user
turn; the loader normalises it to the structured format above.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"

# Maximum *text* tokens per sample.  Qwen2.5-VL dynamically resizes images
# internally, so we only cap the text side.  Raise this on an A100.
MAX_SEQ_LENGTH: int = 1024  # VRAM-saving: keeps KV-cache small on a T4

# Minimum pixels per image processed by the vision encoder.  Lowering this
# value shrinks the number of visual tokens and saves VRAM significantly.
MIN_PIXELS: int = 256 * 28 * 28        # ~200 K pixels  --  VRAM-saving
MAX_PIXELS: int = 512 * 28 * 28        # ~400 K pixels  --  VRAM-saving
# A100 users can raise these:
#   MIN_PIXELS = 256 * 28 * 28
#   MAX_PIXELS = 1280 * 28 * 28

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _normalise_user_content(content: Any) -> list[dict[str, str]]:
    """Ensure the user turn ``content`` is in the structured list format.

    Accepts either:
    *   A plain string with optional ``<image>`` placeholders.
    *   An already-structured list of ``{"type": "image"|"text", ...}`` dicts.
    """
    if isinstance(content, list):
        return content

    # Flat string -- split on <image> markers and interleave image/text parts.
    parts: list[dict[str, str]] = []
    segments = str(content).split("<image>")
    for idx, seg in enumerate(segments):
        if idx > 0:
            # Each <image> placeholder becomes an image-type entry.
            # The actual image path must be supplied elsewhere (first image
            # entry in the conversation, or a parallel field).  We use a
            # sentinel that the collator will replace.
            parts.append({"type": "image", "image": ""})
        text = seg.strip()
        if text:
            parts.append({"type": "text", "text": text})
    return parts


def _normalise_assistant_content(content: Any) -> str:
    """Return assistant content as a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Structured list -- concatenate text parts.
        return " ".join(
            part.get("text", "") for part in content if part.get("type") == "text"
        )
    return str(content)


def load_chatml_jsonl(path: str | Path) -> Dataset:
    """Load a JSONL file into a HuggingFace ``Dataset``.

    Each line must have a ``messages`` key.  The function normalises user
    turns to the structured multimodal format expected by Qwen2.5-VL's
    chat template.
    """
    records: list[dict[str, Any]] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", line_no, exc)
                continue

            messages = obj.get("messages")
            if not messages or not isinstance(messages, list):
                logger.warning("Skipping line %d: missing or empty 'messages'", line_no)
                continue

            normalised: list[dict[str, Any]] = []
            for turn in messages:
                role = turn.get("role", "")
                content = turn.get("content", "")

                if role == "user":
                    normalised.append({
                        "role": "user",
                        "content": _normalise_user_content(content),
                    })
                elif role == "assistant":
                    normalised.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": _normalise_assistant_content(content)}],
                    })
                elif role == "system":
                    normalised.append({
                        "role": "system",
                        "content": [{"type": "text", "text": str(content)}],
                    })
                else:
                    logger.warning(
                        "Line %d: unknown role '%s', skipping turn", line_no, role
                    )

            records.append({"messages": normalised})

    logger.info("Loaded %d training samples from '%s'", len(records), path)
    return Dataset.from_list(records)

# ---------------------------------------------------------------------------
# Model & Quantisation
# ---------------------------------------------------------------------------


def build_quantisation_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation -- the single biggest VRAM saving.

    NF4 is information-theoretically optimal for normally-distributed
    weights and halves memory vs FP16 while preserving >99 %% of accuracy
    when combined with LoRA adapters.

    ``bnb_4bit_use_double_quant`` applies a second round of quantisation
    to the quantisation constants themselves, saving another ~0.4 GB on a
    3B-param model.

    ``bnb_4bit_compute_dtype=bfloat16`` keeps forward-pass matmuls in
    BF16 for numerical stability.  Falls back to FP16 on T4 (no BF16 HW)
    but the flag is still safe -- bitsandbytes handles the fallback.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,                           # VRAM-saving: ~2 GB for 3 B params
        bnb_4bit_quant_type="nf4",                   # VRAM-saving: optimal 4-bit format
        bnb_4bit_compute_dtype=torch.bfloat16,       # VRAM-saving: BF16 activations
        bnb_4bit_use_double_quant=True,              # VRAM-saving: ~0.4 GB extra saving
    )


def build_lora_config() -> LoraConfig:
    """LoRA adapter config targeting the attention projections only.

    r=64, alpha=128 is a good trade-off for satellite-imagery tasks on a
    16 GB GPU.  Trainable params ~= 40 M (< 2 %% of the 3 B base).

    # A100 alternative (32-48 GB VRAM):
    #   r=128, lora_alpha=256, target_modules=find_all_linear_names(model)
    #   This doubles the adapter capacity and can improve grounding accuracy.
    """
    return LoraConfig(
        r=64,                                        # Rank -- higher = more capacity
        lora_alpha=128,                              # Scaling factor (alpha / r = 2)
        # A100 alternative:
        #   r=128,
        #   lora_alpha=256,
        target_modules=["q_proj", "v_proj"],         # Attention projections only -- VRAM-saving
        lora_dropout=0.05,                           # Light regularisation
        bias="none",                                 # VRAM-saving: no bias adapters
        task_type=TaskType.CAUSAL_LM,
        # modules_to_save=[] -- we do NOT save/unfreeze the LM head
    )


def load_model_and_processor(
    model_id: str = MODEL_ID,
) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """Load the base VLM in 4-bit and prepare it for LoRA training.

    Returns the PeftModel-wrapped model and the multimodal processor.
    """
    bnb_config = build_quantisation_config()
    lora_config = build_lora_config()

    logger.info("Loading processor from '%s'", model_id)
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    logger.info("Loading model in 4-bit NF4 from '%s'", model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",                           # VRAM-saving: auto shard across devices
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",     # VRAM-saving + speed: FA2 (falls back gracefully)
        trust_remote_code=True,
    )

    # Prepare quantised model for k-bit training:
    # - casts layer norms to FP32 for training stability
    # - freezes base weights
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,             # VRAM-saving: trades compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for LoRA + DDP
    )

    # Wrap with LoRA adapters.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    learning_rate: float = 2e-5,
) -> SFTConfig:
    """Assemble ``SFTConfig`` with T4-safe defaults.

    Every non-obvious choice is annotated with its VRAM / stability
    rationale.
    """
    return SFTConfig(
        output_dir=output_dir,

        # --- schedule -------------------------------------------------------
        num_train_epochs=epochs,
        learning_rate=learning_rate,                 # 2e-5: conservative for QLoRA
        lr_scheduler_type="cosine",                  # Smooth decay after warmup
        warmup_ratio=0.03,                           # ~3 %% of steps as linear warmup
        optim="adamw_torch",                         # AdamW with weight decay
        weight_decay=0.01,

        # --- batching -------------------------------------------------------
        per_device_train_batch_size=batch_size,      # 1 on T4 -- VRAM-saving
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,# Effective batch = bs * accum

        # --- mixed precision ------------------------------------------------
        bf16=torch.cuda.is_bf16_supported(),         # BF16 if hardware supports it
        fp16=not torch.cuda.is_bf16_supported(),     # FP16 fallback (T4)

        # --- checkpointing --------------------------------------------------
        gradient_checkpointing=True,                 # VRAM-saving: recompute activations
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # --- logging & saving -----------------------------------------------
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,                          # Keep only last 3 checkpoints -- disk-saving
        load_best_model_at_end=False,                # We save adapters only

        # --- dataset --------------------------------------------------------
        max_seq_length=MAX_SEQ_LENGTH,               # VRAM-saving: cap text tokens
        dataset_kwargs={
            "skip_prepare_dataset": True,            # We handle preprocessing ourselves
        },

        # --- misc -----------------------------------------------------------
        remove_unused_columns=False,                 # Required for VLMs (pixel_values etc.)
        dataloader_pin_memory=True,
        report_to="none",                            # Set to "wandb" if you have W&B
        seed=42,
    )


def train(
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    learning_rate: float = 2e-5,
) -> None:
    """End-to-end training entry-point.

    1.  Load dataset from JSONL.
    2.  Load model + processor (4-bit, LoRA).
    3.  Build trainer and run.
    4.  Save LoRA adapter weights (not merged).
    """
    # 1. Dataset -----------------------------------------------------------
    dataset = load_chatml_jsonl(dataset_path)

    # 90/10 train/eval split (deterministic seed for reproducibility).
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info("Train: %d  |  Eval: %d", len(train_ds), len(eval_ds))

    # 2. Model & Processor ------------------------------------------------
    model, processor = load_model_and_processor()

    # 3. Training args -----------------------------------------------------
    training_args = build_training_args(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        learning_rate=learning_rate,
    )

    # 4. Trainer -----------------------------------------------------------
    # ``SFTTrainer`` with a multimodal processor automatically handles
    # vision-language collation (pixel_values, image_grid_thw, etc.) when
    # ``processing_class`` is set.  No custom collate_fn needed.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,                  # Handles tokenisation + vision pre-proc
    )

    # 5. Train! ------------------------------------------------------------
    logger.info("Starting training for %d epoch(s)...", epochs)
    trainer.train()

    # 6. Save LoRA adapter (NOT merged weights) ----------------------------
    # Saving only the adapter keeps checkpoint size at ~80 MB instead of
    # ~6 GB for the full merged model.
    final_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("Saved LoRA adapter + processor to '%s'", final_dir)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL-3B-Instruct with LoRA (QLoRA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the training JSONL file (ChatML format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/satquery-lora-r64",
        help="Directory for checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size (1 recommended for T4).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * this).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate for AdamW.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
    )