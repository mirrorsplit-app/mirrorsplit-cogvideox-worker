"""
download_model.py
─────────────────
Run during `docker build` (as a RUN step) to pre-download all 17 model files
into the image layer cache so the container never fetches them at runtime.

This script is NOT called by handler.py — it only runs at build time.

Usage in Dockerfile:
    RUN python download_model.py

Environment variables (all have defaults that match config.py):
    MODEL_ID        HuggingFace repo id  (default: THUDM/CogVideoX-5b-I2V)
    MODEL_CACHE_DIR Local cache path     (default: /runpod-volume/models)

Why this exists
───────────────
CogVideoX-5B-I2V has 17 files totalling ~18 GB:
  • transformer/  — 9 × ~2 GB safetensors shards
  • vae/          — 1 × ~400 MB safetensors
  • text_encoder/ — 1 × ~9 GB safetensors (T5-XXL)
  • tokenizer/    — small JSON/text files
  • scheduler/    — small JSON files
  • model_index.json

Without this script those 17 files are downloaded inside handler() on the
first cold-start request, taking 5–15 minutes and exceeding the RunPod
execution timeout before inference ever starts.

With this script baked into the Docker image the files live in the image
layer.  get_pipeline() finds them instantly via local_files_only=True and
never calls the network.
"""

import logging
import os
import sys
import time

# ── Set cache path BEFORE importing huggingface_hub ───────────────────────────
# Must match the path used by model_manager.py at runtime.
MODEL_ID        = os.environ.get("MODEL_ID",        "THUDM/CogVideoX-5b-I2V")
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")

# Point every HF library at the same directory
os.environ["HF_HOME"]               = MODEL_CACHE_DIR
os.environ["HF_HUB_CACHE"]          = MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = MODEL_CACHE_DIR

# Disable xet / hf-transfer — unreliable in build environments
os.environ["HF_HUB_DISABLE_XET"]        = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("download_model")

# ── Download ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("CogVideoX model download — build-time cache population")
logger.info("  MODEL_ID        : %s", MODEL_ID)
logger.info("  MODEL_CACHE_DIR : %s", MODEL_CACHE_DIR)
logger.info("=" * 60)

t_start = time.perf_counter()

try:
    from huggingface_hub import snapshot_download

    logger.info(">>> Model download started — fetching all 17 files (~18 GB)")
    logger.info("    This runs once at image build time, never during inference.")

    local_dir = snapshot_download(
        repo_id   = MODEL_ID,
        cache_dir = MODEL_CACHE_DIR,
        # Download every file the pipeline needs; ignore unneeded extras
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    elapsed = time.perf_counter() - t_start
    logger.info(">>> Model download complete in %.0fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("    Cached to: %s", local_dir)

    # ── Verify the cache is readable ─────────────────────────────────────────
    logger.info(">>> Verifying cache is readable with local_files_only=True …")
    from diffusers import CogVideoXImageToVideoPipeline
    import torch

    # map_location cpu — no GPU at build time
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype    = torch.bfloat16,
        cache_dir      = MODEL_CACHE_DIR,
        local_files_only = True,
    )
    del pipe  # free memory — we only needed the load check

    logger.info(">>> Cache verification PASSED — pipeline loaded from local files only")
    logger.info(">>> Worker is ready. Inference will begin immediately on cold start.")

except Exception:
    logger.exception("FATAL: model download/verification failed")
    sys.exit(1)
