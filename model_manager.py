"""
model_manager.py
────────────────
Loads CogVideoXImageToVideoPipeline from diffusers (stable PyPI release).
Keeps the pipeline as a module-level singleton so RunPod warm workers
reuse the already-loaded weights without re-downloading or re-initialising.

Pipeline: CogVideoXImageToVideoPipeline
Model   : THUDM/CogVideoX-5b-I2V
Dtype   : torch.bfloat16  (official recommendation)
VAE     : same pipeline — no separate VAE load required
"""

import logging
import os

import torch

import config

logger = logging.getLogger(__name__)

# Disable xet transfer backend and hf-transfer — both cause
# "Background writer channel closed" on RunPod's network environment.
# Must be set BEFORE importing diffusers or huggingface_hub.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ── Singleton ──────────────────────────────────────────────────────────────────
_pipeline = None


def _load_pipeline():
    # Lazy import — diffusers is a heavy package; importing at call time
    # keeps module-load fast and makes import errors surface clearly.
    from diffusers import CogVideoXImageToVideoPipeline

    model_id  = config.MODEL_ID
    cache_dir = config.MODEL_CACHE_DIR
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Loading CogVideoX-5b-I2V pipeline …")
    logger.info("  model_id  : %s", model_id)
    logger.info("  cache_dir : %s", cache_dir)
    logger.info("  device    : %s", device)

    import huggingface_hub as _hfh
    import diffusers as _dffs
    logger.info("  huggingface_hub : %s", _hfh.__version__)
    logger.info("  diffusers       : %s", _dffs.__version__)
    logger.info("  torch           : %s", torch.__version__)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )

    # ── Memory optimisations ───────────────────────────────────────────────────

    if config.VAE_TILING:
        pipe.vae.enable_tiling()
        logger.info("  VAE tiling          : enabled")

    if device == "cuda" and config.CPU_OFFLOAD:
        # Sequential CPU offload — each sub-module moves to GPU only when
        # needed. Required for GPUs with < 24 GB VRAM.
        pipe.enable_sequential_cpu_offload()
        logger.info("  CPU offload         : enabled")
    else:
        pipe.to(device)
        if device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("  GPU VRAM            : %.1f GB (no offload)", vram)

    logger.info("CogVideoX-5b-I2V pipeline ready.")
    return pipe


def get_pipeline():
    """Return the loaded pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def warm_up() -> None:
    """Pre-load the pipeline at container start to eliminate cold-start latency."""
    logger.info("=== CogVideoX worker warm-up started ===")
    get_pipeline()
    logger.info("=== CogVideoX worker warm-up complete ===")
