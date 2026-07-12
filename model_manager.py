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

# ── Redirect ALL cache / temp I/O to the RunPod network volume ────────────────
# This MUST be the very first code that runs — before importing diffusers,
# huggingface_hub, torch, or anything that touches the filesystem.
#
# Reason: the container filesystem is small (~10-20 GB). CogVideoX-5b-I2V
# weights are ~18 GB. Without redirection every download attempt fails with:
#   OSError: [Errno 28] No space left on device
#   (inside huggingface_hub.file_download → temp_file.write(chunk))
#
# The RunPod network volume is mounted at /runpod-volume and has sufficient
# space for both the weights and all temporary download files.

import os
import tempfile

_VOLUME       = "/runpod-volume"
_HF_HOME      = f"{_VOLUME}/huggingface"
_HF_HUB_CACHE = f"{_HF_HOME}/hub"
_TF_CACHE     = f"{_HF_HOME}/transformers"
_CACHE_HOME   = f"{_VOLUME}/cache"
_TMP          = f"{_VOLUME}/tmp"

# Create all directories before setting env vars so huggingface_hub
# never tries to create them on the container filesystem.
for _d in (_HF_HOME, _HF_HUB_CACHE, _TF_CACHE, _CACHE_HOME, _TMP):
    os.makedirs(_d, exist_ok=True)

# Hugging Face cache locations
os.environ["HF_HOME"]             = _HF_HOME
os.environ["HF_HUB_CACHE"]        = _HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_HUB_CACHE   # older alias
os.environ["TRANSFORMERS_CACHE"]   = _TF_CACHE

# Generic XDG / system temp — some libraries fall back to these
os.environ["XDG_CACHE_HOME"]      = _CACHE_HOME
os.environ["TMPDIR"]              = _TMP
os.environ["TEMP"]                = _TMP
os.environ["TMP"]                 = _TMP

# Tell Python's tempfile module to use the same directory.
# Must be done before any tempfile.* call (including those inside diffusers).
tempfile.tempdir = _TMP

# Disable xet / hf-transfer backends — they have their own writer threads
# that can also hit disk-space issues and produce opaque errors.
os.environ["HF_HUB_DISABLE_XET"]        = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ─────────────────────────────────────────────────────────────────────────────
# All other imports come AFTER the env vars are set so every library that
# reads these vars at import time (huggingface_hub, diffusers, torch) picks
# up the correct values.
# ─────────────────────────────────────────────────────────────────────────────

import logging

import torch

import config

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────────
_pipeline = None


def _load_pipeline():
    # Lazy import — diffusers is a heavy package; importing at call time
    # keeps module-load fast and makes import errors surface clearly.
    from diffusers import CogVideoXImageToVideoPipeline

    model_id  = config.MODEL_ID
    cache_dir = _HF_HUB_CACHE   # always use the volume cache
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading CogVideoX-5b-I2V pipeline …")
    logger.info("  model_id        : %s", model_id)
    logger.info("  cache_dir       : %s", cache_dir)
    logger.info("  device          : %s", device)

    # ── Dependency versions ────────────────────────────────────────────────────
    import huggingface_hub as _hfh
    import diffusers as _dffs
    logger.info("  huggingface_hub : %s", _hfh.__version__)
    logger.info("  diffusers       : %s", _dffs.__version__)
    logger.info("  torch           : %s", torch.__version__)

    # ── Load pipeline ──────────────────────────────────────────────────────────
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )

    # ── Memory optimisations ───────────────────────────────────────────────────

    if config.VAE_TILING:
        pipe.vae.enable_tiling()
        logger.info("  VAE tiling      : enabled")

    if device == "cuda" and config.CPU_OFFLOAD:
        pipe.enable_sequential_cpu_offload()
        logger.info("  CPU offload     : enabled")
    else:
        pipe.to(device)
        if device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("  GPU VRAM        : %.1f GB (no offload)", vram)

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
