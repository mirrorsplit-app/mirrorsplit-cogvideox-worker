"""
model_manager.py
────────────────
Loads CogVideoXImageToVideoPipeline (diffusers 0.32.2 — stable PyPI release).

IMPORTANT: diffusers >= 0.33.0 includes WanPipeline which causes
  "infer_schema: Parameter q has unsupported type torch.Tensor"
on import. requirements.txt pins diffusers==0.32.2 (last release before Wan).
A startup guard below enforces this and fails fast with a clear error if
a wrong version is somehow installed.
"""

# ── Redirect ALL cache / temp I/O to the RunPod network volume ────────────────
# Must be the very first code — before importing diffusers or huggingface_hub.
# The container filesystem is too small (~10-20 GB) for the 18 GB model weights.

import os
import tempfile

_VOLUME       = "/runpod-volume"
_HF_HOME      = f"{_VOLUME}/huggingface"
_HF_HUB_CACHE = f"{_HF_HOME}/hub"
_TF_CACHE     = f"{_HF_HOME}/transformers"
_CACHE_HOME   = f"{_VOLUME}/cache"
_TMP          = f"{_VOLUME}/tmp"

for _d in (_HF_HOME, _HF_HUB_CACHE, _TF_CACHE, _CACHE_HOME, _TMP):
    os.makedirs(_d, exist_ok=True)

os.environ["HF_HOME"]               = _HF_HOME
os.environ["HF_HUB_CACHE"]          = _HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"]    = _TF_CACHE
os.environ["XDG_CACHE_HOME"]        = _CACHE_HOME
os.environ["TMPDIR"]                = _TMP
os.environ["TEMP"]                  = _TMP
os.environ["TMP"]                   = _TMP

tempfile.tempdir = _TMP

# Disable xet / hf-transfer — causes "Background writer channel closed"
os.environ["HF_HUB_DISABLE_XET"]        = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ── Remaining imports (after env vars are set) ────────────────────────────────

import logging

import torch

import config

logger = logging.getLogger(__name__)

# ── Singleton ──────────────────────────────────────────────────────────────────
_pipeline = None


def _check_diffusers_version() -> None:
    """
    Abort startup immediately if diffusers >= 0.33.0 is installed.

    diffusers 0.33.0+ includes WanPipeline and AutoencoderKLWan.
    On import diffusers tries to register all pipeline classes, including Wan,
    which triggers flash-attn's infer_schema and fails with:
      "Parameter q has unsupported type torch.Tensor"

    The correct version is 0.32.2 (last release before Wan was merged).
    If you see this error, the image was built with a wrong diffusers version —
    rebuild with requirements.txt pinning diffusers==0.32.2.
    """
    import importlib.metadata
    from packaging.version import Version

    try:
        installed = Version(importlib.metadata.version("diffusers"))
    except Exception:
        logger.warning("Could not determine diffusers version — proceeding anyway")
        return

    logger.info("  diffusers version check: %s", installed)

    if installed >= Version("0.33.0"):
        raise RuntimeError(
            f"diffusers {installed} is installed but this worker requires < 0.33.0. "
            f"diffusers >= 0.33.0 includes WanPipeline which causes import errors "
            f"on RunPod. Rebuild the image with diffusers==0.32.2."
        )


def _load_pipeline():
    # Run version guard before any diffusers import
    _check_diffusers_version()

    # Import ONLY the CogVideoX pipeline — nothing else from diffusers
    from diffusers import CogVideoXImageToVideoPipeline

    model_id  = config.MODEL_ID
    cache_dir = _HF_HUB_CACHE
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading CogVideoX-5b-I2V pipeline …")
    logger.info("  model_id        : %s", model_id)
    logger.info("  cache_dir       : %s", cache_dir)
    logger.info("  device          : %s", device)

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
