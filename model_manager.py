"""
model_manager.py
────────────────
Loads CogVideoXImageToVideoPipeline (diffusers 0.32.2 — stable PyPI release).

Cache contract
──────────────
Both download_model.py (build time) and this file (runtime) use the same
single path for all HuggingFace I/O:

    MODEL_CACHE_DIR  (env var, default /runpod-volume/models)

Every HF-related env var is pointed at MODEL_CACHE_DIR here, at the very top
of this module, before any diffusers / huggingface_hub import.  That guarantees
from_pretrained() reads from the same location the weights were written to
during docker build.

local_files_only=True is passed to from_pretrained().  If the cache is empty
or the path is wrong the call raises immediately with a clear FileNotFoundError
rather than silently falling through to a multi-hour HuggingFace download.

IMPORTANT: diffusers >= 0.33.0 includes WanPipeline which causes
  "infer_schema: Parameter q has unsupported type torch.Tensor"
on import. requirements.txt pins diffusers==0.32.2 (last release before Wan).
A startup guard below enforces this and fails fast with a clear error if
a wrong version is somehow installed.
"""

# ── Step 1: set ALL cache env vars before any HF import ──────────────────────
# MODEL_CACHE_DIR is read here directly from os.environ (not via config.py)
# because config.py imports os too, and we need to guarantee the env vars are
# set before *any* module that touches huggingface_hub is imported.

import os
import tempfile
import time

# Single source of truth for the cache location.
# Must match MODEL_CACHE_DIR in config.py and download_model.py.
_MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/runpod-volume/models")

# Temp dir — use a subdirectory of the volume so large temp files
# don't overflow the small container filesystem.
_TMP_DIR = os.environ.get("TMPDIR_OVERRIDE", "/runpod-volume/tmp")

for _d in (_MODEL_CACHE_DIR, _TMP_DIR):
    os.makedirs(_d, exist_ok=True)

# Point every HF library at the same cache directory.
os.environ["HF_HOME"]               = _MODEL_CACHE_DIR
os.environ["HF_HUB_CACHE"]          = _MODEL_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = _MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = _MODEL_CACHE_DIR

# Temp overrides — keep large intermediates off the small container FS.
os.environ["TMPDIR"] = _TMP_DIR
os.environ["TEMP"]   = _TMP_DIR
os.environ["TMP"]    = _TMP_DIR
tempfile.tempdir     = _TMP_DIR

# Disable xet / hf-transfer — "Background writer channel closed" on RunPod.
os.environ["HF_HUB_DISABLE_XET"]        = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ── Step 2: remaining imports (env vars already set) ─────────────────────────

import logging

import torch

import config

logger = logging.getLogger(__name__)

# Confirm the cache path that will be used — visible in every log line.
logger.debug("model_manager: _MODEL_CACHE_DIR = %s", _MODEL_CACHE_DIR)

# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline = None


# ── Version guard ─────────────────────────────────────────────────────────────

def _check_diffusers_version() -> None:
    """
    Abort immediately if diffusers >= 0.33.0 is installed.

    diffusers 0.33.0+ includes WanPipeline / AutoencoderKLWan.  On import,
    diffusers tries to register all pipeline classes including Wan, which
    triggers flash-attn's infer_schema and crashes with:
        "Parameter q has unsupported type torch.Tensor"

    The correct version is 0.32.2 — the last PyPI release before Wan was merged.
    """
    import importlib.metadata
    from packaging.version import Version

    try:
        installed = Version(importlib.metadata.version("diffusers"))
    except Exception:
        logger.warning("  Could not read diffusers version — proceeding")
        return

    logger.info("  diffusers       : %s", installed)

    if installed >= Version("0.33.0"):
        raise RuntimeError(
            f"diffusers {installed} is installed but this worker requires ==0.32.2. "
            "diffusers >= 0.33.0 includes WanPipeline which causes import errors. "
            "Rebuild the image with requirements.txt pinning diffusers==0.32.2."
        )


# ── Pipeline loader ──────────────────────────────────────────────────────────

def _load_pipeline():
    """
    Load CogVideoXImageToVideoPipeline from the local cache.

    Raises FileNotFoundError (via local_files_only=True) if the cache is empty
    — that gives a clear error rather than a silent multi-hour download.
    """
    t_load = time.perf_counter()

    _check_diffusers_version()

    from diffusers import CogVideoXImageToVideoPipeline
    import huggingface_hub as _hfh
    import diffusers as _dffs

    model_id  = config.MODEL_ID
    cache_dir = _MODEL_CACHE_DIR
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Confirm cache state before loading ────────────────────────────────────
    cache_exists = os.path.isdir(cache_dir)
    if cache_exists:
        try:
            entries = os.listdir(cache_dir)
            cache_size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, files in os.walk(cache_dir)
                for f in files
            ) / (1024 ** 2)
        except OSError:
            entries, cache_size_mb = [], 0.0
    else:
        entries, cache_size_mb = [], 0.0

    logger.info("=" * 55)
    logger.info(">>> MODEL LOAD STARTED")
    logger.info("  model_id          : %s", model_id)
    logger.info("  cache_dir         : %s", cache_dir)
    logger.info("  cache_dir exists  : %s", cache_exists)
    logger.info("  cache entries     : %d  (%.0f MB total)", len(entries), cache_size_mb)
    logger.info("  device            : %s", device)
    logger.info("  huggingface_hub   : %s", _hfh.__version__)
    logger.info("  diffusers         : %s", _dffs.__version__)
    logger.info("  torch             : %s", torch.__version__)
    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        free_vram  = (torch.cuda.get_device_properties(0).total_memory
                      - torch.cuda.memory_allocated(0)) / (1024 ** 3)
        logger.info("  GPU               : %s", torch.cuda.get_device_name(0))
        logger.info("  VRAM total/free   : %.1f GB / %.1f GB", total_vram, free_vram)
    logger.info("=" * 55)

    if cache_size_mb < 100:
        # Cache looks empty or dangerously small — warn loudly.
        # The worker will still try local_files_only and fail fast with a clear
        # FileNotFoundError rather than starting a silent download.
        logger.warning(
            "!!! Cache at %s is nearly empty (%.0f MB). "
            "Expected ~18 000 MB. "
            "The image was probably built without running download_model.py. "
            "Rebuild with: docker build -t cogvideox-worker:latest .",
            cache_dir, cache_size_mb,
        )

    logger.info(">>> Loading weights from local cache (local_files_only=True) …")
    logger.info("    If this raises FileNotFoundError the cache is empty — rebuild the image.")

    t_pretrained = time.perf_counter()
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype      = torch.bfloat16,
        cache_dir        = cache_dir,
        local_files_only = True,   # never touch the network at runtime
    )
    logger.info(">>> from_pretrained() done in %.1fs", time.perf_counter() - t_pretrained)

    # ── Memory optimisations ──────────────────────────────────────────────────
    # VAE slicing — decodes one frame at a time through the VAE decoder.
    # The official docs call this alongside tiling. Without it the full video
    # latent tensor is decoded at once, causing OOM on consumer GPUs and
    # introducing decoding artifacts when the VAE runs out of headroom.
    if config.VAE_SLICING:
        pipe.vae.enable_slicing()
        logger.info("  VAE slicing       : enabled")

    # VAE tiling — splits the spatial dimensions into tiles for the decoder.
    # Additive with slicing: both should be enabled for maximum stability.
    if config.VAE_TILING:
        pipe.vae.enable_tiling()
        logger.info("  VAE tiling        : enabled")

    t_device = time.perf_counter()
    if device == "cuda" and config.CPU_OFFLOAD:
        # enable_model_cpu_offload() — official recommendation for 24 GB GPUs.
        # Keeps the full model on CPU and moves each sub-model to GPU only when
        # it is needed, then immediately moves it back. Uses ~19 GB VRAM.
        #
        # DO NOT use enable_sequential_cpu_offload() for quality work:
        #   - Moves individual LAYERS (not sub-models) CPU↔GPU
        #   - ~4 GB VRAM but 10-20× slower inference
        #   - Extra precision casts between CPU float32 and GPU bfloat16 on
        #     every layer boundary introduce cumulative rounding errors that
        #     manifest as temporal artifacts, unstable details, and washed-out
        #     motion — exactly the symptoms reported.
        pipe.enable_model_cpu_offload()
        logger.info("  CPU offload       : model_cpu_offload enabled (%.1fs)",
                    time.perf_counter() - t_device)
    else:
        pipe.to(device)
        if device == "cuda":
            free_after = (torch.cuda.get_device_properties(0).total_memory
                          - torch.cuda.memory_allocated(0)) / (1024 ** 3)
            logger.info("  Moved to GPU in %.1fs — free VRAM: %.1f GB",
                        time.perf_counter() - t_device, free_after)

    total_load = time.perf_counter() - t_load
    logger.info(">>> MODEL LOADED in %.1fs — worker is ready", total_load)
    logger.info("=" * 55)

    return pipe


# ── Public API ────────────────────────────────────────────────────────────────

def get_pipeline():
    """
    Return the loaded pipeline singleton.

    First call: loads from local cache (~30–90s depending on GPU / CPU offload).
    Subsequent calls: returns the cached object instantly (0ms).
    """
    global _pipeline
    if _pipeline is None:
        logger.info("get_pipeline(): first call — loading model")
        _pipeline = _load_pipeline()
    else:
        logger.debug("get_pipeline(): returning cached singleton")
    return _pipeline


def warm_up() -> None:
    """
    Pre-load the pipeline during container initialisation.

    Called at module level in handler.py so the model is fully loaded before
    RunPod marks the worker as ready and sends the first job.  Without this,
    the first request triggers the load inside handler() and times out.
    """
    logger.info("=== warm_up() started — loading model before first job ===")
    t = time.perf_counter()
    get_pipeline()
    logger.info("=== warm_up() complete in %.1fs — worker is ready to accept jobs ===",
                time.perf_counter() - t)
