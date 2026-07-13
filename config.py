# ─── CogVideoX Worker Configuration ──────────────────────────────────────────
# All values are read from environment variables.
# Defaults here must stay in sync with the ENV declarations in Dockerfile.
#
# Cache path contract
# ───────────────────
# MODEL_CACHE_DIR is the single directory used by every component:
#   • download_model.py  — writes weights here at docker build time
#   • model_manager.py   — reads weights from here at runtime
#   • Dockerfile ENV     — sets this to /runpod-volume/models at image level
#
# Do NOT change this default without also changing:
#   Dockerfile: ENV MODEL_CACHE_DIR=...
#   model_manager.py: os.environ.get("MODEL_CACHE_DIR", ...)

import os

# ── Model ──────────────────────────────────────────────────────────────────────

# Official THUDM CogVideoX-5B Image-to-Video model (Apache 2.0, no token needed).
MODEL_ID: str = os.getenv("MODEL_ID", "THUDM/CogVideoX-5b-I2V")

# Single cache directory for ALL HuggingFace weight files.
# Must match ENV MODEL_CACHE_DIR in Dockerfile and os.environ.get("MODEL_CACHE_DIR")
# in model_manager.py.  All three must point to the same path.
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")

# ── Generation defaults ────────────────────────────────────────────────────────
# CogVideoX-5B-I2V official defaults:
#   resolution : 720 × 480 px
#   num_frames : 49  (~6 s at 8 fps)
#   steps      : 50
#   guidance   : 6.0

DEFAULT_HEIGHT:     int   = int(os.getenv("DEFAULT_HEIGHT",     "480"))
DEFAULT_WIDTH:      int   = int(os.getenv("DEFAULT_WIDTH",      "720"))
DEFAULT_NUM_FRAMES: int   = int(os.getenv("DEFAULT_NUM_FRAMES", "49"))
DEFAULT_STEPS:      int   = int(os.getenv("DEFAULT_STEPS",      "50"))

# guidance_scale=6.0 matches the official CogVideoX-5B-I2V default.
# use_dynamic_cfg=True in handler.py decays this linearly to 1.0 across
# timesteps — so this is the starting value, not a constant CFG.
DEFAULT_GUIDANCE:   float = float(os.getenv("DEFAULT_GUIDANCE", "6.0"))
DEFAULT_SEED:       int   = int(os.getenv("DEFAULT_SEED",       "-1"))   # -1 = random

# Only k*4+1 values are valid for CogVideoX (VAE temporal scale factor = 4).
# Valid sequence: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49.
MAX_NUM_FRAMES:     int   = int(os.getenv("MAX_NUM_FRAMES",     "49"))

# T5 encoder maximum sequence length. Must match the transformer config.
# Without this, long prompts are silently truncated, discarding scene details.
MAX_SEQUENCE_LENGTH: int  = int(os.getenv("MAX_SEQUENCE_LENGTH", "226"))

# ── Memory optimisation ────────────────────────────────────────────────────────

# VAE slicing — decodes one frame at a time through the VAE decoder.
# Official recommendation: call alongside tiling. Without it the full video
# latent tensor is decoded at once, causing OOM on consumer GPUs and
# decoding artifacts when VRAM headroom is tight.
VAE_SLICING: bool = os.getenv("VAE_SLICING", "true").lower() == "true"

# VAE tiling — splits spatial dimensions into tiles for the decoder.
# Additive with slicing: both should be true for maximum stability.
# Safe to leave on everywhere — no quality loss, only VRAM reduction.
VAE_TILING: bool = os.getenv("VAE_TILING", "true").lower() == "true"

# Model CPU offload — official recommendation for 24 GB GPUs.
# Uses ~19 GB VRAM, moves each sub-model to GPU only when needed.
# DO NOT use sequential CPU offload (old setting) — it moves individual
# layers, is 10-20× slower, and introduces precision artifacts that cause
# unstable motion and washed-out scene generation.
CPU_OFFLOAD: bool = os.getenv("CPU_OFFLOAD", "true").lower() == "true"

# ── Output ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/cogvideox_output")
OUTPUT_FPS: int = int(os.getenv("OUTPUT_FPS", "8"))

# ── S3-compatible storage (optional) ──────────────────────────────────────────
# Leave S3_BUCKET empty → handler returns base64-encoded MP4 instead of a URL.

S3_BUCKET:     str = os.getenv("S3_BUCKET",     "")
S3_ENDPOINT:   str = os.getenv("S3_ENDPOINT",   "")   # e.g. Cloudflare R2 endpoint
S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
S3_REGION:     str = os.getenv("S3_REGION",     "us-east-1")
S3_PUBLIC_URL: str = os.getenv("S3_PUBLIC_URL", "")   # CDN prefix, e.g. https://cdn.example.com

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
