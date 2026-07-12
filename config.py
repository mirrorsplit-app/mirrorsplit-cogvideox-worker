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
DEFAULT_GUIDANCE:   float = float(os.getenv("DEFAULT_GUIDANCE", "6.0"))
DEFAULT_SEED:       int   = int(os.getenv("DEFAULT_SEED",       "-1"))   # -1 = random

MAX_NUM_FRAMES:     int   = int(os.getenv("MAX_NUM_FRAMES",     "49"))

# ── Memory optimisation ────────────────────────────────────────────────────────

# VAE tiling — reduces peak VRAM during decoding.  Safe to leave on everywhere.
VAE_TILING: bool = os.getenv("VAE_TILING", "true").lower() == "true"

# Sequential CPU offload — required on GPUs with < 24 GB VRAM.
# Set CPU_OFFLOAD=false on A40 / A100 / H100 for faster inference.
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
