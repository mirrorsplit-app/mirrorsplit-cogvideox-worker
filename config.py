# ─── CogVideoX Worker Configuration ──────────────────────────────────────────
# All settings are read from environment variables.
# Set these in the RunPod template "Environment Variables" section.

import os

# ── Model ──────────────────────────────────────────────────────────────────────

# Official THUDM CogVideoX-5B Image-to-Video model.
# Apache 2.0 license — no token required.
MODEL_ID: str = os.getenv("MODEL_ID", "THUDM/CogVideoX-5b-I2V")

# Local cache directory. Mount a RunPod network volume here to avoid
# re-downloading ~18 GB of weights on every cold start.
MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")

# ── Generation defaults ────────────────────────────────────────────────────────

# CogVideoX-5B-I2V official defaults:
#   resolution : 720 × 480
#   num_frames : 49  (6 seconds at ~8fps)
#   steps      : 50
#   guidance   : 6.0
DEFAULT_HEIGHT:      int   = int(os.getenv("DEFAULT_HEIGHT",     "480"))
DEFAULT_WIDTH:       int   = int(os.getenv("DEFAULT_WIDTH",      "720"))
DEFAULT_NUM_FRAMES:  int   = int(os.getenv("DEFAULT_NUM_FRAMES", "49"))
DEFAULT_STEPS:       int   = int(os.getenv("DEFAULT_STEPS",      "50"))
DEFAULT_GUIDANCE:    float = float(os.getenv("DEFAULT_GUIDANCE", "6.0"))
DEFAULT_SEED:        int   = int(os.getenv("DEFAULT_SEED",       "-1"))  # -1 = random

MAX_NUM_FRAMES:      int   = int(os.getenv("MAX_NUM_FRAMES",     "49"))

# ── Memory optimisation ────────────────────────────────────────────────────────

# Enable VAE tiling — reduces peak VRAM during decoding.
VAE_TILING: bool = os.getenv("VAE_TILING", "true").lower() == "true"

# Enable sequential CPU offloading.
# Required on GPUs with < 24 GB VRAM. Disable on A100/H100 for max speed.
CPU_OFFLOAD: bool = os.getenv("CPU_OFFLOAD", "true").lower() == "true"

# ── Output ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "/tmp/cogvideox_output")
OUTPUT_FPS: int = int(os.getenv("OUTPUT_FPS", "8"))

# ── S3-compatible storage (optional) ──────────────────────────────────────────
# Leave S3_BUCKET empty to return base64-encoded video instead of a URL.

S3_BUCKET:     str = os.getenv("S3_BUCKET",     "")
S3_ENDPOINT:   str = os.getenv("S3_ENDPOINT",   "")
S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
S3_REGION:     str = os.getenv("S3_REGION",     "us-east-1")
S3_PUBLIC_URL: str = os.getenv("S3_PUBLIC_URL", "")

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
