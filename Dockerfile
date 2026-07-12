# ─── CogVideoX-5B-I2V Worker ──────────────────────────────────────────────────
# RunPod Serverless — Image-to-Video generation using diffusers stable release.
#
# Base image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
#   • Verified to exist on Docker Hub
#   • Python 3.11, PyTorch 2.4.0, CUDA 12.1
#   • Uses the -runtime variant (not -devel) — smaller image, no compiler
#
# Build:
#   docker build -t cogvideox-worker:latest .
#
# Smoke-test (no GPU required):
#   docker run --rm cogvideox-worker:latest python verify_imports.py
#
# Full test (GPU required):
#   docker run --rm --gpus all \
#     -e CPU_OFFLOAD=true \
#     cogvideox-worker:latest

# ── Base image ─────────────────────────────────────────────────────────────────
# pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime is a verified, existing tag.
# https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# diffusers 0.32.1 is a stable PyPI release that includes CogVideoXImageToVideoPipeline.
# No git-source installs — fully reproducible builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY verify_imports.py .
COPY config.py         .
COPY model_manager.py  .
COPY handler.py        .

# ── Output directory ───────────────────────────────────────────────────────────
RUN mkdir -p /tmp/cogvideox_output

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# handler.py calls runpod.serverless.start() at module level (not inside
# if __name__ guard) — required by RunPod's scanner and runtime.
CMD ["python", "-u", "/app/handler.py"]
