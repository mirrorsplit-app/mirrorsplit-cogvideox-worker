# ─── CogVideoX-5B-I2V Worker ──────────────────────────────────────────────────
# RunPod Serverless — Image-to-Video generation using diffusers 0.32.2.
#
# Build strategy
# ──────────────
# The 18 GB CogVideoX-5B-I2V weights are baked into the image during build.
# This is the only reliable way to avoid model downloads at job runtime:
#
#   Stage 1 (deps)   — install Python packages into /app/venv layer
#   Stage 2 (model)  — run download_model.py → writes weights into
#                      /runpod-volume/models as a Docker layer
#   Stage 3 (app)    — copy application code on top
#
# At runtime, model_manager.py calls from_pretrained(..., local_files_only=True)
# which reads straight from the baked layer — zero network I/O.
#
# Build:
#   docker build -t cogvideox-worker:latest .
#
# Smoke-test (no GPU, no model needed):
#   docker run --rm cogvideox-worker:latest python verify_imports.py
#
# Full test (GPU required):
#   docker run --rm --gpus all cogvideox-worker:latest
#
# Push:
#   docker tag cogvideox-worker:latest <registry>/cogvideox-worker:latest
#   docker push <registry>/cogvideox-worker:latest

# ── Base image ─────────────────────────────────────────────────────────────────
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

# ── Cache / model path ─────────────────────────────────────────────────────────
# MODEL_CACHE_DIR is the single source of truth for where weights live.
# download_model.py, model_manager.py, and config.py all read this env var.
# Setting it here bakes the default into the image so nothing else needs to
# be configured at runtime.
ENV MODEL_CACHE_DIR=/runpod-volume/models

# All HuggingFace libraries must point to the same directory.
# Setting them here ensures they're correct even if model_manager.py is
# imported in an unusual order or from a subprocess.
ENV HF_HOME=/runpod-volume/models
ENV HF_HUB_CACHE=/runpod-volume/models
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/models
ENV TRANSFORMERS_CACHE=/runpod-volume/models

# Disable xet / hf-transfer — causes "Background writer channel closed".
# Must be set before pip installs huggingface_hub.
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy and install first so changes to application code don't invalidate
# the pip layer (~10 GB of packages).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Bake model weights into the image ─────────────────────────────────────────
# Copy only the files needed by download_model.py, not the full app.
# This keeps the download layer separate from the code layer so pushing a
# code-only update doesn't re-upload 18 GB.
COPY download_model.py .
COPY config.py         .

# Create the cache directory so download_model.py can write into it.
RUN mkdir -p /runpod-volume/models /runpod-volume/tmp

# Download all 17 model files (~18 GB) into /runpod-volume/models.
# This is the slow step — only re-runs when download_model.py or
# requirements.txt changes.  All other code changes are below this line
# and are fast to rebuild and push.
RUN python download_model.py

# ── Application code ───────────────────────────────────────────────────────────
# Copied AFTER the model download so code edits don't invalidate the
# 18 GB model layer above.
COPY verify_imports.py .
COPY model_manager.py  .
COPY handler.py        .

# ── Output / temp directories ──────────────────────────────────────────────────
RUN mkdir -p /tmp/cogvideox_output /runpod-volume/tmp

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# -u = unbuffered stdout/stderr so logs appear in RunPod immediately.
# handler.py calls runpod.serverless.start() at module level — required by
# RunPod's deployment scanner.
CMD ["python", "-u", "/app/handler.py"]
