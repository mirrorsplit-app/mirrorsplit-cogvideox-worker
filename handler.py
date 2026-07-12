"""
handler.py — CogVideoX-5B-I2V RunPod Serverless Worker
────────────────────────────────────────────────────────
RunPod calls handler(job) once per inference request.

Input schema (all optional except prompt):
  prompt          str   Required. Video description. Max 2000 chars.
  image           str   Optional. HTTPS URL or base64 JPEG/PNG.
                        Triggers Image-to-Video mode when provided.
                        Omit for Text-to-Video mode.
  negative_prompt str   Optional. What to avoid.
  num_frames      int   Optional. Default: 49 (≈6s @ 8fps). Max: 49.
  num_steps       int   Optional. Default: 50. Range: 1–100.
  guidance_scale  float Optional. Default: 6.0.
  seed            int   Optional. -1 = random.
  fps             int   Optional. Output FPS. Default: 8.

Output schema (success):
  video_url       str   Publicly accessible URL (when S3 is configured).
  video_b64       str   Base64-encoded MP4 (when S3 is not configured).
  mode            str   "i2v" or "t2v".
  num_frames      int   Frames generated.
  fps             int   Output FPS.
  seed            int   Seed used (for reproducibility).
  generation_time float Seconds spent on inference only (seconds).

Output schema (error):
  error           str   Human-readable error description.
  traceback       str   Full Python traceback (always included on errors).
  stage           str   Which stage failed (validation/model_load/inference/export/upload).
"""

import base64
import io
import logging
import os
import random
import time
import traceback
import uuid

import runpod
import torch

import config
from model_manager import get_pipeline, warm_up


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("cogvideox.handler")
logger.info("=" * 60)
logger.info("CogVideoX worker process starting")
logger.info("  runpod version : %s", runpod.__version__)
logger.info("  torch version  : %s", torch.__version__)
logger.info("  CUDA available : %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("  GPU            : %s", torch.cuda.get_device_name(0))
    logger.info("  VRAM           : %.1f GB",
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
logger.info("  MODEL_ID       : %s", config.MODEL_ID)
logger.info("  CPU_OFFLOAD    : %s", config.CPU_OFFLOAD)
logger.info("  VAE_TILING     : %s", config.VAE_TILING)
logger.info("=" * 60)


# ── Warm-up: load the model NOW, at container startup ─────────────────────────
# This is the fix for the timeout. Without this, the 18 GB model is downloaded
# and loaded inside the first handler() call, which takes 5–15 minutes and
# exceeds RunPod's execution timeout before inference even starts.
#
# With warm_up() here (module level), RunPod's container initialisation phase
# runs this before any job is accepted. The container stays warm between jobs
# and get_pipeline() returns the cached singleton instantly.
_t_warmup_start = time.perf_counter()
logger.info(">>> Starting model warm-up (this will take several minutes on first cold start)")
try:
    warm_up()
    logger.info(">>> Warm-up complete in %.1fs — worker is ready to accept jobs",
                time.perf_counter() - _t_warmup_start)
except Exception as _warmup_exc:
    # Log the full traceback so it appears in RunPod container logs.
    logger.error(">>> Warm-up FAILED — worker will attempt to load model per-request")
    logger.error(traceback.format_exc())
    # Do NOT re-raise: let RunPod start the handler anyway.
    # If the model fails to load, handler() will catch it and return {"error": ...}.


# ── Input helpers ──────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    pass


def _load_image(src: str):
    """Load a PIL Image from an HTTPS URL or base64 string."""
    from PIL import Image
    import requests

    if not src:
        return None

    # data-URI base64
    if src.startswith("data:"):
        _, _, b64 = src.partition(",")
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    # plain base64 (no http prefix)
    if not src.startswith("http"):
        try:
            raw = base64.b64decode(src)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass

    # HTTPS / HTTP URL
    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    raise ValidationError("'image' must be an HTTPS URL or a base64-encoded image")


def validate(job_input: dict) -> dict:
    """Validate and normalise raw RunPod job input."""

    # prompt
    prompt = job_input.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("'prompt' is required and must be a non-empty string")
    prompt = prompt.strip()
    if len(prompt) > 2000:
        raise ValidationError("'prompt' must be 2000 characters or fewer")

    # negative_prompt
    negative_prompt = job_input.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValidationError("'negative_prompt' must be a string")

    # image
    raw_image = job_input.get("image", "")
    pil_image = None
    mode = "t2v"
    if raw_image and isinstance(raw_image, str) and raw_image.strip():
        try:
            pil_image = _load_image(raw_image.strip())
            mode = "i2v"
        except Exception as exc:
            raise ValidationError(f"Failed to load image: {exc}") from exc

    # num_frames
    raw_frames = job_input.get("num_frames", config.DEFAULT_NUM_FRAMES)
    try:
        num_frames = int(raw_frames)
    except (TypeError, ValueError):
        raise ValidationError(f"'num_frames' must be an integer, got: {raw_frames!r}")
    num_frames = max(1, min(num_frames, config.MAX_NUM_FRAMES))

    # num_steps
    raw_steps = job_input.get("num_steps", config.DEFAULT_STEPS)
    try:
        num_steps = int(raw_steps)
    except (TypeError, ValueError):
        raise ValidationError(f"'num_steps' must be an integer, got: {raw_steps!r}")
    num_steps = max(1, min(num_steps, 100))

    # guidance_scale
    raw_guidance = job_input.get("guidance_scale", config.DEFAULT_GUIDANCE)
    try:
        guidance_scale = float(raw_guidance)
    except (TypeError, ValueError):
        raise ValidationError(f"'guidance_scale' must be a float, got: {raw_guidance!r}")

    # seed
    raw_seed = job_input.get("seed", config.DEFAULT_SEED)
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError):
        raise ValidationError(f"'seed' must be an integer, got: {raw_seed!r}")
    if seed == -1:
        seed = random.randint(0, 2 ** 32 - 1)

    # fps
    raw_fps = job_input.get("fps", config.OUTPUT_FPS)
    try:
        fps = int(raw_fps)
    except (TypeError, ValueError):
        fps = config.OUTPUT_FPS
    fps = max(1, min(fps, 60))

    return {
        "prompt":          prompt,
        "negative_prompt": negative_prompt,
        "pil_image":       pil_image,
        "mode":            mode,
        "num_frames":      num_frames,
        "num_steps":       num_steps,
        "guidance_scale":  guidance_scale,
        "seed":            seed,
        "fps":             fps,
    }


# ── Video export ───────────────────────────────────────────────────────────────

def export_video(frames: list, fps: int, output_path: str) -> None:
    """Write list of PIL Images to MP4 using diffusers utility or imageio."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Try diffusers helper first
    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        logger.debug("export_video: used diffusers export_to_video")
        return
    except Exception as exc:
        logger.warning("export_video: diffusers export_to_video failed (%s) — trying imageio", exc)

    # Fallback: imageio
    import imageio
    import numpy as np
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            if hasattr(frame, "convert"):
                arr = np.array(frame.convert("RGB"))
            elif hasattr(frame, "numpy"):
                arr = frame.numpy()
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = np.array(frame)
            writer.append_data(arr)
    finally:
        writer.close()
    logger.debug("export_video: used imageio fallback")


# ── S3 upload ──────────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, key: str) -> str:
    import boto3
    from botocore.config import Config as BotoConfig
    kwargs = {
        "aws_access_key_id":     config.S3_ACCESS_KEY,
        "aws_secret_access_key": config.S3_SECRET_KEY,
        "region_name":           config.S3_REGION,
        "config":                BotoConfig(signature_version="s3v4"),
    }
    if config.S3_ENDPOINT:
        kwargs["endpoint_url"] = config.S3_ENDPOINT
    s3 = boto3.client("s3", **kwargs)
    s3.upload_file(local_path, config.S3_BUCKET, key,
                   ExtraArgs={"ContentType": "video/mp4"})
    if config.S3_PUBLIC_URL:
        return f"{config.S3_PUBLIC_URL.rstrip('/')}/{key}"
    return f"https://{config.S3_BUCKET}.s3.{config.S3_REGION}.amazonaws.com/{key}"


# ── Timing helper ──────────────────────────────────────────────────────────────

def _elapsed(t_start: float) -> str:
    return f"{time.perf_counter() - t_start:.2f}s"


# ── RunPod handler ─────────────────────────────────────────────────────────────

def handler(job):
    """
    RunPod Serverless entry point.
    Receives: job = {"id": str, "input": dict}
    Returns:  dict (success payload or {"error": str, "traceback": str, "stage": str})
    """
    t_job_start = time.perf_counter()
    job_id      = job.get("id", "local")
    job_input   = job.get("input", {})

    logger.info("[%s] ── Job received ──────────────────────────────────", job_id)
    logger.info("[%s] raw input keys: %s", job_id, list(job_input.keys()))

    # ── Stage 1: Validate ──────────────────────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 1/6 — validation", job_id)
    try:
        params = validate(job_input)
    except ValidationError as exc:
        logger.warning("[%s] Validation failed: %s", job_id, exc)
        return {"error": str(exc), "traceback": traceback.format_exc(), "stage": "validation"}

    logger.info(
        "[%s] validation OK in %s — mode=%s frames=%d steps=%d seed=%d",
        job_id, _elapsed(t_stage), params["mode"],
        params["num_frames"], params["num_steps"], params["seed"],
    )

    # ── Stage 2: Load pipeline ─────────────────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 2/6 — load pipeline (should be instant on warm worker)", job_id)
    try:
        pipe = get_pipeline()
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Pipeline load FAILED in %s:\n%s", job_id, _elapsed(t_stage), tb)
        return {"error": f"Failed to load model: {exc}", "traceback": tb, "stage": "model_load"}

    logger.info("[%s] pipeline ready in %s", job_id, _elapsed(t_stage))

    # ── Stage 3: Build inference kwargs ───────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 3/6 — build inference kwargs", job_id)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(params["seed"])

    call_kwargs = {
        "prompt":              params["prompt"],
        "height":              config.DEFAULT_HEIGHT,
        "width":               config.DEFAULT_WIDTH,
        "num_frames":          params["num_frames"],
        "num_inference_steps": params["num_steps"],
        "guidance_scale":      params["guidance_scale"],
        "generator":           generator,
    }
    if params["negative_prompt"]:
        call_kwargs["negative_prompt"] = params["negative_prompt"]
    if params["mode"] == "i2v" and params["pil_image"] is not None:
        call_kwargs["image"] = params["pil_image"]

    logger.info(
        "[%s] inference kwargs: prompt='%s…' height=%d width=%d frames=%d steps=%d guidance=%.1f mode=%s",
        job_id,
        params["prompt"][:60], config.DEFAULT_HEIGHT, config.DEFAULT_WIDTH,
        params["num_frames"], params["num_steps"], params["guidance_scale"], params["mode"],
    )
    if torch.cuda.is_available():
        free_vram = (torch.cuda.get_device_properties(0).total_memory
                     - torch.cuda.memory_allocated(0)) / (1024 ** 3)
        logger.info("[%s] free VRAM before inference: %.1f GB", job_id, free_vram)

    # ── Stage 4: Inference ────────────────────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 4/6 — inference started (this is the long step)", job_id)
    try:
        output = pipe(**call_kwargs)
    except torch.cuda.OutOfMemoryError:
        tb = traceback.format_exc()
        logger.error("[%s] CUDA OOM after %s:\n%s", job_id, _elapsed(t_stage), tb)
        return {
            "error":     "GPU out of memory. Reduce num_frames or enable CPU offload.",
            "traceback": tb,
            "stage":     "inference",
        }
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Inference FAILED after %s:\n%s", job_id, _elapsed(t_stage), tb)
        return {"error": f"Generation failed: {exc}", "traceback": tb, "stage": "inference"}

    generation_time = round(time.perf_counter() - t_stage, 2)
    logger.info("[%s] inference complete in %.2fs", job_id, generation_time)

    # ── Stage 5: Export MP4 ───────────────────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 5/6 — export MP4", job_id)

    frames     = output.frames[0]
    filename   = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(config.OUTPUT_DIR, filename)

    logger.info("[%s] exporting %d frames → %s", job_id, len(frames), video_path)
    try:
        export_video(frames, fps=params["fps"], output_path=video_path)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("[%s] Export FAILED after %s:\n%s", job_id, _elapsed(t_stage), tb)
        return {"error": f"Failed to encode video: {exc}", "traceback": tb, "stage": "export"}

    file_size_mb = os.path.getsize(video_path) / (1024 ** 2)
    logger.info("[%s] export done in %s — file: %.1f MB", job_id, _elapsed(t_stage), file_size_mb)

    # ── Stage 6: Upload or base64 ─────────────────────────────────────────────
    t_stage = time.perf_counter()
    logger.info("[%s] Stage 6/6 — upload / encode output", job_id)

    video_url = ""
    video_b64 = ""

    if config.S3_BUCKET:
        try:
            video_url = upload_to_s3(video_path, f"cogvideox/{filename}")
            logger.info("[%s] S3 upload done in %s — url: %s", job_id, _elapsed(t_stage), video_url)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[%s] S3 upload FAILED after %s — falling back to base64:\n%s",
                         job_id, _elapsed(t_stage), tb)
            # Do NOT return error here — base64 fallback below will handle it

    if not video_url:
        logger.info("[%s] encoding video as base64 (no S3 bucket configured or upload failed)", job_id)
        t_b64 = time.perf_counter()
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        logger.info("[%s] base64 encode done in %s — payload size: %.1f MB",
                    job_id, _elapsed(t_b64), len(video_b64) / (1024 ** 2))

    try:
        os.remove(video_path)
    except OSError:
        pass

    total_time = round(time.perf_counter() - t_job_start, 2)
    logger.info(
        "[%s] ── Job COMPLETE — total: %.2fs  inference: %.2fs  has_url: %s  has_b64: %s ──",
        job_id, total_time, generation_time, bool(video_url), bool(video_b64),
    )

    return {
        "video_url":       video_url,
        "video_b64":       video_b64,
        "mode":            params["mode"],
        "num_frames":      params["num_frames"],
        "fps":             params["fps"],
        "seed":            params["seed"],
        "generation_time": generation_time,
        "total_time":      total_time,
    }


# ── RunPod Serverless start ────────────────────────────────────────────────────
# Module-level call — required by RunPod's deployment scanner AND runtime.
# Do NOT move this inside if __name__ == "__main__" or any other guard.
runpod.serverless.start({"handler": handler})
