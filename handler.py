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
  generation_time float Seconds spent generating.

Output schema (error):
  error           str   Human-readable error description.
"""

import base64
import io
import logging
import os
import random
import time
import uuid

import runpod
import torch

import config
from model_manager import get_pipeline

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("cogvideox.handler")
logger.info("CogVideoX worker starting — runpod version: %s", runpod.__version__)


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

    try:
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)
        return
    except Exception:
        pass

    # Fallback: imageio
    import imageio
    import numpy as np
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
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
    writer.close()


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


# ── RunPod handler ─────────────────────────────────────────────────────────────

def handler(job):
    """
    RunPod Serverless entry point.
    Receives: job = {"id": str, "input": dict}
    Returns:  dict (success payload or {"error": str})
    """
    job_id    = job.get("id", "local")
    job_input = job.get("input", {})

    logger.info("[%s] Job received", job_id)

    # 1. Validate input
    try:
        params = validate(job_input)
    except ValidationError as exc:
        logger.warning("[%s] Validation failed: %s", job_id, exc)
        return {"error": str(exc)}

    logger.info(
        "[%s] mode=%s frames=%d steps=%d seed=%d",
        job_id, params["mode"], params["num_frames"],
        params["num_steps"], params["seed"],
    )

    # 2. Load pipeline (singleton — instant on warm workers)
    try:
        pipe = get_pipeline()
    except Exception as exc:
        logger.exception("[%s] Failed to load pipeline", job_id)
        return {"error": f"Failed to load model: {exc}"}

    # 3. Build generator
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(params["seed"])

    # 4. Build call kwargs
    # CogVideoXImageToVideoPipeline signature:
    #   pipe(prompt, image=None, negative_prompt=None, height, width,
    #        num_frames, num_inference_steps, guidance_scale, generator)
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

    # 5. Generate
    t0 = time.perf_counter()
    try:
        output = pipe(**call_kwargs)
    except torch.cuda.OutOfMemoryError:
        logger.exception("[%s] CUDA OOM", job_id)
        return {"error": "GPU out of memory. Reduce num_frames or enable CPU offload."}
    except Exception as exc:
        logger.exception("[%s] Generation error", job_id)
        return {"error": f"Generation failed: {exc}"}

    generation_time = round(time.perf_counter() - t0, 2)
    logger.info("[%s] Generated in %.1fs", job_id, generation_time)

    # 6. Export to MP4
    frames     = output.frames[0]
    filename   = f"{uuid.uuid4().hex}.mp4"
    video_path = os.path.join(config.OUTPUT_DIR, filename)

    try:
        export_video(frames, fps=params["fps"], output_path=video_path)
    except Exception as exc:
        logger.exception("[%s] Export failed", job_id)
        return {"error": f"Failed to encode video: {exc}"}

    # 7. Upload to S3 or encode as base64
    video_url = ""
    video_b64 = ""

    if config.S3_BUCKET:
        try:
            video_url = upload_to_s3(video_path, f"cogvideox/{filename}")
            logger.info("[%s] Uploaded: %s", job_id, video_url)
        except Exception as exc:
            logger.exception("[%s] S3 upload failed — falling back to base64", job_id)

    if not video_url:
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        os.remove(video_path)
    except OSError:
        pass

    return {
        "video_url":       video_url,
        "video_b64":       video_b64,
        "mode":            params["mode"],
        "num_frames":      params["num_frames"],
        "fps":             params["fps"],
        "seed":            params["seed"],
        "generation_time": generation_time,
    }


# ── RunPod Serverless start ────────────────────────────────────────────────────
# Module-level call — required by RunPod's deployment scanner AND runtime.
# Do NOT move this inside if __name__ == "__main__" or any other guard.
runpod.serverless.start({"handler": handler})
