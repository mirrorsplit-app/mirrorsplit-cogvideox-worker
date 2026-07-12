# CogVideoX Worker

RunPod Serverless worker for **CogVideoX-5B Image-to-Video** generation.

- Model: `THUDM/CogVideoX-5b-I2V`
- Pipeline: `CogVideoXImageToVideoPipeline` (diffusers stable 0.32.1)
- Output: MP4 video via S3 URL or base64

---

## Input schema

```json
{
  "input": {
    "prompt":          "A cat sitting on a window sill, sunlight streaming in",
    "image":           "https://example.com/cat.jpg",
    "negative_prompt": "blurry, distorted",
    "num_frames":      49,
    "num_steps":       50,
    "guidance_scale":  6.0,
    "seed":            -1,
    "fps":             8
  }
}
```

| Field             | Type   | Required | Default | Notes                                    |
|-------------------|--------|----------|---------|------------------------------------------|
| `prompt`          | string | ✅       | —       | Max 2000 chars                           |
| `image`           | string | no       | —       | HTTPS URL or base64. Triggers I2V mode.  |
| `negative_prompt` | string | no       | `""`    |                                          |
| `num_frames`      | int    | no       | `49`    | Max 49 (~6 seconds at 8fps)              |
| `num_steps`       | int    | no       | `50`    | 1–100                                    |
| `guidance_scale`  | float  | no       | `6.0`   |                                          |
| `seed`            | int    | no       | `-1`    | `-1` = random                            |
| `fps`             | int    | no       | `8`     | Output video FPS                         |

---

## Output schema

```json
{
  "video_url":       "https://your-bucket.s3.amazonaws.com/cogvideox/abc.mp4",
  "video_b64":       "",
  "mode":            "i2v",
  "num_frames":      49,
  "fps":             8,
  "seed":            1234567890,
  "generation_time": 45.2
}
```

`video_url` is populated when `S3_BUCKET` is set. Otherwise `video_b64` contains the base64 MP4.

---

## Environment variables

| Variable         | Default                   | Description                                     |
|------------------|---------------------------|-------------------------------------------------|
| `MODEL_ID`       | `THUDM/CogVideoX-5b-I2V`  | HuggingFace model repo                          |
| `MODEL_CACHE_DIR`| `/runpod-volume/models`   | Weight cache (use RunPod network volume)         |
| `DEFAULT_HEIGHT` | `480`                     | Output height in pixels                         |
| `DEFAULT_WIDTH`  | `720`                     | Output width in pixels                          |
| `DEFAULT_NUM_FRAMES` | `49`               | Default frame count (49 = ~6s at 8fps)          |
| `DEFAULT_STEPS`  | `50`                      | Denoising steps                                 |
| `DEFAULT_GUIDANCE` | `6.0`                  | Classifier-free guidance scale                  |
| `CPU_OFFLOAD`    | `true`                    | Sequential CPU offload (disable on A100/H100)   |
| `VAE_TILING`     | `true`                    | Reduces peak VRAM during decoding               |
| `OUTPUT_FPS`     | `8`                       | Default output FPS                              |
| `S3_BUCKET`      | *(empty)*                 | S3 bucket name (leave empty for base64 output)  |
| `S3_ENDPOINT`    | *(empty)*                 | S3-compatible endpoint (e.g. Cloudflare R2)     |
| `S3_ACCESS_KEY`  | *(empty)*                 |                                                 |
| `S3_SECRET_KEY`  | *(empty)*                 |                                                 |
| `S3_REGION`      | `us-east-1`               |                                                 |
| `S3_PUBLIC_URL`  | *(empty)*                 | CDN prefix prepended to object key              |
| `LOG_LEVEL`      | `INFO`                    | `DEBUG` / `INFO` / `WARNING` / `ERROR`          |

---

## Build and deploy

```bash
# Build
docker build -t cogvideox-worker:latest .

# Smoke-test (no GPU required)
docker run --rm cogvideox-worker:latest python verify_imports.py

# Full test (GPU required)
docker run --rm --gpus all \
  -e CPU_OFFLOAD=true \
  cogvideox-worker:latest

# Push to registry
docker tag cogvideox-worker:latest yourname/cogvideox-worker:latest
docker push yourname/cogvideox-worker:latest
```

Then create a RunPod Serverless endpoint pointing to your registry image.

---

## GPU requirements

| GPU           | VRAM  | Config                          |
|---------------|-------|---------------------------------|
| RTX 4090      | 24 GB | `CPU_OFFLOAD=true` + `VAE_TILING=true` |
| A40           | 48 GB | `CPU_OFFLOAD=false`             |
| A100 80GB     | 80 GB | `CPU_OFFLOAD=false`             |
