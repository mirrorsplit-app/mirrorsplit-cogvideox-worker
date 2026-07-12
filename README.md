# CogVideoX Worker

RunPod Serverless worker for **CogVideoX-5B Image-to-Video** generation.

- Model: `THUDM/CogVideoX-5b-I2V`
- Pipeline: `CogVideoXImageToVideoPipeline` (diffusers 0.32.2)
- Output: MP4 video via S3/R2 URL or base64

---

## Input schema

```json
{
  "input": {
    "prompt":          "A cat sitting on a window sill, sunlight streaming in",
    "image":           "https://example.com/cat.jpg",
    "negative_prompt": "blurry, distorted",
    "duration":        6,
    "aspect_ratio":    "16:9",
    "num_steps":       50,
    "guidance_scale":  6.0,
    "seed":            -1,
    "fps":             8
  }
}
```

| Field             | Type   | Required | Default | Notes                                                    |
|-------------------|--------|----------|---------|----------------------------------------------------------|
| `prompt`          | string | ✅       | —       | Max 2000 chars                                           |
| `image`           | string | no       | —       | HTTPS URL or base64. Triggers I2V mode when provided.    |
| `negative_prompt` | string | no       | `""`    |                                                          |
| `duration`        | float  | no       | `6.0`   | Video length in seconds. Converted to frames via fps.    |
| `aspect_ratio`    | string | no       | `"16:9"`| `"16:9"` `"9:16"` `"1:1"` `"4:3"` `"3:4"` `"21:9"`     |
| `num_frames`      | int    | no       | —       | Explicit frame count. Overrides `duration` when present. |
| `num_steps`       | int    | no       | `50`    | Denoising steps. Range 1–100.                            |
| `guidance_scale`  | float  | no       | `6.0`   | CFG guidance scale.                                      |
| `seed`            | int    | no       | `-1`    | `-1` = random                                            |
| `fps`             | int    | no       | `8`     | Output video FPS (also used in duration→frames calc).    |

### duration → num_frames conversion

```
num_frames = round(duration × fps)   # rounded to nearest odd number
num_frames = min(num_frames, MAX_NUM_FRAMES)   # capped at 49
```

Example: `duration=6, fps=8` → `48` → rounded up to odd `49`.

### aspect_ratio → resolution mapping

| aspect_ratio | width × height |
|---|---|
| `16:9` (default) | 720 × 480 |
| `9:16` | 480 × 720 |
| `1:1` | 480 × 480 |
| `4:3` | 640 × 480 |
| `3:4` | 480 × 640 |
| `21:9` | 848 × 360 |

---

## Output schema

### Success
```json
{
  "video_url":       "https://pub-xxx.r2.dev/cogvideox/abc.mp4",
  "video_b64":       "",
  "mode":            "i2v",
  "num_frames":      49,
  "fps":             8,
  "seed":            1234567890,
  "generation_time": 45.2,
  "total_time":      52.1
}
```

`video_url` is populated when `S3_BUCKET` is configured. Otherwise `video_b64` contains the raw base64 MP4.

### Error
```json
{
  "error":     "GPU out of memory.",
  "traceback": "Traceback (most recent call last): ...",
  "stage":     "inference"
}
```

---

## Environment variables

### Model

| Variable          | Default                   | Description                                          |
|-------------------|---------------------------|------------------------------------------------------|
| `MODEL_ID`        | `THUDM/CogVideoX-5b-I2V`  | HuggingFace model repo                               |
| `MODEL_CACHE_DIR` | `/runpod-volume/models`   | Weight cache — must be on a mounted network volume   |

### Generation

| Variable              | Default | Description                                  |
|-----------------------|---------|----------------------------------------------|
| `DEFAULT_HEIGHT`      | `480`   | Fallback output height                       |
| `DEFAULT_WIDTH`       | `720`   | Fallback output width                        |
| `DEFAULT_NUM_FRAMES`  | `49`    | Frame count when neither duration nor num_frames is sent |
| `MAX_NUM_FRAMES`      | `49`    | Hard cap on frame count                      |
| `DEFAULT_STEPS`       | `50`    | Denoising steps                              |
| `DEFAULT_GUIDANCE`    | `6.0`   | CFG guidance scale                           |
| `CPU_OFFLOAD`         | `true`  | Sequential CPU offload. Disable on A40/A100. |
| `VAE_TILING`          | `true`  | Reduces peak VRAM during VAE decode.         |
| `OUTPUT_FPS`          | `8`     | Default output FPS                           |

### S3 / Cloudflare R2 output (recommended)

Set these to avoid the ~5 MB RunPod response-body limit that silently drops base64 payloads.

| Variable        | Default      | Description                                              |
|-----------------|--------------|----------------------------------------------------------|
| `S3_BUCKET`     | *(empty)*    | Bucket name. Leave empty → returns base64 instead.       |
| `S3_ENDPOINT`   | *(empty)*    | S3-compatible endpoint. For R2: `https://<account>.r2.cloudflarestorage.com` |
| `S3_ACCESS_KEY` | *(empty)*    | Access key ID                                            |
| `S3_SECRET_KEY` | *(empty)*    | Secret access key                                        |
| `S3_REGION`     | `us-east-1`  | Region (use `auto` for Cloudflare R2)                    |
| `S3_PUBLIC_URL` | *(empty)*    | Public CDN prefix. For R2: `https://pub-xxx.r2.dev`      |

### Cloudflare R2 — quick setup

1. Create an R2 bucket in the Cloudflare dashboard
2. Create an API token with **Object Read & Write** on that bucket
3. Enable public access on the bucket and note the `pub-xxx.r2.dev` URL
4. Set these in your RunPod endpoint **Environment Variables**:

```
S3_BUCKET      = your-bucket-name
S3_ENDPOINT    = https://<account_id>.r2.cloudflarestorage.com
S3_ACCESS_KEY  = <r2_access_key_id>
S3_SECRET_KEY  = <r2_secret_access_key>
S3_REGION      = auto
S3_PUBLIC_URL  = https://pub-xxx.r2.dev
```

With R2 configured, the worker uploads the MP4 and returns `video_url` — no base64,
no payload size limit, and the browser streams the video directly from R2.

### Logging

| Variable    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `LOG_LEVEL` | `INFO`  | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Build and deploy

```bash
# Build (downloads ~18 GB model into image — runs once, cached after)
docker build -t cogvideox-worker:latest .

# Smoke-test (no GPU required)
docker run --rm cogvideox-worker:latest python verify_imports.py

# Full test (GPU required)
docker run --rm --gpus all \
  -e CPU_OFFLOAD=true \
  cogvideox-worker:latest

# Push
docker tag cogvideox-worker:latest yourname/cogvideox-worker:latest
docker push yourname/cogvideox-worker:latest
```

---

## GPU requirements

| GPU       | VRAM  | Recommended config                          |
|-----------|-------|---------------------------------------------|
| RTX 4090  | 24 GB | `CPU_OFFLOAD=true` + `VAE_TILING=true`      |
| A40       | 48 GB | `CPU_OFFLOAD=false` + `VAE_TILING=true`     |
| A100 80GB | 80 GB | `CPU_OFFLOAD=false` + `VAE_TILING=false`    |

---

## RunPod endpoint settings

Set **Execution Timeout** to at least `600` seconds (10 minutes).
CogVideoX inference takes 2–8 minutes depending on GPU and frame count.
The default 3-minute timeout will kill jobs mid-inference.
