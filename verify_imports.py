"""
verify_imports.py
─────────────────
Smoke-test script — run manually after the image is built to verify
all critical imports work before the first real job.

Usage (no GPU required):
  docker run --rm cogvideox-worker:latest python verify_imports.py

Expected output when everything is correct:
  [OK] diffusers.CogVideoXImageToVideoPipeline
  [OK] diffusers.utils.export_to_video
  [OK] transformers
  [OK] accelerate
  [OK] runpod
  [OK] PIL
  [OK] imageio
  [OK] torch

  diffusers       : 0.32.1
  transformers    : 4.52.4
  accelerate      : 1.6.0
  huggingface_hub : <resolved by pip>
  torch           : 2.4.x
  CUDA available  : True/False

  All checks passed.
"""

import sys

errors = []


def check(label, fn):
    try:
        fn()
        print(f"[OK] {label}")
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}", file=sys.stderr)
        errors.append(label)


# ── Critical: CogVideoX pipeline ──────────────────────────────────────────────
check(
    "diffusers.CogVideoXImageToVideoPipeline",
    lambda: __import__(
        "diffusers", fromlist=["CogVideoXImageToVideoPipeline"]
    ).CogVideoXImageToVideoPipeline,
)

check(
    "diffusers.utils.export_to_video",
    lambda: __import__(
        "diffusers.utils", fromlist=["export_to_video"]
    ).export_to_video,
)

# ── Supporting stack ───────────────────────────────────────────────────────────
check("transformers",  lambda: __import__("transformers"))
check("accelerate",    lambda: __import__("accelerate"))
check("runpod",        lambda: __import__("runpod"))
check("PIL",           lambda: __import__("PIL"))
check("imageio",       lambda: __import__("imageio"))
check("torch",         lambda: __import__("torch"))

# ── Version summary ────────────────────────────────────────────────────────────
if not errors:
    import diffusers, transformers, accelerate, huggingface_hub, torch
    print()
    print(f"diffusers       : {diffusers.__version__}")
    print(f"transformers    : {transformers.__version__}")
    print(f"accelerate      : {accelerate.__version__}")
    print(f"huggingface_hub : {huggingface_hub.__version__}")
    print(f"torch           : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print()
    print("All checks passed.")
    sys.exit(0)
else:
    print(f"\n{len(errors)} check(s) FAILED: {errors}", file=sys.stderr)
    sys.exit(1)
