#!/usr/bin/env python3
"""Benchmark a HuggingFace object-detection model — no ROS required.

Usage examples
--------------
# Single image, 100 runs, auto device
python scripts/benchmark.py --model PekingU/rtdetr_v2_r18vd --image madison.jpg

# Directory of images, 200 runs, GPU only
python scripts/benchmark.py --model facebook/detr-resnet-50 \\
    --image /path/to/images/ --runs 200 --device cuda

# Smaller resolution
python scripts/benchmark.py --model hustvl/yolos-tiny --image test.jpg \\
    --image-size 512 --runs 50
"""

from venv_setup import auto_inject_venv
auto_inject_venv(packages=['torch', 'transformers'])

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# model_registry lives in the package; make it importable when running the
# script directly from the repo root (python scripts/benchmark.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers_bridge.model_registry import resolve_model_config


# ── Helpers ──────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def collect_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if not imgs:
            sys.exit(f"No images found in directory: {p}")
        return imgs
    sys.exit(f"Path does not exist: {p}")


def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def format_mem(bytes_: int) -> str:
    return f"{bytes_ / 1024**2:.1f} MiB"


def markdown_table(rows: list[dict]) -> str:
    headers = list(rows[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]

    def fmt_row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    lines = [fmt_row(headers), sep] + [fmt_row([r[h] for h in headers]) for r in rows]
    return "\n".join(lines)


# ── Core benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    images: list[Path],
    runs: int,
    image_size: int,
    device: torch.device,
) -> dict:
    cfg, matched_key = resolve_model_config(model_name)
    processor_cls = cfg["processor_cls"]
    model_cls = cfg["model_cls"]

    label = matched_key or "auto (fallback)"
    if matched_key is None:
        print(f"[warn] '{model_name}' not in registry — using Auto classes")
    elif not cfg.get("tested"):
        print(f"[warn] model type '{matched_key}' is untested in this package")

    print(f"  registry match : {label}")
    print(f"  processor      : {processor_cls.__name__}")
    print(f"  model          : {model_cls.__name__}")
    print(f"  device         : {device}")
    print(f"  image_size     : {image_size}")
    print()

    print("Loading model …", flush=True)
    processor = processor_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name).to(device).eval()

    # ── Sanity check: detections must be non-empty on the first image ─────────
    print("Sanity check …", flush=True)
    first_img = load_bgr(images[0])
    inputs = processor(
        images=first_img,
        return_tensors="pt",
        size={"height": image_size, "width": image_size},
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor(
            [(first_img.shape[0], first_img.shape[1])], device=device
        ),
        threshold=0.1,  # low threshold for the sanity check
    )[0]
    n_detections = int(results["scores"].shape[0])
    if n_detections == 0:
        print(f"[WARN] Sanity check FAILED — no detections on '{images[0].name}' "
              f"(threshold=0.1). The model may not suit this image, or loading failed.")
    else:
        print(f"[OK]   Sanity check passed — {n_detections} detection(s) on "
              f"'{images[0].name}' at threshold=0.1")
    print()

    # ── Warm-up (excluded from timing) ───────────────────────────────────────
    print("Warming up …", flush=True)
    dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    warm_inputs = processor(
        images=dummy,
        return_tensors="pt",
        size={"height": image_size, "width": image_size},
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(3):
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(**warm_inputs)
            else:
                model(**warm_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print("Warm-up done.\n", flush=True)

    # ── Timed runs ────────────────────────────────────────────────────────────
    n_images = len(images)
    latencies: list[float] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Running {runs} inference(s) …", flush=True)
    for i in range(runs):
        img = load_bgr(images[i % n_images])
        inputs = processor(
            images=img,
            return_tensors="pt",
            size={"height": image_size, "width": image_size},
        ).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(**inputs)
            else:
                model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        if (i + 1) % 10 == 0 or i == runs - 1:
            print(f"  {i + 1:>{len(str(runs))}}/{runs}   "
                  f"last={latencies[-1]:.1f} ms", flush=True)

    peak_mem = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    return {
        "latencies": latencies,
        "peak_mem_bytes": peak_mem,
        "n_detections_sanity": n_detections,
        "processor_cls": processor_cls.__name__,
        "model_cls": model_cls.__name__,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(
    model_name: str,
    image_size: int,
    device: torch.device,
    runs: int,
    data: dict,
) -> None:
    lats = data["latencies"]
    mean_ms = statistics.mean(lats)
    std_ms  = statistics.stdev(lats) if len(lats) > 1 else 0.0
    min_ms  = min(lats)
    max_ms  = max(lats)
    fps     = 1000.0 / mean_ms

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model      : {model_name}")
    print(f"  Device     : {device}")
    print(f"  Image size : {image_size}px")
    print(f"  Runs       : {runs}")
    print()
    print(f"  Mean       : {mean_ms:7.2f} ms")
    print(f"  Std        : {std_ms:7.2f} ms")
    print(f"  Min        : {min_ms:7.2f} ms")
    print(f"  Max        : {max_ms:7.2f} ms")
    print(f"  FPS        : {fps:7.1f}")
    if data["peak_mem_bytes"]:
        print(f"  Peak VRAM  : {format_mem(data['peak_mem_bytes'])}")
    print()

    # Per-run latency (compact — one value per line)
    print("Per-run latencies (ms):")
    for i, lat in enumerate(lats):
        print(f"  [{i:>{len(str(runs - 1))}}] {lat:.2f}")

    # Markdown table
    row = {
        "Model": model_name,
        "Device": str(device),
        "Size": f"{image_size}px",
        "Runs": runs,
        "Mean (ms)": f"{mean_ms:.2f}",
        "Std (ms)": f"{std_ms:.2f}",
        "Min (ms)": f"{min_ms:.2f}",
        "Max (ms)": f"{max_ms:.2f}",
        "FPS": f"{fps:.1f}",
        "Peak VRAM": format_mem(data["peak_mem_bytes"]) if data["peak_mem_bytes"] else "N/A",
    }

    print("\n--- Markdown table (copy into README) ---\n")
    print(markdown_table([row]))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark a HuggingFace object-detection model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--image", required=True, help="Image file or directory of images")
    p.add_argument("--runs", type=int, default=100, help="Number of timed inference runs (default: 100)")
    p.add_argument("--image-size", type=int, default=640, dest="image_size",
                   help="Resize images to this square size before inference (default: 640)")
    p.add_argument("--device", default="auto",
                   help="Device: auto | cpu | cuda | cuda:N (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    images = collect_images(args.image)

    print(f"\nTransformers Bridge Benchmark")
    print(f"{'=' * 40}")
    print(f"  model  : {args.model}")
    print(f"  images : {len(images)} file(s)  [{images[0].name}{' …' if len(images) > 1 else ''}]")
    print(f"  runs   : {args.runs}")
    print()

    data = run_benchmark(
        model_name=args.model,
        images=images,
        runs=args.runs,
        image_size=args.image_size,
        device=device,
    )

    print_report(
        model_name=args.model,
        image_size=args.image_size,
        device=device,
        runs=args.runs,
        data=data,
    )


if __name__ == "__main__":
    main()
