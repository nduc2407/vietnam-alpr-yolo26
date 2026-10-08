"""Do do tre (latency) va FPS.

Chay:
    python -m src bench --n 100            # chi detector (preprocess + detect)
    python -m src bench --n 100 --full     # ca pipeline gom OCR
"""
import argparse
import json
import time

import cv2
import numpy as np

from src.config import BEST_MODEL, DATASET_DIR, EXPERIMENTS_DIR

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _stats(values):
    t = np.array(values, dtype=float)
    return {"mean_ms": round(float(t.mean()), 2), "p50_ms": round(float(np.percentile(t, 50)), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2)}


def _device_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def benchmark(weights=BEST_MODEL, n=100, warmup=5, split="test", full=False, end2end=True):
    from src.inference.pipeline import ALPRPipeline

    paths = sorted(p for p in (DATASET_DIR / split / "images").glob("*") if p.suffix.lower() in IMG_EXT)[:n]
    if not paths:
        raise FileNotFoundError("Khong tim thay anh de benchmark")
    pipe = ALPRPipeline(weights, use_ocr=full, end2end=end2end)
    frames = [cv2.imread(str(p)) for p in paths]
    for f in frames[:warmup]:
        pipe.process(f)

    stages = {"preprocess": [], "detect": [], "ocr": [], "total": []}
    for f in frames:
        timing = pipe.process(f)["timing_ms"]
        for k in stages:
            stages[k].append(timing[k])

    result = {
        "mode": "full (preprocess+detect+OCR)" if full else "detector (preprocess+detect)",
        "device": _device_name(), "images": len(frames), "warmup": warmup,
        "end2end": end2end,
        "stages": {k: _stats(v) for k, v in stages.items() if k != "ocr" or full},
        "fps": round(1000 / float(np.mean(stages["total"])), 1),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPERIMENTS_DIR / ("benchmark_full.json" if full else "benchmark_detect.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BEST_MODEL))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--nms", action="store_true")
    a = ap.parse_args()
    benchmark(a.weights, a.n, split=a.split, full=a.full, end2end=not a.nms)


if __name__ == "__main__":
    main()
