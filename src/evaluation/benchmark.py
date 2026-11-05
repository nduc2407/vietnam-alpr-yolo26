"""Đo độ trễ (latency) và FPS của hệ thống ALPR."""
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


def _device_name(device=None):
    if device == "cpu":
        import platform, sys
        cpu_name = platform.processor()
        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            except Exception:
                pass
        return f"CPU: {cpu_name}"

    try:
        import torch
        if torch.cuda.is_available() and device != "cpu":
            return torch.cuda.get_device_name(0)
        return "CPU"
    except Exception:
        return "unknown"


def benchmark(weights=BEST_MODEL, n=100, warmup=5, split="test", full=False, end2end=True, device=None, out_name=None):
    from src.inference.pipeline import ALPRPipeline

    paths = sorted(p for p in (DATASET_DIR / split / "images").glob("*") if p.suffix.lower() in IMG_EXT)[:n]
    if not paths:
        raise FileNotFoundError("Khong tim thay anh de benchmark")
    pipe = ALPRPipeline(weights, use_ocr=full, end2end=end2end, device=device)
    frames = [cv2.imread(str(p)) for p in paths]
    for f in frames[:warmup]:
        pipe.process(f)

    stages = {"preprocess": [], "detect": [], "ocr": [], "total": []}
    for f in frames:
        timing = pipe.process(f)["timing_ms"]
        for k in stages:
            stages[k].append(timing[k])

    dev_str = _device_name(device)
    is_cpu = device == "cpu" or "CPU" in dev_str

    result = {
        "mode": "full (preprocess+detect+OCR)" if full else "detector (preprocess+detect)",
        "device": dev_str, "images": len(frames), "warmup": warmup,
        "end2end": end2end,
        "stages": {k: _stats(v) for k, v in stages.items() if k != "ocr" or full},
        "fps": round(1000 / float(np.mean(stages["total"])), 1),
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_name:
        out = EXPERIMENTS_DIR / out_name if not Path(out_name).is_absolute() else Path(out_name)
    else:
        suffix = "_cpu.json" if is_cpu else ".json"
        prefix = "benchmark_full" if full else "benchmark_detect"
        out = EXPERIMENTS_DIR / f"{prefix}{suffix}"

    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Da luu ket qua vao: {out}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BEST_MODEL))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--device", default=None, help="'cpu' hoac '0' (GPU)")
    ap.add_argument("--out", default=None, help="Ten file json luu ket qua (vd benchmark_detect_cpu.json)")
    a = ap.parse_args()
    benchmark(a.weights, a.n, split=a.split, full=a.full, end2end=not a.nms, device=a.device, out_name=a.out)


if __name__ == "__main__":
    main()
