"""Danh gia model tren tap valid/test: precision, recall, F1, mAP50, mAP50-95.

Chay:
    python -m src eval --split test
    python -m src eval --split test --nms --iou 0.5      # duong NMS (end2end=False)
"""
import argparse
import json

from src.config import BEST_MODEL, EXPERIMENTS_DIR, IMG_SIZE
from src.data.dataset import make_data_yaml


def evaluate(weights=BEST_MODEL, split="test", imgsz=IMG_SIZE, batch=16, device=None,
             conf=0.001, iou=0.7, end2end=True):
    from ultralytics import YOLO
    from src.training.train import default_device

    model = YOLO(str(weights))
    kwargs = {} if end2end else {"end2end": False}
    res = model.val(
        data=str(make_data_yaml()),
        split="val" if split == "valid" else split,
        imgsz=imgsz, batch=batch, device=device or default_device(),
        conf=conf, iou=iou,
        project=str(EXPERIMENTS_DIR), name=f"eval_{split}", exist_ok=True,
        **kwargs,
    )
    p, r = float(res.box.mp), float(res.box.mr)
    metrics = {
        "split": split,
        "mode": "end2end (NMS-free)" if end2end else f"one-to-many + NMS (iou={iou})",
        "precision": p,
        "recall": r,
        "f1": (2 * p * r / (p + r)) if (p + r) else 0.0,
        "mAP50": float(res.box.map50),
        "mAP50-95": float(res.box.map),
    }
    suffix = "" if end2end else "_nms"
    out = EXPERIMENTS_DIR / f"metrics_{split}{suffix}.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BEST_MODEL))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--nms", action="store_true", help="end2end=False (dung NMS)")
    a = ap.parse_args()
    evaluate(a.weights, a.split, a.imgsz, a.batch, a.device, iou=a.iou, end2end=not a.nms)


if __name__ == "__main__":
    main()
