"""Phan tich nguong tin cay (confidence) va NMS bang so lieu thuc nghiem.

Voi moi che do (end2end khong NMS | one-to-many + NMS voi cac muc IoU) va moi nguong conf,
tinh TP/FP/FN (ghep GT theo IoU>=0.5) -> precision, recall, F1. Luu CSV + JSON.

Chay:
    python -m src sweep --split test
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import BEST_MODEL, DATASET_DIR, EXPERIMENTS_DIR, IMG_SIZE

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFS = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_NMS_IOUS = [0.3, 0.5, 0.7]


def yolo_to_xyxy(label_path: Path, w: int, h: int) -> np.ndarray:
    boxes = []
    if Path(label_path).exists():
        for line in Path(label_path).read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_predictions(pred_boxes, pred_confs, gt_boxes, conf_thr, iou_thr=0.5):
    """Ghep tham lam theo conf giam dan. Tra ve (tp, fp, fn)."""
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    pred_confs = np.asarray(pred_confs, dtype=float)
    keep = pred_confs >= conf_thr
    boxes, confs = pred_boxes[keep], pred_confs[keep]
    boxes = boxes[np.argsort(-confs)]
    ious = iou_matrix(boxes, gt_boxes)
    used, tp = set(), 0
    for i in range(len(boxes)):
        if ious.shape[1] == 0:
            break
        row = ious[i].copy()
        if used:
            row[list(used)] = -1
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used.add(j)
            tp += 1
    return tp, len(boxes) - tp, len(gt_boxes) - tp


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r)) if p + r else 0.0


def _collect(model, images, labels_dir, imgsz, device, lo, iou, end2end, enhance=False):
    kwargs = {} if end2end else {"end2end": False}
    data = []
    for img in images:
        src = str(img)
        if enhance:  # ablation: CLAHE truoc khi detect (cung kich thuoc nen GT khong doi)
            import cv2
            from src.inference.preprocess import enhance_contrast
            src = enhance_contrast(cv2.imread(str(img)))
        res = model.predict(src, conf=lo, iou=iou, imgsz=imgsz, device=device, verbose=False, **kwargs)[0]
        h, w = res.orig_shape
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,))
        data.append((boxes, confs, yolo_to_xyxy(labels_dir / f"{img.stem}.txt", w, h)))
    return data


def sweep(weights=BEST_MODEL, split="test", confs=None, nms_ious=None, imgsz=IMG_SIZE, device=None, enhance=False):
    from ultralytics import YOLO
    from src.training.train import default_device

    confs = confs or DEFAULT_CONFS
    nms_ious = nms_ious or DEFAULT_NMS_IOUS
    device = device or default_device()
    root = DATASET_DIR / split
    images = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        raise FileNotFoundError(f"Khong co anh trong {root / 'images'}")
    model = YOLO(str(weights))
    modes = [("end2end (NMS-free)", True, 0.7)] + [(f"NMS iou={i}", False, i) for i in nms_ious]

    rows = []
    for name, e2e, iou in modes:
        try:
            data = _collect(model, images, root / "labels", imgsz, device, min(confs), iou, e2e, enhance)
        except Exception as exc:
            print(f"[bo qua] che do '{name}': {exc}")
            continue
        for c in confs:
            tp = fp = fn = 0
            for boxes, cf, gt in data:
                a, b, d = match_predictions(boxes, cf, gt, c)
                tp, fp, fn = tp + a, fp + b, fn + d
            p, r, f1 = prf(tp, fp, fn)
            rows.append({"mode": name, "conf": c, "precision": round(p, 4), "recall": round(r, 4),
                         "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn})

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_clahe" if enhance else ""
    csv_path = EXPERIMENTS_DIR / f"threshold_sweep{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "conf", "precision", "recall", "f1", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(rows)
    best = {}
    for r in rows:
        if r["mode"] not in best or r["f1"] > best[r["mode"]]["f1"]:
            best[r["mode"]] = r
    (EXPERIMENTS_DIR / f"threshold_sweep{tag}.json").write_text(
        json.dumps({"split": split, "clahe": enhance, "images": len(images), "best_by_f1": best, "rows": rows}, indent=2))
    for name, r in best.items():
        print(f"{name:22s} best conf={r['conf']}  P={r['precision']}  R={r['recall']}  F1={r['f1']}")
    print("Da luu:", csv_path)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BEST_MODEL))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--device", default=None)
    ap.add_argument("--enhance", action="store_true", help="ablation: CLAHE truoc khi detect")
    a = ap.parse_args()
    sweep(a.weights, a.split, imgsz=a.imgsz, device=a.device, enhance=a.enhance)


if __name__ == "__main__":
    main()
