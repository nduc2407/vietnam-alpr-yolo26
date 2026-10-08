"""Do chinh xac OCR (exact-match, CER) co/khong tien xu ly anh bien.

Can file CSV nhan chuoi bien so that (cot: image,plate). Anh bien duoc cat theo bbox GT trong
file nhan YOLO nen ket qua khong phu thuoc detector.

Tao khung CSV de dien tay:
    python -m src ocr-eval --template 100 --split test
Chay danh gia:
    python -m src ocr-eval --csv experiments/ocr_gt.csv --split test
"""
import argparse
import csv
import json
from pathlib import Path

import cv2

from src.config import DATASET_DIR, EXPERIMENTS_DIR
from src.inference.ocr import PlateOCR, normalize_plate

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    return levenshtein(pred, gt) / max(len(gt), 1)


def crop_from_label(img, label_path: Path):
    h, w = img.shape[:2]
    lines = [ln.split() for ln in label_path.read_text().splitlines() if len(ln.split()) >= 5]
    if not lines:
        return None
    cx, cy, bw, bh = map(float, lines[0][1:5])
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def evaluate_ocr(gt_csv, split="test", dataset_dir=DATASET_DIR, ocr_factory=None):
    ocr_factory = ocr_factory or (lambda pre: PlateOCR(preprocess=pre).load())
    rows = list(csv.DictReader(Path(gt_csv).open(encoding="utf-8")))
    rows = [r for r in rows if r.get("plate", "").strip()]
    if not rows:
        raise ValueError("CSV chua co dong nao co cot 'plate' duoc dien")
    img_dir, lbl_dir = Path(dataset_dir) / split / "images", Path(dataset_dir) / split / "labels"

    samples = []
    for r in rows:
        p = Path(r["image"]) if Path(r["image"]).is_absolute() else img_dir / r["image"]
        img = cv2.imread(str(p))
        crop = crop_from_label(img, lbl_dir / f"{p.stem}.txt") if img is not None else None
        if crop is not None:
            samples.append((crop, normalize_plate(r["plate"])))

    result = {"split": split, "samples": len(samples)}
    for name, pre in (("without_preprocess", False), ("with_preprocess", True)):
        ocr = ocr_factory(pre)
        preds = [ocr.read(c)[0] for c, _ in samples]
        result[name] = {
            "exact_match": round(sum(p == g for p, (_, g) in zip(preds, samples)) / len(samples), 4),
            "mean_cer": round(sum(cer(p, g) for p, (_, g) in zip(preds, samples)) / len(samples), 4),
        }
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "ocr_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def make_template(n=100, split="test", dataset_dir=DATASET_DIR, out=None):
    imgs = sorted(p.name for p in (Path(dataset_dir) / split / "images").iterdir() if p.suffix.lower() in IMG_EXT)[:n]
    out = Path(out or EXPERIMENTS_DIR / "ocr_gt.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate"])
        w.writerows([[name, ""] for name in imgs])
    print("Da tao khung CSV:", out, "- hay dien cot 'plate' (vd 59X112345)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EXPERIMENTS_DIR / "ocr_gt.csv"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--template", type=int, default=0, help="tao khung CSV voi N anh dau")
    a = ap.parse_args()
    if a.template:
        make_template(a.template, a.split)
    else:
        evaluate_ocr(a.csv, a.split)


if __name__ == "__main__":
    main()
