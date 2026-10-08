"""Gop bo anh bien so DO NHOM TU CHUP + TU GAN NHAN (dinh dang YOLO) vao dataset chinh.

Dau vao (1 trong 2 cau truc):
  A) <src>/images + <src>/labels                      -> tu chia train/valid/test theo ti le
  B) <src>/{train,valid,test}/{images,labels}         -> giu nguyen split
Moi anh duoc doi ten them tien to `own_`, nhan duoc kiem tra/chuyen polygon -> bbox,
va moi ban ghi duoc ghi vao data/own_manifest.csv (de chung minh nguon goc du lieu).

Chay:
    python -m src merge --src D:\\anh_nhom_chup --note "Chup tai TP.HCM 10/2026 bang dien thoai X"
"""
import argparse
import csv
import random
import shutil
import time
from pathlib import Path

from src.config import DATASET_DIR, ROOT
from src.data.dataset import SPLITS, convert_line

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _images(folder: Path):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT) if folder.exists() else []


def _plan(src: Path, val_ratio: float, test_ratio: float, seed: int):
    items = []  # (image, label_dir, split)
    if (src / "train" / "images").exists():
        for s in SPLITS:
            items += [(p, src / s / "labels", s) for p in _images(src / s / "images")]
        return items
    imgs = _images(src / "images")
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    for i, p in enumerate(imgs):
        split = "valid" if i < n_val else "test" if i < n_val + n_test else "train"
        items.append((p, src / "labels", split))
    return items


def merge(src, dest=DATASET_DIR, val_ratio=0.1, test_ratio=0.1, seed=0, prefix="own_", note="",
          manifest=None):
    src, dest = Path(src), Path(dest)
    manifest = Path(manifest) if manifest else ROOT / "data" / "own_manifest.csv"
    stats = {"copied": 0, "skipped_no_label": 0, "skipped_exists": 0, "dropped_lines": 0}
    rows = []
    for img, label_dir, split in _plan(src, val_ratio, test_ratio, seed):
        label = label_dir / f"{img.stem}.txt"
        if not label.exists():
            stats["skipped_no_label"] += 1
            continue
        new_lines = []
        for line in label.read_text().splitlines():
            out, status = convert_line(line)
            if out:
                new_lines.append(out)
            elif status == "invalid":
                stats["dropped_lines"] += 1
        if not new_lines:
            stats["skipped_no_label"] += 1
            continue
        name = f"{prefix}{img.name}"
        dst_img = dest / split / "images" / name
        dst_lbl = dest / split / "labels" / f"{prefix}{img.stem}.txt"
        if dst_img.exists():
            stats["skipped_exists"] += 1
            continue
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(new_lines) + "\n")
        rows.append([name, split, str(src), note, time.strftime("%Y-%m-%d")])
        stats["copied"] += 1

    if rows:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        new_file = not manifest.exists()
        with manifest.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["file", "split", "source_dir", "note", "date"])
            w.writerows(rows)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", default=str(DATASET_DIR))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="", help="mo ta nguon goc: ai chup, o dau, khi nao, thiet bi")
    a = ap.parse_args()
    print(merge(a.src, a.dest, a.val_ratio, a.test_ratio, a.seed, note=a.note))


if __name__ == "__main__":
    main()
