"""Sinh du lieu MAU TONG HOP (anh bien so ve bang OpenCV) de chay thu/CI ma khong can dataset that.
Day KHONG phai du lieu huan luyen. Thay bang vai anh that cua nhom khi demo.

    python scripts/make_sample_data.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
TEXTS = ["59-X1 123.45", "29-A1 234.56", "51F-678.90", "30-B2 111.22", "43-C1 999.99", "92-D1 050.50"]


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, text in enumerate(TEXTS):
        img = rng.integers(60, 140, (480, 640, 3), dtype=np.uint8)
        w, h = int(rng.integers(220, 300)), int(rng.integers(70, 90))
        x, y = int(rng.integers(40, 640 - w - 40)), int(rng.integers(40, 480 - h - 40))
        cv2.rectangle(img, (x, y), (x + w, y + h), (245, 245, 245), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (20, 20, 20), 2)
        cv2.putText(img, text, (x + 8, y + h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2)
        cv2.imwrite(str(OUT / "images" / f"sample_{i}.jpg"), img)
        cx, cy, bw, bh = (x + w / 2) / 640, (y + h / 2) / 480, w / 640, h / 480
        (OUT / "labels" / f"sample_{i}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print("Da tao", len(TEXTS), "anh mau tai", OUT)


if __name__ == "__main__":
    main()
