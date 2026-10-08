import cv2
import numpy as np

# pyrefly: ignore [missing-import]
from src.data.merge import merge

POLY = "0 0.3 0.4 0.4 0.5 0.6 0.6 0.5 0.3 0.3 0.4"


def test_merge_flat_dataset(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for i in range(10):
        cv2.imwrite(str(src / "images" / f"{i}.jpg"), np.zeros((20, 20, 3), np.uint8))
        (src / "labels" / f"{i}.txt").write_text(POLY if i == 0 else "0 0.5 0.5 0.2 0.2")
    cv2.imwrite(str(src / "images" / "nolabel.jpg"), np.zeros((20, 20, 3), np.uint8))

    manifest = tmp_path / "m.csv"
    stats = merge(src, dest, manifest=manifest, note="test")
    assert stats["copied"] == 10 and stats["skipped_no_label"] == 1
    assert manifest.exists() and len(manifest.read_text().splitlines()) == 11
    assert all(p.name.startswith("own_") for p in dest.rglob("*.jpg"))
    for lbl in dest.rglob("*.txt"):
        assert len(lbl.read_text().split()) == 5          # polygon da duoc chuyen thanh bbox

    again = merge(src, dest, manifest=manifest)             # chay lai: khong nhan doi
    assert again["copied"] == 0 and again["skipped_exists"] == 10
