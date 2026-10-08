import csv

import cv2
import numpy as np

from src.evaluation import ocr_eval
from src.evaluation.ocr_eval import cer, evaluate_ocr, levenshtein
from src.inference.ocr import PlateOCR


def test_levenshtein_and_cer():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("59X112345", "59X112345") == 0
    assert abs(cer("59X11234", "59X112345") - 1 / 9) < 1e-9


def test_evaluate_ocr_with_fake_recognizer(tmp_path, monkeypatch):
    monkeypatch.setattr(ocr_eval, "EXPERIMENTS_DIR", tmp_path / "exp")
    ds = tmp_path / "ds"
    (ds / "test" / "images").mkdir(parents=True)
    (ds / "test" / "labels").mkdir(parents=True)
    cv2.imwrite(str(ds / "test" / "images" / "a.jpg"), np.full((100, 200, 3), 200, np.uint8))
    (ds / "test" / "labels" / "a.txt").write_text("0 0.5 0.5 0.8 0.4\n")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate"])
        w.writerow(["a.jpg", "59-X1 123.45"])

    def factory(pre):
        # gia lap: chi khi co tien xu ly moi doc dung
        return PlateOCR(preprocess=pre, recognizer=lambda img: ("59X112345" if pre else "59X11234", 1.0))

    res = evaluate_ocr(gt, "test", ds, factory)
    assert res["samples"] == 1
    assert res["with_preprocess"]["exact_match"] == 1.0
    assert res["without_preprocess"]["exact_match"] == 0.0
    assert res["with_preprocess"]["mean_cer"] < res["without_preprocess"]["mean_cer"]
