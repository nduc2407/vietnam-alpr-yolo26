import numpy as np

# pyrefly: ignore [missing-import]
from src.evaluation.sweep import iou_matrix, match_predictions, prf


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0


def test_match_counts():
    gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], float)
    pred = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], float)
    conf = np.array([0.9, 0.8])
    assert match_predictions(pred, conf, gt, 0.5) == (1, 1, 1)
    assert match_predictions(pred, conf, gt, 0.85) == (1, 0, 1)   # nguong cao loai bo du doan thu 2


def test_duplicate_prediction_counts_as_fp():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    assert match_predictions(pred, np.array([0.9, 0.8]), gt, 0.1) == (1, 1, 0)


def test_prf():
    p, r, f1 = prf(8, 2, 2)
    assert abs(p - 0.8) < 1e-9 and abs(r - 0.8) < 1e-9 and abs(f1 - 0.8) < 1e-9
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
