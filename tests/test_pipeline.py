import numpy as np

from src.inference.pipeline import ALPRPipeline
from src.inference.postprocess import BoundingBox, RawDetection


class FakeDetector:
    def __init__(self):
        self.seen_shape = None

    def detect(self, image, encode_crop=False):
        self.seen_shape = image.shape[:2]
        crop = np.full((40, 100, 3), 200, np.uint8)
        return [RawDetection(BoundingBox(10, 20, 110, 60), 0.9, 0, "BienSoXe", crop,
                             "Zm9v" if encode_crop else None)]


class FakeOCR:
    def read_detailed(self, crop):
        return "59X112345", 0.8, True


def test_bbox_is_scaled_back_to_original_image():
    img = np.zeros((1000, 2000, 3), np.uint8)
    pipe = ALPRPipeline(detector=FakeDetector(), ocr=FakeOCR(), max_side=1000, enhance=False)
    out = pipe.process(img)
    assert pipe.detector.seen_shape == (500, 1000)            # detector thay anh da thu nho
    p = out["plates"][0]
    assert p["bbox"] == [20, 40, 220, 120]                    # toa do tra ve theo anh GOC
    assert p["plate_formatted"] == "59-X1 123.45" and p["valid"] is True
    assert set(out["timing_ms"]) == {"preprocess", "detect", "ocr", "total"}


def test_crop_base64_only_when_requested():
    img = np.zeros((100, 200, 3), np.uint8)
    pipe = ALPRPipeline(detector=FakeDetector(), ocr=FakeOCR(), enhance=False)
    assert "crop_base64" not in pipe.process(img)["plates"][0]
    assert pipe.process(img, encode_crop=True)["plates"][0]["crop_base64"] == "Zm9v"


def test_without_ocr_returns_detection_only():
    pipe = ALPRPipeline(detector=FakeDetector(), use_ocr=False, enhance=False)
    p = pipe.process(np.zeros((100, 200, 3), np.uint8))["plates"][0]
    assert "plate" not in p and p["det_conf"] == 0.9
