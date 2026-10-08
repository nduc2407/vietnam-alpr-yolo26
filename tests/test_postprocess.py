import base64

import cv2
import numpy as np

from src.inference.postprocess import BoundingBox, PostProcessor


class _T:  # gia lap tensor
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)

    def item(self):
        return self.v


class _Box:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = [_T(xyxy)], [_T(conf)], [_T(cls)]


class _Res:
    def __init__(self, boxes, names=None):
        self.boxes, self.names = boxes, names or {0: "BienSoXe"}


def _img():
    return np.full((100, 200, 3), 128, np.uint8)


def test_crop_roi_clamps_to_image_and_copies():
    pp = PostProcessor(crop_padding=4)
    img = _img()
    crop = pp.crop_roi(img, BoundingBox(-10, -10, 20, 20))
    assert crop.shape[:2] == (24, 24)                 # x: 0..24
    crop[:] = 0
    assert img.max() == 128                            # la ban sao, khong sua anh goc
    edge = pp.crop_roi(img, BoundingBox(190, 90, 400, 300))
    assert edge.shape[:2] == (14, 14)                  # bi cat o mep anh


def test_extract_sorts_by_conf_and_maps_names():
    boxes = [_Box([10, 10, 60, 40], 0.5, 0), _Box([100, 20, 180, 60], 0.9, 0)]
    dets = PostProcessor().extract_detections([_Res(boxes)], _img(), encode_crop=False)
    assert [d.confidence for d in dets] == [0.9, 0.5]
    assert dets[0].class_name == "BienSoXe" and dets[0].crop_base64 is None
    assert dets[0].cropped_image.size > 0


def test_extract_empty_inputs():
    pp = PostProcessor()
    assert pp.extract_detections([], _img()) == []
    assert pp.extract_detections([_Res([])], _img()) == []
    assert pp.extract_detections([_Res(None)], _img()) == []


def test_base64_roundtrip_and_to_dict():
    pp = PostProcessor()
    det = pp.extract_detections([_Res([_Box([10, 10, 60, 40], 0.8, 0)])], _img(), encode_crop=True)[0]
    raw = base64.b64decode(det.crop_base64)
    assert cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None
    d = det.to_dict(include_crop=True)
    assert d["bbox"]["x1"] == 10 and "crop_base64" in d and "cropped_image" not in d
    assert PostProcessor.encode_image_to_base64(np.zeros((0, 0, 3), np.uint8)) == ""
