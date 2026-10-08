"""Phat hien bien so bang YOLO26 (YOLO -> PostProcessor -> RawDetection).

Luu y: YOLO26 mac dinh la end-to-end (NMS-free). Dat end2end=False de dung head one-to-many
kem NMS truyen thong (tham so `iou` luc nay moi co tac dung).
"""
from pathlib import Path

import numpy as np

from src.config import BEST_MODEL, CONF_THRES, IMG_SIZE, IOU_THRES
from src.inference.postprocess import PostProcessor, RawDetection


class PlateDetector:
    def __init__(self, weights=BEST_MODEL, conf=CONF_THRES, iou=IOU_THRES, imgsz=IMG_SIZE,
                 device=None, end2end=True, postprocessor=None):
        from ultralytics import YOLO
        from src.training.train import default_device

        if not Path(weights).exists():
            raise FileNotFoundError(f"Khong tim thay weights: {weights}. Hay train truoc.")
        self.model = YOLO(str(weights))
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.device = device or default_device()
        self.end2end = end2end
        self.post = postprocessor or PostProcessor()

    def detect(self, image: np.ndarray, encode_crop: bool = False) -> list[RawDetection]:
        """image: BGR. Tra ve RawDetection (toa do theo `image`), sap xep theo do tin cay giam dan."""
        kwargs = {} if self.end2end else {"end2end": False}
        res = self.model.predict(image, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False, **kwargs)
        return self.post.extract_detections(res, image, encode_crop=encode_crop)
