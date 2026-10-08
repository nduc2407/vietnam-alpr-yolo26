"""Hau xu ly: boc tach bounding box, cat ROI an toan va encode Base64 (module cua nhom + sua nho).

Khac ban goc: crop_roi tra ve BAN SAO (khong giu tham chieu toi ca anh goc), RawDetection co to_dict()
de tra JSON (ndarray khong serialize duoc), ket qua sap xep theo do tin cay giam dan, kiem tra rong an toan.
"""
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> Dict[str, float]:
        return {"x1": round(self.x1, 2), "y1": round(self.y1, 2), "x2": round(self.x2, 2), "y2": round(self.y2, 2)}

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class RawDetection:
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    cropped_image: Optional[np.ndarray] = field(default=None, repr=False)
    crop_base64: Optional[str] = None

    def to_dict(self, include_crop: bool = False) -> Dict[str, Any]:
        d = {"bbox": self.box.to_dict(), "confidence": self.confidence,
             "class_id": self.class_id, "class_name": self.class_name}
        if include_crop and self.crop_base64:
            d["crop_base64"] = self.crop_base64
        return d


class PostProcessor:
    """Hau xu ly ket qua YOLO -> danh sach RawDetection kem anh crop."""

    def __init__(self, crop_padding: int = 4):
        self.crop_padding = crop_padding

    def crop_roi(self, image: np.ndarray, box: BoundingBox, padding: Optional[int] = None) -> np.ndarray:
        """Cat ROI co padding va kiem tra bien. Tra ve ban sao."""
        pad = self.crop_padding if padding is None else padding
        height, width = image.shape[:2]
        x1, y1 = max(0, int(box.x1) - pad), max(0, int(box.y1) - pad)
        x2, y2 = min(width, int(box.x2) + pad), min(height, int(box.y2) + pad)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, image_format: str = ".jpg") -> str:
        if image is None or image.size == 0:
            return ""
        ok, buf = cv2.imencode(image_format, image)
        return base64.b64encode(buf).decode("utf-8") if ok else ""

    def extract_detections(self, yolo_results: List[Any], original_image: np.ndarray,
                           encode_crop: bool = True) -> List[RawDetection]:
        """Trich xuat bien so + anh crop. `original_image` phai la anh DA dua vao model (cung he toa do)."""
        if not yolo_results:
            return []
        result = yolo_results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        names = getattr(result, "names", {}) or {}

        detections: List[RawDetection] = []
        for b in boxes:
            x1, y1, x2, y2 = (round(float(v), 2) for v in b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            bbox = BoundingBox(x1, y1, x2, y2)
            crop = self.crop_roi(original_image, bbox)
            detections.append(RawDetection(
                box=bbox, confidence=round(float(b.conf[0].item()), 4), class_id=cls_id,
                class_name=names.get(cls_id, "license_plate"), cropped_image=crop,
                crop_base64=self.encode_image_to_base64(crop) if (encode_crop and crop.size > 0) else None,
            ))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)
