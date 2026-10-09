"""Pipeline ALPR: tien xu ly -> detect -> crop -> OCR.

Chay:
    python -m src infer --source path/anh.jpg --save out
    python -m src infer --source folder/ --save out
    python -m src infer --source 0          # webcam
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import BEST_MODEL, CONF_THRES, IOU_THRES
from src.inference.detector import PlateDetector
from src.inference.ocr import PlateOCR, format_plate, is_valid_plate
from src.inference.preprocess import preprocess

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ALPRPipeline:
    def __init__(self, weights=BEST_MODEL, conf=CONF_THRES, iou=IOU_THRES, use_ocr=True,
                 enhance=None, max_side=1280, end2end=True, detector=None, ocr=None):
        """enhance=None -> doc bien moi truong PREPROCESS_CLAHE (mac dinh tat: model train khong dung CLAHE,
        chi bat khi `python -m src sweep --enhance` cho thay co loi)."""
        self.detector = detector or PlateDetector(weights, conf=conf, iou=iou, end2end=end2end)
        self.ocr = ocr if ocr is not None else (PlateOCR().load() if use_ocr else None)
        if enhance is None:
            enhance = os.environ.get("PREPROCESS_CLAHE", "0") == "1"
        self.enhance, self.max_side = enhance, max_side

    def process(self, image: np.ndarray, encode_crop: bool = False) -> dict:
        """Tra ve {'plates': [...], 'timing_ms': {...}}. bbox theo toa do anh GOC."""
        t0 = time.perf_counter()
        proc, scale = preprocess(image, self.max_side, self.enhance)
        t1 = time.perf_counter()
        dets = self.detector.detect(proc, encode_crop=encode_crop)
        t2 = time.perf_counter()
        plates = []
        for det in dets:
            item = {"bbox": [int(round(v / scale)) for v in det.box.to_list()], "det_conf": det.confidence}
            if encode_crop and det.crop_base64:
                item["crop_base64"] = det.crop_base64
            if self.ocr:
                text, c, two = self.ocr.read_detailed(det.cropped_image)
                item.update(plate=text, plate_formatted=format_plate(text, two),
                            ocr_conf=round(c, 4), valid=is_valid_plate(text))
            plates.append(item)
        t3 = time.perf_counter()
        timing = {
            "preprocess": round((t1 - t0) * 1000, 2),
            "detect": round((t2 - t1) * 1000, 2),
            "ocr": round((t3 - t2) * 1000, 2),
            "total": round((t3 - t0) * 1000, 2),
        }
        return {"plates": plates, "timing_ms": timing}

    def run(self, image: np.ndarray) -> list[dict]:
        return self.process(image)["plates"]

    @staticmethod
    def annotate(image: np.ndarray, results: list[dict]) -> np.ndarray:
        out = image.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = (0, 200, 0) if r.get("valid", True) else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = r.get("plate_formatted") or f"plate {r['det_conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="anh | thu muc | 0 (webcam)")
    ap.add_argument("--weights", default=str(BEST_MODEL))
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    ap.add_argument("--iou", type=float, default=IOU_THRES, help="chi co tac dung khi --nms")
    ap.add_argument("--nms", action="store_true", help="dung head one-to-many + NMS (end2end=False)")
    ap.add_argument("--enhance", action="store_true", help="CLAHE tang tuong phan truoc khi detect")
    ap.add_argument("--save", default=None, help="thu muc luu anh ket qua")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()

    pipe = ALPRPipeline(a.weights, a.conf, a.iou, not a.no_ocr, a.enhance or None, end2end=not a.nms)
    save = Path(a.save) if a.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    if a.source.isdigit():
        cap = cv2.VideoCapture(int(a.source))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("ALPR (q de thoat)", pipe.annotate(frame, pipe.run(frame)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return

    src = Path(a.source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        out = pipe.process(img)
        print(p.name, [(r.get("plate_formatted"), r["det_conf"]) for r in out["plates"]], out["timing_ms"]["total"], "ms")
        if save:
            cv2.imwrite(str(save / p.name), pipe.annotate(img, out["plates"]))


if __name__ == "__main__":
    main()
