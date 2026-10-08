"""OCR bien so xe Viet Nam bang PaddleOCR (PP-OCRv5) + hau xu ly theo dinh dang bien so.


Bien moi truong:
    OCR_MODEL       ten model nhan dang (mac dinh en_PP-OCRv5_mobile_rec)
    OCR_DEVICE      cpu | gpu:0 (mac dinh cpu - on dinh, tranh xung dot CUDA voi PyTorch)
    OCR_PREPROCESS  1/0 bat tat tien xu ly anh bien (mac dinh 1)
"""
import os
import re

import numpy as np

# pyrefly: ignore [missing-import]
from src.inference.preprocess import prepare_plate_crop

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
# 2 so tinh + 1-2 ky tu seri + 4-5 so
PLATE_RE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")
DEFAULT_MODEL = "en_PP-OCRv5_mobile_rec"


def normalize_plate(text: str) -> str:
    """Chuan hoa: chi giu A-Z0-9, sua nham lan ky tu theo vi tri."""
    s = re.sub(r"[^A-Z0-9]", "", text.upper())
    chars = list(s)
    for i, c in enumerate(chars):
        if i < 2:                    # ma tinh: 2 chu so
            chars[i] = TO_DIGIT.get(c, c)
        elif i == 2:                 # seri: chu cai
            chars[i] = TO_LETTER.get(c, c)
        elif i >= 4:                 # phan so cuoi
            chars[i] = TO_DIGIT.get(c, c)
    return "".join(chars)


def format_plate(s: str) -> str:
    """59X112345 -> 59-X1 123.45 (neu hop le); nguoc lai tra ve nguyen chuoi."""
    if not PLATE_RE.match(s):
        return s
    head, tail = s[:4], s[4:]
    if len(s) == 9:  # 4 + 5
        return f"{head[:2]}-{head[2:]} {tail[:3]}.{tail[3:]}"
    return f"{head[:2]}-{head[2:]} {tail}"


def is_valid_plate(s: str) -> bool:
    return bool(PLATE_RE.match(s))


class PlateOCR:
    def __init__(self, model_name=None, device=None, preprocess=None, recognizer=None):
        """recognizer: ham (anh_bgr) -> (text, score); dung de test hoac thay backend."""
        self.model_name = model_name or os.environ.get("OCR_MODEL", DEFAULT_MODEL)
        self.device = device or os.environ.get("OCR_DEVICE", "cpu")
        if preprocess is None:
            preprocess = os.environ.get("OCR_PREPROCESS", "1") != "0"
        self.preprocess = preprocess
        self._recognizer = recognizer
        self._model = None

    def load(self):
        """Nap model (tai ve lan dau). Goi som de request dau tien khong bi cham."""
        if self._recognizer is None and self._model is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import TextRecognition

            self._model = TextRecognition(model_name=self.model_name, device=self.device)
        return self

    def _recognize(self, img: np.ndarray) -> tuple[str, float]:
        if self._recognizer is not None:
            return self._recognizer(img)
        out = self.load()._model.predict(input=img, batch_size=1)
        if not out:
            return "", 0.0
        res = out[0]
        return str(res["rec_text"]), float(res["rec_score"])

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """Tra ve (text_da_chuan_hoa, do_tin_cay). Bien 2 dong (xe may) duoc cat doi."""
        if crop is None or crop.size == 0:
            return "", 0.0
        img = prepare_plate_crop(crop) if self.preprocess else crop
        h, w = img.shape[:2]
        if w / h < 2.5:  # bien vuong / 2 dong
            (t1, c1), (t2, c2) = self._recognize(img[: h // 2]), self._recognize(img[h // 2:])
            text, conf = t1 + t2, (c1 + c2) / 2
        else:
            text, conf = self._recognize(img)
        return normalize_plate(text), conf
