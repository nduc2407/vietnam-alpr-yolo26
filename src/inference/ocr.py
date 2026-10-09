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


def format_plate(s: str, two_line: bool | None = None) -> str:
    """Dinh dang bien so VN.
    two_line=True  (xe may, 2 dong): 59X112345 -> 59-X1 123.45 ; 59X11234 -> 59-X1 1234
    two_line=False (o to, 1 dong)  : 51F12345 -> 51F-123.45 ; 51A1234 -> 51A-1234 ; 29LD12345 -> 29LD-123.45
    two_line=None  : tu doan (khong chac chan voi chuoi 8 ky tu). Khong hop le thi tra nguyen chuoi."""
    if not PLATE_RE.match(s):
        return s
    prov, rest = s[:2], s[2:]
    if two_line is None:
        two_line = len(s) == 9 and rest[1].isdigit()
    if two_line:
        series, num = rest[:2], rest[2:]
    else:
        n = 2 if rest[1].isalpha() else 1
        series, num = rest[:n], rest[n:]
    if len(num) not in (4, 5) or not num.isdigit():
        return s
    num = f"{num[:3]}.{num[3:]}" if len(num) == 5 else num
    return f"{prov}-{series} {num}" if two_line else f"{prov}{series}-{num}"


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

    def _read_lines(self, img: np.ndarray, two_lines: bool) -> tuple[str, float]:
        if two_lines:
            h = img.shape[0]
            (t1, c1), (t2, c2) = self._recognize(img[: h // 2]), self._recognize(img[h // 2:])
            return t1 + t2, (c1 + c2) / 2
        return self._recognize(img)

    def read_detailed(self, crop: np.ndarray) -> tuple[str, float, bool]:
        """Tra ve (text_da_chuan_hoa, do_tin_cay, la_bien_2_dong)."""
        if crop is None or crop.size == 0:
            return "", 0.0, False
        img = prepare_plate_crop(crop) if self.preprocess else crop
        h, w = img.shape[:2]
        ratio = w / h
        if ratio >= 3.5:
            options = [False]          # chac chan 1 dong
        elif ratio <= 1.5:
            options = [True]           # chac chan 2 dong
        else:
            options = [False, True]    # mo ho (vd bien nghieng): thu ca hai
        results = []
        for two in options:
            text, conf = self._read_lines(img, two)
            norm = normalize_plate(text)
            results.append((is_valid_plate(norm), conf, norm, two))
        _, conf, norm, two = max(results)   # uu tien chuoi hop le, roi den do tin cay
        return norm, conf, two

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """Giu tuong thich: chi tra (text, conf)."""
        text, conf, _ = self.read_detailed(crop)
        return text, conf