"""OCR bien so xe Viet Nam bang PaddleOCR (PP-OCRv5) + hau xu ly theo dinh dang bien so.


Bien moi truong:
    OCR_MODEL       ten model nhan dang (mac dinh en_PP-OCRv5_mobile_rec)
    OCR_DEVICE      cpu | gpu:0 (mac dinh cpu - on dinh, tranh xung dot CUDA voi PyTorch)
    OCR_PREPROCESS  1/0 bat tat tien xu ly anh bien (mac dinh 1)
"""
import os
import re

import cv2
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
    """Dinh dang bien so VN:
    - O to 5 so (1 dong hoac 2 dong vuong): 51L18297 -> 51L-182.97 ; 51F12345 -> 51F-123.45
    - O to 4 so cu: 51A1234 -> 51A-1234
    - O to lien doanh/dac biet: 29LD12345 -> 29LD-123.45
    - Xe may 5 so (2 dong): 15P103737 -> 15-P1 037.37 ; 59X112345 -> 59-X1 123.45
    - Xe may 4 so cu: 59X11234 -> 59-X1 1234
    """
    if not PLATE_RE.match(s):
        return s
    prov, rest = s[:2], s[2:]

    # 1. Seri 2 chu cai (LD, DA, NN, NG, QT, CV, MK...)
    if len(rest) >= 2 and rest[:2].isalpha():
        series, num = rest[:2], rest[2:]
        num_str = f"{num[:3]}.{num[3:]}" if len(num) == 5 else num
        return f"{prov}{series}-{num_str}"

    # 2. O to 5 so: 1 chu cai + 5 so (Tong rest co do dai 6, vd: L18297, F12345)
    if len(rest) == 6 and rest[0].isalpha() and rest[1:].isdigit():
        series, num = rest[0], rest[1:]
        return f"{prov}{series}-{num[:3]}.{num[3:]}"

    # 3. Xe may 5 so: 1 chu cai + 1 so/chu + 5 so (Tong rest co do dai 7, vd: P103737, X112345)
    if len(rest) == 7 and rest[0].isalpha() and rest[2:].isdigit():
        series, num = rest[:2], rest[2:]
        return f"{prov}-{series} {num[:3]}.{num[3:]}"

    # 4. O to 4 so cu: 1 chu cai + 4 so (Tong rest co do dai 5, vd: A1234)
    if len(rest) == 5 and rest[0].isalpha() and rest[1:].isdigit():
        series, num = rest[0], rest[1:]
        return f"{prov}{series}-{num}"

    # 5. Xe may 4 so cu (vd: 59X11234 -> 59-X1 1234)
    if len(rest) == 6 and rest[0].isalpha() and rest[2:].isdigit():
        if two_line is True or (two_line is None and rest[1].isdigit()):
            series, num = rest[:2], rest[2:]
            return f"{prov}-{series} {num}"

    # Mac dinh
    series, num = rest[0], rest[1:]
    num_str = f"{num[:3]}.{num[3:]}" if len(num) == 5 else num
    return f"{prov}{series}-{num_str}"


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
            top_cut = int(round(h * 0.54))
            bot_cut = int(round(h * 0.44))
            (t1, c1) = self._recognize(img[:top_cut])
            (t2, c2) = self._recognize(img[bot_cut:])
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
        best_valid, best_conf, best_norm, best_two = max(results)   # uu tien chuoi hop le, roi den do tin cay

        # TTA Fallback: Neu ket qua chua hop le hoac do goc chup chieu sau khien ky tu bi ep sat / thieu so
        if not best_valid or (not best_two and len(best_norm) == 7 and best_norm[2].isalpha() and best_norm[3:].isdigit()):
            stretched_img = cv2.resize(img, None, fx=1.25, fy=1.0, interpolation=cv2.INTER_CUBIC)
            for two in options:
                text_s, conf_s = self._read_lines(stretched_img, two)
                norm_s = normalize_plate(text_s)
                valid_s = is_valid_plate(norm_s)
                if (valid_s and not best_valid) or (valid_s and len(norm_s) > len(best_norm)):
                    best_valid, best_conf, best_norm, best_two = valid_s, conf_s, norm_s, two
                    break

        return best_norm, best_conf, best_two

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """Giu tuong thich: chi tra (text, conf)."""
        text, conf, _ = self.read_detailed(crop)
        return text, conf