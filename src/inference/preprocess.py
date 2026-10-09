"""Tien xu ly anh (ImagePreprocessor) - ghep tu module cua nhom + cac sua loi da kiem chung.

Hai tang:
  1) Anh dau vao : decode, doi kenh, resize giu ti le, CLAHE (tuy chon)
  2) Anh bien cat: phong to + nan nghieng (enhance_roi) roi khu nhieu/CLAHE/lam net (prepare_plate_crop)


 
  - auto_deskew dung Hough tren cac duong thang ngang (vien/duong chu) lam chinh, minAreaRect chi la du phong:
    minAreaRect tren nhi phan Otsu cho goc 0 voi anh cat tu YOLO (bbox chua nen toi o 4 goc).
  - max_angle mac dinh 20 do (45 qua rong: uoc luong sai se lam hong anh bien).
"""
from typing import Optional, Tuple, Union

import cv2
import numpy as np

MAX_DESKEW_ANGLE = 20.0


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.ndim not in (2, 3):
        raise ValueError("Anh khong hop le")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] != 3:
        raise ValueError("So kenh anh khong ho tro")
    return image


def enhance_contrast(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """CLAHE tren kenh L (LAB). Tao CLAHE moi moi lan goi -> an toan da luong."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def auto_adaptive_lighting(image: np.ndarray) -> np.ndarray:
    """Tu dong can bang sang thich ung:
    - Anh toi (L < 85): Gamma correction (<1.0) + CLAHE de kich sang ro chu
    - Anh choi/sang (L > 175): Gamma correction (>1.0) de giu net chu dam
    - Anh binh thuong: CLAHE nhe de giu nguyen chi tiet tu nhien.
    """
    image = to_bgr(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    mean_l = float(np.mean(l_ch))

    if mean_l < 85:
        # Anh toi -> kich sang
        gamma = max(0.45, mean_l / 110.0)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        l_ch = cv2.LUT(l_ch, table)
        l_ch = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l_ch)
    elif mean_l > 175:
        # Anh choi / nguoc sang -> giam sang de lay lai do tuong phan
        gamma = min(1.6, mean_l / 140.0)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        l_ch = cv2.LUT(l_ch, table)
        l_ch = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l_ch)
    else:
        l_ch = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l_ch)

    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- uoc luong goc nghieng
def _hough_angle(gray: np.ndarray, max_angle: float) -> Optional[float]:
    """Goc nghieng (do) tu cac duong gan nam ngang; None neu khong tim thay duong nao.
    Duong khi duong chay xuong ve ben phai (he toa do anh)."""
    w = gray.shape[1]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=max(8, int(0.25 * w)), maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if a > 90:
            a -= 180
        elif a <= -90:
            a += 180
        if abs(a) <= max_angle:
            angles.append(a)
    return float(np.median(angles)) if angles else None


def _minrect_angle(image: np.ndarray) -> Optional[float]:
    """Du phong (cong thuc cua nhom): minAreaRect tren nhi phan Otsu. Dung cho OpenCV >= 4.5."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return None
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        return -(90 + angle)
    if angle > 45:
        return 90 - angle
    return -angle


def estimate_skew(gray: np.ndarray, max_angle: float = MAX_DESKEW_ANGLE) -> float:
    a = _hough_angle(gray, max_angle)
    return 0.0 if a is None else a


# ---------------------------------------------------------------- class cua nhom
class ImagePreprocessor:
    """Tien xu ly anh: decode nhi phan, CLAHE tren kenh LAB, nan nghieng, tang cuong ROI."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """Decode raw bytes -> BGR. Luu y: KHONG gioi han kich thuoc; API dung `decode_upload` (co kiem tra) truoc."""
        if not image_bytes:
            raise ValueError("Du lieu anh rong.")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Khong the decode anh tu du lieu dau vao.")
        return image

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        return enhance_contrast(image, self.clip_limit, self.tile_grid_size[0])

    def preprocess(self, input_data: Union[bytes, np.ndarray], use_clahe: bool = True) -> np.ndarray:
        if isinstance(input_data, (bytes, bytearray)):
            image = self.decode_image(bytes(input_data))
        elif isinstance(input_data, np.ndarray):
            image = to_bgr(input_data.copy())
        else:
            raise TypeError("Du lieu dau vao phai la bytes hoac np.ndarray.")
        return self.apply_clahe(image) if use_clahe else image

    @staticmethod
    def auto_deskew(image: np.ndarray, max_angle: float = MAX_DESKEW_ANGLE, min_angle: float = 3.0) -> np.ndarray:
        """Nan thang bien so. Bo qua neu lech < min_angle hoac > max_angle."""
        if image is None or image.size == 0:
            return image
        image = to_bgr(image)
        angle = _hough_angle(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), max_angle)
        if angle is None:
            angle = _minrect_angle(image)
        if angle is None or abs(angle) < min_angle or abs(angle) > max_angle:
            return image
        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    @classmethod
    def enhance_roi(cls, roi: np.ndarray, target_min_height: int = 100, max_width: int = 1200) -> np.ndarray:
        """Phong to (noi suy bac 3, co gioi han chieu rong) + nan nghieng."""
        if roi is None or roi.size == 0:
            return roi
        out = to_bgr(roi.copy())
        h, w = out.shape[:2]
        if h < target_min_height:
            scale = min(target_min_height / float(h), max_width / float(w))
            if scale > 1:
                out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cls.auto_deskew(out)


# ---------------------------------------------------------------- ham tien ich cho pipeline
def deskew(image: np.ndarray, max_angle: float = MAX_DESKEW_ANGLE, min_angle: float = 0.5) -> np.ndarray:
    return ImagePreprocessor.auto_deskew(image, max_angle, min_angle)


def resize_max(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image, 1.0
    s = max_side / max(h, w)
    return cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s


def preprocess(image: np.ndarray, max_side: int = 1280, enhance: Union[bool, str] = False):
    """Tra ve (anh_da_xu_ly, scale) voi scale = kich_thuoc_moi / kich_thuoc_goc."""
    out, scale = resize_max(to_bgr(image), max_side)
    if enhance:
        out = auto_adaptive_lighting(out)
    return out, scale


def prepare_plate_crop(crop: np.ndarray, target_h: int = 110, do_deskew: bool = True,
                       denoise: bool = True, sharpen: bool = True, add_border: bool = True) -> np.ndarray:
    """Hinh hoc (phong to + nan nghieng) -> can bang sang thich ung -> padding vien -> khu nhieu -> lam net. Tra ve BGR 3 kenh."""
    img = ImagePreprocessor.enhance_roi(crop, target_h) if do_deskew else to_bgr(crop)
    img = auto_adaptive_lighting(img)
    if add_border:
        # Them padding vien 6px tren duoi va 10px trai phai de ky tu sat viền khong bi mat net
        img = cv2.copyMakeBorder(img, 6, 6, 10, 10, cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise:
        gray = cv2.bilateralFilter(gray, 5, 30, 30)
    if sharpen:
        gray = cv2.addWeighted(gray, 1.4, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.4, 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
