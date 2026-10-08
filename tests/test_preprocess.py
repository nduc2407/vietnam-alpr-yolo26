import numpy as np
import pytest

from src.inference.preprocess import enhance_contrast, preprocess, to_bgr


def test_large_image_is_downscaled_with_scale():
    img = np.zeros((2000, 4000, 3), np.uint8)
    out, scale = preprocess(img, max_side=1000)
    assert max(out.shape[:2]) == 1000 and abs(scale - 0.25) < 1e-6


def test_small_image_unchanged():
    img = np.zeros((100, 200, 3), np.uint8)
    out, scale = preprocess(img, max_side=1000)
    assert out.shape == img.shape and scale == 1.0


def test_gray_and_bgra_to_bgr():
    assert to_bgr(np.zeros((10, 10), np.uint8)).shape == (10, 10, 3)
    assert to_bgr(np.zeros((10, 10, 4), np.uint8)).shape == (10, 10, 3)
    with pytest.raises(ValueError):
        to_bgr(None)


def test_enhance_keeps_shape():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert enhance_contrast(img).shape == img.shape


def _plate_image(angle=0.0):
    import cv2
    img = np.full((120, 360, 3), 90, np.uint8)
    cv2.rectangle(img, (20, 20), (340, 100), (245, 245, 245), -1)
    cv2.rectangle(img, (20, 20), (340, 100), (10, 10, 10), 3)
    cv2.putText(img, "59-X1 123.45", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 2)
    if angle:
        m = cv2.getRotationMatrix2D((180, 60), angle, 1.0)
        img = cv2.warpAffine(img, m, (360, 120), borderMode=cv2.BORDER_REPLICATE)
    return img


def test_deskew_reduces_skew():
    import cv2
    from src.inference.preprocess import deskew, estimate_skew
    tilted = _plate_image(8.0)
    before = abs(estimate_skew(cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)))
    after = abs(estimate_skew(cv2.cvtColor(deskew(tilted), cv2.COLOR_BGR2GRAY)))
    assert before > 4 and after < 2


def test_deskew_leaves_straight_image():
    from src.inference.preprocess import deskew
    straight = _plate_image(0.0)
    assert deskew(straight) is straight


def test_prepare_plate_crop_upscales_and_returns_bgr():
    from src.inference.preprocess import prepare_plate_crop
    out = prepare_plate_crop(_plate_image(0.0)[20:100, 20:340])
    assert out.ndim == 3 and out.shape[2] == 3 and out.shape[0] >= 80


def _tilted_crop(style, theta, clutter, seed=0):
    """Bien so xoay roi cat bang bbox truc chuan (nhu crop tu YOLO): chua nen o 4 goc."""
    import cv2
    rng = np.random.default_rng(seed)
    h, w = 90, 300
    if style == "white":
        plate, fg = np.full((h, w, 3), 245, np.uint8), (10, 10, 10)
    else:
        plate, fg = np.full((h, w, 3), (150, 60, 20), np.uint8), (245, 245, 245)
    cv2.rectangle(plate, (2, 2), (w - 3, h - 3), fg, 3)
    cv2.putText(plate, "59-X1 123.45", (22, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.25, fg, 3)
    H, W = 260, 460
    bg = rng.integers(40, 130, (H, W, 3), dtype=np.uint8) if clutter else np.full((H, W, 3), 110, np.uint8)
    canvas, mask = np.zeros_like(bg), np.zeros((H, W), np.uint8)
    x0, y0 = (W - w) // 2, (H - h) // 2
    canvas[y0:y0 + h, x0:x0 + w], mask[y0:y0 + h, x0:x0 + w] = plate, 255
    m = cv2.getRotationMatrix2D((W / 2, H / 2), theta, 1.0)
    canvas, mask = cv2.warpAffine(canvas, m, (W, H)), cv2.warpAffine(mask, m, (W, H))
    out = np.where(mask[..., None] > 0, canvas, bg)
    ys, xs = np.where(mask > 0)
    return out[max(0, ys.min() - 4):ys.max() + 4, max(0, xs.min() - 4):xs.max() + 4]


def test_auto_deskew_on_yolo_style_crops_with_background():
    """Hoi quy: minAreaRect+Otsu cho goc 0 voi crop co nen toi -> khong nan duoc. Hough phai nan duoc."""
    import cv2
    from src.inference.preprocess import ImagePreprocessor, estimate_skew
    for style in ("white", "blue"):
        for clutter in (False, True):
            for theta in (-15, -8, 8, 15):
                crop = _tilted_crop(style, theta, clutter)
                fixed = ImagePreprocessor.auto_deskew(crop)
                residual = abs(estimate_skew(cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)))
                assert residual < 2.0, (style, clutter, theta, residual)


def test_auto_deskew_ignores_angles_beyond_max():
    from src.inference.preprocess import ImagePreprocessor
    crop = _tilted_crop("white", 35, False)
    assert ImagePreprocessor.auto_deskew(crop, max_angle=20) is crop


def test_minrect_fallback_on_uniform_background():
    import cv2
    from src.inference.preprocess import _minrect_angle
    img = np.full((90, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (2, 2), (297, 87), (10, 10, 10), 3)
    big = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h, w = big.shape[:2]
    big = cv2.warpAffine(big, cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0), (w, h), borderValue=(255, 255, 255))
    assert abs(_minrect_angle(big) - (-10)) < 1.5


def test_image_preprocessor_decode_and_types():
    import cv2
    import pytest
    from src.inference.preprocess import ImagePreprocessor
    pre = ImagePreprocessor()
    ok, buf = cv2.imencode(".png", np.full((20, 30, 3), 90, np.uint8))
    assert pre.preprocess(buf.tobytes(), use_clahe=False).shape == (20, 30, 3)
    with pytest.raises(ValueError):
        pre.decode_image(b"")
    with pytest.raises(ValueError):
        pre.decode_image(b"not an image")
    with pytest.raises(TypeError):
        pre.preprocess("path.jpg")


def test_clahe_is_thread_safe_on_shared_preprocessor():
    """Hoi quy: dung chung 1 doi tuong cv2.CLAHE giua cac luong cho ket qua sai (~14%)."""
    import threading
    from src.inference.preprocess import ImagePreprocessor
    pre = ImagePreprocessor()
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 255, (120 + i * 5, 160 + i * 3, 3), dtype=np.uint8) for i in range(6)]
    ref = [pre.apply_clahe(im) for im in imgs]
    bad = []

    def work(k):
        for _ in range(60):
            if not np.array_equal(pre.apply_clahe(imgs[k]), ref[k]):
                bad.append(k)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not bad


def test_enhance_roi_caps_extreme_upscale():
    from src.inference.preprocess import ImagePreprocessor
    tiny = np.full((3, 200, 3), 128, np.uint8)
    out = ImagePreprocessor.enhance_roi(tiny, target_min_height=100, max_width=1200)
    assert out.shape[1] <= 1200
