import numpy as np

from src.inference.ocr import PlateOCR, format_plate, is_valid_plate, normalize_plate


def test_normalize_strips_noise_and_uppercases():
    assert normalize_plate("59-x1 123.45") == "59X112345"


def test_normalize_fixes_confusions_by_position():
    assert normalize_plate("S9X112345").startswith("59")   # S->5 o vi tri ma tinh
    assert normalize_plate("59X1I2345")[4] == "1"          # I->1 o phan so


def test_format_and_validity():
    assert format_plate("59X112345") == "59-X1 123.45"
    assert is_valid_plate("59X112345")
    assert not is_valid_plate("ABC")
    assert format_plate("ABC") == "ABC"


def test_read_one_line_plate_uses_single_call():
    calls = []

    def fake(img):
        calls.append(img.shape)
        return "59-X1 123.45", 0.9

    crop = np.full((40, 160, 3), 200, np.uint8)       # ti le 4:1 -> bien 1 dong
    text, conf = PlateOCR(recognizer=fake).read(crop)
    assert text == "59X112345" and conf == 0.9 and len(calls) == 1


def test_read_two_line_plate_splits_and_joins():
    seq = iter([("59-X1", 0.8), ("123.45", 0.6)])
    crop = np.full((100, 120, 3), 200, np.uint8)      # gan vuong -> bien 2 dong
    text, conf = PlateOCR(recognizer=lambda img: next(seq)).read(crop)
    assert text == "59X112345" and abs(conf - 0.7) < 1e-9


def test_preprocess_toggle_changes_input_size():
    sizes = {}

    def fake(img):
        sizes.setdefault("h", []).append(img.shape[0])
        return "59X112345", 1.0

    crop = np.full((30, 120, 3), 128, np.uint8)
    PlateOCR(recognizer=fake, preprocess=False).read(crop)
    PlateOCR(recognizer=fake, preprocess=True).read(crop)
    assert sizes["h"][0] == 30 and sizes["h"][1] >= 90   # co tien xu ly -> phong to


def test_empty_crop():
    assert PlateOCR(recognizer=lambda i: ("x", 1.0)).read(np.zeros((0, 0, 3), np.uint8)) == ("", 0.0)
