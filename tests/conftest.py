import cv2
import numpy as np
import pytest


@pytest.fixture
def png_bytes():
    img = np.full((120, 200, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()
