# pyrefly: ignore [missing-import]
from src.data.dataset import convert_line, fix_labels, make_data_yaml, scan_labels

POLY = "0 0.3371 0.4455 0.3704 0.5814 0.5624 0.5508 0.5380 0.4076 0.3371 0.4455"


def test_bbox_line_unchanged():
    out, status = convert_line("0 0.5 0.5 0.2 0.1")
    assert status == "ok" and out.startswith("0 0.500000")


def test_polygon_converted_to_bbox():
    out, status = convert_line(POLY)
    parts = out.split()
    assert status == "converted" and len(parts) == 5
    assert all(0 <= float(v) <= 1 for v in parts[1:])


def test_invalid_line_dropped():
    assert convert_line("0 0.5 0.5")[1] == "invalid"
    assert convert_line("abc 1 2 3 4")[1] == "invalid"


def _make(tmp_path, content):
    (tmp_path / "train" / "labels").mkdir(parents=True)
    f = tmp_path / "train" / "labels" / "a.txt"
    f.write_text(content)
    return f


def test_scan_and_fix(tmp_path):
    f = _make(tmp_path, POLY + "\n")
    assert len(scan_labels(tmp_path)["bad"]) == 1
    stats = fix_labels(tmp_path)
    assert stats["converted_files"] == 1
    assert scan_labels(tmp_path)["bad"] == []
    assert (tmp_path / "_label_backup" / "train" / "a.txt").exists()
    assert len(f.read_text().split()) == 5


def test_make_data_yaml(tmp_path):
    (tmp_path / "test" / "images").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text("names: ['BienSoXe']\n")
    out = make_data_yaml(tmp_path, tmp_path / "data.yaml", tmp_path / "out.yaml")
    text = out.read_text()
    assert "BienSoXe" in text and "test/images" in text
