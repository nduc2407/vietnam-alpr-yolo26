"""Kiểm tra, chuẩn hóa nhãn YOLO và tạo data.yaml với đường dẫn tuyệt đối."""
import argparse
import shutil
from pathlib import Path

import yaml

from src.config import DATASET_DIR

SPLITS = ("train", "valid", "test")


def polygon_to_bbox(coords):
    xs, ys = coords[0::2], coords[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def _clamp(v):
    return max(0.0, min(1.0, v))


def convert_line(line):
    """Tra ve (dong_moi | None, trang_thai) voi trang_thai in ok/converted/invalid."""
    parts = line.split()
    if not parts:
        return None, "empty"
    try:
        cls = int(float(parts[0]))
        vals = [float(p) for p in parts[1:]]
    except ValueError:
        return None, "invalid"
    if len(vals) == 4:
        status = "ok"
        cx, cy, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        cx, cy, w, h = polygon_to_bbox(vals)
        status = "converted"
    else:
        return None, "invalid"
    cx, cy, w, h = map(_clamp, (cx, cy, w, h))
    if w <= 0 or h <= 0:
        return None, "invalid"
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}", status


def scan_labels(root=DATASET_DIR):
    """Dem so file bi loi dinh dang / rong tren moi split."""
    report = {"files": 0, "bad": [], "empty": []}
    for split in SPLITS:
        for f in (Path(root) / split / "labels").glob("*.txt"):
            report["files"] += 1
            text = f.read_text().strip()
            if not text:
                report["empty"].append(str(f))
                continue
            if any(len(ln.split()) != 5 for ln in text.splitlines()):
                report["bad"].append(str(f))
    return report


def fix_labels(root=DATASET_DIR, backup=True):
    """Chuyen polygon -> bbox, bo dong loi. Sao luu file goc vao root/_label_backup."""
    root = Path(root)
    stats = {"converted_files": 0, "dropped_lines": 0}
    for split in SPLITS:
        for f in (root / split / "labels").glob("*.txt"):
            lines = f.read_text().splitlines()
            new, changed = [], False
            for line in lines:
                out, status = convert_line(line)
                if status == "converted":
                    changed = True
                elif status == "invalid":
                    changed = True
                    stats["dropped_lines"] += 1
                if out:
                    new.append(out)
            if changed:
                if backup:
                    dst = root / "_label_backup" / split / f.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                f.write_text("\n".join(new) + ("\n" if new else ""))
                stats["converted_files"] += 1
    return stats


def make_data_yaml(root=DATASET_DIR, src_yaml=None, out_yaml=None):
    """Sinh data.resolved.yaml de tranh loi duong dan tuong doi cua Roboflow."""
    root = Path(root).resolve()
    src_p = Path(src_yaml) if src_yaml is not None else (root / "data.yaml")
    out_p = Path(out_yaml) if out_yaml is not None else (root / "data.resolved.yaml")
    names = ["BienSoXe"]
    if src_p.exists():
        cfg = yaml.safe_load(src_p.read_text(encoding="utf-8")) or {}
        names = cfg.get("names", names)
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    data = {
        "path": str(root),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(names),
        "names": names,
    }
    if (root / "test" / "images").exists():
        data["test"] = "test/images"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    if args.fix:
        print("Fix:", fix_labels())
    if args.scan or not args.fix:
        r = scan_labels()
        print(f"Label files: {r['files']} | Bad: {len(r['bad'])} | Empty: {len(r['empty'])}")
        for p in r["bad"][:10]:
            print("  BAD  ", p)
    print("data.yaml:", make_data_yaml())


if __name__ == "__main__":
    main()
