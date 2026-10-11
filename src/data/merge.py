"""Gộp ảnh và nhãn tự thu thập của nhóm vào dataset YOLO."""

import argparse
import random
import shutil
from pathlib import Path

from src.config import DATASET_DIR
from src.data.dataset import convert_line, make_data_yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_pairs(src_dir: Path):
    """Tim cac cap (file_anh, file_nhan) tu thu muc nguon.
    Ho tro ca thu muc phang va thu muc chia images/labels.
    """
    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Thu muc nguon khong ton tai: {src_dir}")

    # Kiem tra xem co thu muc con images/ hay khong
    images_dir = src_dir / "images" if (src_dir / "images").is_dir() else src_dir
    labels_dir = src_dir / "labels" if (src_dir / "labels").is_dir() else src_dir

    pairs = []
    missing_labels = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() in IMG_EXTS:
            txt_path = labels_dir / f"{img_path.stem}.txt"
            if txt_path.is_file():
                pairs.append((img_path, txt_path))
            else:
                missing_labels.append(img_path)

    return pairs, missing_labels


def validate_and_format_label(txt_path: Path) -> list[str]:
    """Doc va chuan hoa cac dong nhan theo format YOLO detection (cls cx cy w h)."""
    valid_lines = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        cleaned, status = convert_line(line)
        if cleaned and status in ("ok", "converted"):
            valid_lines.append(cleaned)
    return valid_lines


def merge_dataset(
    src: Path,
    dest: Path = DATASET_DIR,
    split: str = "test",
    ratio: tuple[float, float, float] = (0.7, 0.2, 0.1),
    prefix: str = "group_",
    seed: int = 42,
    copy: bool = True,
):
    """Gop anh va nhan cua nhom vao tap dataset dich."""
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    pairs, missing_labels = find_pairs(src)
    if not pairs:
        print(f"[Canh bao] Khong tim thay cap (anh, nhan .txt) hop le nao trong: {src}")
        if missing_labels:
            print(f"Phat hien {len(missing_labels)} anh nhung khong co file .txt tuong ung.")
        return {"merged": 0, "missing_labels": len(missing_labels)}

    if missing_labels:
        print(f"[Luu y] Co {len(missing_labels)} anh khong co nhan .txt (se bi bo qua).")

    # Xac dinh phan bo split
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)

    n_total = len(shuffled_pairs)
    assignments = []  # danh sach tuple (img_path, txt_path, target_split)

    if split in ("train", "valid", "test"):
        assignments = [(img, txt, split) for img, txt in shuffled_pairs]
    elif split == "auto":
        r_train, r_val, r_test = ratio
        total_r = r_train + r_val + r_test
        n_train = int(round(n_total * (r_train / total_r)))
        n_val = int(round(n_total * (r_val / total_r)))

        for i, (img, txt) in enumerate(shuffled_pairs):
            if i < n_train:
                target = "train"
            elif i < n_train + n_val:
                target = "valid"
            else:
                target = "test"
            assignments.append((img, txt, target))
    else:
        raise ValueError(f"Split khong hop le: '{split}'. Chon: train, valid, test hoac auto.")

    # Thuc hien copy/move vao dataset
    stats = {"train": 0, "valid": 0, "test": 0, "fixed_labels": 0}

    for img_path, txt_path, target_split in assignments:
        target_img_dir = dest / target_split / "images"
        target_lbl_dir = dest / target_split / "labels"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        new_stem = f"{prefix}{img_path.stem}"
        new_img_name = f"{new_stem}{img_path.suffix.lower()}"
        new_txt_name = f"{new_stem}.txt"

        dest_img_path = target_img_dir / new_img_name
        dest_txt_path = target_lbl_dir / new_txt_name

        # 1. Chep anh
        if copy:
            shutil.copy2(img_path, dest_img_path)
        else:
            shutil.move(img_path, dest_img_path)

        # 2. Chuan hoa va ghi nhan
        cleaned_lines = validate_and_format_label(txt_path)
        dest_txt_path.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8")

        stats[target_split] += 1

    # Cap nhat file data.resolved.yaml
    yaml_path = make_data_yaml(dest)

    print("=" * 60)
    print("HOAN TAT GOP DU LIEU NHOM")
    print("=" * 60)
    print(f"- Nguon: {src}")
    print(f"- Dich:  {dest}")
    print(f"- Tong so mau hop le: {n_total}")
    print(f"  + train: {stats['train']} anh")
    print(f"  + valid: {stats['valid']} anh")
    print(f"  + test:  {stats['test']} anh")
    print(f"- Data config da cap nhat: {yaml_path}")
    print("=" * 60)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Gop du lieu tu chup cua nhom vao dataset YOLO")
    ap.add_argument("--src", required=True, help="Duong dan thu muc chua anh & nhan cua nhom")
    ap.add_argument("--dest", default=str(DATASET_DIR), help="Thu muc dataset chinh (mac dinh: data/biensoxe)")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test", "auto"],
                    help="Gop vao split nao (hoac 'auto' de chia theo ti le)")
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1],
                    help="Ti le chia train valid test khi dung --split auto (vd: 0.7 0.2 0.1)")
    ap.add_argument("--prefix", default="group_", help="Tien to them vao ten file de tranh trung ten")
    ap.add_argument("--move", action="store_true", help="Di chuyen file thay vi sao chep (copy)")
    a = ap.parse_args()

    merge_dataset(
        src=Path(a.src),
        dest=Path(a.dest),
        split=a.split,
        ratio=tuple(a.ratio),
        prefix=a.prefix,
        copy=not a.move,
    )


if __name__ == "__main__":
    main()
