from pathlib import Path
from src.data.merge import merge_dataset, find_pairs


def test_find_pairs(tmp_path: Path):
    src = tmp_path / "custom"
    src.mkdir()
    (src / "car1.jpg").write_bytes(b"fake_image_bytes")
    (src / "car1.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    (src / "car2.png").write_bytes(b"fake_image_bytes")
    (src / "car2.txt").write_text("0 0.4 0.4 0.3 0.2\n")
    (src / "orphan.jpg").write_bytes(b"fake_image_bytes")

    pairs, missing = find_pairs(src)
    assert len(pairs) == 2
    assert len(missing) == 1
    assert missing[0].name == "orphan.jpg"


def test_merge_dataset_single_split(tmp_path: Path):
    src = tmp_path / "custom"
    dest = tmp_path / "dataset"
    src.mkdir()
    (src / "img_01.jpg").write_bytes(b"dummy")
    (src / "img_01.txt").write_text("0 0.5 0.5 0.2 0.1\n")

    stats = merge_dataset(src=src, dest=dest, split="test", prefix="mygroup_")
    assert stats["test"] == 1
    assert (dest / "test" / "images" / "mygroup_img_01.jpg").exists()
    assert (dest / "test" / "labels" / "mygroup_img_01.txt").exists()


def test_merge_dataset_auto_split(tmp_path: Path):
    src = tmp_path / "custom"
    dest = tmp_path / "dataset"
    src.mkdir()
    for i in range(10):
        (src / f"sample_{i}.jpg").write_bytes(b"dummy")
        (src / f"sample_{i}.txt").write_text("0 0.5 0.5 0.2 0.1\n")

    stats = merge_dataset(src=src, dest=dest, split="auto", ratio=(0.7, 0.2, 0.1), prefix="test_")
    assert stats["train"] + stats["valid"] + stats["test"] == 10
    assert stats["train"] == 7
    assert stats["valid"] == 2
    assert stats["test"] == 1
