import json
from pathlib import Path

from src.evaluation.report import generate_report


def test_generate_report(tmp_path: Path):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    out_file = tmp_path / "docs" / "RESULTS.md"

    # Gia lap metrics_test.json
    metrics = {
        "split": "test",
        "mode": "end2end (NMS-free)",
        "precision": 0.95,
        "recall": 0.91,
        "f1": 0.93,
        "mAP50": 0.96,
        "mAP50-95": 0.78,
    }
    (exp_dir / "metrics_test.json").write_text(json.dumps(metrics))

    # Gia lap benchmark_full.json
    bench = {
        "device": "NVIDIA GeForce RTX 4060",
        "fps": 35.5,
        "stages": {
            "preprocess": {"mean_ms": 2.1, "p50_ms": 1.9, "p95_ms": 3.0},
            "detect": {"mean_ms": 11.2, "p50_ms": 10.5, "p95_ms": 14.1},
            "ocr": {"mean_ms": 15.0, "p50_ms": 14.2, "p95_ms": 18.0},
            "total": {"mean_ms": 28.3, "p50_ms": 26.6, "p95_ms": 35.1},
        },
    }
    (exp_dir / "benchmark_full.json").write_text(json.dumps(bench))

    content = generate_report(exp_dir=exp_dir, out_file=out_file)
    assert out_file.exists()
    assert "Bảng phủ kỹ thuật" in content
    assert "0.96" in content
    assert "35.5 FPS" in content
