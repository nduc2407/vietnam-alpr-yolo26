"""Train YOLO26 phat hien bien so.

Chay:
    python -m src.training.train --epochs 100 --batch 16
"""
import argparse
import shutil
from pathlib import Path

from src.config import BEST_MODEL, EXPERIMENTS_DIR, IMG_SIZE, PRETRAINED
from src.data.dataset import fix_labels, make_data_yaml, scan_labels


def default_device():
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def train(
    model=PRETRAINED,
    epochs=100,
    imgsz=IMG_SIZE,
    batch=16,
    device=None,
    workers=4,
    patience=30,
    name="yolo26n_plates",
    resume=False,
    fix=True,
    **extra,
):
    """Train va copy best.pt sang models/best.pt. Tra ve duong dan best.pt."""
    from ultralytics import YOLO

    if fix:
        report = scan_labels()
        if report["bad"]:
            print(f"Phat hien {len(report['bad'])} file label sai dinh dang -> tu dong sua")
            print(fix_labels())
    data_yaml = make_data_yaml()

    yolo = YOLO(str(model))
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device or default_device(),
        workers=workers,
        patience=patience,
        project=str(EXPERIMENTS_DIR),
        name=name,
        exist_ok=True,
        resume=resume,
        plots=True,
        **extra,
    )

    best = Path(yolo.trainer.best)
    BEST_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BEST_MODEL)
    print(f"Da luu model tot nhat: {BEST_MODEL}")
    return BEST_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PRETRAINED))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="yolo26n_plates")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-fix", action="store_true")
    a = ap.parse_args()
    train(a.model, a.epochs, a.imgsz, a.batch, a.device, a.workers, a.patience, a.name, a.resume, not a.no_fix)


if __name__ == "__main__":
    main()
