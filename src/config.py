"""Cau hinh duong dan va tham so mac dinh cho toan he thong."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env(path=ROOT / ".env"):
    """Doc file .env (KEY=VALUE) vao os.environ, khong can cai them thu vien."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

# Dataset moi tai tu Roboflow -> data/biensoxe (doi bang bien moi truong ALPR_DATASET neu can)
DATASET_NAME = os.environ.get("ALPR_DATASET", "biensoxe")
DATASET_DIR = ROOT / "data" / DATASET_NAME

# Roboflow: https://universe.roboflow.com/thanh-tung-phan/biensoxe-o1lhe
ROBOFLOW_WORKSPACE = "thanh-tung-phan"
ROBOFLOW_PROJECT = "biensoxe-o1lhe"
ROBOFLOW_VERSION = 2
ROBOFLOW_FORMAT = "yolov8"
DATA_YAML = DATASET_DIR / "data.yaml"                 # file goc tu Roboflow
DATA_YAML_RESOLVED = DATASET_DIR / "data.resolved.yaml"  # file duoc sinh tu dong (duong dan tuyet doi)

MODELS_DIR = ROOT / "models"
BEST_MODEL = MODELS_DIR / "best.pt"
# Neu chua co file yolo26n.pt o thu muc goc, Ultralytics se tu tai ve theo ten
_local = ROOT / "yolo26n.pt"
PRETRAINED = _local if _local.exists() else "yolo26n.pt"

EXPERIMENTS_DIR = ROOT / "experiments"

# Tham so suy luan mac dinh
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.5

for _d in (MODELS_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
