"""Tai dataset tu Roboflow roi tu dong sua label + sinh data.resolved.yaml.

API key lay o: Roboflow -> Settings -> API Keys. KHONG hard-code key vao code.
    Windows (PowerShell):  $env:ROBOFLOW_API_KEY="xxxx"

Chay (mac dinh lay thong tin trong src/config.py):
    python -m src.data.download
    python -m src.data.download --version 1
"""
import argparse
import os

from src.config import (DATASET_DIR, ROBOFLOW_FORMAT, ROBOFLOW_PROJECT,
                        ROBOFLOW_VERSION, ROBOFLOW_WORKSPACE)
from src.data.dataset import fix_labels, make_data_yaml, scan_labels


def download(workspace=ROBOFLOW_WORKSPACE, project=ROBOFLOW_PROJECT, version=ROBOFLOW_VERSION,
             fmt=ROBOFLOW_FORMAT, api_key=None, location=DATASET_DIR):
    from roboflow import Roboflow

    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Thieu API key: dat bien moi truong ROBOFLOW_API_KEY hoac dung --api-key")

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(workspace).project(project).version(version).download(fmt, location=str(location))
    print("Da tai ve:", ds.location)

    r = scan_labels(location)
    print(f"Label files: {r['files']} | Bad: {len(r['bad'])} | Empty: {len(r['empty'])}")
    if r["bad"]:
        print(fix_labels(location))
    print("data yaml:", make_data_yaml(location))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=ROBOFLOW_WORKSPACE)
    ap.add_argument("--project", default=ROBOFLOW_PROJECT)
    ap.add_argument("--version", type=int, default=ROBOFLOW_VERSION)
    ap.add_argument("--format", default=ROBOFLOW_FORMAT, help="yolov8 | yolov11 | yolov5 ...")
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    download(a.workspace, a.project, a.version, a.format, a.api_key)


if __name__ == "__main__":
    main()
