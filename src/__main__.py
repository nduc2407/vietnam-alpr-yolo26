"""Diem vao chung: python -m src <lenh> [tham so]

Lenh:
  download   tai dataset tu Roboflow API
  check      kiem tra / sua label + sinh data.resolved.yaml
  train      train YOLO26
  eval       danh gia (valid/test)
  ocr-eval   do chinh xac OCR (exact-match, CER) co/khong tien xu ly
  sweep      quet nguong conf + NMS (precision/recall/F1)
  bench      do do tre / FPS (them --full de gom OCR)
  infer      chay ALPR tren anh / thu muc / webcam
  serve      chay REST API (FastAPI)
  ui         chay Web UI (Streamlit)
Xem tham so cua tung lenh: python -m src <lenh> --help
"""
import sys

COMMANDS = {
    "download": "src.data.download",
    "check": "src.data.dataset",
    "train": "src.training.train",
    "eval": "src.evaluation.evaluate",
    "ocr-eval": "src.evaluation.ocr_eval",
    "sweep": "src.evaluation.sweep",
    "bench": "src.evaluation.benchmark",
    "report": "src.evaluation.report",
    "infer": "src.inference.pipeline",
}


def serve(argv):
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m src serve")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    uvicorn.run("src.backend.app:app", host=a.host, port=a.port)


def run_ui(argv):
    import subprocess
    from pathlib import Path
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)] + argv
    subprocess.run(cmd)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "serve":
        return serve(rest)
    if cmd == "ui":
        return run_ui(rest)
    if cmd not in COMMANDS:
        print(f"Lenh khong hop le: {cmd}\n{__doc__}")
        sys.exit(1)
    import importlib

    sys.argv = [f"python -m src {cmd}"] + rest
    importlib.import_module(COMMANDS[cmd]).main()


if __name__ == "__main__":
    main()
