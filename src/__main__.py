"""Điểm vào thực thi các module của hệ thống ALPR:
  download   Tải dataset từ Roboflow API
  merge      Gộp dữ liệu tự thu thập vào dataset YOLO
  check      Kiểm tra và chuẩn hóa nhãn YOLO
  train      Huấn luyện mô hình YOLO26
  eval       Đánh giá mô hình trên tập kiểm thử
  ocr-eval   Đo lường độ chính xác mô-đun OCR độc lập
  sweep      Phân tích thực nghiệm ngưỡng tin cậy và NMS
  bench      Đo độ trễ và FPS của hệ thống
  report     Tổng hợp số liệu và sinh báo cáo kết quả
  infer      Suy luận nhận diện biển số
  serve      Khởi chạy REST API
  ui         Khởi chạy giao diện Web
"""
import sys

COMMANDS = {
    "download": "src.data.download",
    "merge": "src.data.merge",
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
    uvicorn.run("src.api:app", host=a.host, port=a.port)


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
