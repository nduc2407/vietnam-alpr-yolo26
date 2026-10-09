# ALPR YOLO26 – Nhận dạng biển số xe Việt Nam

Luồng hệ thống: **dữ liệu (Roboflow API + ảnh nhóm tự chụp/gán nhãn) → kiểm tra/sửa nhãn → huấn luyện YOLO26 → đánh giá (P/R/mAP) → phân tích ngưỡng conf & NMS → đo độ trễ/FPS → dịch vụ REST API (Docker)**.

Dataset gốc: <https://universe.roboflow.com/thanh-tung-phan/biensoxe-o1lhe> (CC BY 4.0, 1 lớp `BienSoXe`).


## Chạy local (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU; bỏ qua nếu chỉ CPU
pip install -r requirements.txt   # gồm paddlepaddle (CPU) + paddleocr

$env:ROBOFLOW_API_KEY = (Read-Host "Dan API key").Trim()   # hoặc ghi vào .env
python -m src download                       # -> data/biensoxe
python -m src train --epochs 50 --batch 8 --workers 2       # -> models/best.pt

python -m src eval --split test              # P, R, F1, mAP
python -m src eval --split test --nms        # đường NMS (end2end=False)
python -m src sweep --split test             # quét ngưỡng conf và NMS
python -m src ocr-eval --template 100        # tạo CSV để điền chuỗi biển số thật, rồi: python -m src ocr-eval
python -m src bench --n 100 --full           # độ trễ từng bước + FPS
python -m src report                         # gom số liệu -> docs/RESULTS.md

python -m src infer --source data\sample\images --save out
python -m src serve                          # API tại :8000
```

## Cấu trúc

```
src/
  config.py               đường dẫn, tham số, đọc .env
  data/                   download.py (Roboflow API) · dataset.py (kiểm tra/sửa nhãn) · merge.py (gộp ảnh nhóm)
  training/train.py       huấn luyện YOLO26
  evaluation/             evaluate.py · sweep.py (conf/NMS) · ocr_eval.py · benchmark.py (latency/FPS) · report.py
  inference/              preprocess.py (ImagePreprocessor) · detector.py · postprocess.py (PostProcessor) · ocr.py (PaddleOCR) · pipeline.py
  backend/app.py          FastAPI (xác thực, giới hạn upload, rate limit)
tests/                    pytest
data/sample/              ảnh mẫu TỔNG HỢP để chạy thử (không dùng để huấn luyện)
## Tiền xử lý và hậu xử lý

Luồng: `preprocess (resize, CLAHE tùy chọn) → YOLO → PostProcessor (bbox, cắt ROI, base64) → enhance_roi (phóng to, nắn nghiêng) → lọc nhiễu/CLAHE/làm nét → PaddleOCR → chuẩn hóa biển số`.

- **CLAHE trước khi detect mặc định tắt.** Model được huấn luyện trên ảnh không qua CLAHE, nên bật CLAHE lúc suy luận có thể làm lệch phân phối. Chỉ bật khi `python -m src sweep --split test --enhance` cho F1 cao hơn bản không CLAHE (bảng so sánh nằm trong `docs/RESULTS.md`).
- **Nắn nghiêng dùng Hough** (đường viền và dòng chữ) làm chính, `minAreaRect` làm dự phòng. Trên 24 ảnh biển tổng hợp xoay ±5°–20° kèm nền (cắt bbox trục chuẩn như crop của YOLO), `minAreaRect` + Otsu nắn đúng 0/24 ảnh, Hough đúng 24/24. Có test hồi quy trong `tests/test_preprocess.py`.
- **CLAHE tạo mới mỗi lần gọi:** dùng chung một đối tượng `cv2.CLAHE` giữa các luồng cho kết quả sai (330/2400 lần trong thử nghiệm 8 luồng), mà FastAPI chạy endpoint đồng bộ trên threadpool.
- **`include_crops=true`** trả thêm ảnh crop từng biển (base64); mặc định tắt để giảm tải.

## OCR: PaddleOCR (PP-OCRv5)

Chỉ dùng module nhận dạng (`TextRecognition`) trên ảnh biển đã cắt bởi YOLO. Lần chạy đầu PaddleOCR tải model về `~/.paddlex` (cần Internet). Mặc định chạy CPU (`OCR_DEVICE=cpu`) để không xung đột CUDA với PyTorch; đổi model bằng `OCR_MODEL` (vd `PP-OCRv5_mobile_rec`).
Nếu `import cv2` lỗi sau khi cài (nhiều gói OpenCV chồng nhau): `pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless` rồi `pip install opencv-python`.

## Lưu ý kỹ thuật về NMS

YOLO26 mặc định là end-to-end, **không dùng NMS** (tham số `iou` khi đó không có tác dụng). Để đáp ứng yêu cầu phân tích NMS, repo đo cả hai đường: mặc định (NMS-free) và `end2end=False` kèm NMS với các mức IoU 0.3/0.5/0.7. Báo cáo nên nêu kết quả so sánh và lựa chọn cuối cùng.
## Installation and Usage
Bước 1: Clone repository

Mở PowerShell hoặc Terminal và chạy các lệnh sau để tải mã nguồn dự án từ GitHub về máy:

git clone https://github.com/nduc2407/vietnam-alpr-yolo26.git
cd vietnam-alpr-yolo26
Bước 2: Tạo môi trường ảo

Tạo môi trường Python riêng cho dự án và kích hoạt môi trường đó:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Môi trường ảo giúp cô lập các thư viện của dự án, tránh xung đột với những dự án Python khác trên máy tính.

Bước 3: Cài đặt các thư viện cần thiết

Cập nhật pip và cài đặt các thư viện được khai báo trong file requirements.txt:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Đảm bảo quá trình cài đặt hoàn tất thành công trước khi chuyển sang bước tiếp theo.

Bước 4: Chạy nhận diện biển số xe

Đảm bảo file trọng số mô hình đã huấn luyện (best.pt) và ảnh mẫu tồn tại đúng vị trí mà hệ thống yêu cầu.

Chạy lệnh sau để nhận diện biển số trên ảnh mẫu:

python -m src infer --source "data\sample\images\sample_9.jpg"

Để xử lý một ảnh khác, thay đường dẫn ảnh mẫu bằng đường dẫn đến ảnh cần nhận diện:

python -m src infer --source "path\to\your\image.jpg"

Trong đó, path\to\your\image.jpg là đường dẫn ví dụ. Hãy thay bằng đường dẫn thực tế đến ảnh trên máy tính của bạn.

## Ví dụ
python -m src infer --source data\biensoxe\test\images --save out
## Test riêng từng ảnh
 python -m src infer --source "data\sample\images\sample_9.jpg"
