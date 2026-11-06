# VIETNAM ALPR – HỆ THỐNG NHẬN DẠNG BIỂN SỐ XE VIỆT NAM (YOLO26 & PADDLEOCR)

---

## 1. Tên dự án và Giới thiệu chung

* **Tên dự án:** Vietnam ALPR (Automatic License Plate Recognition) – Nhận dạng biển số xe Việt Nam thời gian thực ứng dụng YOLO26 (NMS-Free End-to-End) và PaddleOCR.
* **Mô tả ngắn:** Dự án xây dựng một giải pháp thông minh tự động phát hiện vị trí biển số xe và trích xuất chuỗi ký tự chuẩn xác từ hình ảnh/video. Hệ thống giải quyết triệt để các thách thức đặc thù của giao thông Việt Nam như biển số vuông 2 dòng của xe máy, biển bị mờ, nghiêng góc, thiếu sáng hoặc biển 4 số cũ. Sản phẩm hướng tới phục vụ các hệ thống quản lý bãi đỗ xe thông minh, trạm thu phí tự động (ETC) và camera giám sát an ninh trật tự đô thị.
* **Sơ đồ luồng hoạt động tổng thể (Two-Stage Pipeline):**

```text
[Hình ảnh đầu vào / Webcam]
           │
           ▼
[Tiền xử lý khung hình: Resize, chuẩn hóa kênh màu]
           │
           ▼
[Mô hình YOLO26n phát hiện vùng biển số (Cơ chế End-to-End, không cần lọc NMS)]
           │
           ▼
[Cắt vùng ROI biển số & Tự động nắn thẳng góc nghiêng bằng biến đổi Hough (Deskew)]
           │
           ▼
[Phân cụm dòng ảnh xe máy (Horizontal Projection) & Nhận dạng ký tự với PaddleOCR (PP-OCRv5)]
           │
           ▼
[Chuẩn hóa ký tự theo quy tắc Bộ Công an (Post-OCR Heuristic & Regex, sửa O/0, I/1, v.v.)]
           │
           ▼
[Kết quả đầu ra: Hiển thị trên Web Dashboard (Streamlit) hoặc trả lời qua REST API (FastAPI)]
```

---

## 2. Danh sách thành viên (Nhóm 21)

| STT | Họ và tên | Mã sinh viên (MSSV) |
| :---: | :--- | :---: |
| 1 | **Đặng Nhật Đức** | 3123410082 |
| 2 | **Phan Thanh Tùng** | 3123410417 |
| 3 | **Võ Văn Tuấn** | 3123410414 |

---

## 3. Công nghệ và Công cụ sử dụng (Tech Stack)

* **Ngôn ngữ lập trình:** Python 3.11.
* **Mô hình Trí tuệ Nhân tạo (AI/Deep Learning):**
  * **YOLO26n (Ultralytics):** Phát hiện vị trí biển số xe với cơ chế One-to-One Head (End-to-End NMS-Free).
  * **PaddleOCR (PP-OCRv5 Mobile Rec):** Nhận dạng ký tự quang học siêu nhẹ, chạy độc lập trên CPU.
* **Thư viện Xử lý ảnh & Thị giác máy tính:** OpenCV (`cv2`), NumPy, Pillow.
* **Framework Giao diện & Dịch vụ:**
  * **FastAPI & Uvicorn:** Cung cấp Backend REST API bất đồng bộ hiệu năng cao.
  * **Streamlit:** Giao diện Web Dashboard trực quan, tương tác ảnh tĩnh và Webcam thời gian thực.
* **Bộ công cụ DevOps & Đóng gói:** Docker, Docker Compose, Dockerfile đa tầng.
* **Công cụ hỗ trợ quản lý & Thử nghiệm:** Git/GitHub, Roboflow Universe, Swagger UI (OpenAPI), Pytest.

---

## 4. Điều kiện tiên quyết (Prerequisites)

Trước khi tiến hành cài đặt và chạy ứng dụng, môi trường máy tính cần chuẩn bị:

1. **Hệ điều hành:** Windows 10/11, Ubuntu 20.04/22.04 LTS hoặc macOS.
2. **Phiên bản Python:** Python **3.11** (khuyến nghị từ 3.10 đến 3.11 để tương thích tốt nhất với PaddleOCR và PyTorch).
3. **Phần mềm Git:** Đã cài đặt Git để clone mã nguồn.
4. **Phần cứng (Khuyến nghị):**
   * *Chạy GPU:* NVIDIA GTX 1650 (4GB VRAM) trở lên và đã cài NVIDIA CUDA Toolkit + cuDNN (đạt tốc độ ~36.3 FPS).
   * *Chạy CPU:* Tối thiểu CPU 4 nhân (ví dụ: Intel Core i5-9300H hoặc tương đương, đạt tốc độ ~14.9 FPS).
5. **Docker Desktop:** Chỉ cần nếu chọn cách chạy bằng Docker, không cần cài đặt Python/CUDA.

---

## 5. Hướng dẫn cài đặt và chạy (Installation & Running)

### Cách 1: Chạy trực tiếp trên máy (Local - Windows PowerShell)

#### Bước 1: Clone repository về máy
```powershell
git clone https://github.com/nduc2407/vietnam-alpr-yolo26.git
cd vietnam-alpr-yolo26
```

#### Bước 2: Tạo và kích hoạt môi trường ảo Python
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Trên Linux/macOS: `source .venv/bin/activate`)*.

#### Bước 3: Cài đặt các thư viện cần thiết
```powershell
python -m pip install --upgrade pip

# Nếu máy có GPU NVIDIA (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Nếu máy chỉ dùng CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt toàn bộ các thư viện còn lại:
pip install -r requirements.txt
```

#### Bước 4: Hướng dẫn chạy nhận diện (Inference & Chạy Sample)
Dự án đã tích hợp sẵn trọng số mô hình tốt nhất tại `models/best.pt` và các ảnh mẫu thử nghiệm tại `data/sample/images/`:

* **Test riêng từng ảnh mẫu cụ thể:**
  ```powershell
  python -m src infer --source "data\sample\images\sample_9.jpg"
  ```
* **Chạy nhận diện trên toàn bộ thư mục ảnh mẫu và lưu kết quả vào thư mục `out/`:**
  ```powershell
  python -m src infer --source "data\sample\images" --save out
  ```
* **Chạy nhận diện trên toàn bộ tập ảnh test:**
  ```powershell
  python -m src infer --source data\biensoxe\test\images --save out
  ```
* **Chạy với một ảnh bất kỳ trên máy tính của bạn:**
  ```powershell
  python -m src infer --source "duong_dan_den_anh_cua_ban.jpg"
  ```

#### Bước 5: Khởi chạy Giao diện Web Dashboard và REST API
* **Khởi chạy Giao diện Web (Streamlit):**
  ```powershell
  python -m src ui
  ```
  *(Truy cập trình duyệt tại: `http://localhost:8501`. Tại giao diện, bạn có thể chọn ngay ảnh mẫu từ menu dropdown hoặc tải ảnh mới, hoặc bật Webcam trực tiếp).*
* **Khởi chạy REST API Backend (FastAPI):**
  ```powershell
  python -m src serve
  ```
  *(Truy cập tài liệu kiểm thử tương tác Swagger UI tại: `http://localhost:8000/docs`)*.

#### Bước 6: Huấn luyện lại mô hình và Đánh giá thực nghiệm (Tùy chọn)
```powershell
# Đánh giá độ chính xác Precision, Recall, mAP trên tập test:
python -m src eval --split test

# Quét tìm ngưỡng tối ưu (Confidence sweep & NMS sweep):
python -m src sweep --split test

# Đánh giá độc lập mô-đun nhận dạng ký tự OCR (Exact-match, CER):
python -m src ocr-eval

# Đo độ trễ từng pha và đo FPS thực tế (Benchmark):
python -m src bench --n 50                  # Chạy trên GPU
python -m src bench --device cpu --n 50     # Chạy trên CPU
```

---

### Cách 2: Đóng gói và triển khai bằng Docker

Cách này phù hợp nhất cho việc triển khai trên máy chủ hoặc máy chấm thi chỉ với một lệnh duy nhất, không cần cài đặt Python, CUDA hay các thư viện phụ thuộc:

#### 1. Khởi chạy toàn bộ hệ thống bằng Docker Compose
Mở PowerShell hoặc Terminal tại thư mục dự án và chạy:
```powershell
docker-compose up --build -d
```
* **Cờ `--build`:** Tự động xây dựng image `vietnam-alpr:latest` từ `Dockerfile` nếu có thay đổi.
* **Cờ `-d` (Detached mode):** Khởi chạy ngầm cả 2 dịch vụ (`alpr_api` và `alpr_ui`), không chiếm dụng terminal.

#### 2. Kiểm tra trạng thái và theo dõi nhật ký hoạt động (Logs)
```powershell
# Kiểm tra trạng thái hoạt động của các container:
docker-compose ps

# Xem nhật ký hoạt động (logs) thời gian thực:
docker-compose logs -f
```

#### 3. Địa chỉ truy cập dịch vụ đã đóng gói
* **Web UI Dashboard:** `http://localhost:8501` (Giao diện trực quan, hỗ trợ kéo thả ảnh hoặc chọn ảnh mẫu có sẵn).
* **REST API Swagger Docs:** `http://localhost:8000/docs` (Tài liệu OpenAPI và kiểm thử trực tiếp các endpoint).
* **Kiểm tra trạng thái sức khỏe (Health Check):**
  ```powershell
  curl http://localhost:8000/health
  # Kết quả phản hồi: {"status":"ok"}
  ```

#### 4. Dừng hoặc khởi động lại hệ thống
```powershell
# Dừng và gỡ bỏ toàn bộ container của hệ thống:
docker-compose down

# Khởi động lại các container:
docker-compose restart
```

#### 5. Xuất file đóng gói chuyển giao máy khác (Offline / Nộp bài không cần Internet)
Nếu cần chuyển giao dự án sang máy tính khác không có kết nối mạng:
```powershell
# Bước 1 (Trên máy của bạn): Xuất Docker Image thành file nén .tar
docker save -o vietnam-alpr.tar vietnam-alpr:latest

# Bước 2 (Chuyển file vietnam-alpr.tar và file docker-compose.yml sang máy đích)

# Bước 3 (Trên máy đích): Nạp Image và khởi chạy ngay lập tức không cần tải gì từ mạng
docker load -i vietnam-alpr.tar
docker-compose up -d
```

---

## 6. Cấu trúc thư mục dự án (Project Structure)

Dưới đây là sơ đồ tổ chức mã nguồn thực tế của dự án, được phân tách theo chuẩn mô-đun hóa:

```text
vietnam-alpr-yolo26/
├── data/
│   ├── sample/images/           # Bộ 12 ảnh mẫu tổng hợp để chạy thử nghiệm nhanh (sample_0 -> sample_11)
│   └── biensoxe/                # Tập dữ liệu biển số xe chuẩn YOLO (train, valid, test) tải từ Roboflow
├── experiments/                 # Kết quả đo đạc thực nghiệm, ma trận nhầm lẫn, biểu đồ và file JSON benchmark
│   ├── benchmark_detect.json    # Kết quả đo độ trễ phát hiện biển số trên GPU GTX 1650
│   ├── benchmark_detect_cpu.json# Kết quả đo độ trễ phát hiện biển số trên CPU i5-9300H
│   ├── benchmark_full.json      # Kết quả phân rã thời gian trễ toàn trình Pipeline trên GPU
│   ├── benchmark_full_cpu.json  # Kết quả phân rã thời gian trễ toàn trình Pipeline trên CPU
│   ├── ocr_eval.json            # Kết quả đo kiểm chỉ số Exact Match và CER mô-đun OCR
│   └── threshold_sweep.csv      # Bảng kết quả quét ngưỡng Confidence và NMS IoU
├── models/
│   └── best.pt                  # Trọng số tối ưu của mô hình YOLO26n sau khi huấn luyện 100 epochs
├── notebooks/                   # Jupyter Notebooks nghiên cứu thực nghiệm
│   ├── train_yolo26.ipynb       # Notebook quy trình huấn luyện và tinh chỉnh YOLO26
│   └── YOLOv8Training.ipynb     # Notebook huấn luyện mô hình đối chứng Baseline YOLOv8
├── scripts/                     # Các công cụ script tự động hóa đo đạc và kiểm tra dữ liệu
├── src/                         # Toàn bộ mã nguồn chính của ứng dụng
│   ├── __init__.py
│   ├── __main__.py              # Điểm điều hướng thực thi CLI (download, train, eval, infer, serve, ui...)
│   ├── api.py                   # Backend REST API bằng FastAPI, xử lý upload ảnh và trả về JSON
│   ├── config.py                # File cấu hình tham số, đọc biến môi trường (.env) và định nghĩa đường dẫn
│   ├── data/                    # Xử lý dữ liệu
│   │   ├── dataset.py           # Quét, kiểm tra tính hợp lệ và tự động chuẩn hóa nhãn YOLO
│   │   ├── download.py          # Script tải bộ dữ liệu từ Roboflow API
│   │   └── merge.py             # Gộp tập ảnh nhóm tự chụp vào tập huấn luyện
│   ├── evaluation/              # Đo đạc và đánh giá thực nghiệm
│   │   ├── benchmark.py         # Đo độ trễ từng bước suy luận, tính p50, p95 và đo FPS (GPU & CPU)
│   │   ├── evaluate.py          # Đo Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95
│   │   ├── ocr_eval.py          # Đo độ chính xác nhận dạng ký tự (Exact-match, CER)
│   │   └── sweep.py             # Quét thực nghiệm ngưỡng tin cậy (conf) và phân tích NMS IoU
│   ├── inference/               # Pipeline xử lý nhận diện biển số
│   │   ├── detector.py          # Khởi tạo mô hình YOLO26 và suy luận phát hiện ROI biển số
│   │   ├── ocr.py               # Tích hợp PaddleOCR PP-OCRv5 và bộ quy tắc sửa lỗi ký tự Việt Nam
│   │   ├── pipeline.py          # Kết nối tuần tự luồng: Preprocess -> Detector -> Postprocess -> OCR
│   │   ├── postprocess.py       # Trích xuất tọa độ Bounding Box, tính tỷ lệ khung hình và cắt ROI
│   │   └── preprocess.py        # Tiền xử lý ảnh nâng cao: Nắn nghiêng Hough Deskew, cân bằng sáng CLAHE
│   ├── training/                # Module huấn luyện qua CLI
│   │   └── train.py             # Script gọi Ultralytics huấn luyện mô hình YOLO26n
│   └── ui/
│       └── app.py               # Giao diện Web tương tác Streamlit (Kéo thả ảnh, chọn sample, Webcam)
├── tests/                       # Bộ kiểm thử tự động với pytest (10 tệp kiểm thử chuyên sâu)
│   ├── test_api.py              # Kiểm thử API endpoint và kiểm tra tính hợp lệ của dữ liệu phản hồi
│   ├── test_dataset.py          # Kiểm thử chức năng kiểm tra và sửa nhãn YOLO
│   ├── test_merge.py            # Kiểm thử tích hợp gộp ảnh
│   ├── test_ocr_eval.py         # Kiểm thử pipeline đánh giá chất lượng OCR
│   ├── test_postprocess.py      # Kiểm thử cắt ảnh ROI và xử lý tọa độ bbox
│   ├── test_preprocess.py       # Kiểm thử thuật toán nắn nghiêng Hough và cân bằng sáng
│   └── test_sweep.py            # Kiểm thử quy trình quét ngưỡng tin cậy
├── app.py                       # File định tuyến khởi động Streamlit khi triển khai Docker
├── Dockerfile                   # Dockerfile cấu hình môi trường Python 3.11, OpenCV, PyTorch CPU và Paddle
├── docker-compose.yml           # Khởi tạo đồng thời 2 container: alpr_api (:8000) và alpr_ui (:8501)
├── requirements.txt             # Danh sách thư viện Python và phiên bản tương thích
└── README.md                    # Tài liệu hướng dẫn đồ án
```

---

## 7. Lưu ý / Các tính năng đã và đang phát triển (Roadmap)

### 7.1. Danh sách các tính năng đã hoàn thành
* [x] **Thu thập & Chuẩn hóa dữ liệu:** Tổng hợp 1.262 ảnh biển số xe Việt Nam, gán nhãn chuẩn YOLO và làm sạch dữ liệu tự động.
* [x] **Huấn luyện mô hình YOLO26n:** Huấn luyện thành công 100 epochs với cơ chế End-to-End One-to-One Head (mAP@0.5 đạt **98.5%**, F1-Score đạt **98.3%**).
* [x] **Thuật toán Nắn nghiêng hình học (Hough Deskew):** Giải quyết tình trạng chụp lệch góc, đạt độ chính xác nắn thẳng 24/24 ảnh thử nghiệm (vượt trội so với phương pháp `minAreaRect` truyền thống).
* [x] **Tách biển số xe máy 2 dòng (Horizontal Projection):** Chiếu ngang điểm ảnh để tách dòng trên (mã tỉnh/seri) và dòng dưới (số thứ tự) nhằm tránh lỗi đọc dồn chữ của OCR.
* [x] **Tích hợp PaddleOCR PP-OCRv5:** Tối ưu chạy thuần trên CPU để không xung đột bộ nhớ GPU, kết hợp ma trận Heuristic sửa lỗi nhầm chữ/số (`O/0`, `I/1`, `8/B`...) theo quy chuẩn vị trí của Bộ Công an.
* [x] **Giao diện người dùng Web Dashboard (Streamlit):** Trực quan hóa ảnh tải lên, chọn nhanh ảnh mẫu có sẵn, phát hiện thời gian thực qua Webcam và hiển thị chi tiết độ trễ từng chặng.
* [x] **Backend REST API Microservice (FastAPI):** Endpoint chuẩn OpenAPI/Swagger UI, xử lý tác vụ suy luận thông qua threadpool tránh khóa luồng.
* [x] **Đóng gói Docker hoàn chỉnh:** Dựng sẵn 2 container đồng bộ (API và Web UI) qua Docker Compose, tối ưu kích thước gói tải bằng wheel PyTorch CPU, cho phép khởi chạy hệ thống chỉ với một lệnh duy nhất.
* [x] **Kiểm thử tự động (Unit Tests):** 10 file pytest kiểm thử toàn diện từ tiền xử lý, thuật toán OCR đến API.

### 7.2. Kết quả thực nghiệm cốt lõi (Đồng bộ với Báo cáo Đồ án `CCNLTHD.docx`)

#### Bảng 1: Hiệu năng phát hiện của YOLO26n trên tập Test 53 ảnh (Bảng 5.1 & 5.2 trong báo cáo)
| Chỉ số đánh giá | Giá trị thực nghiệm | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| **Precision (P)** | **98.5%** | Tỷ lệ dự đoán vùng biển số là chính xác (hạn chế tối đa báo sai). |
| **Recall (R)** | **98.1%** | Khả năng phát hiện không bỏ sót biển số xe trong khung hình. |
| **F1-Score** | **98.3%** | Trung bình điều hòa cân bằng giữa Precision và Recall. |
| **mAP@0.5** | **98.5%** | Độ chính xác trung bình tại ngưỡng IoU = 0.5. |
| **mAP@0.5:0.95** | **96.2%** | Độ chính xác trung bình trên dải ngưỡng IoU từ 0.5 đến 0.95. |

#### Bảng 2: So sánh hiệu năng pha phát hiện biển số giữa CPU và GPU (Bảng 5.5 trong báo cáo)
| Phần cứng kiểm thử | Độ trễ trung bình | Tốc độ xử lý (FPS) | Độ ổn định (p50 / p95) | Đánh giá khả năng thời gian thực |
| :--- | :---: | :---: | :---: | :--- |
| **GPU NVIDIA GTX 1650** | **27.55 ms** | **36.3 FPS** | 20.33 ms / 78.90 ms | **Vượt chuẩn thời gian thực (> 30 FPS)** |
| **CPU Intel Core i5-9300H** | **67.18 ms** | **14.9 FPS** | 66.91 ms / 70.25 ms | Phù hợp triển khai biên (Edge AI) chi phí thấp |

> **Kết luận:** Chạy trên GPU giúp tăng tốc pha phát hiện gấp **2.44 lần** (+143.6% FPS) so với Baseline trên CPU.

#### Bảng 3: Phân rã thời gian trễ toàn trình Pipeline trên GPU GTX 1650 (Bảng 5.6 trong báo cáo)
| Giai đoạn trong Pipeline | Thời gian trễ | Tỷ lệ (%) | Vai trò & Đánh giá kỹ thuật |
| :--- | :---: | :---: | :--- |
| **1. Tiền xử lý khung hình** | 0.00 ms | 0.0% | Chuẩn hóa kích thước ảnh đầu vào. |
| **2. YOLO26 Phát hiện biển số** | 43.59 ms | 31.3% | Cắt trích xuất vùng ROI biển số. |
| **3. PaddleOCR Nhận dạng ký tự** | 95.89 ms | 68.7% | **Điểm nghẽn độ trễ chính của toàn bộ hệ thống.** |
| **Tổng độ trễ toàn trình** | **139.49 ms** | **100%** | **Tốc độ toàn hệ thống: 7.2 FPS** (~0.14s / lượt xe) |

#### Bảng 4: Đánh giá chất lượng mô-đun OCR trên 53 ảnh kiểm thử (Bảng 5.7 trong báo cáo)
| Phương pháp thử nghiệm | Khớp chính xác hoàn toàn (Exact Match) | Tỷ lệ lỗi ký tự (CER) |
| :--- | :---: | :---: |
| **Ảnh thô (Raw ROI - Không tiền xử lý)** | **77.4%** (41/53 ảnh) | **0.0789** |
| **Kèm tiền xử lý (Hough + CLAHE + Gamma)** | 69.8% (37/53 ảnh) | 0.0920 |

> **Phân tích hiện tượng:** Đối với các ảnh đầu vào đã có ánh sáng chuẩn và độ sắc nét cao, việc áp dụng thêm bộ lọc CLAHE làm tăng nhiễu hạt nền kim loại của biển số, dẫn tới PaddleOCR bị nhầm lẫn ký tự. Do đó hệ thống thiết lập mặc định không lạm dụng CLAHE trước OCR (`OCR_PREPROCESS=0`), chỉ kích hoạt khi phát hiện môi trường thiếu sáng nghiêm trọng.

### 7.3. Các lưu ý kỹ thuật chuyên sâu
1. **Bản chất NMS-Free của YOLO26:** Mô hình áp dụng kỹ thuật gán nhãn Hungarian Matching một-đối-một trong pha suy luận (`end2end=True`). Hệ thống không cần chạy bước lọc NMS trên CPU, giúp loại bỏ hoàn toàn độ trễ biến thiên bất định (2–4 ms) của NMS truyền thống.
2. **Hiện tượng xung đột đa luồng của `cv2.CLAHE`:** Trong môi trường web server đa luồng (FastAPI / Uvicorn), việc dùng chung một instance `cv2.CLAHE` giữa các thread sẽ gây lỗi dữ liệu ảnh (sai sót 330/2400 lần thử nghiệm). Do đó, mã nguồn luôn khởi tạo đối tượng `cv2.createCLAHE()` độc lập trong từng hàm gọi.

### 7.4. Hướng phát triển trong tương lai (Roadmap)
* [ ] **Tối ưu hóa mô-đun OCR:** Huấn luyện mô hình nhận dạng ký tự chuyên biệt cho biển số xe Việt Nam (như CRNN siêu nhẹ hoặc ONNX Runtime lượng tử hóa INT8) nhằm giảm thời gian trễ của pha OCR từ ~95 ms xuống < 20 ms, đưa tốc độ toàn trình đạt > 25 FPS.
* [ ] **Tích hợp cơ chế Tracking phương tiện:** Bổ sung thuật toán ByteTrack/DeepSORT để theo dõi biển số xe chuyển động liên tục qua luồng RTSP từ camera an ninh bãi đỗ xe.
* [ ] **Mở rộng cơ sở dữ liệu:** Bổ sung các loại biển số chuyên dụng đặc thù (biển số quân đội đỏ, biển số ngoại giao NG, biển số xe điện màu xanh lá).
