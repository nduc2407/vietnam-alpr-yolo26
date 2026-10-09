import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Setup python path to root if needed
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BEST_MODEL, CONF_THRES, IOU_THRES
from src.inference.pipeline import ALPRPipeline


# Page configuration
st.set_page_config(
    page_title="Vietnam ALPR - YOLO26",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .plate-badge {
        display: inline-block;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 1.6rem;
        font-weight: 800;
        background: #0F172A;
        color: #F8FAFC;
        padding: 6px 16px;
        border-radius: 8px;
        border: 2px solid #38BDF8;
        letter-spacing: 1.5px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-valid {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-invalid {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang tải model ALPR (YOLO26 & PP-OCRv5)...")
def get_pipeline(weights_path: str, conf: float, iou: float, use_ocr: bool, enhance: bool, end2end: bool):
    """Tải và cache pipeline ALPR."""
    return ALPRPipeline(
        weights=weights_path,
        conf=conf,
        iou=iou,
        use_ocr=use_ocr,
        enhance=enhance,
        end2end=end2end,
    )


def find_sample_images() -> list[Path]:
    """Tìm các file ảnh mẫu trong data/sample/images."""
    sample_dir = ROOT_DIR / "data" / "sample" / "images"
    if sample_dir.exists():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return [p for p in sorted(sample_dir.glob("*")) if p.suffix.lower() in exts]
    return []


def render_sidebar():
    """Vẽ thanh công cụ cấu hình bên trái."""
    st.sidebar.title("⚙️ Cấu hình hệ thống")

    # Weights path
    default_model = str(BEST_MODEL) if BEST_MODEL.exists() else "models/best.pt"
    weights_path = st.sidebar.text_input("Trọng số Model YOLO:", value=default_model)

    # Sliders
    conf_thres = st.sidebar.slider("Detection Confidence", min_value=0.05, max_value=1.0, value=float(CONF_THRES), step=0.05)
    iou_thres = st.sidebar.slider("IoU Threshold (NMS)", min_value=0.05, max_value=1.0, value=float(IOU_THRES), step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Bộ xử lý")
    use_ocr = st.sidebar.checkbox("Bật OCR (PaddleOCR v5)", value=True)
    enhance_clahe = st.sidebar.checkbox("Bật tiền xử lý CLAHE (Tăng tương phản)", value=False)
    end2end_mode = st.sidebar.radio("Chế độ Detection:", ("End-to-End (NMS-free)", "One-to-Many + NMS"), index=0)

    end2end = end2end_mode.startswith("End-to-End")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Mẹo:**\n"
        "- **End-to-End:** Tốc độ nhanh hơn, tối ưu cho YOLO26.\n"
        "- **CLAHE:** Hữu ích khi ảnh bị tối, thiếu sáng hoặc lóa đèn."
    )

    return weights_path, conf_thres, iou_thres, use_ocr, enhance_clahe, end2end


def process_image(image_bgr: np.ndarray, pipeline: ALPRPipeline):
    """Xử lý ảnh và vẽ kết quả."""
    res = pipeline.process(image_bgr)
    plates = res["plates"]
    timing = res["timing_ms"]
    annotated_bgr = pipeline.annotate(image_bgr, plates)
    return plates, timing, annotated_bgr


def tab_image(pipeline: ALPRPipeline):
    st.subheader("🖼️ Nhận diện biển số từ Ảnh")

    samples = find_sample_images()
    sample_options = ["-- Tải ảnh từ máy tính --"] + [f"Ảnh mẫu: {p.name}" for p in samples]
    selected_sample = st.selectbox("Chọn ảnh mẫu hoặc tải ảnh mới:", sample_options)

    image_bgr = None
    if selected_sample == "-- Tải ảnh từ máy tính --":
        uploaded_file = st.file_uploader("Chọn file ảnh (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        sample_idx = sample_options.index(selected_sample) - 1
        sample_path = samples[sample_idx]
        image_bgr = cv2.imread(str(sample_path))

    if image_bgr is not None:
        plates, timing, annotated_bgr = process_image(image_bgr, pipeline)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Hiển thị metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Biển số phát hiện", f"{len(plates)}")
        with m2:
            st.metric("Preprocess", f"{timing['preprocess']} ms")
        with m3:
            st.metric("Detect", f"{timing['detect']} ms")
        with m4:
            st.metric("OCR", f"{timing['ocr']} ms")
        with m5:
            st.metric("Tổng thời gian", f"{timing['total']} ms", delta=f"{round(1000/max(timing['total'], 1), 1)} FPS")

        # Hiển thị hình ảnh so sánh
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(annotated_rgb, caption="Kết quả phát hiện & OCR", use_container_width=True)

        st.markdown("### 📋 Danh sách biển số chi tiết")
        if not plates:
            st.warning("Không tìm thấy biển số nào trong ảnh với ngưỡng thiết lập hiện tại.")
        else:
            for idx, p in enumerate(plates, 1):
                with st.container():
                    c_crop, c_info = st.columns([1, 3])
                    with c_crop:
                        x1, y1, x2, y2 = p["bbox"]
                        h, w, _ = image_bgr.shape
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop = image_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Crop #{idx}", use_container_width=True)

                    with c_info:
                        plate_text = p.get("plate_formatted", "N/A")
                        st.markdown(f"<div class='plate-badge'>{plate_text}</div>", unsafe_allow_html=True)
                        st.write("")
                        valid = p.get("valid", True)
                        status_html = (
                            "<span class='status-valid'>✓ Biển số hợp lệ</span>"
                            if valid
                            else "<span class='status-invalid'>⚠ Chưa chuẩn quy tắc</span>"
                        )
                        st.markdown(f"**Trạng thái:** {status_html}", unsafe_allow_html=True)
                        st.write(
                            f"- **Độ tin cậy Detect:** `{p['det_conf'] * 100:.1f}%`\n"
                            f"- **Độ tin cậy OCR:** `{p.get('ocr_conf', 0) * 100:.1f}%`\n"
                            f"- **Tọa độ BBox:** `{p['bbox']}`"
                        )
                    st.markdown("---")


def tab_video(pipeline: ALPRPipeline):
    st.subheader("🎥 Nhận diện biển số từ Video")
    uploaded_video = st.file_uploader("Tải lên file video (.mp4, .avi, .mov):", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            frame_skip = st.slider("Xử lý mỗi N frame (tăng tốc):", min_value=1, max_value=10, value=2)
        with c_opt2:
            max_frames = st.slider("Giới hạn số frame xử lý:", min_value=30, max_value=max(100, total_frames), value=min(300, total_frames))

        start_btn = st.button("🚀 Bắt đầu xử lý Video", type="primary")

        if start_btn:
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_placeholder = st.empty()
            table_placeholder = st.empty()

            detected_plates_history = []
            seen_plates = set()

            frame_idx = 0
            processed_count = 0
            start_time = time.time()

            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    processed_count += 1
                    plates, timing, annotated_bgr = process_image(frame, pipeline)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True)

                    for p in plates:
                        plate_str = p.get("plate_formatted", "")
                        if plate_str and plate_str not in seen_plates:
                            seen_plates.add(plate_str)
                            detected_plates_history.append({
                                "Frame": frame_idx,
                                "Thời gian (s)": round(frame_idx / fps, 2),
                                "Biển số": plate_str,
                                "Độ tin cậy Detect": f"{p['det_conf']*100:.1f}%",
                                "Độ tin cậy OCR": f"{p.get('ocr_conf', 0)*100:.1f}%",
                                "Hợp lệ": "✓" if p.get("valid", True) else "⚠",
                            })

                    if detected_plates_history:
                        df = pd.DataFrame(detected_plates_history)
                        table_placeholder.dataframe(df, use_container_width=True)

                frame_idx += 1
                progress = min(1.0, processed_count / max_frames)
                progress_bar.progress(progress)
                elapsed = time.time() - start_time
                status_text.text(f"Đang xử lý: {processed_count}/{max_frames} frames | Tốc độ: {round(processed_count / max(elapsed, 0.1), 1)} FPS")

            cap.release()
            st.success(f"Hoàn thành! Đã phát hiện {len(seen_plates)} biển số khác nhau.")


def tab_webcam(pipeline: ALPRPipeline):
    st.subheader("📹 Nhận diện qua Webcam / Live Camera")

    mode = st.radio("Chọn cách nhận diện:", ("📸 Chụp ảnh nhanh từ Webcam (Khuyên dùng)", "🔴 Luồng Video trực tiếp (Live Stream)"), horizontal=True)

    if mode.startswith("📸"):
        camera_photo = st.camera_input("Chụp ảnh phương tiện trước camera")
        if camera_photo is not None:
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            plates, timing, annotated_bgr = process_image(image_bgr, pipeline)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption="Kết quả nhận diện từ Webcam", use_container_width=True)
            if plates:
                for idx, p in enumerate(plates, 1):
                    plate_text = p.get("plate_formatted", "N/A")
                    st.markdown(f"### Biển số #{idx}: `<span class='plate-badge'>{plate_text}</span>`", unsafe_allow_html=True)
            else:
                st.info("Chưa phát hiện thấy biển số trong ảnh chụp.")

    else:
        cam_index = st.number_input("Chỉ số Camera thiết bị (0 là webcam mặc định):", min_value=0, max_value=5, value=0)
        run_live = st.checkbox("Bật luồng Live Camera")

        frame_window = st.empty()
        info_window = st.empty()

        if run_live:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(f"Không thể kết nối đến Camera số {cam_index}. Vui lòng kiểm tra lại thiết bị.")
            else:
                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Mất tín hiệu camera.")
                        break

                    plates, timing, annotated_bgr = process_image(frame, pipeline)
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    frame_window.image(annotated_rgb, use_container_width=True)

                    text_info = f"⚡ FPS: {round(1000/max(timing['total'], 1), 1)} | Phát hiện: {len(plates)} biển số"
                    if plates:
                        plates_list = ", ".join([p.get("plate_formatted", "") for p in plates])
                        text_info += f" | Biển số: {plates_list}"
                    info_window.info(text_info)

                cap.release()


def main():
    st.markdown('<div class="main-header">🇻🇳 Hệ Thống Nhận Diện Biển Số Xe Việt Nam</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Mô hình phát hiện biển số YOLO26 kết hợp nhận dạng ký tự PaddleOCR (PP-OCRv5)</div>', unsafe_allow_html=True)

    # Sidebar
    weights_path, conf_thres, iou_thres, use_ocr, enhance_clahe, end2end = render_sidebar()

    # Load Pipeline
    try:
        pipeline = get_pipeline(
            weights_path=weights_path,
            conf=conf_thres,
            iou=iou_thres,
            use_ocr=use_ocr,
            enhance=enhance_clahe,
            end2end=end2end,
        )
    except Exception as e:
        st.error(f"❌ Lỗi tải mô hình: {e}")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Ảnh (Images)", "🎥 Video", "📹 Webcam / Live Camera"])

    with tab1:
        tab_image(pipeline)
    with tab2:
        tab_video(pipeline)
    with tab3:
        tab_webcam(pipeline)


if __name__ == "__main__":
    main()
