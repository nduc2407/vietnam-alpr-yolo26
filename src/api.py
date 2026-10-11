"""API nhận dạng biển số xe và đo độ trễ."""

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import BEST_MODEL
from src.inference.pipeline import ALPRPipeline

app = FastAPI(title="ALPR API", description="API don gian nhan dang bien so xe va do do tre.")

# Khoi tao pipeline
pipeline = ALPRPipeline(BEST_MODEL)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Nhan 1 file anh -> Kiem tra dau vao -> Tra ve bien so va do tre (ms)."""
    # 1. Kiem tra va doc du lieu anh
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File anh rong.")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Du lieu khong phai la file anh hop le.")

    # 2. Suy luan va do do tre
    out = pipeline.process(image)

    # 3. Tra ket qua
    return {
        "count": len(out["plates"]),
        "plates": out["plates"],
        "timing_ms": out["timing_ms"],
    }
