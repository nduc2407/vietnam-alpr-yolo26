"""REST API cho ALPR (FastAPI) - co cac bien phap bao mat co ban theo OWASP Top 10.

Chay:
    uvicorn src.backend.app:app --host 0.0.0.0 --port 8000
Docs (Swagger): http://localhost:8000/docs

Bien moi truong (xem .env.example):
    API_KEY             neu dat -> /predict yeu cau header X-API-Key
    MAX_UPLOAD_MB       gioi han dung luong upload (mac dinh 5)
    MAX_PIXELS          gioi han so diem anh (mac dinh 25 trieu) - chong decompression bomb
    RATE_LIMIT_PER_MIN  so request/phut/IP (mac dinh 60, 0 = tat)
    ALLOWED_ORIGINS     danh sach origin CORS, phan cach dau phay (mac dinh: khong bat CORS)
"""
import base64
import hmac
import io
import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.config import BEST_MODEL
from src.inference.pipeline import ALPRPipeline

log = logging.getLogger("alpr.api")
state: dict = {"pipe": None}
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["pipe"] = ALPRPipeline(BEST_MODEL)
        log.info("Da nap model %s", BEST_MODEL)
    except Exception as exc:  # model chua train / thieu thu vien -> van cho /health tra loi
        state["pipe"] = None
        log.error("Khong nap duoc model: %s", exc)
    yield
    state["pipe"] = None


app = FastAPI(title="ALPR Vietnam", version="1.1", lifespan=lifespan,
              description="Nhan dang bien so xe Viet Nam (YOLO26 + OCR).")

_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:  # A05: khong dung "*"
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                       allow_headers=["X-API-Key", "Content-Type"])


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Loi khong xu ly duoc")  # A09: log chi tiet o server, khong lo cho client
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


def require_key(request: Request):
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    given = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):  # A07: so sanh hang so thoi gian
        log.warning("API key sai tu %s", request.client.host if request.client else "?")
        raise HTTPException(401, "Invalid or missing API key")


def rate_limit(request: Request):
    limit = int(_float_env("RATE_LIMIT_PER_MIN", 60))
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, "Too many requests")
        q.append(now)


def _magic_ok(data: bytes) -> bool:
    return (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP"))


def decode_upload(data: bytes) -> np.ndarray:
    """Kiem tra ky: magic bytes, so diem anh (truoc khi giai ma), roi moi decode."""
    if not _magic_ok(data):
        raise HTTPException(415, "Only JPEG, PNG or WEBP images are accepted")
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        raise HTTPException(400, "Invalid image") from None
    if w * h > int(_float_env("MAX_PIXELS", 25_000_000)):
        raise HTTPException(413, "Image resolution too large")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")
    return img


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_loaded": state.get("pipe") is not None}


@app.post("/predict", tags=["alpr"], dependencies=[Depends(require_key), Depends(rate_limit)])
def predict(file: UploadFile = File(...), annotate: bool = False, include_crops: bool = False):
    """Upload 1 anh (JPEG/PNG/WEBP) -> danh sach bien so.

    `annotate=true` tra them anh ve khung (base64); `include_crops=true` tra anh crop tung bien (base64).
    """
    pipe = state.get("pipe")
    if pipe is None:
        raise HTTPException(503, "Model not loaded")
    limit = int(_float_env("MAX_UPLOAD_MB", 5) * 1024 * 1024)
    data = file.file.read(limit + 1)  # khong doc qua gioi han
    if len(data) > limit:
        raise HTTPException(413, "File too large")
    img = decode_upload(data)
    out = pipe.process(img, encode_crop=include_crops)
    payload = {"count": len(out["plates"]), "plates": out["plates"], "timing_ms": out["timing_ms"]}
    if annotate:
        ok, buf = cv2.imencode(".jpg", pipe.annotate(img, out["plates"]))
        if ok:
            payload["image_base64"] = base64.b64encode(buf).decode()
    return payload
