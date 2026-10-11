FROM python:3.11-slim

# Cai dat thu vien he thong can thiet cho OpenCV va PaddlePaddle C++ runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy file requirements truoc de tan dung Docker cache layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ma nguon va model da huan luyen
COPY src/ src/
COPY models/ models/
COPY app.py .

# Mo port cho ca API (8000) va Streamlit UI (8501)
EXPOSE 8000 8501

# Mac dinh khoi chay API don gian (Muc 1.B)
CMD ["python", "-m", "src", "serve", "--host", "0.0.0.0", "--port", "8000"]
