FROM dustynv/opencv:4.11.0-r36.4.0-cu128-24.04 AS opencv-src

FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=

# Deps sistema (GStreamer + Nvidia GStreamer plugins)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libportaudio2 portaudio19-dev libsndfile1 \
    libasound2-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libtesseract5 \
    libwebpdemux2 \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

# Instalar todo primero
RUN pip3 install --no-cache-dir "ultralytics>=8.3" lapx "numpy<2.0"
RUN pip3 install --no-cache-dir insightface onnx
# onnxruntime-gpu compilado desde fuente para aarch64/Python3.12/CUDA12.8
COPY onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl /tmp/
RUN pip3 install --no-cache-dir /tmp/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl && \
    rm /tmp/onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl && \
    pip3 install --no-cache-dir "numpy<2.0"
RUN pip3 install --no-cache-dir sounddevice pyalsaaudio openai fastapi "uvicorn[standard]"
RUN pip3 install --no-cache-dir faiss-cpu faster-whisper
RUN pip3 install --no-cache-dir ai-edge-litert || \
    pip3 install --no-cache-dir tflite-runtime || \
    echo "WARN: tflite not available, YAMNet disabled"
# Generador de reportes PDF (endpoint /api/report/pdf en dashboard.py)
RUN pip3 install --no-cache-dir fpdf2

# Copiar cv2 con GStreamer AL FINAL (sobreescribe opencv-headless de insightface)
COPY --from=opencv-src /opt/venv/lib/python3.12/site-packages/cv2 /opt/venv/lib/python3.12/site-packages/cv2

WORKDIR /app
COPY . /app

CMD ["python3", "main.py"]
