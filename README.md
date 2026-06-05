# OXM Vision

> Sistema POC de vigilancia inteligente multi-cámara para **NVIDIA Jetson Orin Nano** — detección de personas y objetos, re-identificación facial, clasificación y transcripción de audio, dashboard web en tiempo real.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jetson](https://img.shields.io/badge/NVIDIA-Jetson%20Orin%20Nano-76b900)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![JetPack](https://img.shields.io/badge/JetPack-R36.4-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)]()

---

## ¿Qué hace?

OXM Vision corre en una Jetson Orin Nano y procesa simultáneamente **dos streams de video** (CSI IMX219 + USB) más **audio en tiempo real** para producir un sistema integral de vigilancia con:

- 🎯 **Detección de objetos** — YOLOv8s en TensorRT FP16 (~21 FPS por cámara)
- 👤 **Re-identificación facial** — InsightFace `buffalo_l` en GPU (5 modelos ONNX)
- 🎙️ **Clasificación de audio** — YAMNet 521 clases (alarmas, gritos, fuego, vidrio, etc.)
- 📝 **Transcripción** — Faster-Whisper `base` en CPU int8 + detección de palabras clave (ayuda, fuego, robo…)
- 🔗 **Correlación voz + cara** — evento `speech_face_event` cuando alguien habla y se detecta su rostro en el mismo frame
- 🧠 **Detección de caídas** — YOLOv8s-Pose con análisis de keypoints
- 📊 **Dashboard FastAPI** — streams MJPEG, telemetría GPU, gráficas de actividad, búsqueda con chat LLM
- 📄 **Reporte PDF diario** — generado con fpdf2, enviado por Telegram a las 8am
- 💾 **Persistencia** — SQLite WAL + carpeta `audio_clips/` + `face_db.pkl` (FAISS)

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────────┐
│                     Jetson Orin Nano                          │
│                                                                │
│   CSI IMX219 ──▶ capture_csi ──┐                              │
│                                ├──▶ infer_loop (YOLO TRT)    │
│   USB Webcam ──▶ capture_usb ──┤      │                       │
│                                │      ▼                       │
│   USB Audio  ──▶ audio_loop ──▶│   reid_worker (InsightFace) │
│                                │      │                       │
│                                │      ▼                       │
│                                └──▶ event correlation        │
│                                       │                       │
│                                       ▼                       │
│   FastAPI dashboard ◀── compose_loop  events.db (SQLite)     │
│   :8080 (MJPEG, /api/*)               face_db.pkl (FAISS)    │
│                                       audio_clips/           │
└──────────────────────────────────────────────────────────────┘
```

**Stack:**
- **GPU compute**: TensorRT 10.x, onnxruntime-gpu 1.23, CUDA 12.8
- **Pipelines**: GStreamer (NVDEC para CSI), OpenCV 4.11 V4L2 (USB), sounddevice (audio)
- **Modelos**: YOLOv8s, InsightFace buffalo_l (det_10g · w600k_r50 · 2d106det · 1k3d68 · genderage), YAMNet, Whisper base
- **Web**: FastAPI + Uvicorn + threading.Condition para MJPEG sin busy-loop
- **Reportes**: fpdf2, Anthropic API (claude-haiku) opcional para chat/reportes

---

## Hardware requerido

| Componente | Modelo de referencia |
|---|---|
| Cómputo | NVIDIA Jetson Orin Nano Developer Kit (8 GB) |
| OS | Ubuntu 22.04 + JetPack R36.4 (CUDA 12.8, TensorRT 10.x) |
| Storage | NVMe ≥ 256 GB (recomendado, runtime hace I/O continuo) |
| Cámara CSI | IMX219 (Raspberry Pi v2) — `/dev/video*` automático |
| Cámara USB | UVC-compatible (YGTek, Logitech, etc.) |
| Audio | Webcam USB con mic, o tarjeta USB Audio |

> ℹ️ El proyecto está pensado para Jetson; puede portarse a x86 + CUDA dGPU eliminando las partes de `nvargus`/`nvarguscamerasrc`/GStreamer-NVDEC.

---

## Instalación

### 1. Pre-requisitos en la Jetson

```bash
# JetPack ya viene con Docker + nvidia-container-toolkit
sudo apt update && sudo apt install -y v4l-utils
# Verificar GPU + cámaras
nvidia-smi
v4l2-ctl --list-devices
```

### 2. Clonar el repo

```bash
git clone https://github.com/OXMtech/oxm-vision.git
cd oxm-vision
```

### 3. Configurar variables

```bash
cp .env.example .env
# Editar .env y rellenar:
#   ANTHROPIC_API_KEY=...   (opcional, para chat)
#   TELEGRAM_TOKEN=...      (opcional, para alertas)
#   TELEGRAM_CHAT_ID=...
```

### 4. Modelos YOLO

Compilar el engine TensorRT desde el `.pt` original (solo se hace una vez, ~5 minutos):

```bash
mkdir -p yolo && cd yolo
# Descargar pesos YOLOv8s
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt -O yolo26s.pt
# Compilar a TensorRT FP16 (dentro del container o con ultralytics en host)
yolo export model=yolo26s.pt format=engine half=True device=0
# También: yolov8s-pose.pt → yolov8s-pose.engine
```

> ⚠️ Los `.engine` son específicos del hardware donde se compilan. **No se pueden copiar entre máquinas.**

### 5. Estructura de carpetas runtime

```bash
mkdir -p data/audio_clips data/metadata_archive data/event_clips
mkdir -p models/insightface models/whisper_cache
touch data/unified_metadata.jsonl data/face_db.pkl
```

### 6. Build + arrancar

```bash
# CRÍTICO: reiniciar nvargus-daemon antes (la CSI camera falla si no)
sudo systemctl restart nvargus-daemon && sleep 3

docker compose up -d --build
docker compose logs -f
```

Abre el dashboard:

```
http://<IP_DE_LA_JETSON>:8080
```

---

## Uso

### Endpoints del dashboard

| Endpoint | Descripción |
|---|---|
| `/` | Dashboard web con streams MJPEG y estadísticas |
| `/stream/csi` `/stream/usb` | MJPEG por cámara (960×540, JPEG q80) |
| `/api/stats` | JSON con personas, audio, alertas |
| `/api/tegra` | Telemetría GPU/CPU/RAM/NVMe |
| `/api/health` | Latido de cámaras, modelos y audio |
| `/api/chat` | POST `{"q":"qué pasó ayer..."}` — respuesta natural con LLM |
| `/api/report/pdf` | Descarga PDF del día actual |
| `/api/clip/<id>` | Stream de un clip de audio guardado |

### Enrolar caras conocidas

```bash
docker exec -it oxm-vision-oxm-vision-1 python3 /app/enroll_live.py --name "Brayan" --cam csi
# Pide pose frontal y lateral. Guarda embeddings en face_db.pkl.
```

### Backup de la base de eventos (sin downtime)

```bash
sqlite3 data/events.db ".backup data/events.db.$(date +%Y%m%d).bak"
```

---

## Configuración fina

Variables clave en `src/main.py`:

| Variable | Default | Descripción |
|---|---|---|
| `STREAM_W, STREAM_H` | `960, 540` | Resolución del MJPEG al navegador |
| `INFER_W, INFER_H` | `640, 360` | Resolución de inferencia YOLO |
| `JPEG_QUALITY` | `80` | Calidad del stream MJPEG (0–100) |
| `THRESH` | `0.4` | Umbral de confianza YOLO |
| `REID_EVERY` | `60` | Frames entre cada pase de InsightFace |
| `POSE_EVERY` | `12` | Frames entre cada pase de pose / fall detection |
| `MIN_SPEECH_CHUNKS` | `5` | ~5s mínimos de speech para guardar clip |
| `KEYWORD_ALERTS` | lista | Palabras clave que disparan alerta Telegram |

---

## Lecciones aprendidas (cicatrices del POC)

### 🔧 MJPEG flashes negros
El keepalive del stream nunca debe emitir un JPEG vacío (`Content-Type: image/jpeg` con 0 bytes) — el navegador lo interpreta como imagen rota y pinta negro. **Solución**: re-emitir el último frame válido cuando no llega uno nuevo en 1s. Aplicado en `dashboard.py`.

### 🔧 onnxruntime-gpu en aarch64/Python3.12
No existe wheel oficial. Hay que compilar desde fuente (~3h) o aceptar CPU para InsightFace. Este repo usa onnxruntime-gpu cuando está disponible y cae a CPU si no.

### 🔧 nvargus-daemon hay que reiniciarlo antes de levantar el container
Si no, el pipeline CSI muere con `Argus context not initialized`. El `Dockerfile` y `docker-compose.yml` no lo arreglan — hay que hacerlo manualmente cada vez que se reinicia la Jetson.

### 🔧 `model.to('cuda')` con TensorRT engines
**No usar.** Los `.engine` ya están en GPU; llamar `.to('cuda')` rompe la inferencia silenciosamente. Solo aplica a `.pt`.

### 🔧 Whisper en CPU intencionalmente
Whisper en GPU compite con YOLO+InsightFace por memoria y compute. `device="cpu"` + `compute_type="int8"` da calidad equivalente en este caso de uso (transcripciones cortas de ~10s).

---

## Estructura del repo

```
oxm-vision/
├── src/
│   ├── main.py            # Pipeline principal (threads, captura, inferencia, audio)
│   ├── dashboard.py       # FastAPI + UI + endpoints
│   ├── face_reid.py       # InsightFace wrapper con FAISS
│   ├── pose.py            # YOLOv8s-Pose + lógica de fall detection
│   ├── db.py              # SQLite WAL helper
│   ├── enroll_live.py     # CLI para enrolar caras desde stream en vivo
│   └── event_recorder.py  # Buffer circular para guardar clips de eventos
├── scripts/
│   └── setup-jetson-nat.sh  # Compartir internet Mac → Jetson por Ethernet
├── docs/                    # (futuro) diagramas, arquitectura, troubleshooting
├── Dockerfile               # Base dustynv/pytorch:2.7-r36.4.0-cu128-24.04
├── docker-compose.yml       # Servicio único oxm-vision
├── .env.example             # Plantilla de variables de entorno
├── .gitignore
├── LICENSE                  # MIT
└── README.md                # Este archivo
```

---

## Roadmap

- [ ] Migración a **NVIDIA DGX Spark (GB10)** con captura RTSP multi-cámara (≥4 streams)
- [ ] Reemplazar Anthropic API externa por **VLM local** (Gemma 26B, Qwen36) para chat y reportes
- [ ] VAD Silero antes de Whisper (eliminar alucinaciones tipo "y y y y")
- [ ] Diarización real con pyannote/speaker-diarization-3.1
- [ ] Galería de rostros conocidos en el dashboard
- [ ] Zona de intrusión configurable (polígono)
- [ ] Deduplicación de personas entre CSI y USB

---

## Créditos

Desarrollado por **[OXMtech](https://www.oxmtech.com)** — Soluciones tecnológicas a medida.

POC inicial: abril 2026 — Jetson Orin Nano.
Evolución a DGX Spark + RTSP multi-cámara: en progreso.

### Modelos y librerías de terceros
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — AGPL-3.0
- [InsightFace](https://github.com/deepinsight/insightface) — MIT
- [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) — Apache-2.0
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) — MIT
- [FastAPI](https://fastapi.tiangolo.com/) — MIT

---

## Licencia

[MIT](LICENSE) © 2026 OXMtech S.A. de C.V.
