import os
# Fix: nvidia-container-runtime cambia NVIDIA_VISIBLE_DEVICES a "void" en PID1
# onnxruntime CUDA provider verifica esta variable; resetearla permite GPU
if os.environ.get("NVIDIA_VISIBLE_DEVICES") == "void":
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu/nvidia:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["PYTHONPATH"] = "/usr/lib/python3.10/dist-packages:" + os.environ.get("PYTHONPATH", "")

import cv2
import numpy as np
import threading
import queue
import time
import json
import wave
import subprocess
import shutil
import datetime
import csv
from scipy.signal import butter as _butter, sosfilt as _sosfilt, iirnotch as _iirnotch, lfilter as _lfilter
import http.client
import ssl
try:
    import alsaaudio as _alsaaudio
    ALSAAUDIO_OK = True
except ImportError:
    ALSAAUDIO_OK = False

try:
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        import tflite_runtime.interpreter as tflite
    YAMNET_AVAILABLE = True
except ImportError:
    YAMNET_AVAILABLE = False
    print("YAMNet no disponible (tflite no instalado)")

try:
    from faster_whisper import WhisperModel as _WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    print("WARN: faster-whisper no disponible, transcripción deshabilitada")

_whisper_model = None
_whisper_lock  = threading.Lock()

def _get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None and WHISPER_OK:
            try:
                # CPU int8: Whisper en CPU libera GPU para YOLO/InsightFace sin degradar calidad
                _whisper_model = _WhisperModel("base", device="cpu", compute_type="int8",
                                              cpu_threads=2, num_workers=1)
                print("Whisper base (CPU int8) cargado", flush=True)
            except Exception as e:
                print(f"Whisper error: {e}", flush=True)
    return _whisper_model
from ultralytics import YOLO
import sys
sys.path.insert(0, "/app")
from face_reid import FaceReID
from pose import PoseFall
import dashboard
import db

# ── Configuración ──────────────────────────────────────────────────────────────
MODEL_PATH   = '/yolo/yolo26s.engine'
POSE_MODEL   = '/yolo/yolov8s-pose.engine'
POSE_EVERY   = 12    # cada N frames por cámara
YAMNET_MODEL  = '/app/yamnet.tflite'
YAMNET_LABELS = '/app/yamnet_labels.txt'
DB_PATH              = '/app/events.db'
METADATA_ROTATE_MB   = 500
METADATA_ROTATE_CHECK_S = 3600
AUDIO_DIR      = '/app/audio_clips'
FACE_SNAPS_DIR = '/app/face_snaps'
AUDIO_MAX_B   = 10 * 1024**3
USB_IDX       = 1
AUDIO_CARD    = 0   # Webcam USB (hw:0,0) — pyalsaaudio cardindex
RATE          = 16000
CHUNK         = 15600
CLIP_CHUNKS   = 123   # ~2 minutos (123 × 0.975s ≈ 120s)
CST           = datetime.timezone(datetime.timedelta(hours=-6))
THRESH        = 0.4
META_FLUSH    = 20
REID_EVERY    = 60
JPEG_QUALITY  = 75
STREAM_W, STREAM_H = 640, 360   # resolución del stream al dashboard
INFER_W,  INFER_H  = 640, 360   # resolución de inferencia YOLO
SCALE_X = STREAM_W / INFER_W    # 1.0 — mismo tamaño, no hay resize extra
SCALE_Y = STREAM_H / INFER_H    # 1.0

COCO_ES = {
    'person':'persona','bicycle':'bicicleta','car':'auto','motorcycle':'moto',
    'airplane':'avión','bus':'autobús','train':'tren','truck':'camión',
    'boat':'bote','traffic light':'semáforo','fire hydrant':'hidrante',
    'stop sign':'alto','parking meter':'parquímetro','bench':'banca',
    'bird':'pájaro','cat':'gato','dog':'perro','horse':'caballo',
    'sheep':'oveja','cow':'vaca','elephant':'elefante','bear':'oso',
    'zebra':'cebra','giraffe':'jirafa','backpack':'mochila','umbrella':'paraguas',
    'handbag':'bolsa','tie':'corbata','suitcase':'maleta','frisbee':'frisbee',
    'skis':'esquís','snowboard':'snowboard','sports ball':'pelota','kite':'papalote',
    'baseball bat':'bat','baseball glove':'guante','skateboard':'patineta',
    'surfboard':'tabla surf','tennis racket':'raqueta','bottle':'botella',
    'wine glass':'copa','cup':'taza','fork':'tenedor','knife':'cuchillo',
    'spoon':'cuchara','bowl':'tazón','banana':'plátano','apple':'manzana',
    'sandwich':'sándwich','orange':'naranja','broccoli':'brócoli','carrot':'zanahoria',
    'hot dog':'hot dog','pizza':'pizza','donut':'dona','cake':'pastel',
    'chair':'silla','couch':'sofá','potted plant':'planta','bed':'cama',
    'dining table':'mesa','toilet':'baño','tv':'televisión','laptop':'laptop',
    'mouse':'ratón','remote':'control','keyboard':'teclado','cell phone':'celular',
    'microwave':'microondas','oven':'horno','toaster':'tostadora','sink':'fregadero',
    'refrigerator':'refrigerador','book':'libro','clock':'reloj','vase':'florero',
    'scissors':'tijeras','teddy bear':'oso de peluche','hair drier':'secadora',
    'toothbrush':'cepillo',
}

KEYWORD_ALERTS = [
    # Emergencia
    'ayuda', 'auxilio', 'socorro', 'emergencia', 'help',
    # Fuego
    'fuego', 'incendio', 'humo', 'fire', 'quemando',
    # Violencia / seguridad
    'robo', 'ladrón', 'asalto', 'ataque', 'golpe', 'disparo', 'arma',
    # Médico
    'ambulancia', 'médico', 'doctor', 'accidente', 'herido', 'sangre',
    # Intrusión
    'intruso', 'alarma', 'policía',
]

SPEECH_CLASSES = ['Speech', 'Conversation', 'Narration', 'Child speech']

# ── Telegram ───────────────────────────────────────────────────────────────────
TG_TOKEN     = os.environ.get('TELEGRAM_TOKEN', '')
TG_CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '')
REPORT_HOUR  = int(os.environ.get('REPORT_HOUR_CST', '8'))  # 8am CST por defecto
_tg_lock    = threading.Lock()
_tg_last    = {'audio': 0.0, 'keyword': 0.0, 'fall': 0.0}
TG_COOLDOWN = {'audio': 60.0, 'keyword': 20.0, 'fall': 30.0}  # segundos entre alertas del mismo tipo

def _tg_send(caption: str, jpg_bytes: bytes | None = None, parse_mode: str | None = None):
    """Envía foto o mensaje a Telegram via http.client (stdlib)."""
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    try:
        import urllib.parse
        ctx  = ssl.create_default_context()
        conn = http.client.HTTPSConnection('api.telegram.org', context=ctx, timeout=15)
        if jpg_bytes:
            bnd  = b'TGBnd'
            body = (
                b'--' + bnd + b'\r\n'
                b'Content-Disposition: form-data; name="chat_id"\r\n\r\n'
                + TG_CHAT_ID.encode() + b'\r\n'
                b'--' + bnd + b'\r\n'
                b'Content-Disposition: form-data; name="caption"\r\n\r\n'
                + caption.encode('utf-8') + b'\r\n'
                b'--' + bnd + b'\r\n'
                b'Content-Disposition: form-data; name="photo"; filename="alert.jpg"\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + jpg_bytes + b'\r\n'
                b'--' + bnd + b'--\r\n'
            )
            conn.request('POST', f'/bot{TG_TOKEN}/sendPhoto', body=body,
                         headers={'Content-Type': 'multipart/form-data; boundary=TGBnd',
                                  'Content-Length': str(len(body))})
        else:
            params = {'chat_id': TG_CHAT_ID, 'text': caption}
            if parse_mode:
                params['parse_mode'] = parse_mode
            body = urllib.parse.urlencode(params).encode()
            conn.request('POST', f'/bot{TG_TOKEN}/sendMessage', body=body,
                         headers={'Content-Type': 'application/x-www-form-urlencoded',
                                  'Content-Length': str(len(body))})
        resp = conn.getresponse()
        conn.close()
        print(f"[TELEGRAM] Enviado ({resp.status}): {caption[:60]}", flush=True)
    except Exception as e:
        print(f"[TELEGRAM] Error: {e}", flush=True)

def send_telegram(alert_type: str, caption: str, frame=None):
    """Envía alerta con cooldown por tipo. Siempre en thread propio."""
    now = time.time()
    with _tg_lock:
        if now - _tg_last.get(alert_type, 0) < TG_COOLDOWN.get(alert_type, 30):
            return
        _tg_last[alert_type] = now
    jpg_bytes = None
    if frame is not None:
        try:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_bytes = buf.tobytes()
        except Exception:
            pass
    threading.Thread(target=_tg_send, args=(caption, jpg_bytes), daemon=True).start()
ALERT_CLASSES  = ['Smoke detector', 'Fire alarm', 'Alarm', 'Smoke', 'Fire',
                  'Screaming', 'Explosion', 'Glass']
BBOX_COLORS = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
               (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
)

os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Modelos ────────────────────────────────────────────────────────────────────
# Calentar contexto CUDA antes de InsightFace — en PID1 CUDA está frío y
# onnxruntime CUDAExecutionProvider falla silenciosamente sin esto.
try:
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.init()
        _ = _torch.zeros(1, device='cuda')
        del _
        print("CUDA context inicializado (torch)", flush=True)
except Exception as _e:
    print(f"WARN cuda warmup: {_e}", flush=True)

print("Cargando FaceReID...")
reid = FaceReID()
dashboard.set_reid_up()

print("Cargando YOLO...")
models = {
    'csi': YOLO(MODEL_PATH, task="detect"),
    'usb': YOLO(MODEL_PATH, task="detect"),
}
labels = models['csi'].names
dashboard.set_yolo_up()

print("Cargando YOLO-Pose...")
try:
    pose = PoseFall(POSE_MODEL)
    print("Pose OK", flush=True)
except Exception as _e:
    pose = None
    print(f"WARN pose: {_e}", flush=True)

if YAMNET_AVAILABLE:
    print("Cargando YAMNet...")
    interp = tflite.Interpreter(model_path=YAMNET_MODEL)
    interp.allocate_tensors()
    yamnet_inp = interp.get_input_details()[0]
    yamnet_out = interp.get_output_details()[0]
    yamnet_labels = []
    with open(YAMNET_LABELS) as f:
        for row in csv.DictReader(f):
            yamnet_labels.append(row['display_name'])
else:
    interp = yamnet_inp = yamnet_out = yamnet_labels = None


# ── Estado compartido ──────────────────────────────────────────────────────────
frames       = {'csi': None, 'usb': None}
results_out  = {'csi': (None, 0), 'usb': (None, 0)}
audio_state  = {'label': '', 'conf': 0.0, 'alert': False, 'speech': False}
person_count = {'csi': 0, 'usb': 0}
lock         = threading.Lock()
results_lock = threading.Lock()
running      = True

# ── Metadata (SQLite) ──────────────────────────────────────────────────────────
db.init(DB_PATH)
os.makedirs(FACE_SNAPS_DIR, exist_ok=True)

def write_meta(record):
    db.write_event(record)

def send_person_photo(uuid_prefix: str) -> str:
    """
    Busca el crop más reciente de la persona cuyo UUID empieza con uuid_prefix
    y lo manda por Telegram. Devuelve mensaje de resultado.
    """
    import glob as _glob
    prefix = uuid_prefix.lower().replace('-', '')[:8]
    candidates = _glob.glob(f"{FACE_SNAPS_DIR}/*.jpg")
    match = None
    for p in candidates:
        fname = os.path.basename(p).replace('-', '')
        if fname.lower().startswith(prefix):
            match = p
            break
    if match is None:
        return f"No hay foto guardada para '{uuid_prefix[:8]}' — la persona no ha sido identificada aún o el snap fue sobreescrito."
    try:
        with open(match, 'rb') as f:
            jpg = f.read()
        uid_short = os.path.basename(match).replace('.jpg', '')[:8]
        _tg_send(f"👤 Persona ID: {uid_short}\nÚltima detección guardada", jpg)
        return f"Foto de {uid_short} enviada a Telegram."
    except Exception as e:
        return f"Error enviando foto: {e}"

def _estimate_speakers(audio, rate, win_s=2.0, threshold=0.28):
    """Estima hablantes por cambios en perfil espectral (numpy puro)."""
    win  = int(win_s * rate)
    hop  = win // 2
    BANDS = 16
    window_fn = np.hanning(win)
    features = []
    for i in range(0, len(audio) - win, hop):
        seg = audio[i:i+win]
        if np.sqrt(np.mean(seg**2)) < 0.005:   # silencio → ignorar
            continue
        fft_mag = np.abs(np.fft.rfft(seg * window_fn))
        half    = len(fft_mag) // 2 + 1
        bands   = np.array_split(fft_mag[:half], BANDS)
        feat    = np.array([np.log(np.mean(b**2) + 1e-8) for b in bands])
        features.append(feat)
    if len(features) < 2:
        return 1
    feats = np.array(features)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    # Clustering aglomerativo: cada segmento vs centroides conocidos
    centroids = [feats[0]]
    for feat in feats[1:]:
        sims = [np.dot(feat, c) / (np.linalg.norm(feat) * np.linalg.norm(c) + 1e-8)
                for c in centroids]
        if max(sims) < (1.0 - threshold):
            centroids.append(feat)
            if len(centroids) >= 8:
                break
    return len(centroids)

def diarize_clip(path_opus, meta_ts):
    try:
        tmp_wav = path_opus.replace('.opus', '_diar.wav')
        ret = subprocess.run(
            ['ffmpeg', '-y', '-i', path_opus, '-ar', str(RATE), '-ac', '1', tmp_wav],
            capture_output=True
        )
        if ret.returncode != 0:
            return
        with wave.open(tmp_wav, 'r') as wf:
            raw   = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        os.remove(tmp_wav)
        n = _estimate_speakers(audio, RATE)
        with open(path_opus.replace('.opus', '.spk'), 'w') as f:
            f.write(str(n))
        write_meta({"timestamp": meta_ts, "type": "speaker_count",
                    "file": path_opus, "n_speakers": n})
        print(f"[DIARIZE] {n} hablante(s) en {os.path.basename(path_opus)}", flush=True)
    except Exception as e:
        print(f"[DIARIZE] Error: {e}", flush=True)

def _correlate_audio_visual(path_opus, meta_ts, duration_s, text):
    """
    Consulta el DB por detecciones y caras durante la ventana del clip
    y escribe un evento speech_visual_event con el contexto combinado.
    Requiere SQLite — sin esto la correlación no tiene rango de tiempo.
    """
    try:
        from collections import Counter as _C
        end_dt   = datetime.datetime.fromisoformat(meta_ts)
        start_dt = end_dt - datetime.timedelta(seconds=duration_s)
        start_s  = start_dt.isoformat()

        visual = db.events_in_range(start_s, meta_ts,
                                    types=['detection', 'face_reid'])
        if not visual:
            return

        persons_seen  = {}   # uuid → {'new': bool, 'cams': set}
        obj_counts    = _C()
        cams_active   = set()

        for ev in visual:
            t = ev.get('type')
            if t == 'face_reid':
                uid = ev.get('person_uuid', '')
                if uid not in persons_seen:
                    persons_seen[uid] = {'new': ev.get('new', False), 'cams': set()}
                persons_seen[uid]['cams'].add(ev.get('cam', ''))
                cams_active.add(ev.get('cam', ''))
            elif t == 'detection':
                obj_counts[ev.get('class', '')] += 1
                cams_active.add(ev.get('cam', ''))

        db.write_event({
            "timestamp": meta_ts,
            "type": "speech_visual_event",
            "file": path_opus,
            "duration_s": round(duration_s, 1),
            "text_snippet": text[:300],
            "window_start": start_s,
            "window_end": meta_ts,
            "n_unique_persons": len(persons_seen),
            "persons": [
                {"uuid": uid, "new": v['new'], "cams": list(v['cams'])}
                for uid, v in persons_seen.items()
            ],
            "top_objects": dict(obj_counts.most_common(6)),
            "cams": list(cams_active),
        })
        print(f"[CORRELATE] {os.path.basename(path_opus)}: "
              f"{len(persons_seen)} persona(s), objetos={dict(obj_counts.most_common(3))}",
              flush=True)
    except Exception as e:
        print(f"[CORRELATE] error: {e}", flush=True)


def transcribe_clip(path_opus, meta_ts, duration_s=120.0):
    if not WHISPER_OK:
        return
    try:
        wm = _get_whisper()
        segments, info = wm.transcribe(
            path_opus,
            language="es",
            beam_size=3,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt="Transcripción de reunión de negocios en español:",
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            print(f"[TRANSCRIPT] {os.path.basename(path_opus)}: sin habla detectada", flush=True)
            return
        # Filtrar alucinaciones de Whisper (ej: "y y y y y y y")
        tokens = text.split()
        if tokens:
            single_char_ratio = sum(1 for t in tokens if len(t.strip("¿?.,!¡")) <= 1) / len(tokens)
            if single_char_ratio > 0.6:
                print(f"[TRANSCRIPT] {os.path.basename(path_opus)}: descartado (alucinación Whisper: '{text[:40]}')", flush=True)
                return
        # Post-procesar con Nemotron: limpiar muletillas, corregir texto
        try:
            from openai import OpenAI as _OAI
            _nemotron_url = os.environ.get('NEMOTRON_URL', 'http://localhost:8003/v1')
            _nc = _OAI(base_url=_nemotron_url, api_key='none', timeout=30.0)
            _nr = _nc.chat.completions.create(
                model='nemotron-omni', max_tokens=600,
                messages=[{
                    'role': 'user',
                    'content': (
                        'Corrige y limpia esta transcripción de audio en español. '
                        'Elimina muletillas ("eh", "um", "este", repeticiones), corrige '
                        'errores de dictado y puntúa correctamente. '
                        'Devuelve SOLO el texto corregido, sin explicaciones.\n\n'
                        + text
                    )
                }]
            )
            text = _nr.choices[0].message.content.strip() or text
            print(f"[TRANSCRIPT] Nemotron limpió transcripción", flush=True)
        except Exception as _ne:
            print(f"[TRANSCRIPT] Nemotron no disponible, usando Whisper raw: {_ne}", flush=True)

        txt_path = path_opus.replace('.opus', '.txt')
        with open(txt_path, 'w') as f:
            f.write(text)
        write_meta({"timestamp": meta_ts, "type": "audio_transcript",
                    "file": path_opus, "text": text,
                    "language": getattr(info, 'language', 'es')})
        print(f"[TRANSCRIPT] {os.path.basename(path_opus)}: {text[:80]}", flush=True)

        # Diarización (en el mismo thread, después de transcripción)
        diarize_clip(path_opus, meta_ts)

        # Detectar palabras clave de alerta
        text_lower = text.lower()
        for kw in KEYWORD_ALERTS:
            if kw in text_lower:
                write_meta({"timestamp": meta_ts, "type": "keyword_alert",
                            "keyword": kw, "file": path_opus, "text": text})
                dashboard.push_alert({
                    "timestamp": meta_ts,
                    "label": f"ALERTA: '{kw}' detectado en audio",
                    "confidence": 1.0
                })
                print(f"[KEYWORD] '{kw}' detectado en: {text[:60]}", flush=True)
                with results_lock:
                    f_csi, _ = results_out['csi']
                    f_usb, _ = results_out['usb']
                _frame = f_csi if f_csi is not None else f_usb
                _cap = (f"⚠️ PALABRA CLAVE DETECTADA\n"
                        f"Palabra: '{kw}'\n"
                        f"Transcripción: {text[:200]}\n"
                        f"{meta_ts[:19]} CST")
                send_telegram('keyword', _cap, _frame)
                break  # una alerta por clip aunque haya varias keywords

        # Correlación audio-visual: qué se vio mientras se habló
        _correlate_audio_visual(path_opus, meta_ts, duration_s, text)
    except Exception as e:
        print(f"[TRANSCRIPT] Error: {e}", flush=True)

# ── Captura CSI ────────────────────────────────────────────────────────────────
def capture_csi():
    """
    Runs the GStreamer/Argus CSI pipeline in an isolated subprocess (csi_worker.py).
    If GStreamer native threads crash with SIGSEGV, only the worker dies — this
    function detects the exit, kills/restarts nvargus-daemon, and relaunches.
    Frames are exchanged via a mmap'd temp file; signals via a dedicated OS pipe
    (not stdout, because GStreamer writes random bytes there).
    """
    import queue as _queue
    import mmap
    import tempfile
    import numpy as np

    _CSI_H, _CSI_W = 720, 1280
    _FRAME_BYTES = _CSI_H * _CSI_W * 3

    # Create persistent temp file for frame exchange
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix='csi_frame_', suffix='.bin')
    tmp.write(b'\x00' * _FRAME_BYTES)
    tmp.flush()
    frame_path = tmp.name
    tmp.close()

    frame_file = open(frame_path, 'r+b')
    mm = mmap.mmap(frame_file.fileno(), _FRAME_BYTES)
    buf_view = np.frombuffer(mm, dtype=np.uint8).reshape((_CSI_H, _CSI_W, 3))

    def _reader(read_fd, q):
        """Read 1-byte signals from the dedicated pipe into a queue."""
        try:
            with os.fdopen(read_fd, 'rb', buffering=0) as pipe:
                while True:
                    b = pipe.read(1)
                    if not b:
                        q.put(b'EOF')
                        return
                    q.put(b)
        except Exception:
            q.put(b'EOF')

    try:
        while running:
            print("CSI: lanzando worker subprocess...", flush=True)
            sig_q = _queue.Queue()

            # Dedicated OS pipe for signals (avoids GStreamer stdout pollution)
            r_fd, w_fd = os.pipe()
            proc = subprocess.Popen(
                ['python3', '/app/csi_worker.py', frame_path],
                env={**os.environ, 'CSI_SIGNAL_FD': str(w_fd)},
                pass_fds=(w_fd,),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.close(w_fd)  # parent closes write end; worker holds it
            threading.Thread(target=_reader, args=(r_fd, sig_q), daemon=True).start()

            # Wait for Ready signal (up to 15s)
            try:
                sig = sig_q.get(timeout=15)
            except _queue.Empty:
                sig = b'TIMEOUT'

            if sig != b'R':
                print(f"CSI: worker no abrió pipeline ({sig})", flush=True)
                proc.kill()
                proc.wait(3)
                dashboard.set_cam_status('csi', False)
                if not _nvargus_restarted:
                    # Primer fallo → reiniciar nvargus-daemon y reintentar
                    print("CSI: reiniciando nvargus-daemon...", flush=True)
                    try:
                        subprocess.run(
                            ['nsenter', '-t', '1', '-m', '--',
                             'systemctl', 'kill', '-s', 'KILL', 'nvargus-daemon'],
                            timeout=3, capture_output=True
                        )
                        time.sleep(1)
                        subprocess.run(
                            ['nsenter', '-t', '1', '-m', '--',
                             'systemctl', 'start', 'nvargus-daemon'],
                            timeout=8, capture_output=True
                        )
                        print("CSI: nvargus-daemon reiniciado", flush=True)
                    except Exception as _e:
                        print(f"CSI: no pudo reiniciar nvargus-daemon: {_e}", flush=True)
                    _nvargus_restarted = True
                    time.sleep(4)
                else:
                    time.sleep(3)
                continue

            print("CSI: OK", flush=True)
            dashboard.set_cam_status('csi', True)
            _csi_reconnecting.clear()

            # Process frame signals until worker dies or frame loss
            while running:
                try:
                    sig = sig_q.get(timeout=1.0)
                except _queue.Empty:
                    if proc.poll() is not None:
                        sig = b'EOF'
                    else:
                        continue

                if sig == b'F':
                    frame_copy = buf_view.copy()
                    with lock:
                        frames['csi'] = frame_copy
                elif sig in (b'E', b'EOF'):
                    ec = proc.poll()
                    print(f"CSI: worker desconectado (sig={sig}, exit={ec}), reconectando...", flush=True)
                    break

            # Worker died — clean up
            _csi_reconnecting.set()
            frames['csi'] = None
            dashboard.set_cam_status('csi', False)

            if proc.poll() is None:
                proc.kill()
            proc.wait(5)

            # Estrategia de reconexión en 2 etapas:
            # 1. Intentar reabrir el pipeline directamente (rápido, ~2s).
            #    Funciona cuando Argus daemon sigue vivo pero solo cayó el socket GStreamer.
            # 2. Si falla, reiniciar nvargus-daemon (costoso, ~8s).
            #    Necesario solo cuando el daemon mismo murió o está corrupto.
            _nvargus_restarted = False
            time.sleep(2)  # pausa mínima antes del intento rápido
    finally:
        mm.close()
        frame_file.close()
        os.unlink(frame_path)

# ── Captura USB ────────────────────────────────────────────────────────────────
def capture_usb():
    fail_count = 0
    while running:
        print("USB: abriendo cámara...")
        cap = cv2.VideoCapture(USB_IDX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            print("USB: ERROR no abre, reintentando en 5s")
            dashboard.set_cam_status('usb', False)
            time.sleep(5)
            continue
        print("USB: OK")
        dashboard.set_cam_status('usb', True)
        fail_count = 0
        while running:
            ret, frame = cap.read()
            if ret and frame is not None:
                fail_count = 0
                with lock:
                    frames['usb'] = frame
            else:
                fail_count += 1
                if fail_count > 10:
                    print("USB: pérdida de frames, reconectando...")
                    break
        cap.release()
        frames['usb'] = None
        dashboard.set_cam_status('usb', False)
        time.sleep(3)

# ── Audio YAMNet ───────────────────────────────────────────────────────────────
MIN_SPEECH_CHUNKS = 5   # ~5s de speech mínimo para guardar el clip
recording_buffer = []
speech_chunks_in_buffer = 0

# ── Micrófono mute toggle ─────────────────────────────────────────────────────
_mic_muted = False
def set_mic_muted(muted: bool):
    global _mic_muted
    _mic_muted = muted
    dashboard.set_audio_muted(muted)
    print(f"[AUDIO] Micrófono {'silenciado' if muted else 'activado'}", flush=True)

# ── Parámetros de calidad de audio ───────────────────────────────────────────
_NOISE_GATE_RMS  = 0.004   # descarta silencio absoluto — bajo para no cortar voz lejana
_GAIN_TARGET_RMS = 0.12    # más alto para compensar micrófono de webcam lejano
_PREEMPH_COEF    = 0.97    # pre-énfasis: realza frecuencias altas de voz
_LIMITER_THRESH  = 0.85    # compresor suave: aplana picos por encima de este nivel

# Filtro bandpass 100Hz–7kHz — elimina 60Hz (red eléctrica) y ruido alto
_BP_SOS = _butter(4, [100 / (RATE / 2), 7000 / (RATE / 2)], btype='band', output='sos')
# Notch 120Hz y 180Hz — armónicos de 60Hz que pasan el bandpass
_N120_B, _N120_A = _iirnotch(120, Q=30, fs=RATE)
_N180_B, _N180_A = _iirnotch(180, Q=30, fs=RATE)

def _process_audio(raw_int16: bytes) -> np.ndarray:
    """S16LE → float32: bandpass 100-7kHz, compresor suave, pre-énfasis, normalización."""
    audio = np.frombuffer(raw_int16, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < CHUNK:
        audio = np.pad(audio, (0, CHUNK - len(audio)))
    else:
        audio = audio[:CHUNK]
    # Bandpass: elimina 60Hz fundamental y ruido >7kHz
    audio = _sosfilt(_BP_SOS, audio)
    # Notch 120Hz y 180Hz: armónicos de zumbido eléctrico (60Hz red México)
    audio = _lfilter(_N120_B, _N120_A, audio)
    audio = _lfilter(_N180_B, _N180_A, audio)
    # Compresor suave: aplana picos fuertes sin distorsionar
    mask = np.abs(audio) > _LIMITER_THRESH
    audio[mask] = np.sign(audio[mask]) * (_LIMITER_THRESH + np.tanh(np.abs(audio[mask]) - _LIMITER_THRESH) * (1 - _LIMITER_THRESH))
    # Pre-énfasis: realza consonantes y frecuencias altas de voz
    audio = np.append(audio[0], audio[1:] - _PREEMPH_COEF * audio[:-1])
    # Normalización: gain alto para compensar micrófono lejano
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms > _NOISE_GATE_RMS:
        gain = min(_GAIN_TARGET_RMS / rms, 15.0)
        audio = audio * gain
    return audio.astype(np.float32)

def audio_loop():
    if not YAMNET_AVAILABLE:
        print("Audio loop deshabilitado (sin tflite)")
        dashboard.set_audio_status(False, device='sin tflite')
        return
    if not ALSAAUDIO_OK:
        print("[AUDIO] pyalsaaudio no disponible, loop deshabilitado")
        dashboard.set_audio_status(False, device='sin alsaaudio')
        return
    global recording_buffer, speech_chunks_in_buffer
    _dev_name = f'Webcam USB (hw:{AUDIO_CARD},0)'
    _audio_ok = False
    _pcm = None
    while running:
        # Mute: pausa el loop sin cerrar el device
        if _mic_muted:
            dashboard.set_audio_status(True, label='silenciado', device=_dev_name)
            time.sleep(0.5)
            continue
        try:
            # Abrir/reabrir PCM si es necesario
            if _pcm is None:
                _pcm = _alsaaudio.PCM(
                    _alsaaudio.PCM_CAPTURE,
                    _alsaaudio.PCM_NORMAL,
                    cardindex=AUDIO_CARD,
                    channels=1,
                    rate=RATE,
                    format=_alsaaudio.PCM_FORMAT_S16_LE,
                    periodsize=CHUNK,
                )
                _audio_ok = False
                print(f"[AUDIO] PCM abierto: {_dev_name}", flush=True)
            length, raw = _pcm.read()
            if length <= 0:
                time.sleep(0.01)
                continue
            if not _audio_ok:
                dashboard.set_audio_status(True, device=_dev_name)
                _audio_ok = True
            # Procesar: pre-énfasis + normalización
            audio = _process_audio(raw)
            # Noise gate — skip silencio absoluto
            raw_rms = float(np.sqrt(np.mean(np.frombuffer(raw, dtype=np.int16).astype(np.float32)**2))) / 32768.0
            if raw_rms < _NOISE_GATE_RMS:
                dashboard.set_audio_status(True, label='Silence', device=_dev_name)
                continue
            interp.set_tensor(yamnet_inp['index'], audio.reshape(yamnet_inp['shape']))
            interp.invoke()
            scores = interp.get_tensor(yamnet_out['index'])[0]
            top5   = np.argsort(scores)[-5:][::-1]
            label  = yamnet_labels[top5[0]]
            conf   = float(scores[top5[0]])
            is_alert  = any(a.lower() in label.lower() for a in ALERT_CLASSES)
            is_speech = any(s.lower() in label.lower() for s in SPEECH_CLASSES)
            now = datetime.datetime.now(CST)
            if is_alert:
                dashboard.push_alert({
                    "timestamp": now.isoformat(),
                    "label": label, "confidence": round(conf, 3)
                })
                with results_lock:
                    f_csi, _ = results_out['csi']
                    f_usb, _ = results_out['usb']
                _frame = f_csi if f_csi is not None else f_usb
                _cap = (f"🚨 ALERTA SONORA\n"
                        f"Tipo: {label} ({conf:.0%})\n"
                        f"{now.strftime('%Y-%m-%d %H:%M:%S')} CST")
                send_telegram('audio', _cap, _frame)
            with lock:
                audio_state.update(label=label, conf=conf,
                                   alert=is_alert, speech=is_speech)
            dashboard.set_audio_status(True, label=label, device=_dev_name)

            # Grabación continua — guarda cada ~2 minutos solo si hubo speech
            recording_buffer.append(audio)
            if is_speech:
                speech_chunks_in_buffer += 1
            if len(recording_buffer) >= CLIP_CHUNKS:
                has_speech = speech_chunks_in_buffer >= MIN_SPEECH_CHUNKS
                ts       = now.strftime('%Y%m%d_%H%M%S')
                path_wav = f"{AUDIO_DIR}/conv_{ts}.wav"
                path     = f"{AUDIO_DIR}/conv_{ts}.opus"
                data = np.concatenate(recording_buffer)
                recording_buffer = []
                speech_chunks_in_buffer = 0
                if not has_speech:
                    print(f"[AUDIO] Clip descartado — sin speech suficiente en los últimos ~2min", flush=True)
                    continue
                with wave.open(path_wav, 'w') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2)
                    wf.setframerate(RATE)
                    wf.writeframes((data * 32767).astype(np.int16).tobytes())
                subprocess.run(["ffmpeg","-y","-i",path_wav,
                                "-c:a","libopus","-b:a","24k",path],
                               capture_output=True)
                os.remove(path_wav)
                total = sum(os.path.getsize(os.path.join(AUDIO_DIR, f))
                            for f in os.listdir(AUDIO_DIR))
                if total > AUDIO_MAX_B:
                    oldest = sorted(os.listdir(AUDIO_DIR))[0]
                    os.remove(os.path.join(AUDIO_DIR, oldest))
                clip_ts = now.isoformat()
                write_meta({"timestamp": clip_ts,
                            "type": "audio_clip", "file": path,
                            "format": "opus", "duration_s": len(data)/RATE})
                threading.Thread(target=transcribe_clip,
                                 args=(path, clip_ts, len(data)/RATE), daemon=True).start()

            write_meta({"timestamp": now.isoformat(),
                        "type": "audio", "label": label,
                        "confidence": round(conf, 3),
                        "alert": is_alert, "speech": is_speech})
        except Exception as e:
            print(f"[AUDIO] Error: {e}", flush=True)
            dashboard.set_audio_status(False, device=_dev_name)
            _audio_ok = False
            if _pcm is not None:
                try:
                    _pcm.close()
                except Exception:
                    pass
                _pcm = None
            time.sleep(2)

# ── REID worker (thread independiente por cámara) ─────────────────────────────
# Cuando CSI reconecta, Argus reinicializa el ISP en GPU. Si InsightFace corre
# al mismo tiempo los CUDA streams colisionan y provocan segfault. Este Event
# pausa el GPU inference en reid_worker durante la ventana de reconexión.
_csi_reconnecting = threading.Event()

reid_queues = {'csi': queue.Queue(maxsize=2), 'usb': queue.Queue(maxsize=2)}

def reid_worker(cam_id):
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while running:
        try:
            frame, ts, persons = reid_queues[cam_id].get(timeout=1.0)
        except queue.Empty:
            continue
        # Pausar reid de CSI mientras reconecta (evita procesar frames stale del pipeline muerto).
        # USB no se pausa — sigue funcionando independientemente.
        if cam_id == 'csi' and _csi_reconnecting.is_set():
            time.sleep(0.5)
            continue
        try:
            # Serializar con gpu_infer_lock si InsightFace corre en GPU (TRT/CUDA)
            # para evitar conflictos de CUDA streams con Pose inference
            if reid.on_gpu:
                with gpu_infer_lock:
                    faces = reid.identify(frame)
            else:
                faces = reid.identify(frame)
            if not faces:
                print(f"[REID] {cam_id}: sin caras detectadas", flush=True)
                continue

            with lock:
                is_speech = audio_state['speech']
                audio_label = audio_state['label']
                audio_conf  = audio_state['conf']

            batch = []
            for face in faces:
                x1, y1, x2, y2 = face["bbox"]
                pid   = face["person_uuid"]
                color = (0, 255, 0) if not face["new"] else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID:{pid[:8]}', (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                # Guardar crop del rostro — se sobreescribe con el más reciente
                try:
                    pad = 20
                    h, w = frame.shape[:2]
                    fx1 = max(0, x1 - pad); fy1 = max(0, y1 - pad)
                    fx2 = min(w, x2 + pad); fy2 = min(h, y2 + pad)
                    crop = frame[fy1:fy2, fx1:fx2]
                    if crop.size > 0:
                        cv2.imwrite(f"{FACE_SNAPS_DIR}/{pid}.jpg", crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                except Exception:
                    pass
                batch.append({"timestamp": ts, "type": "face_reid",
                              "cam": cam_id, "person_uuid": pid,
                              "new": face["new"], "similarity": face["similarity"],
                              "bbox": face["bbox"]})

                # Evento correlacionado: cara + voz al mismo tiempo
                if is_speech and persons > 0:
                    event = {"timestamp": ts, "type": "speech_face_event",
                             "cam": cam_id, "person_uuid": pid,
                             "audio_label": audio_label,
                             "audio_conf": round(audio_conf, 3),
                             "new_face": face["new"]}
                    batch.append(event)
                    dashboard.push_alert({
                        "timestamp": ts,
                        "label": f"Voz+Cara: {pid[:8]} ({audio_label})",
                        "confidence": round(audio_conf, 3)
                    })
                    print(f"[EVENT] {cam_id}: voz+cara pid={pid[:8]} audio={audio_label}", flush=True)

            db.write_events(batch)

            # Actualizar frame con REID overlay en dashboard
            _, jpeg = cv2.imencode('.jpg', frame, enc_params)
            dashboard.update_frame(cam_id, jpeg.tobytes())

        except Exception as e:
            print(f"[REID] {cam_id}: ERROR {e}", flush=True)

# ── Inferencia YOLO en thread por cámara ───────────────────────────────────────
frame_cnt = {'csi': 0, 'usb': 0}
infer_locks = {'csi': threading.Lock(), 'usb': threading.Lock()}
_DET_LOG_INTERVAL = 30.0  # segundos mínimos entre logs DB de la misma detección
_det_last_ts: dict = {}   # (cam_id, track_id|class) -> time.time()
# Serializa Pose (YOLO-pose TRT) e InsightFace (onnxruntime-gpu) — ambos en GPU device 0.
# Sin esto, los dos pueden ejecutar simultáneamente desde threads distintos y
# generar conflictos de CUDA streams que terminan en segfault.
gpu_infer_lock = threading.Lock()

fps_timer = {}
def infer_loop(cam_id):
    fps_t = {"t": time.time(), "cnt": 0}
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while running:
        with lock:
            frame = frames[cam_id]
        if frame is None:
            time.sleep(0.01)
            continue

        # Un solo resize: INFER == STREAM, misma resolución
        infer_frame  = cv2.resize(frame, (INFER_W, INFER_H))
        stream_frame = infer_frame.copy()
        persons = 0
        ts = datetime.datetime.now(CST).isoformat()
        batch = []

        with infer_locks[cam_id]:
            results = models[cam_id].track(infer_frame, verbose=False, persist=True, device=0,
                                           tracker="bytetrack.yaml")

        for box in results[0].boxes:
            conf = box.conf.item()
            if conf < THRESH:
                continue
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            cls      = int(box.cls.item())
            name     = labels[cls]
            color    = BBOX_COLORS[cls % 10]
            track_id = int(box.id.item()) if box.id is not None else -1
            if name == 'person':
                persons += 1
            # Escalar bbox de resolución de inferencia a resolución de stream
            sx1 = int(xmin * SCALE_X); sy1 = int(ymin * SCALE_Y)
            sx2 = int(xmax * SCALE_X); sy2 = int(ymax * SCALE_Y)
            cv2.rectangle(stream_frame, (sx1, sy1), (sx2, sy2), color, 2)
            id_str = f"#{track_id} " if track_id >= 0 else ""
            name_es = COCO_ES.get(name, name)
            lbl = f'{id_str}{name_es}: {int(conf*100)}%'
            lsz, base = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ly = max(sy1, lsz[1] + 10)
            cv2.rectangle(stream_frame, (sx1, ly-lsz[1]-10),
                          (sx1+lsz[0], ly+base-10), color, cv2.FILLED)
            cv2.putText(stream_frame, lbl, (sx1, ly-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            _det_key = (cam_id, track_id if track_id >= 0 else name)
            _det_now = time.time()
            if _det_now - _det_last_ts.get(_det_key, 0) >= _DET_LOG_INTERVAL:
                _det_last_ts[_det_key] = _det_now
                batch.append({"timestamp": ts, "type": "detection", "cam": cam_id,
                              "class": name, "confidence": round(conf, 3),
                              "track_id": track_id,
                              "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]})

        frame_cnt[cam_id] += 1
        fps_t["cnt"] += 1
        elapsed = time.time() - fps_t["t"]
        if elapsed >= 5.0:
            fps_val = fps_t['cnt'] / elapsed
            print(f"[FPS] {cam_id}: {fps_val:.1f} fps", flush=True)
            dashboard.update_fps(cam_id, fps_val)
            fps_t["t"] = time.time(); fps_t["cnt"] = 0
        if frame_cnt[cam_id] % REID_EVERY == 0:
            try:
                reid_queues[cam_id].put_nowait((stream_frame.copy(), ts, persons))
            except queue.Full:
                pass  # REID ocupado, saltar este frame

        # Pose estimation + detección de caídas (throttle por cámara)
        if (pose is not None and frame_cnt[cam_id] % POSE_EVERY == 0
                and persons > 0 and not _csi_reconnecting.is_set()):
            try:
                with gpu_infer_lock:
                    pose_res = pose.analyze(stream_frame, cam_id)
                pose.draw(stream_frame, pose_res)
                for pr in pose_res:
                    if pr["fall_event"]:
                        print(f"[FALL] {cam_id} track={pr['track_id']} angle={pr['torso_angle']}", flush=True)
                        batch.append({
                            "timestamp": ts, "type": "fall", "cam": cam_id,
                            "track_id": pr["track_id"],
                            "torso_angle": round(pr["torso_angle"] or 0, 1),
                            "bbox": pr["bbox"],
                        })
                        dashboard.push_alert({
                            "timestamp": ts,
                            "label": f"CAIDA detectada ({cam_id})",
                            "confidence": 1.0,
                        })
                        _cap = (f"🆘 CAIDA DETECTADA\n"
                                f"Cámara: {cam_id.upper()}\n"
                                f"{ts[:19]} CST")
                        send_telegram('fall', _cap, stream_frame)
            except Exception as e:
                print(f"[POSE] {cam_id}: {e}", flush=True)

        cam_label = 'CSI IMX219' if cam_id == 'csi' else 'USB Webcam'
        cv2.putText(stream_frame, cam_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,255), 2)
        cv2.putText(stream_frame, f'Personas: {persons}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)

        with results_lock:
            results_out[cam_id] = (stream_frame, persons)
        person_count[cam_id] = persons
        dashboard.update_person_count(cam_id, persons)

        # Push frame al dashboard (960×540 quality 80)
        _, jpeg = cv2.imencode('.jpg', stream_frame, enc_params)
        dashboard.update_frame(cam_id, jpeg.tobytes())

        db.write_events(batch)

# ── HUD audio ─────────────────────────────────────────────────────────────────
def draw_audio_hud(frame):
    with lock:
        lbl   = audio_state['label']
        conf  = audio_state['conf']
        alert = audio_state['alert']
        speech= audio_state['speech']
    color = (0,0,255) if alert else (0,255,255) if speech else (180,180,180)
    flag  = ' ALERTA' if alert else ' VOZ' if speech else ''
    txt   = f'Audio: {lbl} ({conf:.0%}){flag}'
    cv2.putText(frame, txt, (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
    return frame

# ── Thread de composición combined ────────────────────────────────────────────
def compose_loop():
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while running:
        with results_lock:
            f_csi, p_csi = results_out['csi']
            f_usb, p_usb = results_out['usb']
            f_csi = f_csi.copy() if f_csi is not None else None
            f_usb = f_usb.copy() if f_usb is not None else None

        panels = []
        total_persons = 0
        if f_csi is not None:
            panels.append(f_csi)
            total_persons += p_csi
        if f_usb is not None:
            panels.append(f_usb)
            total_persons += p_usb

        if not panels:
            time.sleep(0.02)
            continue

        combined = np.hstack(panels) if len(panels) == 2 else panels[0]
        combined = draw_audio_hud(combined)
        cv2.putText(combined, f'Total personas: {total_persons}',
                    (combined.shape[1]//2 - 100, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,0), 2)

        _, jpeg = cv2.imencode('.jpg', combined, enc_params)
        dashboard.update_frame('combined', jpeg.tobytes())
        time.sleep(0.04)  # ~25 fps combined

# ── Reporte diario ────────────────────────────────────────────────────────────
def _build_daily_report(date_str: str) -> str:
    """Lee metadata de las últimas 24h y genera reporte con Claude."""
    from openai import OpenAI as _OpenAI
    from collections import Counter as _C
    cutoff = (datetime.datetime.now(CST) - datetime.timedelta(hours=24)).isoformat()
    records = db.events_in_range(cutoff)
    det, aud, faces, transcripts, kw_alerts, spk_counts = [], [], [], [], [], []
    for r in records:
        t = r.get('type')
        if t == 'detection':          det.append(r)
        elif t == 'audio':            aud.append(r)
        elif t == 'face_reid':        faces.append(r)
        elif t == 'audio_transcript': transcripts.append(r)
        elif t == 'keyword_alert':    kw_alerts.append(r)
        elif t == 'speaker_count':    spk_counts.append(r)

    seen = set()
    for d in det:
        seen.add((COCO_ES.get(d['class'], d['class']), d.get('track_id', -1)))
    obj_count = _C(cls for cls, _ in seen)

    tx_summary = '\n'.join(
        f"[{tx['timestamp'][11:16]}] {tx.get('text','')[:120]}"
        for tx in transcripts if tx.get('text')
    ) or 'Sin habla detectada'

    kw_summary = ', '.join(
        f"'{k['keyword']}'" for k in kw_alerts
    ) or 'Ninguna'

    avg_spk = (sum(s.get('n_speakers', 1) for s in spk_counts) / len(spk_counts)
               if spk_counts else 0)

    ctx = (
        f"Fecha: {date_str}\n"
        f"Objetos únicos detectados: {dict(obj_count.most_common(8))}\n"
        f"Personas únicas (FaceReID): {len(set(f['person_uuid'] for f in faces))}\n"
        f"Eventos de audio: {len(aud)} | Alertas críticas: {len([a for a in aud if a.get('alert')])}\n"
        f"Clips de audio: {len(spk_counts)} | Promedio hablantes/clip: {avg_spk:.1f}\n"
        f"Palabras clave detectadas: {kw_summary}\n"
        f"Transcripciones del día:\n{tx_summary}"
    )
    client = _OpenAI(base_url=os.environ.get('NEMOTRON_URL', 'http://localhost:8003/v1'), api_key="none")
    resp = client.chat.completions.create(
        model="nemotron-omni",
        max_tokens=1200,
        messages=[{"role": "user", "content": (
            "Genera un reporte diario conciso de un sistema de videovigilancia. "
            "Usa emojis. Incluye: resumen ejecutivo (2 líneas), actividad destacada, "
            "alertas importantes, y una observación final. Máximo 350 palabras. "
            "Responde en español.\n\nDatos:\n" + ctx
        )}]
    )
    return resp.choices[0].message.content

def send_daily_report():
    """Genera y envía el reporte diario a Telegram."""
    try:
        now = datetime.datetime.now(CST)
        date_str = now.strftime('%Y-%m-%d')
        print(f"[REPORT] Generando reporte diario {date_str}...", flush=True)
        text = _build_daily_report(date_str)
        msg = f"📊 <b>Reporte Diario — OXM Vision</b>\n📅 {date_str}\n\n{text}"
        _tg_send(msg, parse_mode='HTML')
        db.write_event({"timestamp": now.isoformat(), "type": "daily_report",
                        "date": date_str, "sent": True})
        print(f"[REPORT] Reporte enviado", flush=True)
    except Exception as e:
        print(f"[REPORT] Error: {e}", flush=True)

# Exponer para que dashboard pueda dispararlo manualmente
dashboard.set_report_fn(send_daily_report)
dashboard.set_face_photo_fn(send_person_photo)

def report_loop():
    sent_date = None
    while running:
        time.sleep(60)
        now = datetime.datetime.now(CST)
        today = now.date()
        if now.hour == REPORT_HOUR and sent_date != today:
            sent_date = today
            threading.Thread(target=send_daily_report, daemon=True).start()

def db_rotate_loop():
    while running:
        time.sleep(1800)  # cada 30 min
        db.rotate_if_needed(METADATA_ROTATE_MB)
        db.purge_detections(keep=50_000)

# ── Lanzar threads ─────────────────────────────────────────────────────────────
# Registrar callback de mute (set_mic_muted ya está definida)
dashboard.set_mic_muted_fn(set_mic_muted)
threading.Thread(target=report_loop,    daemon=True).start()
threading.Thread(target=db_rotate_loop, daemon=True).start()
threading.Thread(target=capture_csi,  daemon=True).start()
threading.Thread(target=capture_usb,  daemon=True).start()
threading.Thread(target=audio_loop,   daemon=True).start()
threading.Thread(target=infer_loop,   args=('csi',), daemon=True).start()
threading.Thread(target=infer_loop,   args=('usb',), daemon=True).start()
threading.Thread(target=reid_worker,  args=('csi',), daemon=True).start()
threading.Thread(target=reid_worker,  args=('usb',), daemon=True).start()
threading.Thread(target=compose_loop, daemon=True).start()

# ── Iniciar dashboard en thread ────────────────────────────────────────────────
import uvicorn
dash_thread = threading.Thread(
    target=uvicorn.run,
    kwargs={"app": dashboard.app, "host": "0.0.0.0", "port": 8080, "log_level": "warning"},
    daemon=True
)
dash_thread.start()
print("Dashboard iniciado en http://0.0.0.0:8080")

print("Pipeline headless corriendo. Ctrl+C para salir.")
time.sleep(3)

# ── Loop principal headless ────────────────────────────────────────────────────
try:
    while True:
        time.sleep(10)
        p_csi = person_count.get('csi', 0)
        p_usb = person_count.get('usb', 0)
        with lock:
            a_lbl = audio_state['label']
        print(f"[{datetime.datetime.now(CST).strftime('%H:%M:%S')}] "
              f"Personas CSI={p_csi} USB={p_usb} | Audio: {a_lbl}")
except KeyboardInterrupt:
    pass

running = False
print("Listo. Metadata en:", DB_PATH)
