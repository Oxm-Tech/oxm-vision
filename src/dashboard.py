import asyncio
import json, os, time, threading
from datetime import datetime, timezone, timedelta
from collections import Counter
import re as _re

_UUID_RE = _re.compile(r'^[0-9a-f]{6,40}(-[0-9a-f]{4,})*$', _re.I)
FACE_SNAPS_DIR = '/app/face_snaps'

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

CST = timezone(timedelta(hours=-6))
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
import uvicorn

AUDIO_DIR = '/app/audio_clips'
import db

app = FastAPI()

# ── Frame buffer con Condition para notificar nuevos frames ──────────────────
_frames = {'csi': None, 'usb': None, 'combined': None}
_frame_events = {k: threading.Condition() for k in _frames}

def update_frame(cam_id, jpeg_bytes):
    with _frame_events[cam_id]:
        _frames[cam_id] = jpeg_bytes
        _frame_events[cam_id].notify_all()

def update_combined(jpeg_bytes):
    with _frame_events['combined']:
        _frames['combined'] = jpeg_bytes
        _frame_events['combined'].notify_all()

_fps = {'csi': 0.0, 'usb': 0.0}
_person_count = {'csi': 0, 'usb': 0}
_alerts = []

_health = {
    'csi':   {'up': False, 'fps': 0.0, 'since': None, 'reconnects': 0},
    'usb':   {'up': False, 'fps': 0.0, 'since': None, 'reconnects': 0},
    'audio': {'up': False, 'label': '', 'device': '', 'muted': False},
    'reid':  {'up': False},
    'yolo':  {'up': False},
}

def set_cam_status(cam_id: str, up: bool):
    _health[cam_id]['up'] = up
    _health[cam_id]['since'] = datetime.now(CST).isoformat()
    if up:
        _health[cam_id]['reconnects'] += 1

def set_audio_status(up: bool, label: str = '', device: str = ''):
    _health['audio'].update(up=up, label=label, device=device)

def set_audio_muted(muted: bool):
    _health['audio']['muted'] = muted

_mic_muted_fn = None
def set_mic_muted_fn(fn):
    """main.py registra su función set_mic_muted aquí para que el endpoint la llame."""
    global _mic_muted_fn
    _mic_muted_fn = fn

def set_reid_up():
    _health['reid']['up'] = True

def set_yolo_up():
    _health['yolo']['up'] = True

_report_fn = None
def set_report_fn(fn):
    global _report_fn
    _report_fn = fn

_face_photo_fn = None
def set_face_photo_fn(fn):
    global _face_photo_fn
    _face_photo_fn = fn

def update_fps(cam_id, fps_val):
    _fps[cam_id] = fps_val
    _health[cam_id]['fps'] = round(fps_val, 1)

def update_person_count(cam_id, count):
    _person_count[cam_id] = count

def push_alert(alert):
    _alerts.append(alert)
    if len(_alerts) > 100:
        _alerts.pop(0)

def mjpeg_gen(cam):
    event = _frame_events.get(cam)
    if event is None:
        return
    last_frame = None
    frame_none_since = time.monotonic()
    try:
        while True:
            with event:
                # espera hasta 1s por frame nuevo; si no llega manda keepalive
                got_new = event.wait(timeout=1.0)
                frame = _frames.get(cam)
            if frame is None:
                # Cámara sin un solo frame por >8s: cerrar la conexión en vez de
                # colgarla para siempre. Sin esto el navegador nunca dispara
                # onerror/reconecta, y la conexión zombie ocupa un slot HTTP
                # que puede bloquear otros streams del mismo origen.
                if time.monotonic() - frame_none_since > 8:
                    return
                continue
            frame_none_since = time.monotonic()
            if not got_new and frame is last_frame:
                # Sin frame nuevo: re-emitir el último válido (NO mandar JPEG vacío,
                # eso causa flash negro en el navegador).
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                    + frame + b'\r\n'
                )
                continue
            last_frame = frame
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n'
                + frame + b'\r\n'
            )
    except GeneratorExit:
        pass

@app.get('/logo')
def serve_logo():
    path = '/app/logo-oxmtech.png'
    if not os.path.exists(path):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='image/png')

@app.get('/static/chart.js')
def serve_chartjs():
    # Chart.js local — el dashboard no depende de internet (CDN caído = gráfica muerta)
    path = '/app/chart.umd.min.js'
    if not os.path.exists(path):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='application/javascript')

@app.get('/audio/{filename}')
def serve_audio(filename: str):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path) or not filename.endswith('.opus'):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='audio/ogg')

@app.get('/api/audio/list')
async def audio_list(limit: int = 60):
    return await asyncio.to_thread(_audio_list_sync, limit)

def _audio_list_sync(limit: int = 60):
    if not os.path.exists(AUDIO_DIR):
        return JSONResponse([])

    # Índice: minuto → labels de audio desde DB
    label_by_min: dict = {}
    for r in db.tail_events(2000, types=['audio']):
        min_key = r['timestamp'][:16]
        label_by_min.setdefault(min_key, []).append(r.get('label', ''))

    clips = sorted(os.listdir(AUDIO_DIR))[-limit:][::-1]
    result = []
    for fname in clips:
        # conv_20260422_193028.opus → "2026-04-22T19:30"
        try:
            parts = fname.replace('conv_', '').replace('.opus', '')
            d, t = parts[:8], parts[9:]
            ts = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}"
            mins = int(t[2:4])
            bucket_min = (mins // 5) * 5
            bucket = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{bucket_min:02d}"
            labels = label_by_min.get(ts, [])
            if labels:
                from collections import Counter as _C
                top_label = _C(labels).most_common(1)[0][0]
            else:
                top_label = ''
        except Exception:
            ts, bucket, top_label = '', '', ''
        txt_path = os.path.join(AUDIO_DIR, fname.replace('.opus', '.txt'))
        transcript = ''
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as tf:
                    transcript = tf.read()[:400]
            except Exception:
                pass
        spk_path = os.path.join(AUDIO_DIR, fname.replace('.opus', '.spk'))
        n_speakers = None
        if os.path.exists(spk_path):
            try:
                n_speakers = int(open(spk_path).read().strip())
            except Exception:
                pass
        result.append({'file': fname, 'ts': ts, 'bucket': bucket,
                       'label': top_label, 'transcript': transcript,
                       'n_speakers': n_speakers})
    return JSONResponse(result)

@app.post('/api/audio/mute')
async def toggle_mic():
    """Silencia o activa el micrófono. Devuelve el estado nuevo."""
    if _mic_muted_fn is None:
        return JSONResponse({'muted': _health['audio'].get('muted', False), 'error': 'no_callback'})
    new_state = not _health['audio'].get('muted', False)
    _mic_muted_fn(new_state)
    return JSONResponse({'muted': new_state})

@app.get('/api/tegra')
async def tegra_stats():
    return await asyncio.to_thread(_tegra_sync)

def _tegra_sync():
    import glob as _glob, shutil as _shutil

    def _read(path, default=None):
        try:
            return open(path).read().strip()
        except Exception:
            return default

    # GPU load (0–1000 → %)
    gpu_raw = _read('/sys/devices/platform/bus@0/17000000.gpu/load')
    gpu_load = round(int(gpu_raw) / 10.0, 1) if gpu_raw else None

    # GPU freq MHz
    cur  = _read('/sys/class/devfreq/17000000.gpu/cur_freq')
    mxf  = _read('/sys/class/devfreq/17000000.gpu/max_freq')
    gpu_mhz     = round(int(cur) / 1e6) if cur else None
    gpu_mhz_max = round(int(mxf) / 1e6) if mxf else None

    # Temperatures
    gpu_temp = cpu_temp = soc_temp = None
    for zdir in _glob.glob('/sys/class/thermal/thermal_zone*'):
        try:
            name = open(f'{zdir}/type').read().strip().lower()
            val  = round(int(open(f'{zdir}/temp').read().strip()) / 1000.0, 1)
            if 'gpu' in name and gpu_temp is None:
                gpu_temp = val
            elif 'cpu' in name and cpu_temp is None:
                cpu_temp = val
            elif 'soc0' in name and soc_temp is None:
                soc_temp = val
        except Exception:
            pass

    # RAM (MB)
    ram_total = ram_avail = None
    try:
        for line in open('/proc/meminfo'):
            k, v = line.split(':')
            kb = int(v.strip().split()[0])
            if k == 'MemTotal':    ram_total = kb // 1024
            elif k == 'MemAvailable': ram_avail = kb // 1024
    except Exception:
        pass
    ram_used = (ram_total - ram_avail) if ram_total and ram_avail else None
    ram_pct  = round(ram_used / ram_total * 100, 1) if ram_total else None

    # NVMe / disco
    disk_used = disk_total = disk_pct = None
    try:
        du = _shutil.disk_usage('/app/audio_clips')
        disk_total = round(du.total / 1e9, 1)
        disk_used  = round(du.used  / 1e9, 1)
        disk_pct   = round(du.used  / du.total * 100, 1)
    except Exception:
        pass

    return JSONResponse({
        'gpu_load':    gpu_load,
        'gpu_mhz':     gpu_mhz,
        'gpu_mhz_max': gpu_mhz_max,
        'gpu_temp':    gpu_temp,
        'cpu_temp':    cpu_temp,
        'soc_temp':    soc_temp,
        'ram_used_mb': ram_used,
        'ram_total_mb':ram_total,
        'ram_pct':     ram_pct,
        'disk_used_gb':disk_used,
        'disk_total_gb':disk_total,
        'disk_pct':    disk_pct,
    })

@app.post('/api/report/now')
async def report_now():
    if _report_fn is None:
        return JSONResponse({'error': 'report function not registered'}, status_code=503)
    import threading as _t
    _t.Thread(target=_report_fn, daemon=True).start()
    return JSONResponse({'status': 'generando reporte, llega en ~30s a Telegram'})

@app.get('/api/health')
def health():
    return JSONResponse({
        'cameras': {
            'csi': _health['csi'],
            'usb': _health['usb'],
        },
        'audio':   _health['audio'],
        'reid':    _health['reid'],
        'yolo':    _health['yolo'],
        'persons': _person_count,
        'timestamp': datetime.now(CST).isoformat(),
    })

@app.get('/stream/{cam}')
def stream(cam: str):
    return StreamingResponse(mjpeg_gen(cam),
        media_type='multipart/x-mixed-replace; boundary=frame')

# ── Stats API ─────────────────────────────────────────────────────────────────
def load_data():
    records = db.tail_events(5000, types=['detection', 'audio', 'face_reid'])
    detections, audio_events, face_events = [], [], []
    for r in records:
        t = r.get('type')
        if t == 'detection':   detections.append(r)
        elif t == 'audio':     audio_events.append(r)
        elif t == 'face_reid': face_events.append(r)
    return detections, audio_events, face_events

@app.get('/api/stats')
async def stats():
    det, aud, faces = await asyncio.to_thread(load_data)
    persons   = [d for d in det if d['class'] == 'person']
    speech    = [a for a in aud if a.get('speech')]
    alerts    = [a for a in aud if a.get('alert')]
    # Contar track_ids únicos por clase (no eventos raw) para evitar inflación por frame
    seen_tracks = set()
    for d in det:
        seen_tracks.add((COCO_ES.get(d['class'], d['class']), d.get('track_id', -1)))
    obj_count = Counter(cls for cls, _ in seen_tracks)
    known     = [f for f in faces if not f.get('new')]
    new_faces = [f for f in faces if f.get('new')]

    person_by_min = Counter()
    for d in persons:
        person_by_min[d['timestamp'][:16]] += 1

    speech_by_min = Counter()
    for a in speech:
        speech_by_min[a['timestamp'][:16]] += 1

    speech_ts = set(a['timestamp'][:19] for a in speech)
    person_ts  = set(d['timestamp'][:19] for d in persons)
    overlap    = len(speech_ts & person_ts)

    clips = sorted(os.listdir(AUDIO_DIR))[-10:] if os.path.exists(AUDIO_DIR) else []
    minutes = sorted(set(list(person_by_min.keys()) + list(speech_by_min.keys())))

    return JSONResponse({
        "total_detections":   len(det),
        "total_audio":        len(aud),
        "persons_csi":        sum(1 for d in persons if d['cam']=='csi'),
        "persons_usb":        sum(1 for d in persons if d['cam']=='usb'),
        "speech_seconds":     len(speech),
        "alerts":             len(alerts),
        "speech_with_person": overlap,
        "speech_ratio":       round(overlap/max(len(speech_ts),1)*100),
        "top_objects":        obj_count.most_common(10),
        "timeline_labels":    minutes[-60:],
        "timeline_persons":   [person_by_min.get(m,0) for m in minutes[-60:]],
        "timeline_speech":    [speech_by_min.get(m,0) for m in minutes[-60:]],
        "audio_clips":        clips,
        "known_faces":        len(set(f['person_uuid'] for f in known)),
        "new_faces":          len(set(f['person_uuid'] for f in new_faces)),
        "last_update":        datetime.now(CST).isoformat()
    })

from pydantic import BaseModel as _BM
class ChatReq(_BM):
    question: str

@app.post('/api/chat')
async def chat(req: ChatReq):
    return await asyncio.to_thread(_chat_sync, req)

def _chat_sync(req: ChatReq):
    try:
        from openai import OpenAI as _OpenAI
        from collections import Counter as _C

        # Consultas separadas por tipo
        all_faces        = db.tail_events(300,   types=['face_reid'])
        all_transcripts  = db.tail_events(80,    types=['audio_transcript'])
        all_keywords     = db.tail_events(100,   types=['keyword_alert'])
        all_speakers     = db.tail_events(100,   types=['speaker_count'])
        all_correlations = db.tail_events(80,    types=['speech_visual_event'])
        all_sf_events    = db.tail_events(80,    types=['speech_face_event'])
        recent_audio     = db.tail_events(300,   types=['audio'])
        recent_det       = db.tail_events(500,   types=['detection'])

        oldest_ts, newest_ts, total_ev = db.time_range()
        counts_by_type = db.count_by_type()
        db_range = (f'{oldest_ts[:19]} -> {newest_ts[:19]} CST' if oldest_ts else 'sin datos')

        h = _health
        csi_status   = f'OK {h["csi"]["fps"]:.0f} fps' if h['csi']['up'] else 'DESCONECTADA'
        usb_status   = f'OK {h["usb"]["fps"]:.0f} fps' if h['usb']['up'] else 'DESCONECTADA'
        audio_status = (f'ACTIVO — {h["audio"].get("label","?")!r}' if h['audio']['up'] else 'SIN AUDIO')
        reid_status  = 'ACTIVO' if h['reid']['up'] else 'INICIANDO'

        # Objetos: deduplicados por track_id
        seen_tracks = {}
        for d in recent_det:
            key = (d.get('class','?'), d.get('track_id',-1))
            ts  = d.get('timestamp','')
            if key not in seen_tracks:
                seen_tracks[key] = {'first': ts, 'last': ts}
            else:
                seen_tracks[key]['last'] = ts
        obj_count = _C(cls for cls, _ in seen_tracks)
        obj_lines = ', '.join(
            f'{COCO_ES.get(cls, cls)}: {cnt}'
            for cls, cnt in obj_count.most_common(12)
        ) or 'ninguno'

        # Personas: timeline completo
        persons = {}
        for f in all_faces:
            uid = f.get('person_uuid','')
            ts  = f.get('timestamp','')
            if uid not in persons:
                persons[uid] = {'first':ts,'last':ts,'new':f.get('new',True),
                                'cam':f.get('cam',''),'sim':f.get('similarity',0),'count':1}
            else:
                persons[uid]['last']  = ts
                persons[uid]['count'] += 1
                if f.get('similarity',0) > persons[uid]['sim']:
                    persons[uid]['sim'] = f.get('similarity',0)

        persons_lines = ''
        for uid, p in list(persons.items())[:25]:
            label = 'NUEVA' if p['new'] else 'conocida'
            persons_lines += (
                f'  ID:{uid[:8]} [{label}] '
                f'primera:{p["first"][11:16]} ultima:{p["last"][11:16]} '
                f'apariciones:{p["count"]} cam:{p["cam"]} sim:{p["sim"]:.2f}\n'
            )

        # Audio
        audio_dist   = _C(a.get('label','?') for a in recent_audio)
        audio_alerts = [a for a in recent_audio if a.get('alert')]
        speech_pct   = (audio_dist.get('Speech',0) / len(recent_audio) * 100
                        if recent_audio else 0)

        # Transcripciones completas con texto
        tx_lines = ''
        for tx in all_transcripts:
            text = tx.get('text','').strip()
            if text:
                tx_lines += f'  [{tx.get("timestamp","")[:19]}] {text[:250]}\n'
        tx_lines = tx_lines or '  Sin transcripciones (requiere >30s de habla continua)\n'

        # Hablantes
        spk_lines = ''
        for s in all_speakers[-15:]:
            spk_lines += (
                f'  [{s.get("timestamp","")[:19]}] '
                f'{s.get("n_speakers","?")} hablante(s) | '
                f'archivo:{s.get("filename","?")}\n'
            )
        spk_lines = spk_lines or '  Sin clips procesados\n'

        # Keywords
        kw_lines = ''
        for kw in all_keywords:
            kw_lines += (
                f'  [{kw.get("timestamp","")[:19]}] '
                f'KEYWORD={kw.get("keyword")!r} -> '
                f'{kw.get("text","")[:150]!r}\n'
            )
        kw_lines = kw_lines or '  Ninguna\n'

        # Correlaciones audio-visual
        all_corr = sorted(all_sf_events + all_correlations,
                          key=lambda x: x.get('timestamp',''))
        corr_lines = ''
        for c in all_corr[-15:]:
            corr_lines += (
                f'  [{c.get("timestamp","")[:19]}] '
                f'persona:{c.get("person_uuid","?")[:8]} | '
                f'audio:{c.get("audio_label",c.get("audio","?"))} | '
                f'{c.get("text_snippet",c.get("text",""))[:150]!r}\n'
            )
        corr_lines = corr_lines or '  Sin correlaciones aun\n'

        _plines = persons_lines if persons_lines else '  Ninguna detectada\n'
        ctx = (
            f'=== BASE DE DATOS ===\n'
            f'Periodo completo: {db_range}\n'
            f'Total eventos: {total_ev:,}\n'
            f'Desglose por tipo: {dict(list(counts_by_type.items())[:10])}\n\n'
            f'=== ESTADO DEL SISTEMA ===\n'
            f'Camara CSI: {csi_status}\n'
            f'Camara USB: {usb_status}\n'
            f'Audio: {audio_status}\n'
            f'FaceReID: {reid_status}\n\n'
            f'=== OBJETOS DETECTADOS (muestra reciente, dedup por track_id) ===\n'
            f'{obj_lines}\n'
            f'(Total detecciones en toda la DB: {counts_by_type.get("detection",0):,})\n\n'
            f'=== PERSONAS ({len(persons)} unicas, {len(all_faces)} eventos ReID) ===\n'
            f'{_plines}\n'
            f'=== AUDIO ({len(recent_audio)} eventos clasificados) ===\n'
            f'Distribucion: {dict(audio_dist.most_common(8))}\n'
            f'Porcentaje de habla: {speech_pct:.0f}%\n'
            f'Alertas criticas: {len(audio_alerts)}\n\n'
            f'=== TRANSCRIPCIONES ({len(all_transcripts)} clips transcritos) ===\n'
            f'{tx_lines}\n'
            f'=== HABLANTES POR CLIP ({len(all_speakers)} clips) ===\n'
            f'{spk_lines}\n'
            f'=== ALERTAS PALABRAS CLAVE ({len(all_keywords)}) ===\n'
            f'{kw_lines}\n'
            f'=== CORRELACIONES AUDIO-VISUAL ({len(all_corr)}) ===\n'
            f'{corr_lines}'
        )

        # Solicitud de foto
        photo_ctx = ''
        if _face_photo_fn is not None:
            import re as _re2
            q_lower = req.question.lower()
            _PHOTO_KW = ['foto','imagen','picture','ver','muéstrame','muestrame',
                         'manda','envias','envia','telegram','mostrar','enseña']
            if any(kw in q_lower for kw in _PHOTO_KW):
                uuid_hit = _re2.search(r'[0-9a-f]{6,}', q_lower)
                target_uuid = None
                if uuid_hit:
                    prefix = uuid_hit.group(0)
                    for f in reversed(all_faces):
                        if f.get('person_uuid','').lower().startswith(prefix):
                            target_uuid = f['person_uuid']; break
                elif any(kw in q_lower for kw in ['persona','alguien','quien','quién',
                                                   'ultimo','última','reciente']):
                    if all_faces: target_uuid = all_faces[-1]['person_uuid']
                if target_uuid:
                    photo_ctx = f'\n\n=== SOLICITUD DE FOTO ===\n{_face_photo_fn(target_uuid)}\n'

        SYSTEM_PROMPT = (
            'Eres el asistente de un sistema de vision inteligente OXM Vision.\n'
            'El sistema monitorea un salon con varias mesas donde hay una junta de negocios.\n'
            'Se esta presentando una demostracion sobre nuevas tecnologias de inteligencia\n'
            'artificial y vision por computadora de NVIDIA. Hay un auditor en la reunion\n'
            'y el tema central es vision inteligente.\n'
            'Cuando describes lo que ocurre, hazlo de forma natural desde lo que observas:\n'
            'personas sentadas en mesas, actividad de la presentacion, conversaciones.\n'
            'No menciones que recibiste instrucciones sobre el contexto.\n\n'

            'ESTILO: prosa natural de colega tecnico, sin headers ni bullets, sin reportes.\n'
            'Frases cortas. Muletillas: mira, fijate, pues. Sin disculpas corporativas.\n\n'
            'CONTENIDO: usa todo el contexto disponible. Cita transcripciones textualmente\n'
            'con su hora y conéctalas a la junta o presentacion cuando tenga sentido.\n'
            'Personas: da hora de aparicion, cuantas veces aparecio, si es conocida o nueva.\n'
            'Correlaciones audio-visual: conecta lo que se dijo con quien estaba en frame.\n'
            'Narrativa temporal: "a las 10 habia..., luego a las 14...".\n'
            'Se especifico con IDs y horarios exactos. No inventes nada.'
        )

        client = _OpenAI(base_url=os.environ.get('NEMOTRON_URL', 'http://localhost:8003/v1'), api_key='none', timeout=60.0)
        resp = client.chat.completions.create(
            model='nemotron-omni',
            max_tokens=1024,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': (
                    f'Datos del sistema:\n{ctx}{photo_ctx}\n\nPregunta: {req.question}'
                )}
            ]
        )
        return JSONResponse({'answer': resp.choices[0].message.content})
    except Exception as e:
        return JSONResponse({'answer': f'Error: {e}'}, status_code=500)

@app.get('/api/report/pdf')
async def report_pdf():
    from fastapi.responses import Response
    result = await asyncio.to_thread(_build_pdf_sync)
    if isinstance(result, JSONResponse):
        return result
    date_str = datetime.now(CST).strftime('%Y-%m-%d')
    return Response(content=result, media_type='application/pdf',
                    headers={'Content-Disposition': f'attachment; filename="oxm-reporte-{date_str}.pdf"'})

def _build_pdf_sync():
    try:
        from fpdf import FPDF
    except ImportError:
        return JSONResponse({'error': 'fpdf2 no instalado'}, status_code=500)

    date_str = datetime.now(CST).strftime('%Y-%m-%d')
    now_str  = datetime.now(CST).strftime('%Y-%m-%d %H:%M:%S')

    # ── Cargar datos ─────────────────────────────────────────
    records = db.tail_events(10000)
    det, aud, faces, transcripts, kw_alerts = [], [], [], [], []
    for r in records:
        t = r.get('type')
        if   t == 'detection':          det.append(r)
        elif t == 'audio':              aud.append(r)
        elif t == 'face_reid':          faces.append(r)
        elif t == 'audio_transcript':   transcripts.append(r)
        elif t == 'keyword_alert':      kw_alerts.append(r)
    seen_tracks = set()
    for d in det:
        seen_tracks.add((COCO_ES.get(d['class'], d['class']), d.get('track_id', -1)))
    obj_count  = Counter(cls for cls, _ in seen_tracks)
    top_obj    = obj_count.most_common(8)
    persons    = [d for d in det if d['class'] == 'person']
    alerts_lst = [a for a in aud if a.get('alert')]
    known_ids  = len(set(f['person_uuid'] for f in faces if not f.get('new')))
    audio_dist = Counter(a.get('label', '') for a in aud if a.get('label')).most_common(6)

    # Actividad por hora
    by_hour = Counter()
    for d in persons:
        try: by_hour[int(d['timestamp'][11:13])] += 1
        except Exception: pass

    # ── PDF ──────────────────────────────────────────────────
    class PDF(FPDF):
        def footer(self):
            self.set_y(-12)
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(120, 140, 170)
            self.cell(0, 6,
                f'OXM Vision  ·  OXMtech  ·  {now_str} CST  ·  Pág. {self.page_no()}',
                align='C')

    pdf = PDF()
    pdf.set_margins(14, 14, 14)
    pdf.add_page()
    pdf.set_auto_page_break(True, margin=20)
    W = pdf.w - pdf.l_margin - pdf.r_margin   # 182 mm

    # ── Header bar ───────────────────────────────────────────
    pdf.set_fill_color(8, 18, 35)
    pdf.rect(0, 0, 210, 32, 'F')
    pdf.set_fill_color(30, 86, 200)
    pdf.rect(0, 31, 210, 1.5, 'F')

    logo = '/app/logo-oxmtech.png'
    if os.path.exists(logo):
        pdf.image(logo, x=14, y=6, h=20)

    pdf.set_xy(100, 7)
    pdf.set_font('Helvetica', 'B', 15)
    pdf.set_text_color(220, 235, 255)
    pdf.cell(96, 9, 'Reporte de Vigilancia', align='R')
    pdf.set_xy(100, 17)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(90, 140, 200)
    pdf.cell(96, 7, f'{date_str}  ·  Jetson Orin Nano', align='R')
    pdf.set_y(40)

    # ── Stat boxes ───────────────────────────────────────────
    stats_def = [
        ('PERSONAS',       len(set(d.get('track_id',-1) for d in persons)), (77,159,255)),
        ('CARAS CONOCIDAS',known_ids,                                        (167,139,250)),
        ('ALERTAS',        len(alerts_lst),                                  (240,92,110)),
        ('EVENTOS AUDIO',  len(aud),                                         (45,201,126)),
    ]
    bw = (W - 9) / 4
    bh = 22
    y0 = pdf.get_y()
    for i, (lbl, val, col) in enumerate(stats_def):
        x = pdf.l_margin + i * (bw + 3)
        pdf.set_fill_color(14, 24, 42)
        pdf.set_draw_color(25, 45, 75)
        pdf.rect(x, y0, bw, bh, 'FD')
        pdf.set_fill_color(*col)
        pdf.rect(x, y0, bw, 1.5, 'F')
        pdf.set_xy(x, y0 + 3)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(*col)
        pdf.cell(bw, 11, str(val), align='C')
        pdf.set_xy(x, y0 + 14)
        pdf.set_font('Helvetica', 'B', 6)
        pdf.set_text_color(110, 140, 180)
        pdf.cell(bw, 5, lbl, align='C')
    pdf.set_y(y0 + bh + 8)

    # ── Helpers ──────────────────────────────────────────────
    def section_title(title):
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(70, 120, 200)
        pdf.cell(0, 5, title.upper(), new_x='LMARGIN', new_y='NEXT')
        y = pdf.get_y()
        pdf.set_draw_color(25, 50, 90)
        pdf.line(pdf.l_margin, y, pdf.l_margin + W, y)
        pdf.set_y(y + 4)

    def horiz_bars(x0, y0, width, items, color=(30, 90, 200)):
        label_w = 34
        cnt_w   = 14
        bar_w   = width - label_w - cnt_w - 3
        max_val = items[0][1] if items else 1
        y = y0
        for name, val in items:
            pdf.set_xy(x0, y)
            pdf.set_font('Helvetica', '', 7)
            pdf.set_text_color(180, 200, 230)
            pdf.cell(label_w, 5, str(name)[:16])
            bx = x0 + label_w + 1
            pdf.set_fill_color(20, 35, 60)
            pdf.rect(bx, y + 1, bar_w, 3.5, 'F')
            fw = max(2, int(bar_w * val / max_val)) if max_val else 2
            pdf.set_fill_color(*color)
            pdf.rect(bx, y + 1, fw, 3.5, 'F')
            pdf.set_xy(bx + bar_w + 1, y)
            pdf.set_text_color(90, 140, 200)
            pdf.cell(cnt_w, 5, str(val), align='R')
            y += 6
        return y

    # ── Dos columnas: Objetos | Audio ─────────────────────────
    col_w = (W - 6) / 2
    y_2col = pdf.get_y()

    # Columna izquierda — Objetos
    pdf.set_xy(pdf.l_margin, y_2col)
    section_title('Objetos Detectados')
    if top_obj:
        y_left = horiz_bars(pdf.l_margin, pdf.get_y(), col_w, top_obj, (30, 90, 200))
    else:
        pdf.set_font('Helvetica', 'I', 7.5); pdf.set_text_color(100,120,150)
        pdf.cell(col_w, 6, 'Sin detecciones', new_x='LMARGIN', new_y='NEXT')
        y_left = pdf.get_y()

    # Columna derecha — Audio
    x_right = pdf.l_margin + col_w + 6
    pdf.set_xy(x_right, y_2col)
    section_title('Clasificación de Audio')
    if audio_dist:
        y_right = horiz_bars(x_right, pdf.get_y(), col_w, audio_dist, (20, 160, 100))
    else:
        pdf.set_font('Helvetica', 'I', 7.5); pdf.set_text_color(100,120,150)
        pdf.cell(col_w, 6, 'Sin datos de audio', new_x='LMARGIN', new_y='NEXT')
        y_right = pdf.get_y()

    pdf.set_y(max(y_left, y_right) + 8)

    # ── Actividad por hora ────────────────────────────────────
    if by_hour:
        section_title('Actividad de Personas por Hora del Día')
        max_h = max(by_hour.values()) or 1
        bar_total = W
        bar_slot  = bar_total / 24
        y_chart   = pdf.get_y()
        chart_h   = 18
        # Fondo
        pdf.set_fill_color(14, 24, 42)
        pdf.rect(pdf.l_margin, y_chart, bar_total, chart_h + 6, 'F')
        for hour in range(24):
            cnt = by_hour.get(hour, 0)
            bh2 = int(chart_h * cnt / max_h) if cnt else 0
            x   = pdf.l_margin + hour * bar_slot
            if bh2 > 0:
                pdf.set_fill_color(30, 90, 200)
                pdf.rect(x + 0.5, y_chart + chart_h - bh2 + 3, bar_slot - 1, bh2, 'F')
            if hour % 4 == 0:
                pdf.set_xy(x, y_chart + chart_h + 3)
                pdf.set_font('Helvetica', '', 5.5)
                pdf.set_text_color(90, 120, 160)
                pdf.cell(bar_slot * 4, 4, f'{hour:02d}h', align='C')
        pdf.set_y(y_chart + chart_h + 10)

    # ── Transcripciones ───────────────────────────────────────
    recent_tx = [t for t in transcripts[-12:] if t.get('text','').strip()]
    if recent_tx:
        pdf.add_page() if pdf.get_y() > 220 else None
        section_title(f'Transcripciones Recientes ({len(recent_tx)} clips)')
        for tx in recent_tx:
            ts   = tx.get('timestamp', '')[:19]
            text = tx.get('text', '').strip()[:300]
            n_sp = tx.get('n_speakers')
            pdf.set_font('Helvetica', 'B', 7)
            pdf.set_text_color(77, 159, 255)
            spk_str = f'  · {n_sp} hablante{"s" if n_sp != 1 else ""}' if n_sp else ''
            pdf.cell(0, 5, f'{ts}{spk_str}', new_x='LMARGIN', new_y='NEXT')
            pdf.set_font('Helvetica', '', 7.5)
            pdf.set_text_color(180, 205, 230)
            pdf.set_fill_color(14, 24, 42)
            pdf.multi_cell(W, 4.5, text, fill=True, new_x='LMARGIN', new_y='NEXT')
            pdf.set_y(pdf.get_y() + 3)

    # ── Alertas por palabras clave ────────────────────────────
    if kw_alerts:
        section_title(f'Alertas por Palabras Clave ({len(kw_alerts)} total)')
        for kw in kw_alerts[-15:]:
            ts  = kw.get('timestamp', '')[:19]
            kwd = kw.get('keyword', '')
            txt = kw.get('text', '')[:200]
            pdf.set_font('Helvetica', 'B', 7)
            pdf.set_text_color(240, 92, 110)
            pdf.cell(0, 5, f'[{ts}]  "{kwd}"', new_x='LMARGIN', new_y='NEXT')
            if txt:
                pdf.set_font('Helvetica', 'I', 7)
                pdf.set_text_color(200, 160, 170)
                pdf.multi_cell(W, 4.5, txt, new_x='LMARGIN', new_y='NEXT')
            pdf.set_y(pdf.get_y() + 2)

    # ── Sin actividad placeholder ─────────────────────────────
    if not det and not aud:
        pdf.set_font('Helvetica', 'I', 9)
        pdf.set_text_color(100, 120, 150)
        pdf.cell(0, 8, 'Sin actividad registrada en el período.', align='C', new_x='LMARGIN', new_y='NEXT')

    return bytes(pdf.output())

# ── Persons API ───────────────────────────────────────────────────────────────
@app.get('/api/persons')
async def persons_list():
    return await asyncio.to_thread(_persons_sync)

def _persons_sync():
    records = db.tail_events(10000, types=['face_reid'])
    persons = {}
    for r in records:
        uid = r.get('person_uuid')
        if not uid:
            continue
        ts = r.get('timestamp', '')
        if uid not in persons:
            persons[uid] = {
                'uuid': uid,
                'first_seen': ts,
                'last_seen': ts,
                'cams': set(),
                'known': not r.get('new', True),
                'best_similarity': float(r.get('similarity') or 0),
                'n_events': 0,
            }
        p = persons[uid]
        if ts > p['last_seen']:
            p['last_seen'] = ts
        if ts and ts < p['first_seen']:
            p['first_seen'] = ts
        cam = r.get('cam')
        if cam:
            p['cams'].add(cam)
        p['n_events'] += 1
        if not r.get('new', True):
            p['known'] = True
        sim = float(r.get('similarity') or 0)
        if sim > p['best_similarity']:
            p['best_similarity'] = sim

    result = []
    for uid, p in sorted(persons.items(), key=lambda x: x[1]['last_seen'], reverse=True):
        has_snap = os.path.exists(f'{FACE_SNAPS_DIR}/{uid}.jpg')
        result.append({
            'uuid': uid,
            'uuid_short': uid[:8],
            'first_seen': p['first_seen'][:19],
            'last_seen': p['last_seen'][:19],
            'cams': sorted(p['cams']),
            'known': p['known'],
            'similarity': round(p['best_similarity'], 2),
            'n_events': p['n_events'],
            'has_snap': has_snap,
        })

    return JSONResponse({'persons': result, 'total': len(result)})

@app.get('/face_snap/{uuid}')
def face_snap(uuid: str):
    if not _UUID_RE.match(uuid):
        return JSONResponse({'error': 'invalid'}, status_code=400)
    path = f'{FACE_SNAPS_DIR}/{uuid}.jpg'
    if not os.path.exists(path):
        return JSONResponse({'error': 'not found'}, status_code=404)
    return FileResponse(path, media_type='image/jpeg')

@app.get('/', response_class=HTMLResponse)
def dashboard():
    # Cache-Control estricto: el HTML incluye JS/CSS inline que cambia con cada
    # deploy. Sin esto algunos navegadores sirven una versión vieja cacheada
    # y los fixes no se ven aunque el servidor ya los tenga.
    return HTMLResponse(content='''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OXM Vision</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<script src="/static/chart.js"></script>
<style>
:root{
  --bg:#080c12;
  --surface:#0e1420;
  --surface2:#141b27;
  --border:#1e2a3a;
  --border2:#243045;
  --text:#e2eaf5;
  --muted:#7d93b8;
  --blue:#45b3ff;
  --blue-dim:rgba(69,179,255,.12);
  --green:#2dc97e;
  --green-dim:rgba(45,201,126,.12);
  --purple:#a78bfa;
  --purple-dim:rgba(167,139,250,.12);
  --red:#f05c6e;
  --red-dim:rgba(240,92,110,.12);
  --amber:#f59e0b;
  --amber-dim:rgba(245,158,11,.12);
  --radius:12px;
  --radius-sm:8px;
}
body.light{
  --bg:#f0f4f8;
  --surface:#ffffff;
  --surface2:#e8edf5;
  --border:#c8d4e3;
  --border2:#b0bfd2;
  --text:#0f1c2e;
  --muted:#5a7099;
  --blue:#1a56db;
  --blue-dim:rgba(26,86,219,.10);
  --green:#059669;
  --green-dim:rgba(5,150,105,.10);
  --purple:#6d28d9;
  --purple-dim:rgba(109,40,217,.10);
  --red:#dc2626;
  --red-dim:rgba(220,38,38,.10);
  --amber:#d97706;
  --amber-dim:rgba(217,119,6,.10);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:
    radial-gradient(1100px 480px at 50% -12%, rgba(69,179,255,.075), transparent 62%),
    var(--bg);
  color:var(--text);font-family:\'Inter\',system-ui,sans-serif;min-height:100vh;
}

/* ── Header ─────────────────────────────────────────────── */
.header{
  display:grid;grid-template-columns:1fr auto 1fr;align-items:center;
  padding:12px 24px;
  background:linear-gradient(90deg,#0a1628 0%,#0e1c32 100%);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:50;
  backdrop-filter:blur(12px);
}
.header::after{
  content:\'\';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--blue) 30%,#8fd0ff 50%,var(--blue) 70%,transparent);
  opacity:.8;
}
.header-left{display:flex;align-items:center;gap:10px;justify-content:flex-start}
.header-center{display:flex;justify-content:center;align-items:center}
.header-right{display:flex;justify-content:flex-end;align-items:center;gap:10px}
.logo-pill{
  background:#fff;padding:7px 18px;border-radius:12px;
  display:flex;align-items:center;
  box-shadow:0 0 0 1px var(--border),0 4px 18px rgba(0,0,0,.35),0 0 24px rgba(77,159,255,.08);
}
body.light .logo-pill{box-shadow:0 0 0 1px var(--border),0 2px 10px rgba(15,28,46,.12)}
.oxm-logo{height:30px;width:auto;object-fit:contain;display:block}
.subtitle{font-size:.72em;color:var(--muted);font-weight:400}
.clock{font-size:.8em;color:var(--muted);font-variant-numeric:tabular-nums;
  background:var(--surface2);padding:5px 12px;border-radius:20px;border:1px solid var(--border)}

/* ── Layout ─────────────────────────────────────────────── */
.page{padding:20px 24px;max-width:1600px;margin:0 auto;display:flex;flex-direction:column;gap:18px}

/* ── Health bar ──────────────────────────────────────────── */
.health-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.hchip{
  display:flex;align-items:center;gap:6px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:20px;padding:5px 14px;font-size:.72em;font-weight:500;
  transition:border-color .2s;
}
.hchip:hover{border-color:var(--border2)}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-ok{background:var(--green);box-shadow:0 0 6px var(--green)}
.dot-err{background:var(--red);box-shadow:0 0 6px var(--red)}
.dot-warn{background:var(--amber);box-shadow:0 0 6px var(--amber)}
.dot-pulse{animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.hlbl{color:var(--muted)}
.hval{color:var(--text);margin-left:2px}
.mic-btn{
  margin-left:4px;background:none;border:none;cursor:pointer;
  font-size:1em;padding:0 2px;line-height:1;
  opacity:.8;transition:opacity .15s,transform .15s;
}
.mic-btn:hover{opacity:1;transform:scale(1.2)}

/* ── Streams ─────────────────────────────────────────────── */
.streams-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.stream-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;
  transition:border-color .2s;
}
.stream-card:hover{border-color:var(--border2)}
.stream-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:10px 14px;background:var(--surface2);border-bottom:1px solid var(--border);
}
.stream-title{font-size:.78em;font-weight:600;letter-spacing:.3px;text-transform:uppercase;color:var(--muted)}
.stream-badge{
  display:flex;align-items:center;gap:5px;
  font-size:.65em;font-weight:600;padding:2px 9px;border-radius:12px;
  letter-spacing:.5px;
}
.badge-live{background:rgba(45,201,126,.15);color:var(--green);border:1px solid rgba(45,201,126,.3)}
.badge-err{background:var(--red-dim);color:var(--red);border:1px solid rgba(240,92,110,.3)}
.badge-live .dot,.badge-err .dot{width:5px;height:5px}
.stream-img-wrap{position:relative;background:#000;aspect-ratio:16/9}
.stream-img-wrap img{position:relative;z-index:3;width:100%;height:100%;object-fit:cover;display:block}
.stream-footer{padding:8px 14px;font-size:.7em;color:var(--muted);
  display:flex;justify-content:space-between;align-items:center}

/* ── Stat cards ──────────────────────────────────────────── */
.cards-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}
.stat-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px 14px;
  display:flex;flex-direction:column;gap:6px;
  transition:border-color .2s,transform .1s;
  position:relative;overflow:hidden;
}
.stat-card:hover{border-color:var(--border2);transform:translateY(-1px)}
.stat-card::before{
  content:\'\';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--accent,var(--blue));opacity:.6;
}
.stat-card.green{--accent:var(--green)}
.stat-card.purple{--accent:var(--purple)}
.stat-card.red{--accent:var(--red)}
.stat-card.amber{--accent:var(--amber)}
.stat-icon{font-size:1.1em;line-height:1}
.stat-val{font-size:1.7em;font-weight:700;color:var(--accent,var(--blue));line-height:1.1}
.stat-lbl{font-size:.72em;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.4px}

/* ── Section title ───────────────────────────────────────── */
.section-title{
  font-size:.72em;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.6px;
  display:flex;align-items:center;gap:8px;
  margin-bottom:10px;
}
.section-title::after{content:\'\';flex:1;height:1px;background:var(--border)}

/* ── Panel ───────────────────────────────────────────────── */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:16px}

/* ── Chart ───────────────────────────────────────────────── */
.chart-wrap{position:relative}

/* ── Objects ─────────────────────────────────────────────── */
.obj-list{display:flex;flex-direction:column;gap:6px}
.obj-item{display:flex;align-items:center;gap:10px;font-size:.8em}
.obj-name{width:90px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0}
.obj-track{flex:1;background:var(--surface2);border-radius:4px;height:6px;overflow:hidden}
.obj-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,#1a56db,var(--blue));transition:width .4s ease}
.obj-cnt{font-size:.75em;color:var(--muted);min-width:30px;text-align:right}

/* ── Clips ───────────────────────────────────────────────── */
.clips-list{list-style:none;max-height:400px;overflow-y:auto;display:flex;flex-direction:column;gap:6px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.clips-list::-webkit-scrollbar{width:4px}
.clips-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
.clip-group-hdr{
  font-size:.68em;color:var(--muted);font-weight:600;
  padding:6px 8px 4px;border-bottom:1px solid var(--border);
  margin-top:4px;display:flex;align-items:center;gap:6px;
}
.clip-item{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:8px 10px;
  display:flex;flex-direction:column;gap:4px;
}
.clip-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.clip-time{font-size:.7em;font-weight:600;color:var(--blue)}
.clip-tag{
  font-size:.6em;padding:1px 7px;border-radius:10px;font-weight:500;
  background:var(--surface);border:1px solid var(--border);color:var(--muted);
}
.clip-transcript{font-size:.68em;color:#94b8d8;line-height:1.5;white-space:pre-wrap;margin-top:2px}
.clip-item audio{width:100%;height:24px;margin-top:4px;opacity:.85}
.clip-item audio:hover{opacity:1}

/* ── Chat ────────────────────────────────────────────────── */
.chat-messages{
  min-height:80px;max-height:220px;overflow-y:auto;
  display:flex;flex-direction:column;gap:8px;
  margin-bottom:12px;padding:2px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent
}
.msg-user{
  align-self:flex-end;
  background:linear-gradient(135deg,#1a56db,#2563eb);
  color:#dbeafe;padding:8px 14px;border-radius:14px 14px 3px 14px;
  font-size:.82em;max-width:80%;line-height:1.4;
}
.msg-ai{
  align-self:flex-start;
  background:var(--surface2);border:1px solid var(--border);
  color:var(--text);padding:8px 14px;border-radius:14px 14px 14px 3px;
  font-size:.82em;max-width:80%;white-space:pre-wrap;line-height:1.4;
}
.msg-loading{color:var(--muted);font-size:.78em;font-style:italic;align-self:flex-start;padding:4px 0}
.chat-form{display:flex;gap:8px}
.chat-form input{
  flex:1;background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:9px 14px;color:var(--text);
  font-size:.85em;font-family:inherit;outline:none;transition:border-color .2s;
}
.chat-form input:focus{border-color:var(--blue)}
.chat-form input::placeholder{color:var(--muted)}
.btn{
  border:none;border-radius:var(--radius-sm);padding:9px 18px;
  color:#fff;cursor:pointer;font-size:.82em;font-family:inherit;font-weight:600;
  transition:filter .15s,opacity .15s;white-space:nowrap;
}
.btn:hover{filter:brightness(1.1)}
.btn:disabled{opacity:.4;cursor:not-allowed;filter:none}
.btn-blue{background:linear-gradient(135deg,#1a56db,#2563eb)}
.btn-green{background:linear-gradient(135deg,#059669,#10b981)}
.btn-indigo{background:linear-gradient(135deg,#4f46e5,#6d66f5)}

/* ── Telemetría ──────────────────────────────────────────── */
.tegra-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}
.tegra-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:12px 14px;
  display:flex;flex-direction:column;gap:6px;
  transition:border-color .2s;
}
.tegra-card:hover{border-color:var(--border2)}
.tegra-header{display:flex;justify-content:space-between;align-items:center}
.tegra-label{font-size:.68em;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.tegra-value{font-size:1.1em;font-weight:700}
.tegra-sub{font-size:.65em;color:var(--muted);margin-top:-2px}
.tegra-bar-bg{height:5px;background:var(--surface2);border-radius:3px;overflow:hidden}
.tegra-bar-fill{height:100%;border-radius:3px;transition:width .6s ease}
.bar-blue{background:linear-gradient(90deg,#1a56db,var(--blue))}
.bar-green{background:linear-gradient(90deg,#059669,var(--green))}
.bar-amber{background:linear-gradient(90deg,#b45309,var(--amber))}
.bar-red{background:linear-gradient(90deg,#b91c1c,var(--red))}
.temp-ok{color:var(--green)} .temp-warm{color:var(--amber)} .temp-hot{color:var(--red)}

/* ── Bottom row ──────────────────────────────────────────── */
.bottom-row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.actions-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.last-update{font-size:.68em;color:var(--muted)}

/* ── Alertas en vivo (toasts) ────────────────────────────── */
.toast-stack{position:fixed;top:70px;right:20px;z-index:100;display:flex;flex-direction:column;gap:8px;max-width:340px}
.toast{
  display:flex;align-items:flex-start;gap:10px;
  background:var(--surface);border:1px solid var(--border2);
  border-left:3px solid var(--blue);border-radius:var(--radius-sm);
  padding:10px 14px;font-size:.8em;line-height:1.4;color:var(--text);
  box-shadow:0 8px 24px rgba(0,0,0,.45);
  animation:toastIn .3s cubic-bezier(.21,1.02,.73,1);
}
.toast.out{animation:toastOut .25s ease forwards}
.toast-red{border-left-color:var(--red);background:linear-gradient(90deg,var(--red-dim),var(--surface) 45%)}
.toast-amber{border-left-color:var(--amber);background:linear-gradient(90deg,var(--amber-dim),var(--surface) 45%)}
.toast-green{border-left-color:var(--green)}
@keyframes toastIn{from{transform:translateX(110%);opacity:0}to{transform:none;opacity:1}}
@keyframes toastOut{to{transform:translateX(110%);opacity:0}}
@keyframes alertPulse{0%,100%{box-shadow:0 0 0 0 rgba(240,92,110,.35)}50%{box-shadow:0 0 0 10px rgba(240,92,110,0)}}
.stream-card.alerted{border-color:var(--red);animation:alertPulse 1.4s ease-out infinite}
.stream-card.alerted .stream-header{background:var(--red-dim)}
.stat-card.red.has-alerts{animation:alertPulse 2.2s ease-out infinite}

/* ── Skeleton de carga de streams ────────────────────────── */
.stream-img-wrap::before{
  content:\'\';position:absolute;inset:0;z-index:1;
  background:linear-gradient(110deg,#0c1220 8%,#16202f 18%,#0c1220 33%);
  background-size:200% 100%;animation:shimmer 1.4s linear infinite;
  opacity:1;transition:opacity .5s;pointer-events:none;
}
.stream-img-wrap.loaded::before{opacity:0}
@keyframes shimmer{to{background-position-x:-200%}}
.badge-warn{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(245,158,11,.3)}

/* ── Accesibilidad: foco por teclado ─────────────────────── */
.tab-btn:focus-visible,.filter-btn:focus-visible,.btn:focus-visible,
.mic-btn:focus-visible,#theme-btn:focus-visible{outline:2px solid var(--blue);outline-offset:2px}

/* ══ Acabado visual ═══════════════════════════════════════ */

/* Panel "maquinado": filo de luz arriba + sombra flotante */
.panel,.stat-card,.tegra-card,.stream-card,.tab-nav,.person-card,.clip-item,.transcript-item{
  box-shadow:inset 0 1px 0 rgba(255,255,255,.035), 0 10px 28px rgba(0,0,0,.28);
}
body.light .panel,body.light .stat-card,body.light .tegra-card,body.light .stream-card,
body.light .tab-nav,body.light .person-card,body.light .clip-item,body.light .transcript-item{
  box-shadow:inset 0 1px 0 rgba(255,255,255,.8), 0 6px 18px rgba(15,28,46,.08);
}

/* Esquinas de visor en los streams — firma visual del producto */
.vf{position:absolute;width:20px;height:20px;z-index:2;pointer-events:none;
  border:0 solid rgba(255,255,255,.5);opacity:.9;transition:border-color .3s}
.vf-tl{top:10px;left:10px;border-top-width:2px;border-left-width:2px;border-top-left-radius:3px}
.vf-tr{top:10px;right:10px;border-top-width:2px;border-right-width:2px;border-top-right-radius:3px}
.vf-bl{bottom:10px;left:10px;border-bottom-width:2px;border-left-width:2px;border-bottom-left-radius:3px}
.vf-br{bottom:10px;right:10px;border-bottom-width:2px;border-right-width:2px;border-bottom-right-radius:3px}
.stream-card.alerted .vf{border-color:var(--red);opacity:1}
.stream-card:hover .vf{border-color:var(--blue)}

/* Stat cards: valor protagonista, icono fantasma al fondo */
.stat-card{min-height:92px;justify-content:flex-end}
.stat-card .stat-icon{position:absolute;top:8px;right:12px;font-size:2.1em;opacity:.13;line-height:1}
.stat-card .stat-val{font-family:\'JetBrains Mono\',monospace;font-size:1.85em;letter-spacing:-.5px;font-variant-numeric:tabular-nums}

/* Marcador de capítulo en títulos de sección */
.section-title::before{content:\'\';width:3px;height:12px;border-radius:2px;background:var(--blue);flex-shrink:0}

/* Botones con peso físico */
.btn{box-shadow:0 2px 10px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.12);
  letter-spacing:.2px;transition:filter .15s,transform .15s,box-shadow .15s}
.btn:hover{filter:brightness(1.08);transform:translateY(-1px);box-shadow:0 5px 16px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.12)}
.btn:active{transform:translateY(0)}
.tegra-card:hover{transform:translateY(-1px)}
.tegra-card{transition:border-color .2s,transform .15s}

/* Datos en monoespaciada — lectura de telemetría */
.clock,.hval,.tegra-value,.stream-title,.stream-footer,.clip-time,.transcript-ts,.person-id{
  font-family:\'JetBrains Mono\',monospace;font-variant-numeric:tabular-nums;
}
.stream-title{font-size:.72em;letter-spacing:.8px}

/* Entrada escalonada del contenido */
@keyframes fadeUp{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:none}}
.tab-content.active>*{animation:fadeUp .4s cubic-bezier(.21,.9,.35,1) both}
.tab-content.active>*:nth-child(2){animation-delay:.05s}
.tab-content.active>*:nth-child(3){animation-delay:.1s}
.tab-content.active>*:nth-child(4){animation-delay:.15s}
.tab-content.active>*:nth-child(5){animation-delay:.2s}
.tab-content.active>*:nth-child(6){animation-delay:.25s}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}
}

/* Firma de pie de página */
.footer-sig{
  text-align:center;padding:20px 0 28px;font-size:.62em;
  letter-spacing:3px;text-transform:uppercase;color:var(--muted);opacity:.55;
  font-family:\'JetBrains Mono\',monospace;
}

/* ── Responsive ──────────────────────────────────────────── */
@media(max-width:768px){
  .streams-grid,.bottom-row{grid-template-columns:1fr}
  .page{padding:14px 14px}
  .header{padding:12px 14px}
}

/* ── Tabs ────────────────────────────────────────────────── */
.tab-nav{
  display:flex;gap:4px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:4px;
  align-self:flex-start;
}
.tab-btn{
  background:none;border:none;color:var(--muted);
  padding:8px 18px;border-radius:8px;cursor:pointer;
  font-size:.8em;font-weight:600;font-family:inherit;
  transition:background .15s,color .15s;white-space:nowrap;
}
.tab-btn:hover{background:var(--surface2);color:var(--text)}
.tab-btn.active{background:var(--surface2);color:var(--text);border:1px solid var(--border)}
.tab-content{display:none;flex-direction:column;gap:18px}
.tab-content.active{display:flex}

/* ── Persons grid ────────────────────────────────────────── */
.persons-toolbar{
  display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:10px;
}
.filter-btns{display:flex;gap:4px}
.filter-btn{
  background:var(--surface2);border:1px solid var(--border);
  color:var(--muted);padding:6px 14px;border-radius:20px;
  cursor:pointer;font-size:.75em;font-weight:600;font-family:inherit;
  transition:all .15s;
}
.filter-btn:hover{border-color:var(--border2);color:var(--text)}
.filter-btn.active{background:var(--blue-dim);border-color:var(--blue);color:var(--blue)}
.filter-btn.active-green{background:var(--green-dim);border-color:var(--green);color:var(--green)}
.filter-btn.active-amber{background:var(--amber-dim);border-color:var(--amber);color:var(--amber)}
.persons-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
  gap:12px;
}
.person-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);overflow:hidden;
  cursor:pointer;transition:border-color .2s,transform .1s;
  display:flex;flex-direction:column;
}
.person-card:hover{border-color:var(--border2);transform:translateY(-2px)}
.person-card.known{border-top:2px solid var(--green)}
.person-card.new-person{border-top:2px solid var(--amber)}
.person-photo-wrap{
  background:var(--surface2);aspect-ratio:1;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
}
.person-photo{width:100%;height:100%;object-fit:cover;display:block}
.person-no-photo{font-size:2.5em;opacity:.3}
.person-info{padding:10px;display:flex;flex-direction:column;gap:4px}
.person-id{font-size:.85em;font-weight:700;color:var(--text);letter-spacing:.5px}
.person-status{
  display:inline-flex;align-items:center;gap:4px;
  font-size:.65em;font-weight:600;padding:2px 8px;border-radius:10px;
  width:fit-content;
}
.status-known{background:var(--green-dim);color:var(--green);border:1px solid rgba(45,201,126,.3)}
.status-new{background:var(--amber-dim);color:var(--amber);border:1px solid rgba(245,158,11,.3)}
.person-meta{font-size:.68em;color:var(--muted);display:flex;flex-direction:column;gap:2px;margin-top:2px}
.person-meta span{display:flex;align-items:center;gap:4px}
.persons-empty{text-align:center;color:var(--muted);padding:40px;font-size:.9em;grid-column:1/-1}

/* ── Audio tab ───────────────────────────────────────────── */
.transcript-item{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius-sm);padding:12px;
  display:flex;flex-direction:column;gap:8px;
}
.transcript-header{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.transcript-ts{font-size:.75em;font-weight:600;color:var(--blue)}
.transcript-text{font-size:.8em;color:var(--text);line-height:1.6;white-space:pre-wrap}
.transcript-audio{width:100%;height:24px;margin-top:4px;opacity:.8}
.transcript-empty{color:var(--muted);font-size:.9em;text-align:center;padding:40px}
.transcript-corr{
  font-size:.7em;color:var(--muted);border-top:1px solid var(--border);
  padding-top:6px;display:flex;gap:6px;flex-wrap:wrap;
}
.corr-tag{
  background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:2px 8px;font-size:.9em;
}
</style>
</head>
<body>

<header class="header">
  <div class="header-left">
    <div class="subtitle">Jetson Orin Nano &middot; YOLO TRT &middot; InsightFace</div>
  </div>
  <div class="header-center">
    <div class="logo-pill"><img src="/logo" alt="OXMtech" class="oxm-logo"></div>
  </div>
  <div class="header-right">
    <button id="theme-btn" onclick="toggleTheme()" title="Cambiar tema" style="background:none;border:1px solid var(--border);color:var(--muted);border-radius:8px;padding:5px 10px;cursor:pointer;font-size:.85em;transition:all .2s">&#9790;</button>
    <div class="clock" id="clock">--:--:--</div>
  </div>
</header>

<div class="page">

  <!-- Health chips -->
  <div class="health-row" id="health-bar"></div>

  <!-- Tab navigation -->
  <div class="tab-nav">
    <button class="tab-btn active" id="tbtn-cams" onclick="switchTab(\'cams\')">&#128247; C&#225;maras</button>
    <button class="tab-btn" id="tbtn-personas" onclick="switchTab(\'personas\')">&#128100; Personas</button>
    <button class="tab-btn" id="tbtn-audio" onclick="switchTab(\'audio\')">&#127908; Audio</button>
  </div>

  <!-- ╔══ Tab: Cámaras ══╗ -->
  <div id="tab-cams" class="tab-content active">

    <!-- Streams -->
    <div class="streams-grid">
      <div class="stream-card">
        <div class="stream-header">
          <span class="stream-title">&#128247; CSI &mdash; IMX219</span>
          <span class="stream-badge badge-live" id="status-csi"><span class="dot dot-ok dot-pulse"></span> LIVE</span>
        </div>
        <div class="stream-img-wrap">
          <img id="img-csi" alt="CSI stream">
          <div class="vf vf-tl"></div><div class="vf vf-tr"></div><div class="vf vf-bl"></div><div class="vf vf-br"></div>
        </div>
        <div class="stream-footer"><span id="fps-csi">-- fps</span><span>640&times;360</span></div>
      </div>
      <div class="stream-card">
        <div class="stream-header">
          <span class="stream-title">&#128249; USB &mdash; Webcam</span>
          <span class="stream-badge badge-live" id="status-usb"><span class="dot dot-ok dot-pulse"></span> LIVE</span>
        </div>
        <div class="stream-img-wrap">
          <img id="img-usb" alt="USB stream">
          <div class="vf vf-tl"></div><div class="vf vf-tr"></div><div class="vf vf-bl"></div><div class="vf vf-br"></div>
        </div>
        <div class="stream-footer"><span id="fps-usb">-- fps</span><span>640&times;360</span></div>
      </div>
    </div>

    <!-- Stat cards -->
    <div class="cards-grid" id="cards"></div>

    <!-- Reportes -->
    <div class="panel">
      <div class="actions-row">
        <div class="section-title" style="margin-bottom:0;flex:1">&#128202; Reportes</div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <span class="last-update" id="refresh"></span>
          <button class="btn btn-indigo" id="report-btn" onclick="sendReport()">&#128202; Reporte &rarr; Telegram</button>
          <a class="btn btn-green" id="pdf-btn" href="/api/report/pdf" download style="text-decoration:none">&#128196; Descargar PDF</a>
        </div>
      </div>
    </div>

    <!-- Chat -->
    <div class="panel">
      <div class="section-title">&#129302; Consulta inteligente</div>
      <div class="chat-messages" id="chat-messages"></div>
      <div class="chat-form">
        <input type="text" id="chat-input" placeholder="Ej: ¿Cuántas personas se detectaron hoy?" />
        <button class="btn btn-blue" id="chat-btn" onclick="sendChat()">Enviar</button>
      </div>
    </div>

    <!-- Telemetría hardware -->
    <div class="tegra-row" id="tegra-row"></div>

    <!-- Objects + Clips -->
    <div class="bottom-row">
      <div class="panel">
        <div class="section-title">&#128065; Objetos detectados</div>
        <div class="obj-list" id="objects"></div>
      </div>
      <div class="panel">
        <div class="section-title">&#127908; Clips de audio</div>
        <ul class="clips-list" id="clips"></ul>
      </div>
    </div>

    <!-- Timeline chart -->
    <div class="panel">
      <div class="section-title">&#128200; Actividad por minuto</div>
      <div class="chart-wrap">
        <canvas id="timeline" height="70"></canvas>
      </div>
    </div>

  </div><!-- /tab-cams -->

  <!-- ╔══ Tab: Personas ══╗ -->
  <div id="tab-personas" class="tab-content">

    <div class="persons-toolbar">
      <div class="filter-btns">
        <button class="filter-btn active" id="fb-all"   onclick="filterPersons(\'all\')">Todos</button>
        <button class="filter-btn"        id="fb-known" onclick="filterPersons(\'known\')">&#9989; Conocidos</button>
        <button class="filter-btn"        id="fb-new"   onclick="filterPersons(\'new\')">&#10024; Nuevos</button>
      </div>
      <span id="persons-count" style="color:var(--muted);font-size:.8em;font-weight:500"></span>
    </div>

    <div id="persons-grid" class="persons-grid">
      <div class="persons-empty">Cargando personas...</div>
    </div>

  </div><!-- /tab-personas -->

  <!-- ╔══ Tab: Audio ══╗ -->
  <div id="tab-audio" class="tab-content">

    <div class="panel">
      <div class="section-title">&#127908; Clips &amp; Transcripciones</div>
      <div id="transcripts-list" style="display:flex;flex-direction:column;gap:10px;max-height:700px;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent">
        <div class="transcript-empty">Cargando clips...</div>
      </div>
    </div>

  </div><!-- /tab-audio -->

</div><!-- /page -->

<div class="footer-sig">OXM Vision &middot; Edge AI Surveillance &middot; Jetson Orin Nano</div>

<div class="toast-stack" id="toast-stack" aria-live="polite"></div>

<script>
// ── Clock ──────────────────────────────────────────────────
function tickClock(){
  const now = new Date();
  document.getElementById(\'clock\').textContent =
    now.toLocaleTimeString(\'es-MX\',{hour12:false});
}
tickClock(); setInterval(tickClock,1000);

// ── Toasts (alertas en vivo) ───────────────────────────────
function showToast(msg, type, ms){
  const stack=document.getElementById(\'toast-stack\');
  if(!stack) return;
  const t=document.createElement(\'div\');
  t.className=\'toast\'+(type?\' toast-\'+type:\'\');
  t.textContent=msg;
  stack.appendChild(t);
  setTimeout(()=>{t.classList.add(\'out\');setTimeout(()=>t.remove(),260)},ms||6000);
}

// ── Streams ────────────────────────────────────────────────
function setupStream(imgId, baseSrc) {
  const img = document.getElementById(imgId);
  const cam = imgId.replace(\'img-\',\'\');
  let retryDelay = 2000;
  img.onerror = () => {
    img.parentElement.classList.remove(\'loaded\');
    const b=document.getElementById(\'status-\'+cam);
    if(b){b.innerHTML=\'<span class="dot dot-warn dot-pulse"></span> RECONECTANDO\';b.className=\'stream-badge badge-warn\';}
    setTimeout(() => {
      img.src = baseSrc + \'?t=\' + Date.now();
      retryDelay = Math.min(retryDelay * 1.5, 10000);
    }, retryDelay);
  };
  img.onload = () => { retryDelay = 2000; img.parentElement.classList.add(\'loaded\'); };
  img.src = baseSrc;
  // Salvavidas: algunos navegadores no disparan onload en streams MJPEG multipart
  // tras el primer frame. Sin esto el skeleton se queda tapando el video indefinidamente.
  setTimeout(() => { img.parentElement.classList.add(\'loaded\'); }, 4000);
}
setupStream(\'img-csi\', \'/stream/csi\');
setupStream(\'img-usb\', \'/stream/usb\');

// ── Health ─────────────────────────────────────────────────
function setStreamBadge(statusId, up, fps) {
  const el = document.getElementById(statusId);
  if (!el) return;
  if (up && fps != null && fps < 8) {
    el.innerHTML = \'<span class="dot dot-warn dot-pulse"></span> DEGRADADO\';
    el.className = \'stream-badge badge-warn\';
  } else if (up) {
    el.innerHTML = \'<span class="dot dot-ok dot-pulse"></span> LIVE\';
    el.className = \'stream-badge badge-live\';
  } else {
    el.innerHTML = \'<span class="dot dot-err"></span> SIN SEÑAL\';
    el.className = \'stream-badge badge-err\';
  }
}
let _camPrev = {};
async function updateHealth() {
  try {
    const h = await fetch(\'/api/health\').then(r=>r.json());
    const chip = (dot,label,val,extra=\'\') =>
      `<div class="hchip"><div class="dot ${dot}"></div><span class="hlbl">${label}</span><span class="hval">${val}</span>${extra}</div>`;
    const camDot = c => c.up ? \'dot-ok\' : \'dot-err\';
    const camVal = c => c.up ? `${c.fps} fps` : \'sin señal\';
    document.getElementById(\'health-bar\').innerHTML =
      chip(camDot(h.cameras.csi),\'CSI\',camVal(h.cameras.csi)) +
      chip(camDot(h.cameras.usb),\'USB\',camVal(h.cameras.usb)) +
      chip(h.audio.muted?\'dot-warn\':(h.audio.up?\'dot-ok\':\'dot-err\'),\'Audio\',h.audio.muted?\'silenciado\':(h.audio.up?(h.audio.label||\'OK\'):\'sin audio\'),`<button class="mic-btn" onclick="toggleMic()" title="${h.audio.muted?\'Activar micrófono\':\'Silenciar micrófono\'}">${h.audio.muted?\'🔇\':\'🎤\'}</button>`) +
      chip(h.yolo.up?\'dot-ok\':\'dot-warn\',\'YOLO\',h.yolo.up?\'OK\':\'iniciando\') +
      chip(h.reid.up?\'dot-ok\':\'dot-warn\',\'FaceReID\',h.reid.up?\'OK\':\'iniciando\');
    setStreamBadge(\'status-csi\', h.cameras.csi.up, h.cameras.csi.fps);
    setStreamBadge(\'status-usb\', h.cameras.usb.up, h.cameras.usb.fps);
    [[\'csi\',h.cameras.csi.up],[\'usb\',h.cameras.usb.up]].forEach(([c,up])=>{
      if(_camPrev[c]===true && !up) showToast(\'\\ud83d\\udcf7 C\\u00e1mara \'+c.toUpperCase()+\' sin se\\u00f1al\',\'amber\');
      if(_camPrev[c]===false && up) showToast(\'\\u2705 C\\u00e1mara \'+c.toUpperCase()+\' recuperada\',\'green\',4000);
      _camPrev[c]=up;
    });
    document.getElementById(\'fps-csi\').textContent = h.cameras.csi.up ? h.cameras.csi.fps+\' fps\' : \'sin señal\';
    document.getElementById(\'fps-usb\').textContent = h.cameras.usb.up ? h.cameras.usb.fps+\' fps\' : \'sin señal\';
  } catch(e) {}
}

async function toggleMic() {
  try {
    const r = await fetch(\'/api/audio/mute\', {method:\'POST\'}).then(r=>r.json());
    const btn = document.querySelector(\'.mic-btn\');
    if (btn) btn.textContent = r.muted ? \'🔇\' : \'🎤\';
    updateHealth();
  } catch(e) {}
}

// ── Telemetría GPU ─────────────────────────────────────────
function tempClass(t) {
  if (t == null) return \'\';
  return t < 60 ? \'temp-ok\' : t < 75 ? \'temp-warm\' : \'temp-hot\';
}
function tegraCard(icon, label, value, sub, barPct, barClass) {
  const pct = Math.min(100, Math.max(0, barPct || 0));
  return `<div class="tegra-card">
    <div class="tegra-header">
      <span class="tegra-label">${icon} ${label}</span>
      <span class="tegra-value">${value}</span>
    </div>
    <div class="tegra-sub">${sub}</div>
    <div class="tegra-bar-bg"><div class="tegra-bar-fill ${barClass}" style="width:${pct}%"></div></div>
  </div>`;
}
async function updateTegra() {
  try {
    const t = await fetch(\'/api/tegra\').then(r => r.json());
    const gpuLoad  = t.gpu_load  != null ? t.gpu_load + \'%\'  : \'--\';
    const gpuTemp  = t.gpu_temp  != null ? t.gpu_temp + \'°C\' : \'--\';
    const cpuTemp  = t.cpu_temp  != null ? t.cpu_temp + \'°C\' : \'--\';
    const ramUsed  = t.ram_used_mb  != null ? (t.ram_used_mb/1024).toFixed(1)  + \' GB\' : \'--\';
    const ramTotal = t.ram_total_mb != null ? (t.ram_total_mb/1024).toFixed(1) + \' GB\' : \'--\';
    const diskUsed = t.disk_used_gb != null ? t.disk_used_gb + \' GB\' : \'--\';
    const diskTotal= t.disk_total_gb!= null ? t.disk_total_gb+ \' GB\' : \'--\';
    const gpuMhz   = t.gpu_mhz != null ? t.gpu_mhz + \' MHz\' : \'\';
    const gpuBarClass = (t.gpu_load||0) > 80 ? \'bar-red\' : (t.gpu_load||0) > 50 ? \'bar-amber\' : \'bar-blue\';
    const ramBarClass = (t.ram_pct||0) > 85 ? \'bar-red\' : (t.ram_pct||0) > 65 ? \'bar-amber\' : \'bar-green\';
    const dskBarClass = (t.disk_pct||0) > 85 ? \'bar-red\' : (t.disk_pct||0) > 65 ? \'bar-amber\' : \'bar-green\';
    document.getElementById(\'tegra-row\').innerHTML =
      tegraCard(\'🎮\', \'GPU\', gpuLoad, gpuMhz, t.gpu_load, gpuBarClass) +
      tegraCard(\'🌡\', \'GPU Temp\', `<span class="${tempClass(t.gpu_temp)}">${gpuTemp}</span>`,
                `CPU ${cpuTemp}`, (t.gpu_temp||0)/100*100, t.gpu_temp>75?\'bar-red\':t.gpu_temp>60?\'bar-amber\':\'bar-blue\') +
      tegraCard(\'💾\', \'RAM\', ramUsed, `de ${ramTotal} · ${t.ram_pct||\'--\'}%`, t.ram_pct, ramBarClass) +
      tegraCard(\'💿\', \'NVMe\', diskUsed, `de ${diskTotal} · ${t.disk_pct||\'--\'}%`, t.disk_pct, dskBarClass);
  } catch(e) {}
}

// ── Stats ──────────────────────────────────────────────────
let chart;
const CARD_DEFS = [
  {key:(d)=>d.persons_csi+d.persons_usb, lbl:\'Personas totales\', icon:\'&#128100;\', cls:\'\'},
  {key:(d)=>d.persons_csi,               lbl:\'Personas CSI\',     icon:\'&#128247;\', cls:\'\'},
  {key:(d)=>d.persons_usb,               lbl:\'Personas USB\',     icon:\'&#128249;\', cls:\'\'},
  {key:(d)=>d.known_faces,               lbl:\'Caras conocidas\',  icon:\'&#128064;\', cls:\'purple\'},
  {key:(d)=>d.new_faces,                 lbl:\'Caras nuevas\',     icon:\'&#10024;\',  cls:\'purple\'},
  {key:(d)=>d.speech_seconds+\'s\',        lbl:\'Voz detectada\',    icon:\'&#127908;\', cls:\'green\'},
  {key:(d)=>d.speech_ratio+\'%\',          lbl:\'Voz c/ persona\',   icon:\'&#127897;\', cls:\'green\'},
  {key:(d)=>d.alerts,                    lbl:\'Alertas\',          icon:\'&#9888;\',   cls:\'red\'},
];
let _lastAlerts = null;
async function update(){
  try {
    const d = await fetch(\'/api/stats\').then(r=>r.json());
    if(_lastAlerts!==null && d.alerts>_lastAlerts){
      showToast(\'\\u26a0\\ufe0f Nueva alerta detectada \\u2014 \'+d.alerts+\' hoy\',\'red\',8000);
      [\'csi\',\'usb\'].forEach(c=>{
        const img=document.getElementById(\'img-\'+c);
        const card=img&&img.closest(\'.stream-card\');
        if(card){card.classList.add(\'alerted\');setTimeout(()=>card.classList.remove(\'alerted\'),15000);}
      });
    }
    _lastAlerts = d.alerts;
    document.getElementById(\'cards\').innerHTML = CARD_DEFS.map(def=>`
      <div class="stat-card ${def.cls}${def.cls===\'red\'&&d.alerts>0?\' has-alerts\':\'\'}">
        <div class="stat-icon">${def.icon}</div>
        <div class="stat-val">${def.key(d)}</div>
        <div class="stat-lbl">${def.lbl}</div>
      </div>`).join(\'\');

    const maxObj = d.top_objects[0]?.[1]||1;
    document.getElementById(\'objects\').innerHTML = d.top_objects.map(([cls,cnt])=>`
      <div class="obj-item">
        <span class="obj-name">${cls}</span>
        <div class="obj-track"><div class="obj-fill" style="width:${Math.round(cnt/maxObj*100)}%"></div></div>
        <span class="obj-cnt">${cnt.toLocaleString()}</span>
      </div>`).join(\'\');

    document.getElementById(\'refresh\').textContent = \'Act. \' + new Date(d.last_update).toLocaleTimeString(\'es-MX\',{hour12:false});

    const labels = d.timeline_labels.map(l=>l.slice(11,16));
    if(chart){
      chart.data.labels = labels;
      chart.data.datasets[0].data = d.timeline_persons;
      chart.data.datasets[1].data = d.timeline_speech;
      chart.update(\'none\');
    } else {
      chart = new Chart(document.getElementById(\'timeline\'),{
        type:\'line\',
        data:{
          labels,
          datasets:[
            {label:\'Personas\',data:d.timeline_persons,
             borderColor:\'#45b3ff\',backgroundColor:\'rgba(69,179,255,.08)\',
             tension:.35,fill:true,pointRadius:2,borderWidth:2},
            {label:\'Voz (seg)\',data:d.timeline_speech,
             borderColor:\'#2dc97e\',backgroundColor:\'rgba(45,201,126,.08)\',
             tension:.35,fill:true,pointRadius:2,borderWidth:2}
          ]
        },
        options:{
          animation:{duration:300},
          plugins:{
            legend:{labels:{color:\'#7d93b8\',font:{size:11},boxWidth:12}},
            tooltip:{backgroundColor:\'#0e1420\',borderColor:\'#1e2a3a\',borderWidth:1,
              titleColor:\'#e2eaf5\',bodyColor:\'#94b8d8\'}
          },
          scales:{
            x:{ticks:{color:\'#7d93b8\',maxTicksLimit:10,font:{size:10}},
               grid:{color:\'rgba(30,42,58,.6)\'}},
            y:{ticks:{color:\'#7d93b8\',font:{size:10}},
               grid:{color:\'rgba(30,42,58,.6)\'}}
          }
        }
      });
    }
  } catch(e) {}
}

// ── Clips ──────────────────────────────────────────────────
async function loadClips(){
  try {
    const clips = await fetch(\'/api/audio/list?limit=60\').then(r=>r.json());
    const groups={};
    clips.forEach(c=>{ const b=c.bucket||\'Sin fecha\'; (groups[b]=groups[b]||[]).push(c); });
    let html=\'\';
    Object.keys(groups).sort().reverse().forEach(bucket=>{
      const hh=bucket.slice(11,16);
      const [hh2,mm]=hh.split(\':\').map(Number);
      const mEnd=mm+4;
      html+=`<div class="clip-group-hdr">&#128337; ${hh} &ndash; ${hh2}:${String(mEnd).padStart(2,\'0\')} <span style="margin-left:auto;opacity:.6">${groups[bucket].length} clips</span></div>`;
      groups[bucket].forEach(c=>{
        const time=c.ts?c.ts.slice(11,19):\'\';
        const lbl=c.label?`<span class="clip-tag">&#127908; ${c.label}</span>`:\'\';
        const spk=c.n_speakers!=null?`<span class="clip-tag">&#128101; ${c.n_speakers} hlbt${c.n_speakers!==1?\'s\':\'\'}</span>`:\'\';
        const tx=c.transcript?`<div class="clip-transcript">&#128172; ${c.transcript}</div>`:\'\';
        html+=`<li class="clip-item">
          <div class="clip-meta"><span class="clip-time">&#128266; ${time}</span>${lbl}${spk}</div>
          ${tx}
          <audio controls preload="none"><source src="/audio/${c.file}" type="audio/ogg"></audio>
        </li>`;
      });
    });
    document.getElementById(\'clips\').innerHTML=html;
  } catch(e) {}
}

// ── Chat ───────────────────────────────────────────────────
document.getElementById(\'chat-input\').addEventListener(\'keydown\',e=>{ if(e.key===\'Enter\') sendChat(); });
function toggleTheme(){
  const light=document.body.classList.toggle(\'light\');
  localStorage.setItem(\'theme\',light?\'light\':\'dark\');
  document.getElementById(\'theme-btn\').textContent=light?\'\u2600\':\'\u263e\';
}
(function(){
  const stored=localStorage.getItem(\'theme\');
  const b=document.getElementById(\'theme-btn\');
  if(stored==='light'){
    document.body.classList.add(\'light\');
    if(b) b.textContent=\'\u2600\';
  } else if(!stored){
    // Sin preferencia guardada: tema claro de 7am a 7pm
    const h=new Date().getHours();
    if(h>=7&&h<19){ document.body.classList.add(\'light\'); if(b) b.textContent=\'\u2600\'; }
  }
})();
async function sendChat(){
  const input=document.getElementById(\'chat-input\');
  const btn=document.getElementById(\'chat-btn\');
  const msgs=document.getElementById(\'chat-messages\');
  const q=input.value.trim(); if(!q) return;
  input.value=\'\'; btn.disabled=true;
  const um=document.createElement(\'div\'); um.className=\'msg-user\'; um.textContent=q; msgs.appendChild(um);
  const lm=document.createElement(\'div\'); lm.className=\'msg-loading\'; lm.textContent=\'Analizando datos…\'; msgs.appendChild(lm);
  msgs.scrollTop=msgs.scrollHeight;
  try {
    const ctrl=new AbortController(); const tid=setTimeout(()=>ctrl.abort(),70000);
    const res=await fetch(\'/api/chat\',{method:\'POST\',headers:{\'Content-Type\':\'application/json\'},body:JSON.stringify({question:q}),signal:ctrl.signal}).then(r=>r.json());
    clearTimeout(tid); lm.remove();
    const am=document.createElement(\'div\'); am.className=\'msg-ai\'; am.textContent=res.answer||\'Sin respuesta.\'; msgs.appendChild(am);
  } catch(e){ lm.textContent=e.name===\'AbortError\'?\'Tiempo de espera agotado (70s).\': \'Error al consultar.\'; }
  msgs.scrollTop=msgs.scrollHeight; btn.disabled=false;
}

// ── Report ─────────────────────────────────────────────────
async function sendReport(){
  const btn=document.getElementById(\'report-btn\');
  btn.disabled=true; btn.textContent=\'⏳ Generando…\';
  try {
    await fetch(\'/api/report/now\',{method:\'POST\'});
    btn.textContent=\'✅ Enviado a Telegram\';
  } catch(e){ btn.textContent=\'❌ Error\'; }
  setTimeout(()=>{ btn.disabled=false; btn.innerHTML=\'&#128202; Reporte &rarr; Telegram\'; },5000);
}

// ── Tab switching ──────────────────────────────────────────
function switchTab(tab) {
  [\'cams\',\'personas\',\'audio\'].forEach(t => {
    document.getElementById(\'tab-\'+t).classList.toggle(\'active\', t===tab);
    document.getElementById(\'tbtn-\'+t).classList.toggle(\'active\', t===tab);
  });
  if (tab === \'personas\') loadPersons();
  if (tab === \'audio\')    loadTranscripts();
}

// ── Persons tab ────────────────────────────────────────────
let _allPersons = [];
let _personFilter = \'all\';

async function loadPersons() {
  try {
    const data = await fetch(\'/api/persons\').then(r=>r.json());
    _allPersons = data.persons || [];
    renderPersons();
  } catch(e) {
    document.getElementById(\'persons-grid\').innerHTML = \'<div class="persons-empty">Error cargando personas</div>\';
  }
}

function filterPersons(f) {
  _personFilter = f;
  [\'all\',\'known\',\'new\'].forEach(x => {
    const btn = document.getElementById(\'fb-\'+x);
    btn.classList.remove(\'active\',\'active-green\',\'active-amber\');
  });
  if (f===\'all\')   document.getElementById(\'fb-all\').classList.add(\'active\');
  if (f===\'known\') document.getElementById(\'fb-known\').classList.add(\'active-green\');
  if (f===\'new\')   document.getElementById(\'fb-new\').classList.add(\'active-amber\');
  renderPersons();
}

function renderPersons() {
  const grid = document.getElementById(\'persons-grid\');
  let list = _allPersons;
  if (_personFilter === \'known\') list = list.filter(p=>p.known);
  if (_personFilter === \'new\')   list = list.filter(p=>!p.known);

  document.getElementById(\'persons-count\').textContent =
    list.length + \' persona\' + (list.length!==1?\'s\':\'\') +
    \' · \' + _allPersons.length + \' total\';

  if (!list.length) {
    grid.innerHTML = \'<div class="persons-empty">Sin personas en este filtro</div>\';
    return;
  }

  grid.innerHTML = list.map(p => {
    const cls   = p.known ? \'known\' : \'new-person\';
    const badge = p.known
      ? \'<span class="person-status status-known">&#9989; CONOCIDO</span>\'
      : \'<span class="person-status status-new">&#10024; NUEVO</span>\';
    const photo = p.has_snap
      ? `<img class="person-photo" src="/face_snap/${p.uuid}" alt="ID ${p.uuid_short}" loading="lazy" onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">`
        + `<div class="person-no-photo" style="display:none">&#128100;</div>`
      : `<div class="person-no-photo">&#128100;</div>`;
    const cams = p.cams.map(c => c === \'csi\' ? \'📷CSI\' : \'🎥USB\').join(\' \');
    const sim  = p.similarity > 0 ? `<span>&#127919; ${(p.similarity*100).toFixed(0)}% sim</span>` : \'\';
    return `<div class="person-card ${cls}">
      <div class="person-photo-wrap">${photo}</div>
      <div class="person-info">
        <div class="person-id">ID: ${p.uuid_short}</div>
        ${badge}
        <div class="person-meta">
          <span>&#128336; ${p.last_seen.slice(11,19)}</span>
          <span>&#128247; ${cams}</span>
          ${sim}
          <span>&#128203; ${p.n_events} eventos</span>
        </div>
      </div>
    </div>`;
  }).join(\'\');
}

// ── Audio tab ──────────────────────────────────────────────
async function loadTranscripts() {
  try {
    const clips = await fetch(\'/api/audio/list?limit=80\').then(r=>r.json());
    const el = document.getElementById(\'transcripts-list\');
    if (!clips || !clips.length) {
      el.innerHTML = \'<div class="transcript-empty">Sin clips de audio grabados aún</div>\';
      return;
    }
    const withText = clips.filter(c=>c.transcript);
    const noText   = clips.filter(c=>!c.transcript);
    let html = \'\';
    if (withText.length) {
      withText.forEach(c => {
        const time  = c.ts ? c.ts.slice(11,19) : \'\';
        const date  = c.ts ? c.ts.slice(0,10)  : \'\';
        const lbl   = c.label ? `<span class="clip-tag">&#127908; ${c.label}</span>` : \'\';
        const spk   = c.n_speakers != null ? `<span class="clip-tag">&#128101; ${c.n_speakers} hlbt</span>` : \'\';
        html += `<div class="transcript-item">
          <div class="transcript-header">
            <span class="transcript-ts">&#128337; ${date} ${time}</span>
            ${lbl}${spk}
          </div>
          <div class="transcript-text">${c.transcript}</div>
          <audio class="transcript-audio" controls preload="none">
            <source src="/audio/${c.file}" type="audio/ogg">
          </audio>
        </div>`;
      });
    }
    if (noText.length) {
      html += `<div style="color:var(--muted);font-size:.75em;padding:8px 0">
        ${noText.length} clips sin transcribir aún (Whisper los procesará en orden)
      </div>`;
    }
    if (!html) {
      html = \'<div class="transcript-empty">Sin transcripciones disponibles</div>\';
    }
    el.innerHTML = html;
  } catch(e) {
    document.getElementById(\'transcripts-list\').innerHTML =
      \'<div class="transcript-empty">Error cargando transcripciones</div>\';
  }
}

// ── Init ───────────────────────────────────────────────────
update();       setInterval(update,5000);
updateHealth(); setInterval(updateHealth,4000);
updateTegra();  setInterval(updateTegra,3000);
loadClips();    setInterval(loadClips,30000);
</script>
</body></html>''', headers={
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
    })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
