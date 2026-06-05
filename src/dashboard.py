"""
dashboard.py — OXM Vision Dashboard (minimalista)
FastAPI + MJPEG + Tabs (Streams, Timeline, Audio, Personas, Reporte)
Puerto 8080 — Jetson Orin Nano JetPack R36.4
"""

import json, os, time, asyncio, threading, pickle
from datetime import datetime
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# ─── Paths ────────────────────────────────────────────────────────────────────
METADATA  = '/app/unified_metadata.jsonl'
AUDIO_DIR = '/app/audio_clips'
FACE_DB   = '/app/face_db.pkl'
LOGO      = '/app/logo-oxmtech.png'

app = FastAPI()

# ─── Estado compartido ────────────────────────────────────────────────────────
_frames        = {'csi': None, 'usb': None, 'combined': None}
_fps           = {'csi': 0.0, 'usb': 0.0}
_person_count  = {'csi': 0, 'usb': 0}
_alerts        = []
_alerts_lock   = threading.Lock()
_report_fn     = None
_reid_ref      = None

_frame_conds = {
    'csi':      threading.Condition(),
    'usb':      threading.Condition(),
    'combined': threading.Condition(),
}

_status = {
    'reid': False, 'yolo': False,
    'csi': False, 'usb': False,
    'audio': False, 'audio_label': '', 'audio_device': ''
}

# ─── API de estado ────────────────────────────────────────────────────────────
def set_reid_up():    _status['reid'] = True
def set_yolo_up():    _status['yolo'] = True
def set_cam_status(cam_id: str, ok: bool): _status[cam_id] = ok
def set_audio_status(ok: bool, label: str = '', device: str = ''):
    _status['audio'] = ok; _status['audio_label'] = label; _status['audio_device'] = device
def set_report_fn(fn):
    global _report_fn
    _report_fn = fn

def set_reid_ref(r):
    global _reid_ref
    _reid_ref = r

def update_frame(cam_id: str, jpeg_bytes: bytes):
    _frames[cam_id] = jpeg_bytes
    cond = _frame_conds.get(cam_id)
    if cond:
        with cond:
            cond.notify_all()

def update_fps(cam_id: str, fps: float):       _fps[cam_id] = fps
def update_person_count(cam_id: str, count: int): _person_count[cam_id] = count

def push_alert(event: dict):
    with _alerts_lock:
        _alerts.append(event)
        if len(_alerts) > 200:
            _alerts.pop(0)

# ─── Helpers ──────────────────────────────────────────────────────────────────
TAIL_LINES = 5000

def _tail_file(path: str, n: int):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(n * 150, size)
            f.seek(max(0, size - block))
            return f.read().decode('utf-8', errors='ignore').splitlines()[-n:]
    except Exception:
        return []

def _read_gpu_load() -> float:
    for path in ('/sys/devices/platform/bus@0/17000000.gpu/load', '/sys/devices/gpu.0/load'):
        try:
            v = int(open(path).read().strip())
            return round(v / 10, 1)
        except Exception:
            continue
    return -1.0

_cpu_prev = {'idle': 0, 'total': 0}
def _read_cpu_load() -> float:
    try:
        with open('/proc/stat') as f:
            line = f.readline()
        parts = [int(x) for x in line.split()[1:]]
        idle  = parts[3] + parts[4]
        total = sum(parts)
        d_idle  = idle  - _cpu_prev['idle']
        d_total = total - _cpu_prev['total']
        _cpu_prev['idle']  = idle
        _cpu_prev['total'] = total
        if d_total <= 0:
            return -1.0
        return round((1 - d_idle / d_total) * 100, 1)
    except Exception:
        return -1.0

def _read_ram_pct() -> float:
    try:
        with open('/proc/meminfo') as f:
            mem = {}
            for line in f:
                k, v = line.split(':', 1)
                mem[k.strip()] = int(v.strip().split()[0])
        total = mem.get('MemTotal', 1)
        avail = mem.get('MemAvailable', total)
        return round((1 - avail / total) * 100, 1)
    except Exception:
        return -1.0

def _read_temp() -> float:
    best = -1.0
    for p in ('/sys/class/thermal/thermal_zone0/temp',
              '/sys/class/thermal/thermal_zone1/temp',
              '/sys/class/thermal/thermal_zone2/temp'):
        try:
            v = int(open(p).read().strip())
            t = v / 1000.0
            if t > best:
                best = t
        except Exception:
            continue
    return round(best, 1)

def _load_data(tail: int = TAIL_LINES):
    detections, audio_events, face_events = [], [], []
    for line in _tail_file(METADATA, tail):
        try:
            r = json.loads(line)
            t = r.get('type')
            if t == 'detection':   detections.append(r)
            elif t == 'audio':     audio_events.append(r)
            elif t == 'face_reid': face_events.append(r)
        except Exception:
            pass
    return detections, audio_events, face_events

def _load_all_events(tail: int = TAIL_LINES):
    events = []
    for line in _tail_file(METADATA, tail):
        try:
            events.append(json.loads(line))
        except Exception:
            pass
    return events

def _read_face_db():
    try:
        with open(FACE_DB, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}

def _write_face_db(db):
    with open(FACE_DB, 'wb') as f:
        pickle.dump(db, f)

# ─── MJPEG ────────────────────────────────────────────────────────────────────
def _mjpeg_generator(cam: str):
    cond = _frame_conds.get(cam)
    while True:
        if cond:
            with cond:
                cond.wait(timeout=2.0)
        frame = _frames.get(cam)
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)

@app.get('/stream/{cam}')
def stream(cam: str):
    if cam not in ('csi', 'usb', 'combined'):
        raise HTTPException(status_code=404)
    return StreamingResponse(
        _mjpeg_generator(cam),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )

# ─── /logo.png ────────────────────────────────────────────────────────────────
@app.get('/logo.png')
def logo():
    if not os.path.exists(LOGO):
        raise HTTPException(status_code=404)
    return FileResponse(LOGO, media_type='image/png')

# ─── /api/status ──────────────────────────────────────────────────────────────
@app.get('/api/status')
def api_status():
    return JSONResponse(_status)

# ─── /api/stats ───────────────────────────────────────────────────────────────
@app.get('/api/stats')
def api_stats():
    det, aud, faces = _load_data()
    persons = [d for d in det if d.get('class') == 'person']
    speech  = [a for a in aud if a.get('speech')]
    alerts  = [a for a in aud if a.get('alert')]
    obj_count = Counter(d.get('class', '') for d in det)
    known     = [f for f in faces if not f.get('new')]
    new_faces = [f for f in faces if f.get('new')]

    speech_ts = set(a.get('timestamp', '')[:19] for a in speech)
    person_ts = set(d.get('timestamp', '')[:19] for d in persons)
    overlap   = len(speech_ts & person_ts)

    with _alerts_lock:
        recent_alerts = list(_alerts[-5:])

    return JSONResponse({
        'persons_csi_now':    _person_count['csi'],
        'persons_usb_now':    _person_count['usb'],
        'total_detections':   len(det),
        'total_audio':        len(aud),
        'speech_seconds':     len(speech),
        'alerts':             len(alerts),
        'speech_with_person': overlap,
        'top_objects':        obj_count.most_common(8),
        'known_faces':        len(set(f.get('person_uuid','') for f in known)),
        'new_faces':          len(set(f.get('person_uuid','') for f in new_faces)),
        'fps_csi':            round(_fps['csi'], 1),
        'fps_usb':            round(_fps['usb'], 1),
        'cpu_load':           _read_cpu_load(),
        'gpu_load':           _read_gpu_load(),
        'ram_pct':            _read_ram_pct(),
        'temp_c':             _read_temp(),
        'recent_alerts':      recent_alerts,
        'last_update':        datetime.now().isoformat(),
    })

# ─── /api/timeline ────────────────────────────────────────────────────────────
@app.get('/api/timeline')
def api_timeline(limit: int = 80, types: str = '', cam: str = '', person: str = ''):
    allowed_types = set(types.split(',')) if types else set()
    person_lower  = person.strip().lower()
    events = _load_all_events(TAIL_LINES)
    result = []
    for ev in reversed(events):
        t = ev.get('type', '')
        if allowed_types and t not in allowed_types:
            continue
        if cam and ev.get('cam', '') != cam:
            continue
        if person_lower:
            pid = str(ev.get('person_uuid', '')).lower()
            if person_lower not in pid:
                continue
        result.append(ev)
        if len(result) >= limit:
            break
    return JSONResponse(result)

# ─── /api/audio ───────────────────────────────────────────────────────────────
@app.get('/api/audio/list')
def api_audio_list(limit: int = 30):
    if not os.path.exists(AUDIO_DIR):
        return JSONResponse([])
    files = sorted(
        (f for f in os.listdir(AUDIO_DIR) if f.endswith('.opus')),
        reverse=True,
    )[:limit]
    result = []
    for fname in files:
        fpath = os.path.join(AUDIO_DIR, fname)
        size_kb = round(os.path.getsize(fpath) / 1024, 1)
        txt_path = fpath.replace('.opus', '.txt')
        transcript = ''
        if os.path.exists(txt_path):
            try:
                transcript = open(txt_path, encoding='utf-8', errors='ignore').read().strip()
            except Exception:
                pass
        spk_path = fpath.replace('.opus', '.spk')
        speakers = ''
        if os.path.exists(spk_path):
            try:
                speakers = open(spk_path, encoding='utf-8', errors='ignore').read().strip()
            except Exception:
                pass
        result.append({
            'filename':   fname,
            'transcript': transcript,
            'speakers':   speakers,
            'size_kb':    size_kb,
        })
    return JSONResponse(result)

@app.get('/api/audio/{filename}')
def api_audio_file(filename: str):
    if '..' in filename or '/' in filename:
        raise HTTPException(status_code=400)
    fpath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404)
    return FileResponse(fpath, media_type='audio/ogg; codecs=opus')

# ─── /api/faces/db ────────────────────────────────────────────────────────────
@app.get('/api/faces/db')
def api_faces_db():
    db = _read_face_db()
    result = []
    for fid in db:
        sid = str(fid)
        is_name = ' ' in sid or len(sid) > 10
        result.append({'id': sid, 'is_name': is_name})
    return JSONResponse(result)

@app.delete('/api/faces/db/{face_id}')
def api_faces_db_delete(face_id: str):
    db = _read_face_db()
    found = False
    for key in list(db.keys()):
        if str(key) == face_id:
            del db[key]
            found = True
            break
    if not found:
        raise HTTPException(status_code=404)
    try:
        _write_face_db(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error guardando: {e}')
    return JSONResponse({'ok': True, 'deleted': face_id})

class EnrollReq(BaseModel):
    name: str
    cam: str = 'usb'

@app.post('/api/faces/enroll')
def api_faces_enroll(req: EnrollReq):
    if _reid_ref is None:
        raise HTTPException(status_code=503, detail='FaceReID no inicializado')
    cam = req.cam if req.cam in ('csi', 'usb') else 'usb'
    import numpy as np, cv2
    raw = _frames.get(cam)
    if raw is None:
        raise HTTPException(status_code=503, detail=f'Sin frame de cámara {cam}')
    try:
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=503, detail='Frame inválido')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error decodificando frame: {e}')
    result = _reid_ref.enroll(req.name, frame)
    if not result.get('ok'):
        raise HTTPException(status_code=422, detail=result.get('error', 'Error en enroll'))
    return JSONResponse(result)

# ─── /api/report ──────────────────────────────────────────────────────────────
@app.get('/api/report/latest')
def api_report_latest():
    lines = _tail_file(METADATA, TAIL_LINES)
    for line in reversed(lines):
        try:
            r = json.loads(line)
            if r.get('type') == 'daily_report':
                return JSONResponse(r)
        except Exception:
            pass
    return JSONResponse({'type': 'daily_report', 'content': None, 'timestamp': None})

@app.post('/api/report/generate')
def api_report_generate():
    if _report_fn is None:
        raise HTTPException(status_code=503, detail='report_fn no configurado')
    threading.Thread(target=_report_fn, daemon=True).start()
    return JSONResponse({'ok': True, 'message': 'Generando reporte en background'})

@app.get('/report/print', response_class=HTMLResponse)
def report_print():
    lines = _tail_file(METADATA, TAIL_LINES)
    report_data = None
    for line in reversed(lines):
        try:
            r = json.loads(line)
            if r.get('type') == 'daily_report':
                report_data = r
                break
        except Exception:
            pass
    det, aud, faces = _load_data()
    obj_count = Counter(d.get('class', '') for d in det)
    known_set  = {f.get('person_uuid', '') for f in faces if not f.get('new')}
    new_set    = {f.get('person_uuid', '') for f in faces if f.get('new')}
    alerts_cnt = len([a for a in aud if a.get('alert')])
    speech_cnt = len([a for a in aud if a.get('speech')])
    ts_now     = datetime.now().strftime('%Y-%m-%d %H:%M')
    content    = (report_data or {}).get('content') or 'Sin reporte generado. Usa "Generar reporte" primero.'
    rpt_ts     = (report_data or {}).get('timestamp', '')
    top_objs   = ''.join(
        f'<tr><td>{cls}</td><td>{cnt}</td></tr>'
        for cls, cnt in obj_count.most_common(8)
    )
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>OXM Vision — Reporte</title>
<style>
  @media print {{ .no-print{{ display:none }} }}
  body{{ font-family: system-ui, sans-serif; margin: 2cm; color: #111; font-size: 12pt }}
  h1{{ font-size: 1.4em; margin-bottom: 4px }}
  h2{{ font-size: 1em; color: #444; margin: 1.2em 0 .4em }}
  .meta{{ color:#666; font-size:.85em; margin-bottom: 1em }}
  .kpis{{ display:flex; gap:24px; margin-bottom:1em; flex-wrap:wrap }}
  .kpi{{ background:#f5f5f5; padding:10px 18px; border-radius:6px; text-align:center }}
  .kpi .v{{ font-size:1.8em; font-weight:700 }}
  .kpi .k{{ font-size:.75em; color:#555; text-transform:uppercase }}
  table{{ border-collapse:collapse; width:100%; margin-bottom:1em }}
  th,td{{ border:1px solid #ddd; padding:6px 10px; text-align:left }}
  th{{ background:#f0f0f0; font-weight:600 }}
  .report{{ white-space:pre-wrap; background:#f9f9f9; padding:12px; border-radius:6px; font-size:.92em; line-height:1.6 }}
  .btn{{ background:#0070f3; color:#fff; border:none; padding:10px 20px; border-radius:5px; cursor:pointer; font-size:1em; margin-bottom:1em }}
</style>
</head>
<body>
<div class="no-print"><button class="btn" onclick="window.print()">Imprimir / Guardar PDF</button></div>
<h1>OXM Vision — Reporte de Videovigilancia</h1>
<div class="meta">Generado: {ts_now} | Reporte IA: {rpt_ts[:19] if rpt_ts else '—'}</div>
<div class="kpis">
  <div class="kpi"><div class="v">{len(det)}</div><div class="k">Detecciones</div></div>
  <div class="kpi"><div class="v">{len(aud)}</div><div class="k">Eventos audio</div></div>
  <div class="kpi"><div class="v">{len(known_set)}</div><div class="k">Caras conocidas</div></div>
  <div class="kpi"><div class="v">{len(new_set)}</div><div class="k">Caras nuevas</div></div>
  <div class="kpi"><div class="v">{alerts_cnt}</div><div class="k">Alertas audio</div></div>
  <div class="kpi"><div class="v">{speech_cnt}</div><div class="k">Eventos de voz</div></div>
</div>
<h2>Objetos más detectados (últimas 5 000 líneas)</h2>
<table><thead><tr><th>Clase</th><th>Conteo</th></tr></thead><tbody>{top_objs}</tbody></table>
<h2>Reporte IA</h2>
<div class="report">{content}</div>
</body></html>""")

# ─── /api/chat ────────────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    question: str

@app.post('/api/chat')
def api_chat(req: ChatReq):
    try:
        import anthropic as _anthropic
        lines = _tail_file(METADATA, 2000)
        det, aud, faces = [], [], []
        for line in lines:
            try:
                r = json.loads(line)
                t = r.get('type')
                if t == 'detection':   det.append(r)
                elif t == 'audio':     aud.append(r)
                elif t == 'face_reid': faces.append(r)
            except Exception:
                pass
        named_faces = sorted({f.get('person_uuid','') for f in faces
                              if ' ' in str(f.get('person_uuid','')) or len(str(f.get('person_uuid','')))>10})
        ctx = (
            f"Sistema: {len(det)} detecciones, {len(aud)} eventos audio, {len(faces)} face_reid.\n"
            f"Top clases: {Counter(d.get('class','') for d in det).most_common(5)}\n"
            f"Personas únicas: {len({f.get('person_uuid','') for f in faces})}\n"
            f"Nombrados detectados: {named_faces[:10]}\n"
            f"Última detección: {det[-1].get('timestamp','') if det else '-'}\n"
            f"Última etiqueta audio: {aud[-1].get('label','') if aud else '-'}"
        )
        client = _anthropic.Anthropic()
        resp = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            messages=[{
                'role': 'user',
                'content': (
                    'Eres un asistente de análisis de videovigilancia. Responde breve y directo.\n'
                    f'Datos en tiempo real:\n{ctx}\n\n'
                    f'Pregunta: {req.question}'
                ),
            }],
        )
        return JSONResponse({'answer': resp.content[0].text})
    except Exception as e:
        return JSONResponse({'answer': f'Error: {e}'}, status_code=500)

# ─── HTML DASHBOARD (minimalista) ─────────────────────────────────────────────
@app.get('/', response_class=HTMLResponse)
def dashboard():
    return _HTML

_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OXM Vision</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0b0f14; --surface:#11161d; --surface2:#161c25; --border:#1f2630;
  --text:#e6edf3; --muted:#8b949e; --accent:#3fb950; --accent2:#58a6ff;
  --warn:#d29922; --err:#f85149;
}
body{background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,sans-serif;min-height:100vh}

/* ── Header ── */
header{background:var(--surface);border-bottom:1px solid var(--border);padding:10px 20px;
  display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.brand img{height:36px;width:auto}
.brand h1{font-size:1.05em;font-weight:600;letter-spacing:.5px}
.brand .sub{color:var(--muted);font-size:.72em;margin-top:2px}

.metrics{display:flex;gap:18px;flex-wrap:wrap;margin-left:auto}
.metric{display:flex;flex-direction:column;align-items:center;min-width:62px}
.metric .lbl{font-size:.65em;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.metric .val{font-size:1.05em;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.bar{width:60px;height:3px;background:var(--surface2);border-radius:2px;overflow:hidden;margin-top:4px}
.bar > span{display:block;height:100%;background:var(--accent);transition:width .4s,background .2s}
.bar.warn > span{background:var(--warn)}
.bar.err > span{background:var(--err)}

.pills{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:10px;
  font-size:.7em;background:var(--surface2);color:var(--muted);border:1px solid var(--border)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--err)}
.pill.ok .dot{background:var(--accent)}
.pill.ok{color:var(--text)}

/* ── Tabs ── */
nav.tabs{background:var(--surface);border-bottom:1px solid var(--border);
  display:flex;gap:0;padding:0 12px;overflow-x:auto;scrollbar-width:none}
nav.tabs::-webkit-scrollbar{display:none}
.tab{background:none;border:none;border-bottom:2px solid transparent;color:var(--muted);
  cursor:pointer;padding:11px 16px;font-size:.85em;white-space:nowrap;transition:all .15s}
.tab:hover{color:var(--text)}
.tab.active{color:var(--accent2);border-bottom-color:var(--accent2)}

main{padding:18px}
.view{display:none}
.view.active{display:block}

/* ── Cards & grids ── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:14px}
.card h3{font-size:.8em;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;
  margin-bottom:10px;font-weight:500}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.kpi{background:var(--surface2);border-radius:6px;padding:12px;text-align:center}
.kpi .v{font-size:1.6em;font-weight:600;font-variant-numeric:tabular-nums}
.kpi .k{font-size:.7em;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-top:3px}

/* ── Streams ── */
.streams{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.streams{grid-template-columns:1fr}}
.stream{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.stream .hdr{padding:8px 12px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center}
.stream .hdr .name{font-size:.8em;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.stream .hdr .fps{font-size:.75em;color:var(--accent);font-variant-numeric:tabular-nums}
.stream img{display:block;width:100%;height:auto;background:#000}

/* ── Timeline ── */
.events{max-height:600px;overflow-y:auto}
.event{display:grid;grid-template-columns:80px 90px 60px 1fr;gap:10px;padding:8px 0;
  border-bottom:1px solid var(--border);font-size:.85em;align-items:center}
.event:last-child{border-bottom:none}
.event .t{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.85em}
.event .type{font-size:.7em;padding:2px 8px;border-radius:4px;text-align:center;text-transform:uppercase;letter-spacing:.3px}
.event .type.detection{background:#1f3a5c;color:#79b8ff}
.event .type.audio{background:#3a2f1f;color:#e3b341}
.event .type.face_reid{background:#1f3a2f;color:#7ee787}
.event .cam{color:var(--muted);font-size:.8em}

.filters{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filters select,.filters button{background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:5px;padding:6px 10px;font-size:.8em;cursor:pointer}
.filters button:hover{border-color:var(--accent2)}

/* ── Audio ── */
.audiolist .clip{padding:10px 0;border-bottom:1px solid var(--border)}
.audiolist .clip:last-child{border-bottom:none}
.audiolist .clip .top{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.audiolist .clip .name{font-size:.8em;color:var(--muted);font-family:ui-monospace,monospace}
.audiolist .clip .size{margin-left:auto;font-size:.7em;color:var(--muted)}
.audiolist audio{width:100%;height:32px}
.audiolist .txt{font-size:.85em;color:var(--text);margin-top:6px;line-height:1.5;
  background:var(--surface2);padding:8px 10px;border-radius:5px}
.audiolist .spk{font-size:.7em;color:var(--accent2);margin-top:4px}

/* ── Personas ── */
.faces{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.face{background:var(--surface2);border-radius:6px;padding:10px;display:flex;justify-content:space-between;align-items:center}
.face .id{font-family:ui-monospace,monospace;font-size:.85em;word-break:break-all}
.face.named .id{color:var(--accent);font-family:inherit;font-weight:600}
.face .del{background:none;border:none;color:var(--err);cursor:pointer;font-size:1.2em;padding:0 6px}
.face .del:hover{opacity:.7}

/* ── Enroll form ── */
.enroll-form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
.enroll-form input[type=text]{flex:1;min-width:160px;background:var(--surface2);color:var(--text);
  border:1px solid var(--border);border-radius:5px;padding:7px 10px;font-size:.85em;font-family:inherit}
.enroll-form input[type=text]:focus{outline:none;border-color:var(--accent)}
.enroll-form select{background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:5px;padding:7px 10px;font-size:.8em;cursor:pointer}
.enroll-msg{font-size:.8em;padding:6px 10px;border-radius:5px;margin-top:4px;display:none}
.enroll-msg.ok{background:#1f3a2f;color:#7ee787;display:block}
.enroll-msg.err{background:#3a1f1f;color:#f85149;display:block}

/* ── Filtro texto ── */
.filters input[type=text]{background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:5px;padding:6px 10px;font-size:.8em;font-family:inherit;min-width:140px}
.filters input[type=text]:focus{outline:none;border-color:var(--accent2)}

/* ── Reporte ── */
.report{white-space:pre-wrap;line-height:1.6;font-size:.9em;
  background:var(--surface2);padding:14px;border-radius:6px;max-height:600px;overflow-y:auto}
.report-meta{font-size:.75em;color:var(--muted)}
.btn-primary{background:var(--accent2);color:#fff;border:none;border-radius:5px;
  padding:8px 16px;font-size:.85em;cursor:pointer}
.btn-primary:hover{opacity:.9}
.btn-primary:disabled{opacity:.5;cursor:wait}

.empty{color:var(--muted);text-align:center;padding:30px;font-size:.85em}

/* ── Chat ── */
.chat{display:flex;flex-direction:column;height:340px}
.chat .log{flex:1;overflow-y:auto;padding:8px;background:var(--surface2);border-radius:6px;
  display:flex;flex-direction:column;gap:8px;margin-bottom:10px}
.chat .msg{max-width:88%;padding:8px 12px;border-radius:10px;font-size:.85em;line-height:1.45;
  white-space:pre-wrap;word-wrap:break-word}
.chat .msg.user{align-self:flex-end;background:var(--accent2);color:#fff}
.chat .msg.bot{align-self:flex-start;background:#1f2630;color:var(--text);border:1px solid var(--border)}
.chat .msg.sys{align-self:center;color:var(--muted);font-size:.75em;font-style:italic;background:none}
.chat .input{display:flex;gap:8px}
.chat input{flex:1;background:var(--surface2);color:var(--text);border:1px solid var(--border);
  border-radius:5px;padding:8px 10px;font-size:.85em;font-family:inherit}
.chat input:focus{outline:none;border-color:var(--accent2)}
.chat button{background:var(--accent2);color:#fff;border:none;border-radius:5px;
  padding:0 16px;font-size:.85em;cursor:pointer}
.chat button:disabled{opacity:.5;cursor:wait}
</style>
</head>
<body>

<header>
  <div class="brand">
    <img src="/logo.png" alt="OXMtech" onerror="this.style.display='none'">
    <div>
      <h1>OXM Vision</h1>
      <div class="sub">Jetson Orin Nano</div>
      <div class="pills" id="pills"></div>
    </div>
  </div>
  <div class="metrics">
    <div class="metric"><span class="lbl">CPU</span><span class="val" id="m-cpu">—</span><div class="bar" id="b-cpu"><span></span></div></div>
    <div class="metric"><span class="lbl">GPU</span><span class="val" id="m-gpu">—</span><div class="bar" id="b-gpu"><span></span></div></div>
    <div class="metric"><span class="lbl">RAM</span><span class="val" id="m-ram">—</span><div class="bar" id="b-ram"><span></span></div></div>
    <div class="metric"><span class="lbl">Temp</span><span class="val" id="m-temp">—</span><div class="bar" id="b-temp"><span></span></div></div>
    <div class="metric"><span class="lbl">FPS CSI</span><span class="val" id="m-fps-csi">—</span></div>
    <div class="metric"><span class="lbl">FPS USB</span><span class="val" id="m-fps-usb">—</span></div>
  </div>
</header>

<nav class="tabs">
  <button class="tab active" data-view="streams">Streams</button>
  <button class="tab" data-view="timeline">Timeline</button>
  <button class="tab" data-view="audio">Audio</button>
  <button class="tab" data-view="personas">Personas</button>
  <button class="tab" data-view="reporte">Reporte</button>
</nav>

<main>

<!-- STREAMS -->
<section class="view active" id="view-streams">
  <div class="card">
    <h3>Resumen en vivo</h3>
    <div class="kpis">
      <div class="kpi"><div class="v" id="k-csi">0</div><div class="k">Personas CSI</div></div>
      <div class="kpi"><div class="v" id="k-usb">0</div><div class="k">Personas USB</div></div>
      <div class="kpi"><div class="v" id="k-known">0</div><div class="k">Caras conocidas</div></div>
      <div class="kpi"><div class="v" id="k-new">0</div><div class="k">Caras nuevas</div></div>
      <div class="kpi"><div class="v" id="k-det">0</div><div class="k">Detecciones (5k)</div></div>
      <div class="kpi"><div class="v" id="k-aud">0</div><div class="k">Eventos audio (5k)</div></div>
    </div>
  </div>
  <div class="streams">
    <div class="stream">
      <div class="hdr"><span class="name">CSI · /dev/video0</span><span class="fps" id="fps-csi-pill">— fps</span></div>
      <img id="img-csi" src="/stream/csi" alt="CSI">
    </div>
    <div class="stream">
      <div class="hdr"><span class="name">USB · /dev/video1</span><span class="fps" id="fps-usb-pill">— fps</span></div>
      <img id="img-usb" src="/stream/usb" alt="USB">
    </div>
  </div>

  <div class="card" style="margin-top:14px">
    <h3>Asistente IA · pregunta sobre lo que está pasando</h3>
    <div class="chat">
      <div class="log" id="chat-log">
        <div class="msg sys">Pregunta sobre detecciones, audio o personas. Ej: "¿quién apareció hoy?", "¿hubo gritos?"</div>
      </div>
      <form class="input" id="chat-form">
        <input type="text" id="chat-input" placeholder="Escribe tu pregunta..." autocomplete="off">
        <button type="submit" id="chat-send">Enviar</button>
      </form>
    </div>
  </div>
</section>

<!-- TIMELINE -->
<section class="view" id="view-timeline">
  <div class="card">
    <h3>Eventos recientes</h3>
    <div class="filters">
      <select id="f-type">
        <option value="">Todos los tipos</option>
        <option value="detection">Detection</option>
        <option value="audio">Audio</option>
        <option value="face_reid">Face REID</option>
      </select>
      <select id="f-cam">
        <option value="">Ambas cámaras</option>
        <option value="csi">CSI</option>
        <option value="usb">USB</option>
      </select>
      <input type="text" id="f-person" placeholder="Buscar persona..." style="min-width:150px">
      <button onclick="loadTimeline()">Refrescar</button>
    </div>
    <div class="events" id="events"></div>
  </div>
</section>

<!-- AUDIO -->
<section class="view" id="view-audio">
  <div class="card">
    <h3>Clips de audio recientes</h3>
    <div class="audiolist" id="audiolist"></div>
  </div>
</section>

<!-- PERSONAS -->
<section class="view" id="view-personas">
  <div class="card">
    <h3>Registrar persona desde cámara</h3>
    <div class="enroll-form">
      <input type="text" id="enroll-name" placeholder="Nombre completo..." maxlength="60">
      <select id="enroll-cam">
        <option value="usb">USB cam</option>
        <option value="csi">CSI cam</option>
      </select>
      <button class="btn-primary" id="btn-enroll" onclick="enrollFace()">Registrar</button>
    </div>
    <div class="enroll-msg" id="enroll-msg"></div>
  </div>
  <div class="card">
    <h3>Base de datos facial</h3>
    <div class="faces" id="faces"></div>
  </div>
</section>

<!-- REPORTE -->
<section class="view" id="view-reporte">
  <div class="card">
    <h3>Reporte</h3>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:12px;flex-wrap:wrap">
      <div class="report-meta" id="report-meta">Sin reporte generado</div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn-primary" id="btn-gen-report" onclick="generateReport()">Generar reporte</button>
        <button class="btn-primary" style="background:var(--accent)" onclick="window.open('/report/print','_blank')">Imprimir / PDF</button>
      </div>
    </div>
    <div class="report" id="report"></div>
  </div>
</section>

</main>

<script>
// ── Tabs ──
document.querySelectorAll('.tab').forEach(b => {
  b.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.view').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById('view-' + b.dataset.view).classList.add('active');
    if (b.dataset.view === 'timeline') loadTimeline();
    if (b.dataset.view === 'audio')    loadAudio();
    if (b.dataset.view === 'personas') loadFaces();
    if (b.dataset.view === 'reporte')  loadReport();
  };
});

// ── Helpers ──
function colorBar(el, v) {
  const span = el.querySelector('span');
  span.style.width = Math.max(0, Math.min(100, v)) + '%';
  el.classList.toggle('warn', v >= 70 && v < 85);
  el.classList.toggle('err',  v >= 85);
}
function fmtTs(ts) {
  if (!ts) return '';
  const t = ts.split('T')[1] || ts;
  return t.split('.')[0].slice(0, 8);
}

// ── Stats ──
async function loadStats() {
  try {
    const r = await fetch('/api/stats').then(r => r.json());
    document.getElementById('m-cpu').textContent  = r.cpu_load >= 0 ? r.cpu_load.toFixed(0) + '%' : '—';
    document.getElementById('m-gpu').textContent  = r.gpu_load >= 0 ? r.gpu_load.toFixed(0) + '%' : '—';
    document.getElementById('m-ram').textContent  = r.ram_pct  >= 0 ? r.ram_pct.toFixed(0)  + '%' : '—';
    document.getElementById('m-temp').textContent = r.temp_c   >= 0 ? r.temp_c.toFixed(0)   + '°C' : '—';
    if (r.cpu_load >= 0) colorBar(document.getElementById('b-cpu'),  r.cpu_load);
    if (r.gpu_load >= 0) colorBar(document.getElementById('b-gpu'),  r.gpu_load);
    if (r.ram_pct  >= 0) colorBar(document.getElementById('b-ram'),  r.ram_pct);
    if (r.temp_c   >= 0) colorBar(document.getElementById('b-temp'), Math.min(100, r.temp_c * 1.1));

    document.getElementById('m-fps-csi').textContent = r.fps_csi.toFixed(1);
    document.getElementById('m-fps-usb').textContent = r.fps_usb.toFixed(1);
    document.getElementById('fps-csi-pill').textContent = r.fps_csi.toFixed(1) + ' fps';
    document.getElementById('fps-usb-pill').textContent = r.fps_usb.toFixed(1) + ' fps';

    document.getElementById('k-csi').textContent   = r.persons_csi_now;
    document.getElementById('k-usb').textContent   = r.persons_usb_now;
    document.getElementById('k-known').textContent = r.known_faces;
    document.getElementById('k-new').textContent   = r.new_faces;
    document.getElementById('k-det').textContent   = r.total_detections;
    document.getElementById('k-aud').textContent   = r.total_audio;
  } catch(e) {}
}

async function loadStatus() {
  try {
    const s = await fetch('/api/status').then(r => r.json());
    const items = [
      ['REID',  s.reid],
      ['YOLO',  s.yolo],
      ['CSI',   s.csi],
      ['USB',   s.usb],
      ['AUDIO', s.audio],
    ];
    document.getElementById('pills').innerHTML = items.map(
      ([n, ok]) => `<span class="pill ${ok?'ok':''}"><span class="dot"></span>${n}</span>`
    ).join('');
  } catch(e) {}
}

// ── Timeline ──
async function loadTimeline() {
  const tp     = document.getElementById('f-type').value;
  const cam    = document.getElementById('f-cam').value;
  const person = document.getElementById('f-person').value.trim();
  const qs     = new URLSearchParams();
  if (tp)     qs.set('types', tp);
  if (cam)    qs.set('cam', cam);
  if (person) qs.set('person', person);
  qs.set('limit', '80');
  const evs = await fetch('/api/timeline?' + qs).then(r => r.json());
  if (!evs.length) {
    document.getElementById('events').innerHTML = '<div class="empty">Sin eventos</div>';
    return;
  }
  document.getElementById('events').innerHTML = evs.map(e => {
    const t  = e.type || '';
    const ts = fmtTs(e.timestamp);
    const cm = e.cam || '';
    let txt = '';
    if (t === 'detection') {
      txt = `${e.class||''} (conf ${(e.confidence||0).toFixed(2)})`;
    } else if (t === 'audio') {
      txt = e.label || '';
      if (e.transcript) txt += ' — ' + e.transcript.slice(0,80);
    } else if (t === 'face_reid') {
      const pid = e.person_uuid || '';
      const dpy = (pid.includes(' ') || pid.length > 10) ? pid : 'ID:' + pid;
      txt = `${dpy} sim=${(e.sim||0).toFixed(2)} ${e.new?'(nuevo)':''}`;
    } else {
      txt = JSON.stringify(e).slice(0, 100);
    }
    return `<div class="event">
      <span class="t">${ts}</span>
      <span class="type ${t}">${t}</span>
      <span class="cam">${cm}</span>
      <span>${txt}</span>
    </div>`;
  }).join('');
}
document.getElementById('f-type').onchange   = loadTimeline;
document.getElementById('f-cam').onchange    = loadTimeline;
document.getElementById('f-person').oninput  = loadTimeline;

// ── Audio ──
async function loadAudio() {
  const clips = await fetch('/api/audio/list?limit=20').then(r => r.json());
  if (!clips.length) {
    document.getElementById('audiolist').innerHTML = '<div class="empty">Sin clips</div>';
    return;
  }
  document.getElementById('audiolist').innerHTML = clips.map(c => `
    <div class="clip">
      <div class="top">
        <span class="name">${c.filename}</span>
        <span class="size">${c.size_kb} KB</span>
      </div>
      <audio controls preload="none" src="/api/audio/${c.filename}"></audio>
      ${c.transcript ? `<div class="txt">${c.transcript}</div>` : ''}
      ${c.speakers ? `<div class="spk">${c.speakers}</div>` : ''}
    </div>
  `).join('');
}

// ── Personas ──
async function enrollFace() {
  const name = document.getElementById('enroll-name').value.trim();
  const cam  = document.getElementById('enroll-cam').value;
  const msg  = document.getElementById('enroll-msg');
  const btn  = document.getElementById('btn-enroll');
  if (!name) { msg.className='enroll-msg err'; msg.textContent='Escribe un nombre primero'; return; }
  btn.disabled = true; btn.textContent = 'Registrando...';
  msg.className = 'enroll-msg'; msg.textContent = '';
  try {
    const r = await fetch('/api/faces/enroll', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({name, cam})
    });
    const d = await r.json();
    if (r.ok) {
      msg.className = 'enroll-msg ok';
      msg.textContent = `OK — '${d.name}' registrado (${d.faces_found} cara(s) detectada(s))`;
      document.getElementById('enroll-name').value = '';
      loadFaces();
    } else {
      msg.className = 'enroll-msg err';
      msg.textContent = d.detail || 'Error desconocido';
    }
  } catch(e) {
    msg.className = 'enroll-msg err';
    msg.textContent = 'Error de red: ' + e.message;
  } finally {
    btn.disabled = false; btn.textContent = 'Registrar';
  }
}

async function loadFaces() {
  const list = await fetch('/api/faces/db').then(r => r.json());
  if (!list.length) {
    document.getElementById('faces').innerHTML = '<div class="empty">DB vacía</div>';
    return;
  }
  list.sort((a,b) => (b.is_name?1:0) - (a.is_name?1:0));
  document.getElementById('faces').innerHTML = list.map(f => `
    <div class="face ${f.is_name?'named':''}">
      <span class="id">${f.is_name ? f.id : 'ID:' + f.id.slice(0,8)}</span>
      <button class="del" onclick="delFace('${f.id}')" title="Eliminar">×</button>
    </div>
  `).join('');
}
async function delFace(id) {
  if (!confirm('¿Eliminar ' + id + '?')) return;
  await fetch('/api/faces/db/' + encodeURIComponent(id), { method: 'DELETE' });
  loadFaces();
}

// ── Reporte ──
async function loadReport() {
  const r = await fetch('/api/report/latest').then(r => r.json());
  if (r.content) {
    document.getElementById('report-meta').textContent = 'Generado: ' + (r.timestamp || '');
    document.getElementById('report').textContent      = r.content;
  } else {
    document.getElementById('report-meta').textContent = 'Sin reporte generado';
    document.getElementById('report').textContent      = '';
  }
}
async function generateReport() {
  const btn = document.getElementById('btn-gen-report');
  btn.disabled = true;
  btn.textContent = 'Generando...';
  try {
    await fetch('/api/report/generate', { method: 'POST' });
    setTimeout(() => { loadReport(); btn.disabled = false; btn.textContent = 'Generar reporte'; }, 8000);
  } catch(e) {
    btn.disabled = false; btn.textContent = 'Generar reporte';
  }
}

// ── Chat ──
const chatLog   = document.getElementById('chat-log');
const chatInput = document.getElementById('chat-input');
const chatSend  = document.getElementById('chat-send');
function chatAdd(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}
document.getElementById('chat-form').onsubmit = async (e) => {
  e.preventDefault();
  const q = chatInput.value.trim();
  if (!q) return;
  chatAdd('user', q);
  chatInput.value = '';
  chatSend.disabled = true;
  const pending = chatAdd('bot', '...');
  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: q})
    }).then(r => r.json());
    pending.textContent = r.answer || '(sin respuesta)';
  } catch(err) {
    pending.textContent = 'Error: ' + err.message;
  } finally {
    chatSend.disabled = false;
    chatInput.focus();
  }
};

// ── Stream auto-reconnect ──
['csi','usb'].forEach(c => {
  const img = document.getElementById('img-' + c);
  setInterval(() => {
    if (img.naturalWidth === 0) {
      img.src = '/stream/' + c + '?t=' + Date.now();
    }
  }, 6000);
});

// ── Init ──
loadStats(); loadStatus();
setInterval(loadStats, 2000);
setInterval(loadStatus, 5000);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
