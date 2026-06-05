import collections
import datetime
import os
import subprocess
import threading
import time

EVENT_DIR     = '/app/event_clips'
RING_SECONDS  = 5     # pre-roll
POST_SECONDS  = 5     # post-roll
FPS           = 15    # framerate asumido para el .mp4


class EventRecorder:
    """Mantiene ring buffer de JPEGs por cámara y genera .mp4 de 10s
    (5s pre + 5s post) cuando se dispara un evento."""

    def __init__(self, cams, cst_tz):
        self.cst = cst_tz
        maxlen = (RING_SECONDS + POST_SECONDS + 2) * FPS
        self.buffers = {c: collections.deque(maxlen=maxlen) for c in cams}
        self.locks   = {c: threading.Lock()                 for c in cams}
        self.active  = {c: False                             for c in cams}
        os.makedirs(EVENT_DIR, exist_ok=True)

    def push(self, cam, jpeg_bytes):
        with self.locks[cam]:
            self.buffers[cam].append((time.time(), jpeg_bytes))

    def trigger(self, cam, event_type, on_ready=None):
        """Dispara grabación asíncrona. on_ready(path) se llama al terminar."""
        if self.active.get(cam):
            return None  # ya hay clip en curso para esta cámara
        start = time.time() - RING_SECONDS
        end   = time.time() + POST_SECONDS
        ts    = datetime.datetime.now(self.cst).strftime('%Y%m%d_%H%M%S')
        path  = f"{EVENT_DIR}/evt_{cam}_{event_type}_{ts}.mp4"
        threading.Thread(target=self._encode,
                         args=(cam, path, start, end, on_ready),
                         daemon=True).start()
        return path

    def _encode(self, cam, path, start_ts, end_ts, on_ready):
        self.active[cam] = True
        try:
            wait = end_ts - time.time()
            if wait > 0:
                time.sleep(wait)
            with self.locks[cam]:
                frames = [(ts, jpg) for ts, jpg in self.buffers[cam]
                          if start_ts <= ts <= end_ts]
            if len(frames) < 5:
                print(f"[EVENT_REC] {cam}: insuficientes frames ({len(frames)}), skip", flush=True)
                return
            proc = subprocess.Popen([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'image2pipe', '-framerate', str(FPS),
                '-c:v', 'mjpeg', '-i', '-',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                path
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for _, jpg in frames:
                    proc.stdin.write(jpg)
                proc.stdin.close()
                proc.wait(timeout=30)
            except Exception as e:
                print(f"[EVENT_REC] {cam}: ffmpeg error {e}", flush=True)
                proc.kill()
                return
            if proc.returncode != 0:
                err = proc.stderr.read().decode(errors='ignore')[:200] if proc.stderr else ''
                print(f"[EVENT_REC] {cam}: ffmpeg rc={proc.returncode} {err}", flush=True)
                return
            sz = os.path.getsize(path)
            print(f"[EVENT_REC] {path} ({sz//1024} KB, {len(frames)} frames)", flush=True)
            if on_ready:
                try:
                    on_ready(path)
                except Exception as e:
                    print(f"[EVENT_REC] on_ready error: {e}", flush=True)
        finally:
            self.active[cam] = False
