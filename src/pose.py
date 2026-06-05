import math
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO

# COCO-17 keypoint indices
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
L_ANK, R_ANK = 15, 16
L_KNEE, R_KNEE = 13, 14

SKELETON = [
    (5,7),(7,9),(6,8),(8,10),        # arms
    (5,6),(5,11),(6,12),(11,12),     # torso
    (11,13),(13,15),(12,14),(14,16), # legs
    (0,5),(0,6),                     # head-shoulders
]

KP_CONF_MIN  = 0.30
FALL_ANGLE   = 60.0   # degrees from vertical
STAND_ANGLE  = 40.0
FALL_HOLD_S  = 1.2
COOLDOWN_S   = 15.0


def _torso_angle(kpts, kconf):
    if kconf[L_SHO] < KP_CONF_MIN or kconf[R_SHO] < KP_CONF_MIN: return None
    if kconf[L_HIP] < KP_CONF_MIN or kconf[R_HIP] < KP_CONF_MIN: return None
    sho = (kpts[L_SHO] + kpts[R_SHO]) * 0.5
    hip = (kpts[L_HIP] + kpts[R_HIP]) * 0.5
    dx, dy = hip[0] - sho[0], hip[1] - sho[1]
    if abs(dx) < 1e-3 and abs(dy) < 1e-3: return None
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def _bbox_ratio(bbox):
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return w / h


class PoseFall:
    def __init__(self, engine_path, imgsz=640, conf=0.35):
        self.model = YOLO(engine_path, task="pose")
        self.imgsz = imgsz
        self.conf  = conf
        self.lock  = threading.Lock()
        self.state = {}   # key=(cam,tid) → {"since": t, "alerted": t_last_alert}

    def analyze(self, frame, cam_id):
        """Ejecuta pose sobre el frame dado y devuelve lista de dicts:
            {bbox, kpts, kconf, track_id, torso_angle, fall_now, fall_event}
        fall_event=True solo la primera vez que se confirma (después del hold).
        """
        try:
            with self.lock:
                res = self.model.track(frame, imgsz=self.imgsz, conf=self.conf,
                                       device=0, verbose=False, persist=True,
                                       tracker="bytetrack.yaml")
        except Exception as e:
            print(f"[POSE] error inferencia: {e}", flush=True)
            return []

        out = []
        if not res:
            return out
        r = res[0]
        if r.keypoints is None or r.boxes is None:
            return out

        kpts_all  = r.keypoints.xy.cpu().numpy()    # (N,17,2)
        kconf_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones((len(kpts_all),17))
        boxes     = r.boxes.xyxy.cpu().numpy().astype(int)
        ids       = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(boxes), -1)

        now = time.time()
        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            kpts = kpts_all[i]
            kconf = kconf_all[i]
            tid   = int(ids[i])
            ang   = _torso_angle(kpts, kconf)
            ratio = _bbox_ratio(bbox)
            # Caída: requiere keypoints visibles + torso horizontal.
            # bbox ratio sirve solo como confirmación (no dispara por sí solo).
            horizontal = (ang is not None) and (ang > FALL_ANGLE) and (ratio > 1.0)

            key = (cam_id, tid)
            st  = self.state.get(key, {"since": None, "alerted": 0.0})
            fall_event = False
            fall_now   = False

            if horizontal:
                if st["since"] is None:
                    st["since"] = now
                elapsed = now - st["since"]
                if elapsed >= FALL_HOLD_S:
                    fall_now = True
                    if now - st["alerted"] > COOLDOWN_S:
                        fall_event = True
                        st["alerted"] = now
            else:
                if ang is not None and ang < STAND_ANGLE:
                    st["since"] = None

            self.state[key] = st

            out.append({
                "bbox": bbox, "kpts": kpts, "kconf": kconf, "track_id": tid,
                "torso_angle": ang, "ratio": ratio,
                "fall_now": fall_now, "fall_event": fall_event,
            })

        self._gc_state(now)
        return out

    def _gc_state(self, now, ttl=30.0):
        drop = [k for k, v in self.state.items()
                if v.get("since") is None and (now - v.get("alerted", 0)) > ttl]
        for k in drop:
            self.state.pop(k, None)

    @staticmethod
    def draw(frame, results):
        """Dibuja esqueleto + bbox roja si caída detectada."""
        for r in results:
            kpts, kconf = r["kpts"], r["kconf"]
            color = (0, 0, 255) if r["fall_now"] else (0, 255, 180)
            # skeleton
            for a, b in SKELETON:
                if kconf[a] < KP_CONF_MIN or kconf[b] < KP_CONF_MIN:
                    continue
                pa = tuple(map(int, kpts[a]))
                pb = tuple(map(int, kpts[b]))
                cv2.line(frame, pa, pb, color, 2)
            # keypoints
            for i in range(len(kpts)):
                if kconf[i] < KP_CONF_MIN: continue
                cv2.circle(frame, tuple(map(int, kpts[i])), 3, color, -1)
            # bbox if fall
            if r["fall_now"]:
                x1, y1, x2, y2 = r["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "CAIDA", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame
