import pickle, os, uuid, threading, time
import numpy as np
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("WARN: faiss no disponible, usando búsqueda lineal")

DB_PATH       = '/app/face_db.pkl'
THRESHOLD     = 0.35   # 0.45 era demasiado estricto para distintos ángulos/iluminación
EMB_ALPHA     = 0.15   # peso del embedding nuevo en EMA (aprendizaje gradual)
EMB_DIM       = 512
SAVE_INTERVAL = 30     # segundos entre writes a disco

class FaceReID:
    def __init__(self):
        available = onnxruntime.get_available_providers()
        use_trt   = 'TensorrtExecutionProvider' in available
        use_cuda  = 'CUDAExecutionProvider' in available

        if use_trt:
            # TensorRT: mejor rendimiento en Jetson Orin (FP16 nativo)
            os.makedirs('/tmp/trt_insightface', exist_ok=True)
            trt_opts = {
                'device_id': 0,
                'trt_max_workspace_size': 256 * 1024 * 1024,
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': '/tmp/trt_insightface',
            }
            cuda_opts = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 256 * 1024 * 1024,
            }
            providers = [('TensorrtExecutionProvider', trt_opts),
                         ('CUDAExecutionProvider', cuda_opts),
                         'CPUExecutionProvider']
            ctx_id = 0
            self.mode = 'TRT'
        elif use_cuda:
            # CUDA: fallback si TRT no disponible
            # kSameAsRequested evita arena gigante que falla en Jetson con memoria fragmentada
            cuda_opts = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 512 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
            ctx_id = 0
            self.mode = 'CUDA'
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            self.mode = 'CPU'

        self.on_gpu = self.mode != 'CPU'
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        print(f"FaceReID ({self.mode}): det_size=320x320")

        self.db          = self._load_db()
        self._dirty      = False
        self._lock       = threading.Lock()
        # InsightFace (self.app.get) no es thread-safe para llamadas concurrentes.
        # Este lock serializa inferencias de CSI y USB reid_workers sobre el mismo objeto.
        self._infer_lock = threading.Lock()
        self._build_index()
        print(f"FaceReID: {len(self.db)} personas en DB {'(FAISS)' if FAISS_OK else '(lineal)'}")

        threading.Thread(target=self._flush_loop, daemon=True).start()

    def _load_db(self):
        if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
            try:
                with open(DB_PATH, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"[FaceReID] ERROR: face_db.pkl corrupto ({e}), "
                      f"buscando backup más reciente...", flush=True)
                backup = self._latest_backup()
                if backup:
                    print(f"[FaceReID] Restaurando desde {backup}", flush=True)
                    with open(backup, 'rb') as f:
                        return pickle.load(f)
                print("[FaceReID] Sin backups disponibles, iniciando DB vacía", flush=True)
        return {}

    def _latest_backup(self):
        backup_dir = os.path.join(os.path.dirname(DB_PATH), 'backups')
        if not os.path.isdir(backup_dir):
            return None
        candidates = sorted(
            (f for f in os.listdir(backup_dir) if f.startswith('face_db_') and f.endswith('.pkl')),
            reverse=True
        )
        return os.path.join(backup_dir, candidates[0]) if candidates else None

    def _save_db(self):
        # Write atómico: escribe a un .tmp y renombra — nunca deja DB_PATH
        # a medio escribir si el proceso muere durante el dump (causa raíz
        # de corrupción vista en producción: pkl truncado a 0 bytes).
        tmp_path = DB_PATH + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(self.db, f)
        os.replace(tmp_path, DB_PATH)
        print(f"[FaceReID] DB guardada ({len(self.db)} personas)", flush=True)

    def _flush_loop(self):
        while True:
            time.sleep(SAVE_INTERVAL)
            with self._lock:
                if self._dirty:
                    self._save_db()
                    self._rebuild_index_unsafe()  # reconstruye con embeddings actualizados
                    self._dirty = False

    def _rebuild_index_unsafe(self):
        """Llama solo dentro de self._lock."""
        self.id_list = list(self.db.keys())
        if FAISS_OK:
            self.index = faiss.IndexFlatIP(EMB_DIM)
            if self.id_list:
                embs = np.stack([self.db[pid] for pid in self.id_list]).astype('float32')
                self.index.add(embs)

    def _build_index(self):
        self.id_list = list(self.db.keys())
        if FAISS_OK:
            self.index = faiss.IndexFlatIP(EMB_DIM)
            if self.id_list:
                embs = np.stack([self.db[pid] for pid in self.id_list]).astype('float32')
                self.index.add(embs)
        else:
            self.index = None

    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def identify(self, frame):
        """Retorna lista de {person_uuid, bbox, similarity, new}"""
        results = []
        with self._infer_lock:
            faces = self.app.get(frame)
        for face in faces:
            emb = face.normed_embedding.astype('float32')
            best_id, best_sim = None, 0.0

            with self._lock:
                if FAISS_OK and self.index is not None and self.index.ntotal > 0:
                    D, I = self.index.search(emb.reshape(1, -1), 1)
                    best_sim = float(D[0][0])
                    if best_sim >= THRESHOLD:
                        best_id = self.id_list[I[0][0]]
                else:
                    for pid, stored_emb in self.db.items():
                        sim = self._cosine(emb, stored_emb)
                        if sim > best_sim:
                            best_sim, best_id = sim, pid
                    if best_sim < THRESHOLD:
                        best_id = None

                if best_id is not None:
                    person_uuid = best_id
                    is_new = False
                    # Actualizar embedding con promedio exponencial (se adapta a cambios de ángulo/luz)
                    updated = (1 - EMB_ALPHA) * self.db[best_id] + EMB_ALPHA * emb
                    updated /= (np.linalg.norm(updated) + 1e-6)
                    self.db[best_id] = updated
                    self._dirty = True
                else:
                    person_uuid = str(uuid.uuid4())[:8]
                    self.db[person_uuid] = emb
                    self.id_list.append(person_uuid)
                    if FAISS_OK and self.index is not None:
                        self.index.add(emb.reshape(1, -1))
                    self._dirty = True
                    is_new = True

            bbox = face.bbox.astype(int).tolist()
            results.append({
                'person_uuid': person_uuid,
                'bbox': bbox,
                'similarity': round(best_sim, 3),
                'new': is_new
            })
        return results
