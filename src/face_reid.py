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
THRESHOLD     = 0.35  # similitud mínima para match
EMB_ALPHA     = 0.15   # peso del embedding nuevo en EMA (aprendizaje gradual)
EMB_DIM       = 512
SAVE_INTERVAL = 30     # segundos entre writes a disco

class FaceReID:
    def __init__(self):
        available = onnxruntime.get_available_providers()
        use_cuda  = 'CUDAExecutionProvider' in available

        if use_cuda:
            # kSameAsRequested evita arena gigante que falla en Jetson con memoria fragmentada
            cuda_opts = {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 512 * 1024 * 1024,
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }
            providers = [('CUDAExecutionProvider', cuda_opts), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.app = FaceAnalysis(providers=providers)
        if use_cuda:
            self.app.prepare(ctx_id=0, det_size=(320, 320))
            print("FaceReID (GPU): det_size=320x320")
        else:
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            print("FaceReID (CPU): det_size=320x320")

        self.db     = self._load_db()
        self._dirty = False
        self._lock  = threading.Lock()
        self._build_index()
        print(f"FaceReID: {len(self.db)} personas en DB {'(FAISS)' if FAISS_OK else '(lineal)'}")

        threading.Thread(target=self._flush_loop, daemon=True).start()

    def _load_db(self):
        if os.path.exists(DB_PATH):
            with open(DB_PATH, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_db(self):
        with open(DB_PATH, 'wb') as f:
            pickle.dump(self.db, f)
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

    def enroll(self, name: str, frame) -> dict:
        """Enrolla una persona por nombre desde un frame BGR.
        Usa la cara más grande detectada. Retorna {ok, name, faces_found} o {ok=False, error}."""
        if not name or not name.strip():
            return {'ok': False, 'error': 'Nombre vacío'}
        name = name.strip()
        try:
            faces = self.app.get(frame)
        except Exception as e:
            return {'ok': False, 'error': f'Detección falló: {e}'}
        if not faces:
            return {'ok': False, 'error': 'No se detectó ningún rostro en el frame'}
        # Usar cara más grande (mayor área de bbox)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb  = face.normed_embedding.astype('float32')
        with self._lock:
            self.db[name] = emb
            self._rebuild_index_unsafe()
            self._dirty = True
        print(f"[FaceReID] Enrolled: '{name}' ({len(faces)} cara(s) detectada(s))", flush=True)
        return {'ok': True, 'name': name, 'faces_found': len(faces)}
