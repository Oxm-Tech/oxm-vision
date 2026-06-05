#!/usr/bin/env python3
"""
Enrollment de personas desde cámara USB en vivo.
Captura 15 frames con cara visible, promedia embeddings, guarda en face_db.pkl.
Uso:
    python3 /app/enroll_live.py --name "Ximena Vieyra" --device 1 --frames 15
"""
import argparse
import os
import sys
import pickle
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--db', default='/app/face_db.pkl')
    parser.add_argument('--device', type=int, default=1, help='V4L2 device index (1=USB)')
    parser.add_argument('--frames', type=int, default=15, help='Frames a capturar')
    parser.add_argument('--interval', type=float, default=0.5, help='Segundos entre capturas')
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        print("ERROR: nombre vacío"); sys.exit(1)

    import onnxruntime
    import insightface
    from insightface.app import FaceAnalysis
    import cv2

    # Forzar CPU para evitar OOM compartiendo GPU con main.py (que está detenido, pero por seguridad)
    providers = ['CPUExecutionProvider']

    print(f"Cargando InsightFace (CPU)...")
    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"Abriendo cámara USB /dev/video{args.device}...")
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir /dev/video{args.device}")
        sys.exit(1)

    print(f"\n*** Pon a {name} frente a la cámara USB ***")
    print(f"Capturando {args.frames} frames con intervalo de {args.interval}s...\n")
    time.sleep(2)  # Dar tiempo para posicionarse

    embeddings = []
    attempts = 0
    max_attempts = args.frames * 4  # Intentar hasta 4x más frames en caso de fallos

    while len(embeddings) < args.frames and attempts < max_attempts:
        # Descartar frames en buffer leyendo dos veces
        cap.grab()
        ret, frame = cap.read()
        attempts += 1

        if not ret or frame is None:
            print(f"  Frame {attempts}: SKIP (error de captura)")
            time.sleep(0.2)
            continue

        faces = app.get(frame)
        if not faces:
            print(f"  Frame {attempts}: SKIP (sin cara detectada)")
            time.sleep(0.3)
            continue

        # Tomar la cara más grande
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        emb = face.normed_embedding.astype('float32')
        embeddings.append(emb)
        print(f"  Frame {attempts}: OK  det_score={face.det_score:.3f}  [{len(embeddings)}/{args.frames}]")

        if len(embeddings) < args.frames:
            time.sleep(args.interval)

    cap.release()

    if not embeddings:
        print("\nERROR: no se capturó ningún embedding.")
        sys.exit(1)

    print(f"\nEmbeddings capturados: {len(embeddings)}")

    # Promediar y renormalizar
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb /= (np.linalg.norm(avg_emb) + 1e-6)

    # Cargar DB
    if os.path.exists(args.db):
        with open(args.db, 'rb') as f:
            db = pickle.load(f)
        print(f"DB cargada: {len(db)} entradas")
    else:
        db = {}
        print("DB nueva")

    # Mostrar similitud del nuevo embedding con el anterior si existe
    if name in db:
        sim = float(np.dot(db[name], avg_emb))
        print(f"'{name}' ya existe — similitud foto vs cámara: {sim:.3f}")
        # Reemplazar completamente (no EMA) porque el anterior era de fotos (dominio distinto)
        db[name] = avg_emb
        print(f"Reemplazado: '{name}' (embedding desde cámara, dominio correcto)")
    else:
        db[name] = avg_emb
        print(f"Registrado: '{name}' (nuevo)")

    with open(args.db, 'wb') as f:
        pickle.dump(db, f)

    print(f"\nDB guardada: {len(db)} personas en {args.db}")

    # Verificar que quedó bien — similitud consigo mismo debe ser ~1.0
    sim_self = float(np.dot(db[name], avg_emb))
    print(f"Verificación auto-similitud: {sim_self:.4f} (debe ser ~1.0)")

if __name__ == '__main__':
    main()
