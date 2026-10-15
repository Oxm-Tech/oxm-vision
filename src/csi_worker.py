#!/usr/bin/env python3
"""
CSI capture worker — runs isolated from main.py.
GStreamer/Argus segfaults only crash THIS script; main.py continues.

argv[1]: path to a pre-created temp file used as mmap frame buffer (720*1280*3 bytes)

Signals are written to the file descriptor in env var CSI_SIGNAL_FD (not stdout,
because GStreamer may write random bytes to stdout/stderr):
  b'R' = pipeline ready
  b'F' = new frame written to mmap file
  b'E' = error / frame loss (worker will exit)

Exit codes: 1=open failed, 2=frame loss
"""
import os
import sys
import mmap
import cv2
import numpy as np

_H, _W = 720, 1280
_FRAME_BYTES = _H * _W * 3

# Tolerancia: 10 fallos consecutivos antes de declarar desconexión.
# Con ~30 fps, 10 fallos = ~333ms de ventana antes de reconectar.
# Antes era 2 (66ms) → demasiado agresivo con drops transitorios de Argus.
FAIL_THRESHOLD = 10

CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! "
    "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
)


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    frame_path = sys.argv[1]

    sig_fd = int(os.environ.get('CSI_SIGNAL_FD', '-1'))
    if sig_fd < 0:
        sys.exit(1)
    sig_out = os.fdopen(sig_fd, 'wb', buffering=0)

    with open(frame_path, 'r+b') as f:
        mm = mmap.mmap(f.fileno(), _FRAME_BYTES)

    buf = np.frombuffer(mm, dtype=np.uint8).reshape((_H, _W, 3))

    cap = cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        sig_out.write(b'E')
        mm.close()
        sys.exit(1)

    sig_out.write(b'R')

    fail = 0
    while True:
        ret, frame = cap.read()
        if ret and frame is not None:
            fail = 0
            buf[:] = frame  # copia directa al mmap sin tobytes()
            sig_out.write(b'F')
        else:
            fail += 1
            if fail >= FAIL_THRESHOLD:
                sig_out.write(b'E')
                break

    # Skip cap.release() intentionally — calling it triggers GStreamer native
    # cleanup which floods Argus errors and corrupts the heap, causing SIGSEGV.
    # The OS will reclaim all resources when this process exits.
    mm.close()
    sys.exit(2)


if __name__ == '__main__':
    main()
