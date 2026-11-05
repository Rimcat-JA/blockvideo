"""AI makeup virtual camera: real-time makeup on your webcam for online calls."""

import os

# Before anything imports NumPy. Its OpenBLAS would start one worker per logical CPU, but the pipeline
# keeps this process on one hardware thread per core: a large solve (the photo analysis) then oversubscribes
# the CPU so badly that the video and the GUI stall for over a second. One thread is also faster for what
# this app solves (303 unknowns: 0.9 ms against 420 ms and more). The user can still override it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.1.0"
