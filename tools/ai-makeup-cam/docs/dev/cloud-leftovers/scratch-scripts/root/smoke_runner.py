"""Smoke test: real CameraSource / VirtualCameraOutput in a container without devices."""
import logging, time, numpy as np
from makeup_cam.camera import list_cameras
from makeup_cam.pipeline import ProcessResult
from makeup_cam.runner import CameraRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

class Passthrough:
    bypass = False
    def process(self, frame, timestamp_ms=None):
        return ProcessResult(frame, frame, frame, False, {})
    def set_settings(self, s): pass

t = time.monotonic(); print("list_cameras:", list_cameras(), f"{(time.monotonic()-t)*1000:.1f} ms")
frames = []
runner = CameraRunner(Passthrough(), on_frame=frames.append)
runner.start()
time.sleep(2.5)
s = runner.status
print("status:", s.camera_ok, "|", s.camera_error, "|", s.virtual_camera, "|", s.virtual_camera_error)
print("on_frame calls in 2.5 s:", len(frames), frames[-1].output.shape if frames else None)
t = time.monotonic(); runner.stop(); print(f"stop took {(time.monotonic()-t)*1000:.1f} ms")
