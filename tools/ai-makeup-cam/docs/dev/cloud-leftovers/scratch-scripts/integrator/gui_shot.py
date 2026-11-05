"""GUI screenshot: real MakeupPipeline (all built-in effects, natural preset) + real
CameraRunner fed by a fake camera serving frame_720p.png; the virtual camera is a
fake sink named like OBS so the status bar shows the normal state."""
from __future__ import annotations

import os, sys, tempfile, time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AI_MAKEUP_CAM_HOME"] = tempfile.mkdtemp(prefix="gui-shot-")
SCRATCH = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
os.environ["AI_MAKEUP_CAM_MODELS_DIR"] = str(SCRATCH / "models")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from makeup_cam.camera import CameraInfo
from makeup_cam.gui import GuiOptions, MainWindow
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.presets import load_preset
from makeup_cam.runner import CameraRunner

FRAME = cv2.imread(str(SCRATCH / "testimgs" / "frame_720p.png"))
timings: list[dict] = []


class StillCamera:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.size = (width, height)

    def open(self) -> None: ...

    def read(self):
        time.sleep(1 / 30)
        return FRAME.copy() if self.size == (1280, 720) else cv2.resize(FRAME, self.size)

    def close(self) -> None: ...


class FakeVirtualCamera:
    device = "OBS Virtual Camera"

    def __init__(self, width, height, fps): ...
    def send(self, frame) -> None: ...
    def close(self) -> None: ...


app = QApplication([])
settings = load_preset("natural")
pipeline = MakeupPipeline(settings)


def factory(on_frame, on_status):
    def frame_hook(result):
        timings.append(result.timings_ms)
        on_frame(result)
    return CameraRunner(pipeline, on_frame=frame_hook, on_status=on_status,
                        camera_factory=StillCamera, vcam_factory=FakeVirtualCamera)


cameras = [CameraInfo(0, "カメラ 0", 1280, 720), CameraInfo(1, "カメラ 1", 640, 480)]
window = MainWindow(factory, settings=settings, preset_name="natural", options=GuiOptions(),
                    camera_lister=lambda in_use: cameras)
window.resize(1400, 820)
window.show()
window.start()
out = Path(sys.argv[1])
QTimer.singleShot(int(float(sys.argv[2]) * 1000) if len(sys.argv) > 2 else 8000,
                  lambda: (window.grab().save(str(out)), window.close()))
app.exec()
print("fps:", window._fps_status.text(), "| time:", window._time_status.text(), "| face:", window._face_status.text())
print("runner stopped:", window.shutdown())
pipeline.close()
import json, statistics
keys = sorted({k for t in timings[10:] for k in t})
print("frames:", len(timings))
for k in keys:
    vals = [t[k] for t in timings[10:] if k in t]
    print(f"  {k:13s} median {statistics.median(vals):6.1f}  p90 {sorted(vals)[int(len(vals)*0.9)]:6.1f}  n={len(vals)}")
