"""Smoke test: MainWindow + real CameraRunner + real MakeupPipeline (no effects yet).

The camera is a fake serving frame_720p.png at ~30 fps; the virtual camera is
the real one (fails here: no v4l2loopback), so the error path is exercised.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AI_MAKEUP_CAM_HOME"] = tempfile.mkdtemp(prefix="gui-smoke-")
SCRATCH = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
os.environ["AI_MAKEUP_CAM_MODELS_DIR"] = str(SCRATCH / "models")
sys.path.insert(0, "/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from makeup_cam.config import MakeupSettings
from makeup_cam.gui import GuiOptions, MainWindow
from makeup_cam.pipeline import MakeupPipeline
from makeup_cam.runner import CameraRunner

FRAME = cv2.imread(str(SCRATCH / "testimgs" / "frame_720p.png"))


class StillCamera:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.width, self.height = width, height

    def open(self) -> None: ...

    def read(self):
        time.sleep(1 / 30)
        return cv2.resize(FRAME, (self.width, self.height)) if (self.width, self.height) != (1280, 720) else FRAME.copy()

    def close(self) -> None: ...


app = QApplication([])
settings = MakeupSettings()
settings.general.show_landmarks = True
pipeline = MakeupPipeline(settings, effects=[])


def factory(on_frame, on_status):
    return CameraRunner(pipeline, on_frame=on_frame, on_status=on_status, camera_factory=StillCamera)


window = MainWindow(factory, settings=settings, options=GuiOptions(), camera_lister=lambda in_use: [])
window.show()
window.start()
QTimer.singleShot(6000, lambda: (window.grab().save(str(SCRATCH / "renders" / "gui" / "real_runner.png")), window.close()))
started = time.monotonic()
app.exec()
closed = time.monotonic()
print("vcam label:", window._vcam_status.text())
print("vcam tooltip:", window._vcam_status.toolTip()[:120])
print("fps label:", window._fps_status.text(), "| time:", window._time_status.text(), "| face:", window._face_status.text())
print("runner stopped:", window.shutdown())
pipeline.close()
