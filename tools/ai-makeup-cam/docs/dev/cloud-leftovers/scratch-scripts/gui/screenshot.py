"""Render the settings window offscreen with a real frame for layout review."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["AI_MAKEUP_CAM_HOME"] = tempfile.mkdtemp(prefix="gui-shot-")

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path("/home/user/blockvideo-upgrade-project/tools/ai-makeup-cam")
sys.path.insert(0, str(ROOT))
from makeup_cam.gui import MainWindow, GuiOptions  # noqa: E402
from makeup_cam.config import MakeupSettings  # noqa: E402

SCRATCH = Path("/tmp/claude-0/-home-user-blockvideo-upgrade-project/e0146503-3c18-585c-b30a-5c8eb08c9019/scratchpad")
OUT = SCRATCH / "renders" / "gui"


@dataclass
class Result:
    output: np.ndarray
    preview: np.ndarray
    original_preview: np.ndarray
    face_found: bool = True
    timings_ms: dict = field(default_factory=dict)


@dataclass
class Status:
    fps: float = 29.7
    camera_ok: bool = True
    camera_error: str | None = None
    virtual_camera: str | None = "OBS Virtual Camera"
    virtual_camera_error: str | None = None
    face_found: bool = True
    processing_ms: float = 13.4
    processing_error: str | None = None


@dataclass
class Camera:
    index: int
    name: str
    width: int
    height: int


class Runner:
    def __init__(self, on_frame, on_status):
        self.on_frame, self.on_status = on_frame, on_status

    def start(self): ...
    def stop(self, timeout=2.0): ...
    def set_settings(self, s): ...
    def set_bypass(self, b): ...
    def set_camera(self, i, w=None, h=None): ...
    def set_virtual_camera(self, e): ...


def main() -> None:
    app = QApplication.instance() or QApplication([])
    runners = []
    window = MainWindow(
        lambda f, s: runners.append(Runner(f, s)) or runners[-1],
        settings=MakeupSettings(),
        camera_lister=lambda in_use: [Camera(0, "FaceTime HD カメラ", 1280, 720), Camera(1, "USB カメラ", 1920, 1080)],
        options=GuiOptions(),
    )
    runner = runners[0]
    frame = cv2.imread(str(SCRATCH / "testimgs" / "frame_720p.png"))
    mirrored = np.ascontiguousarray(frame[:, ::-1])
    window.show()
    window.start()
    for _ in range(50):
        app.processEvents()
    runner.on_frame(Result(mirrored, mirrored, mirrored))
    runner.on_status(Status())
    app.processEvents()

    def shot(name: str) -> None:
        for _ in range(5):
            app.processEvents()
        window.grab().save(str(OUT / name))
        print("wrote", OUT / name)

    shot("window.png")
    for section, name in (("lips", "tab_lips.png"), ("skin", "tab_skin.png"), ("hair", "tab_hair.png")):
        window._tabs.setCurrentWidget(window.section_page(section))
        shot(name)

    window._tabs.setCurrentWidget(window.section_page("blush"))
    window.section_page("blush").editor("enabled").check_box.setChecked(False)
    window.section_page("blush").editor("enabled").check_box.toggled.emit(False)
    runner.on_frame(Result(mirrored, mirrored, mirrored, face_found=False))
    runner.on_status(Status(virtual_camera=None, virtual_camera_error="仮想カメラを開始できませんでした。OBS Studio（26 以降）をインストールし、ビデオ会議アプリを再起動してください。", face_found=False))
    shot("states_noface_vcam_error.png")

    runner.on_status(Status(camera_ok=False, camera_error="カメラの映像が届いていません。再接続を試みています…"))
    window._bypass_button.setChecked(True)
    shot("states_camera_error.png")

    window.resize(980, 600)
    window._bypass_button.setChecked(False)
    runner.on_status(Status())
    runner.on_frame(Result(mirrored, mirrored, mirrored))
    window._tabs.setCurrentWidget(window.section_page("general"))
    shot("window_min.png")
    window.shutdown()


main()
