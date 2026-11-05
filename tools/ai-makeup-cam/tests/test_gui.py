"""Settings window tests, run offscreen with a fake runner (no camera or models)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import argparse
import signal
import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from makeup_cam import config, gui
from makeup_cam.config import SECTION_LABELS, SECTIONS, MakeupSettings
from makeup_cam.gui import (
    APP_TITLE,
    FACE_HINT,
    ChoiceEditor,
    ColorEditor,
    FloatEditor,
    GuiOptions,
    GuiState,
    IntEditor,
    MainWindow,
    bgr_to_qimage,
    options_from_args,
)
from makeup_cam.presets import find_preset, load_last_session, load_preset, save_preset

# ---- fakes ------------------------------------------------------------------


@dataclass
class FakeResult:
    """Shaped like pipeline.ProcessResult."""

    output: np.ndarray
    preview: np.ndarray
    original_preview: np.ndarray
    face_found: bool = True
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class FakeStatus:
    """Shaped like runner.RunnerStatus."""

    fps: float = 30.0
    camera_ok: bool = True
    camera_error: str | None = None
    virtual_camera: str | None = "OBS Virtual Camera"
    virtual_camera_error: str | None = None
    face_found: bool = True
    processing_ms: float = 10.0
    processing_error: str | None = None


@dataclass
class FakeCamera:
    """Shaped like camera.CameraInfo."""

    index: int
    name: str
    width: int
    height: int


class FakeRunner:
    def __init__(self, on_frame: Callable[[FakeResult], None], on_status: Callable[[FakeStatus], None]) -> None:
        self.on_frame = on_frame
        self.on_status = on_status
        self.started = False
        self.stopped = False
        self.stop_delay_s = 0.0
        self.pushed: list[MakeupSettings] = []
        self.bypass: list[bool] = []
        self.cameras: list[tuple[int, int | None, int | None]] = []
        self.virtual_camera: list[bool] = []

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 2.0) -> None:
        time.sleep(self.stop_delay_s)
        self.stopped = True

    def set_settings(self, settings: MakeupSettings) -> None:
        self.pushed.append(settings)

    def set_bypass(self, bypass: bool) -> None:
        self.bypass.append(bypass)

    def set_camera(self, index: int, width: int | None = None, height: int | None = None) -> None:
        self.cameras.append((index, width, height))

    def set_virtual_camera(self, enabled: bool) -> None:
        self.virtual_camera.append(enabled)


class FakeDialogs(gui.Dialogs):
    """Answers every question from attributes and records what was shown."""

    def __init__(self) -> None:
        super().__init__()
        self.confirm_answer = True
        self.text_answer: str | None = None
        self.questions: list[str] = []
        self.errors: list[str] = []

    def confirm(self, title: str, text: str) -> bool:
        self.questions.append(text)
        return self.confirm_answer

    def ask_text(self, title: str, label: str, text: str = "") -> str | None:
        return self.text_answer

    def show_error(self, title: str, text: str) -> None:
        self.errors.append(text)


@dataclass
class Harness:
    window: MainWindow
    runner: FakeRunner
    dialogs: FakeDialogs


# ---- fixtures and helpers ------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point user data and built-in presets at empty temp dirs; returns the built-in dir."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    monkeypatch.setattr(config, "BUILTIN_PRESETS_DIR", builtin)
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))
    return builtin


@pytest.fixture
def make_window(qapp: QApplication) -> Iterator[Callable[..., Harness]]:
    windows: list[MainWindow] = []

    def make(
        settings: MakeupSettings | None = None,
        preset_name: str | None = None,
        camera_lister: Callable[[Collection[int]], list[FakeCamera]] | None = None,
    ) -> Harness:
        runners: list[FakeRunner] = []

        def factory(on_frame: Callable, on_status: Callable) -> FakeRunner:
            runners.append(FakeRunner(on_frame, on_status))
            return runners[-1]

        dialogs = FakeDialogs()
        window = MainWindow(
            factory,
            settings=settings or MakeupSettings(),
            preset_name=preset_name,
            camera_lister=camera_lister,
            options=GuiOptions(),
            dialogs=dialogs,
        )
        windows.append(window)
        return Harness(window, runners[0], dialogs)

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    QApplication.processEvents()


def wait_until(condition: Callable[[], bool], timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        QApplication.processEvents()
        time.sleep(0.005)


def solid(bgr: tuple[int, int, int], width: int = 64, height: int = 36) -> np.ndarray:
    return np.full((height, width, 3), bgr, np.uint8)


def result(
    preview_bgr: tuple[int, int, int], original_bgr: tuple[int, int, int], face_found: bool = True
) -> FakeResult:
    preview = solid(preview_bgr)
    return FakeResult(output=preview, preview=preview, original_preview=solid(original_bgr), face_found=face_found)


def shown_color(window: MainWindow) -> tuple[int, int, int]:
    image = window.preview.image
    assert image is not None
    color = image.pixelColor(0, 0)
    return (color.blue(), color.green(), color.red())


# ---- conversion -----------------------------------------------------------------


def test_bgr_to_qimage_maps_channels_and_copies_pixels() -> None:
    frame = solid((255, 0, 0), width=5, height=3)
    image = bgr_to_qimage(frame)
    frame[:] = 0
    assert (image.width(), image.height()) == (5, 3)
    assert image.pixelColor(4, 2) == QColor(0, 0, 255)


def test_bgr_to_qimage_handles_non_contiguous_views() -> None:
    frame = np.zeros((4, 8, 3), np.uint8)
    frame[:, ::2] = (0, 255, 0)
    image = bgr_to_qimage(frame[:, ::2])
    assert image.width() == 4
    assert image.pixelColor(3, 3) == QColor(0, 255, 0)


def test_bgr_to_qimage_rejects_float_frames() -> None:
    with pytest.raises(ValueError):
        bgr_to_qimage(np.zeros((2, 2, 3), np.float32))


# ---- generated controls -----------------------------------------------------------


def test_window_has_title_and_one_tab_per_section(make_window: Callable[..., Harness]) -> None:
    window = make_window().window
    tabs = window._tabs
    assert window.windowTitle() == APP_TITLE
    assert [tabs.tabText(i) for i in range(tabs.count())] == [SECTION_LABELS[name] for name in SECTIONS]
    assert isinstance(window.section_page("lips").editor("opacity"), FloatEditor)
    assert isinstance(window.section_page("general").editor("segmentation_interval"), IntEditor)
    assert isinstance(window.section_page("lips").editor("color"), ColorEditor)
    assert isinstance(window.section_page("lips").editor("finish"), ChoiceEditor)


def test_widgets_show_the_initial_settings(make_window: Callable[..., Harness]) -> None:
    settings = MakeupSettings()
    settings.lips.opacity = 0.83
    settings.blush.shape = "lifted"
    window = make_window(settings).window
    opacity = window.section_page("lips").editor("opacity")
    assert opacity.spin_box.value() == pytest.approx(0.83)
    assert opacity.slider.value() == 83
    assert window.section_page("blush").editor("shape").value() == "lifted"


def test_slider_updates_settings_spin_box_and_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("lips").editor("opacity")

    editor.slider.setValue(70)

    assert editor.spin_box.value() == pytest.approx(0.70)
    assert harness.window.settings().lips.opacity == pytest.approx(0.70)
    wait_until(lambda: bool(harness.runner.pushed))
    assert harness.runner.pushed[-1].lips.opacity == pytest.approx(0.70)


def test_spin_box_updates_slider_and_signed_ranges(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("skin").editor("brightness")  # range -0.3 .. 0.3

    editor.spin_box.setValue(-0.12)

    assert editor.slider.value() == 18
    assert harness.window.settings().skin.brightness == pytest.approx(-0.12)


def test_int_field_stores_an_int(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.window.section_page("general").editor("segmentation_interval").spin_box.setValue(4)
    value = harness.window.settings().general.segmentation_interval
    assert value == 4 and isinstance(value, int)


def test_rapid_changes_are_throttled_but_the_last_value_arrives(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    slider = harness.window.section_page("lips").editor("opacity").slider
    for position in range(10, 60, 5):
        slider.setValue(position)
    wait_until(lambda: bool(harness.runner.pushed) and harness.runner.pushed[-1].lips.opacity == pytest.approx(0.55))
    assert len(harness.runner.pushed) <= 2


def test_color_hex_entry_updates_settings_and_rejects_bad_input(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    editor = harness.window.section_page("lips").editor("color")

    editor.hex_edit.setText("aa3344")
    editor.hex_edit.editingFinished.emit()
    assert harness.window.settings().lips.color == "#AA3344"
    assert editor.hex_edit.text() == "#AA3344"

    editor.hex_edit.setText("#12")
    editor.hex_edit.editingFinished.emit()
    assert harness.window.settings().lips.color == "#AA3344"
    assert editor.hex_edit.text() == "#AA3344"


def test_choice_updates_settings(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    combo = harness.window.section_page("lips").editor("finish").combo_box
    combo.setCurrentIndex(combo.findData("gloss"))
    assert harness.window.settings().lips.finish == "gloss"


def test_reset_button_restores_the_default(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    page = harness.window.section_page("eyeliner")
    default = MakeupSettings().eyeliner.wing
    assert not page.reset_button("wing").isEnabled()

    page.editor("wing").slider.setValue(90)
    assert page.reset_button("wing").isEnabled()
    page.reset_button("wing").click()

    assert harness.window.settings().eyeliner.wing == default
    assert page.editor("wing").value() == default
    assert not page.reset_button("wing").isEnabled()


def test_disabling_a_section_greys_out_its_controls(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    page = harness.window.section_page("blush")

    page.editor("enabled").check_box.setChecked(False)

    assert harness.window.settings().blush.enabled is False
    assert not page.editor("opacity").isEnabled()
    page.editor("enabled").check_box.setChecked(True)
    assert page.editor("opacity").isEnabled()


def test_wheel_over_an_unfocused_slider_does_not_change_it(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    slider = harness.window.section_page("lips").editor("opacity").slider
    before = slider.value()
    wheel = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(slider, wheel)
    assert slider.value() == before


# ---- presets ------------------------------------------------------------------------


def test_preset_save_as_load_and_delete_roundtrip(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    window, dialogs = harness.window, harness.dialogs
    opacity = window.section_page("lips").editor("opacity")

    opacity.slider.setValue(80)
    dialogs.text_answer = "面接/用"
    window._save_as_button.click()
    assert window.preset_name == "面接用"
    assert load_preset("面接用").lips.opacity == pytest.approx(0.80)
    assert window._preset_combo.currentText() == "面接用"
    assert not window.is_modified()
    assert window._delete_button.isEnabled()

    opacity.slider.setValue(30)
    assert window.is_modified() and window._save_button.isEnabled()
    window._save_button.click()
    assert load_preset("面接用").lips.opacity == pytest.approx(0.30)
    assert not window.is_modified()

    window._defaults_button.click()
    assert window.settings() == MakeupSettings()
    assert window.preset_name is None and window._preset_combo.currentIndex() == -1

    pushes_before = len(harness.runner.pushed)
    combo = window._preset_combo
    combo.activated.emit(combo.findData("面接用"))
    assert window.settings().lips.opacity == pytest.approx(0.30)
    assert opacity.slider.value() == 30
    assert len(harness.runner.pushed) == pushes_before + 1
    assert harness.runner.pushed[-1].lips.opacity == pytest.approx(0.30)

    window._delete_button.click()
    assert find_preset("面接用") is None
    assert window.preset_name is None
    assert combo.findData("面接用") == -1
    assert window.settings().lips.opacity == pytest.approx(0.30)  # current settings are kept


def test_builtin_preset_is_marked_and_cannot_be_overwritten_or_deleted(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    (isolated_dirs / "natural.json").write_text(MakeupSettings().to_json(), encoding="utf-8")
    window = make_window(preset_name="natural").window

    assert window._preset_combo.currentText() == "natural（標準）"
    window.section_page("lips").editor("opacity").slider.setValue(90)
    assert window.is_modified()
    assert not window._save_button.isEnabled()
    assert not window._delete_button.isEnabled()


def test_saving_under_a_builtin_name_asks_and_overrides_it(
    make_window: Callable[..., Harness], isolated_dirs: Path
) -> None:
    (isolated_dirs / "natural.json").write_text(MakeupSettings().to_json(), encoding="utf-8")
    harness = make_window(preset_name="natural")
    harness.window.section_page("lips").editor("opacity").slider.setValue(90)
    harness.dialogs.text_answer = "natural"

    harness.dialogs.confirm_answer = False
    harness.window._save_as_button.click()
    assert find_preset("natural").builtin  # declined: still the built-in one

    harness.dialogs.confirm_answer = True
    harness.window._save_as_button.click()
    assert "標準プリセット" in harness.dialogs.questions[-1]
    assert find_preset("natural").builtin is False
    assert harness.window._delete_button.isEnabled()


def test_choosing_a_preset_with_unsaved_changes_asks_first(make_window: Callable[..., Harness]) -> None:
    save_preset("A", MakeupSettings())
    harness = make_window()
    harness.window.section_page("lips").editor("opacity").slider.setValue(95)
    harness.dialogs.confirm_answer = False

    combo = harness.window._preset_combo
    combo.activated.emit(combo.findData("A"))

    assert harness.dialogs.questions
    assert harness.window.settings().lips.opacity == pytest.approx(0.95)
    assert harness.window.preset_name is None
    assert combo.currentIndex() == -1


def test_broken_preset_shows_an_error_and_keeps_settings(make_window: Callable[..., Harness]) -> None:
    save_preset("broken", MakeupSettings())
    find_preset("broken").path.write_text("{", encoding="utf-8")
    harness = make_window()

    assert harness.window.apply_preset("broken") is False
    assert "broken" in harness.dialogs.errors[-1]
    assert harness.window.settings() == MakeupSettings()


# ---- preview ---------------------------------------------------------------------------


def test_frame_updates_preview_and_face_hint(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    assert harness.window.preview.notice  # "starting" until the first frame

    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0), face_found=False))

    assert shown_color(harness.window) == (10, 20, 30)
    assert harness.window.preview.hint == FACE_HINT
    assert harness.window.preview.notice == ""
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0), face_found=True))
    assert harness.window.preview.hint == ""


def test_compare_button_shows_the_original_while_pressed(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (200, 150, 100)))
    button = harness.window._compare_button

    button.pressed.emit()
    assert shown_color(harness.window) == (200, 150, 100)
    assert harness.window.preview.badge
    harness.runner.on_frame(result((11, 21, 31), (201, 151, 101)))
    assert shown_color(harness.window) == (201, 151, 101)

    button.released.emit()
    assert shown_color(harness.window) == (11, 21, 31)
    assert harness.window.preview.badge == ""


def test_space_key_holds_compare(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (200, 150, 100)))

    QTest.keyPress(harness.window, Qt.Key.Key_Space)
    assert shown_color(harness.window) == (200, 150, 100)
    QTest.keyRelease(harness.window, Qt.Key.Key_Space)
    assert shown_color(harness.window) == (10, 20, 30)


def test_frames_from_a_worker_thread_keep_only_the_latest(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    shown: list[int] = []
    harness.window.preview.set_image = lambda image: shown.append(image.pixelColor(0, 0).red())

    def produce() -> None:
        for value in range(1, 41):
            harness.runner.on_frame(result((0, 0, value), (0, 0, 0)))

    worker = threading.Thread(target=produce)
    worker.start()
    worker.join()
    wait_until(lambda: bool(shown) and shown[-1] == 40)
    assert len(shown) < 40


def test_bypass_and_virtual_camera_toggles_reach_the_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0)))

    harness.window._bypass_button.click()
    assert harness.runner.bypass == [True]
    assert harness.window.preview.badge == "メイクOFF中"

    harness.window._vcam_check.setChecked(False)
    assert harness.runner.virtual_camera == [False]
    assert harness.window._vcam_status.text() == "仮想カメラ OFF"


# ---- status bar -------------------------------------------------------------------------


def test_status_bar_shows_rates_and_detection(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_status(FakeStatus(fps=29.84, processing_ms=12.345, face_found=False))
    window = harness.window
    assert window._fps_status.text() == "FPS 29.8"
    assert window._time_status.text() == "処理 12.3 ms"
    assert "NG" in window._face_status.text()
    assert "OBS Virtual Camera" in window._vcam_status.text()


def test_virtual_camera_error_is_red_with_full_guidance(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    guidance = "OBS Studio をインストールしてから、このアプリを再起動してください。"
    harness.runner.on_status(FakeStatus(virtual_camera=None, virtual_camera_error=guidance))

    label = harness.window._vcam_status
    assert "使用できません" in label.text() and gui._ERROR_COLOR in label.text()
    assert label.toolTip() == guidance
    label.linkActivated.emit("details")
    assert harness.dialogs.errors == [guidance]


def test_camera_error_is_shown_on_the_preview_and_status_bar(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.on_frame(result((10, 20, 30), (0, 0, 0)))
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="他のアプリがカメラを使用中です"))

    assert "他のアプリがカメラを使用中です" in harness.window.preview.notice
    assert harness.window._camera_status.isVisibleTo(harness.window)
    harness.runner.on_status(FakeStatus())
    assert harness.window.preview.notice == ""
    assert not harness.window._camera_status.isVisibleTo(harness.window)


# ---- camera selection ------------------------------------------------------------------


def test_cameras_are_listed_in_the_background_and_selectable(make_window: Callable[..., Harness]) -> None:
    asked: list[Collection[int]] = []

    def lister(in_use: Collection[int]) -> list[FakeCamera]:
        asked.append(in_use)
        return [FakeCamera(0, "内蔵カメラ", 1280, 720), FakeCamera(2, "USB カメラ", 1920, 1080)]

    harness = make_window(camera_lister=lister)
    harness.window.start()
    combo = harness.window._camera_combo
    wait_until(lambda: combo.count() == 2)
    assert harness.runner.started
    assert list(asked[0]) == [0]
    assert combo.itemText(1) == "USB カメラ"

    combo.setCurrentIndex(1)
    combo.activated.emit(1)
    resolution = harness.window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    resolution.activated.emit(resolution.currentIndex())  # unchanged: no reconnect

    assert harness.runner.cameras == [(2, 1280, 720), (2, 640, 480)]


def test_selected_camera_stays_listed_when_the_probe_misses_it(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    assert harness.window._camera_combo.itemData(0) == 0


# ---- shutdown and restore ----------------------------------------------------------------


def test_close_saves_session_and_stops_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.window.section_page("hair").editor("gloss").slider.setValue(77)

    harness.window.close()

    assert harness.runner.stopped
    session = load_last_session()
    assert session is not None and session.hair.gloss == pytest.approx(0.77)


def test_close_never_waits_long_for_a_stuck_runner(make_window: Callable[..., Harness]) -> None:
    harness = make_window()
    harness.runner.stop_delay_s = 10.0
    started = time.monotonic()
    assert harness.window.shutdown() is False
    assert time.monotonic() - started < gui.RUNNER_STOP_DEADLINE_S + 0.5


def test_remembered_state_roundtrip(make_window: Callable[..., Harness]) -> None:
    save_preset("A", MakeupSettings())
    harness = make_window(preset_name="A")
    harness.window.close()

    state = GuiState()
    assert state.preset() == "A"
    assert state.camera_index() == 0
    assert state.resolution() == (1280, 720)
    assert state.geometry() is not None


def test_options_merge_command_line_and_remembered_state(qapp: QApplication) -> None:
    state = GuiState()
    state.save(camera_index=2, resolution=(960, 540), preset=None, geometry=QByteArray(), splitter=QByteArray())
    remembered = options_from_args(None, state)
    assert (remembered.camera_index, remembered.resolution, remembered.virtual_camera) == (2, (960, 540), True)

    args = argparse.Namespace(camera=0, width=640, height=480, fps=15, no_vcam=True, preset="面接")
    explicit = options_from_args(args, state)
    assert explicit == GuiOptions(camera_index=0, width=640, height=480, fps=15, virtual_camera=False, preset="面接")


def test_prepare_models_reports_download_errors(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from makeup_cam import models

    calls: list[str] = []

    def failing_download(progress: Callable[[str, int, int], None]) -> None:
        progress("face_landmarker", 1_000_000, 4_000_000)
        calls.append("download")
        raise models.ModelDownloadError("インターネットに接続できません")

    monkeypatch.setattr(models, "models_dir", lambda: tmp_path)
    monkeypatch.setattr(models, "ensure_all_models", failing_download)
    assert gui.prepare_models() == "インターネットに接続できません"
    assert calls == ["download"]

    monkeypatch.setattr(models, "ensure_all_models", lambda progress: calls.append("ok"))
    assert gui.prepare_models() is None
    for spec in models.MODEL_SPECS.values():
        (tmp_path / spec.filename).write_bytes(b"model")
    assert gui.prepare_models() is None
    assert calls == ["download", "ok"]  # present files: no download attempt


def test_ctrl_c_in_the_terminal_closes_the_window(qapp: QApplication) -> None:
    window = QMainWindow()
    window.show()
    previous = signal.getsignal(signal.SIGINT)
    with gui.closed_by_ctrl_c(window):
        signal.raise_signal(signal.SIGINT)
        qapp.processEvents()
        assert not window.isVisible()
    assert signal.getsignal(signal.SIGINT) is previous
