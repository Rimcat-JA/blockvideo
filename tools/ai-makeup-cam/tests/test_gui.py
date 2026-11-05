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
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

from makeup_cam import camera, config, gui
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
from makeup_cam.runner import CameraRunner

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
        self.camera_names: list[str | None] = []
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

    def set_camera(
        self, index: int, width: int | None = None, height: int | None = None, name: str | None = None
    ) -> None:
        self.cameras.append((index, width, height))
        self.camera_names.append(name)

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
    monkeypatch.setattr(camera, "_device_names", dict)  # the cameras of the PC running the tests
    return builtin


@pytest.fixture
def make_window(qapp: QApplication) -> Iterator[Callable[..., Harness]]:
    windows: list[MainWindow] = []

    def make(
        settings: MakeupSettings | None = None,
        preset_name: str | None = None,
        camera_lister: Callable[[Collection[int]], list[FakeCamera]] | None = None,
        options: GuiOptions | None = None,
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
            options=options or GuiOptions(),
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


def test_rescan_moves_a_failing_virtual_camera_selection_to_a_real_camera(make_window: Callable[..., Harness]) -> None:
    found: list[FakeCamera] = []
    harness = make_window(camera_lister=lambda in_use: list(found))
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="「OBS Virtual Camera」は仮想カメラなので…"))
    assert "仮想カメラ" in window.preview.notice
    assert harness.runner.cameras == []  # nothing real to move to yet

    found.append(FakeCamera(1, "HD Pro Webcam C920", 1280, 720))  # she plugged in the webcam and pressed the button
    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert window._camera_combo.currentData() == 1
    assert window._camera_combo.currentText() == "HD Pro Webcam C920"
    assert harness.runner.cameras == [(1, 1280, 720)]


def test_a_camera_that_is_still_starting_is_not_replaced(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(1, "HD Pro Webcam C920", 1280, 720)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert harness.window._camera_combo.currentData() == 0
    assert harness.runner.cameras == []


def test_real_camera_name_is_remembered_with_its_index(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "HD Pro Webcam C920", 1280, 720)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (0, "HD Pro Webcam C920")


def test_a_selection_known_not_to_be_a_real_camera_does_not_replace_the_remembered_one(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    state = GuiState()
    state.save(
        camera_index=1,
        camera_name="HD Pro Webcam C920",
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    # The webcam is unplugged today: index 0 is OBS (hidden), another camera was found at index 3.
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(3, "Other", 640, 480)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, "HD Pro Webcam C920")


def test_choosing_a_camera_tells_the_runner_its_name(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "内蔵カメラ", 1280, 720), FakeCamera(2, "USB", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)

    combo.setCurrentIndex(1)
    combo.activated.emit(1)

    assert harness.runner.cameras == [(2, 1280, 720)]
    assert harness.runner.camera_names == ["USB"]


def test_a_placeholder_is_not_passed_on_as_a_device_name(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "カメラ 0", 640, 480), FakeCamera(1, "カメラ 1", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)

    combo.setCurrentIndex(1)
    combo.activated.emit(1)

    assert harness.runner.cameras == [(1, 1280, 720)]
    assert harness.runner.camera_names == [None]


def test_the_probe_is_told_what_is_in_use_by_name_once_it_is_known(make_window: Callable[..., Harness]) -> None:
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return [FakeCamera(1, "USB Camera", 1280, 720)]

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert asked == [[1]]  # no name yet: only the index to go on

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert asked == [[1], ["USB Camera"]]


def test_a_failing_camera_without_a_name_is_really_probed(make_window: Callable[..., Harness]) -> None:
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return []

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="カメラ 1 を開けませんでした"))
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert asked == [[]]  # nothing is open, so there is nothing to protect from being probed


def test_rescan_follows_the_selected_camera_to_its_new_number(make_window: Callable[..., Harness]) -> None:
    found = [FakeCamera(0, "Built-in", 1280, 720), FakeCamera(1, "USB Camera", 1280, 720)]
    harness = make_window(camera_lister=lambda in_use: list(found), options=GuiOptions(camera_index=1))
    window, runner, combo = harness.window, harness.runner, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)
    assert combo.currentText() == "USB Camera"

    found[:] = [FakeCamera(0, "USB Camera", 0, 0), FakeCamera(1, "Other Camera", 640, 480)]  # Built-in went away
    window._rescan_button.click()
    wait_until(lambda: combo.itemText(1) == "Other Camera")

    assert (combo.currentData(), combo.currentText()) == (0, "USB Camera")
    assert combo.count() == 2  # no made-up entry for the old number
    assert runner.cameras == []  # the same camera: no reconnect

    combo.activated.emit(combo.currentIndex())
    assert runner.cameras == []  # choosing it again is still no reason to reconnect

    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    assert runner.cameras == [(0, 640, 480)]
    assert runner.camera_names == ["USB Camera"]


def test_a_live_camera_the_probe_missed_stays_selected_under_its_own_name(make_window: Callable[..., Harness]) -> None:
    options = GuiOptions(camera_index=1, camera_name="USB Camera")
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(1, "Other Camera", 640, 480)], options=options)
    window, combo = harness.window, harness.window._camera_combo
    assert combo.currentText() == "USB Camera"  # known from the start
    window.start()
    wait_until(lambda: combo.count() == 2)

    assert combo.currentText() == "USB Camera"  # not Other Camera, which has the same number now
    assert harness.runner.cameras == []
    window.close()

    assert GuiState().camera_name() is None  # an unplugged device replaces nothing


def test_a_failing_camera_that_is_gone_moves_to_a_real_one_on_rescan(make_window: Callable[..., Harness]) -> None:
    found = [FakeCamera(1, "Other Camera", 640, 480)]
    options = GuiOptions(camera_index=1, camera_name="USB Camera")
    harness = make_window(camera_lister=lambda in_use: list(found), options=options)
    harness.runner.on_status(FakeStatus(camera_ok=False, camera_error="「USB Camera」が見つかりません"))

    harness.window._rescan_button.click()
    wait_until(lambda: harness.window._rescan_button.isEnabled())

    assert harness.window._camera_combo.currentText() == "Other Camera"
    assert harness.runner.cameras == [(1, 1280, 720)]
    assert harness.runner.camera_names == ["Other Camera"]


def test_two_identical_cameras_keep_their_order_on_rescan(make_window: Callable[..., Harness]) -> None:
    cameras = [FakeCamera(0, "USB Camera", 640, 480), FakeCamera(1, "USB Camera", 640, 480)]
    asked: list[list[object]] = []

    def lister(in_use: Collection[object]) -> list[FakeCamera]:
        asked.append(list(in_use))
        return list(cameras)

    harness = make_window(camera_lister=lister, options=GuiOptions(camera_index=1))
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: combo.count() == 2)
    assert combo.currentData() == 1

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert combo.currentData() == 1
    assert asked == [[1], [1]]  # the name cannot tell them apart: only the number says which one is live


class NoDevice:
    """cv2.VideoCapture with nothing there."""

    def __init__(self, index: int, api: int) -> None:
        pass

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass


def save_camera(index: int, name: str) -> None:
    state = GuiState()
    state.save(
        camera_index=index,
        camera_name=name,
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )


def test_the_saved_camera_name_survives_a_run_with_the_webcam_unplugged(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    # No device at all is listed: the selected number is listed anyway in case it is the one in use.
    save_camera(1, "HD Pro Webcam C920")
    options = options_from_args(None, GuiState())
    assert (options.camera_index, options.camera_name) == (1, None)
    harness = make_window(
        camera_lister=lambda in_use: camera.list_cameras(in_use=in_use, capture_factory=NoDevice), options=options
    )
    window = harness.window
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert window._camera_combo.currentText() == "カメラ 1"
    assert "見つかりませんでした" in window._camera_combo.toolTip()
    window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, "HD Pro Webcam C920")


def test_a_bare_number_is_still_remembered_where_there_are_never_names(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    cameras = [FakeCamera(1, "カメラ 1", 640, 480)]
    harness = make_window(camera_lister=lambda in_use: list(cameras), options=GuiOptions(camera_index=1))
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (1, None)


def test_the_not_found_hint_goes_away_when_the_listed_camera_turns_out_to_work(
    make_window: Callable[..., Harness],
) -> None:
    # Where there are no names, a camera in use is listed without being probed: it may or may not exist.
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "カメラ 0", 0, 0)])
    window, combo = harness.window, harness.window._camera_combo
    window.start()
    wait_until(lambda: window._rescan_button.isEnabled())
    assert "見つかりませんでした" in combo.toolTip()

    harness.runner.on_status(FakeStatus())

    assert combo.toolTip() == ""


def test_a_camera_that_was_probed_is_never_reported_as_not_found(make_window: Callable[..., Harness]) -> None:
    harness = make_window(camera_lister=lambda in_use: [FakeCamera(0, "カメラ 0", 640, 480)])
    harness.window.start()
    wait_until(lambda: harness.window._rescan_button.isEnabled())
    assert harness.window._camera_combo.toolTip() == ""


def test_options_look_up_the_remembered_camera_by_name(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    state = GuiState()
    state.save(
        camera_index=1,
        camera_name="HD Pro Webcam C920",
        resolution=(1280, 720),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    # Renumbered since the last run; index 1 is the virtual camera now.
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "HD Pro Webcam C920", 1: "OBS Virtual Camera"})

    options = options_from_args(None, state)
    assert (options.camera_index, options.camera_name) == (0, "HD Pro Webcam C920")

    # --camera is taken literally (CameraSource refuses a virtual camera).
    explicit = options_from_args(argparse.Namespace(camera=1), state)
    assert (explicit.camera_index, explicit.camera_name) == (1, None)


def test_remembered_virtual_camera_index_is_not_used(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    state = GuiState()
    state.save(camera_index=0, resolution=(1280, 720), preset=None, geometry=QByteArray(), splitter=QByteArray())
    monkeypatch.setattr(camera, "_device_names", lambda: {0: "OBS Virtual Camera", 1: "USB"})

    assert options_from_args(None, state).camera_index == 1


# ---- a renumbered camera, with the real runner and camera source -------------------------


DEVICE_VALUES = {"Built-in": 50, "USB Camera": 100, "Other Camera": 150, "OBS Virtual Camera": 250}


class DeviceWorld:
    """Windows as the app sees it: every capture is bound to the device that held the index when it was opened."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = dict(names)
        self.opens: list[tuple[str, str]] = []  # (device, name of the thread that opened it)
        self.stopped: list[str] = []  # devices stopped by the release of another capture, as on DirectShow
        self.sent_value: int | None = None
        self.device = "Fake Virtual Camera"

    def capture(self, index: int, api: int) -> DeviceCapture:
        return DeviceCapture(self, self.names.get(index))

    def send(self, frame: np.ndarray) -> None:
        self.sent_value = int(frame[0, 0, 0])

    def close(self) -> None:
        pass

    def by_probe(self) -> list[str]:
        return [device for device, thread in self.opens if thread == "camera-probe"]

    def by_runner(self) -> list[str]:
        return [device for device, thread in self.opens if thread != "camera-probe"]


class DeviceCapture:
    def __init__(self, world: DeviceWorld, device: str | None) -> None:
        self.world = world
        self.device = device
        if device is not None:
            world.opens.append((device, threading.current_thread().name))

    def isOpened(self) -> bool:
        return self.device is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.005)
        return True, np.full((48, 64, 3), DEVICE_VALUES[str(self.device)], np.uint8)

    def set(self, _prop: int, _value: float) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return 30.0

    def release(self) -> None:
        if threading.current_thread().name == "camera-probe":
            self.world.stopped.append(str(self.device))


class PassThroughPipeline:
    bypass = False

    def process(self, frame: np.ndarray, _timestamp_ms: int) -> FakeResult:
        return FakeResult(output=frame, preview=frame, original_preview=frame)

    def set_settings(self, _settings: MakeupSettings) -> None:
        pass


@pytest.fixture
def real_window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., tuple]]:
    windows: list[MainWindow] = []

    def make(names: dict[int, str], index: int, name: str) -> tuple[MainWindow, DeviceWorld]:
        world = DeviceWorld(names)
        monkeypatch.setattr(camera, "host_os", lambda: "windows")
        monkeypatch.setattr(camera, "_device_names", lambda: dict(world.names))
        monkeypatch.setattr(camera, "_msmf_device_names", dict)

        def factory(on_frame: Callable, on_status: Callable) -> CameraRunner:
            return CameraRunner(
                PassThroughPipeline(),
                camera_index=index,
                camera_name=name,
                camera_factory=partial(camera.CameraSource, capture_factory=world.capture, reopen_interval_s=0.05),
                vcam_factory=lambda **_: world,
                on_frame=on_frame,
                on_status=on_status,
                retry_interval_s=0.05,
            )

        window = MainWindow(
            factory,
            settings=MakeupSettings(),
            camera_lister=lambda in_use: camera.list_cameras(in_use=in_use, capture_factory=world.capture),
            options=GuiOptions(camera_index=index, camera_name=name),
            dialogs=FakeDialogs(),
        )
        windows.append(window)
        window.start()
        wait_until(lambda: world.sent_value == DEVICE_VALUES[name] and window._rescan_button.isEnabled())
        return window, world

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    QApplication.processEvents()


@pytest.mark.parametrize(
    "takes_the_old_number",
    ["OBS Virtual Camera", "Other Camera"],  # refused and blanked the call / silently the wrong camera
)
def test_a_new_resolution_keeps_the_camera_that_was_renumbered(real_window: Callable, takes_the_old_number: str) -> None:
    window, world = real_window({0: "Built-in", 1: "USB Camera", 2: takes_the_old_number}, 1, "USB Camera")
    world.names = {0: "USB Camera", 1: takes_the_old_number}  # the built-in camera went away

    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())

    wait_until(lambda: len(world.by_runner()) == 2)
    assert world.by_runner() == ["USB Camera", "USB Camera"]
    wait_until(lambda: world.sent_value == DEVICE_VALUES["USB Camera"])
    time.sleep(0.05)
    assert world.sent_value == DEVICE_VALUES["USB Camera"]
    assert window._last_status is None or window._last_status.camera_error is None
    assert window._camera_combo.currentText() == "USB Camera"


def test_a_rescan_never_opens_the_live_camera_under_its_new_number(real_window: Callable) -> None:
    window, world = real_window({0: "Built-in", 1: "USB Camera", 2: "OBS Virtual Camera"}, 1, "USB Camera")
    combo = window._camera_combo
    world.names = {0: "USB Camera", 1: "OBS Virtual Camera"}
    world.opens.clear()
    world.stopped.clear()

    window._rescan_button.click()
    wait_until(lambda: window._rescan_button.isEnabled())

    assert world.by_probe() == []
    assert world.stopped == []  # on DirectShow this would have stopped the live capture
    assert [(combo.itemData(row), combo.itemText(row)) for row in range(combo.count())] == [(0, "USB Camera")]
    assert (combo.currentData(), combo.currentText()) == (0, "USB Camera")
    assert world.by_runner() == []  # no reconnect
    window.close()
    state = GuiState()
    assert (state.camera_index(), state.camera_name()) == (0, "USB Camera")


# ---- a video file instead of a camera ---------------------------------------------------


def test_video_file_is_the_one_camera_entry_and_survives_reselection(make_window: Callable[..., Harness]) -> None:
    options = GuiOptions(camera_file=Path("素材") / "面接の練習.flv")
    harness = make_window(options=options)
    window, runner = harness.window, harness.runner
    window.start()

    combo = window._camera_combo
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")
    assert "テスト用" in combo.toolTip()
    assert not window._rescan_button.isEnabled()
    assert runner.started

    combo.activated.emit(0)  # working: choosing it again changes nothing
    assert runner.cameras == []

    runner.on_status(FakeStatus(camera_ok=False, camera_error="動画ファイルを読み込めませんでした"))
    combo.activated.emit(0)  # failing: choosing it again reconnects
    assert runner.cameras == [(0, 1280, 720)]
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")

    runner.on_status(FakeStatus())
    resolution = window._resolution_combo
    resolution.setCurrentIndex(resolution.findText("640×480"))
    resolution.activated.emit(resolution.currentIndex())
    assert runner.cameras == [(0, 1280, 720), (0, 640, 480)]
    assert (combo.count(), combo.currentText()) == (1, "動画ファイル: 面接の練習.flv")


def test_video_file_does_not_replace_the_remembered_camera(
    make_window: Callable[..., Harness], qapp: QApplication
) -> None:
    state = GuiState()
    state.save(
        camera_index=2,
        camera_name="USB",
        resolution=(960, 540),
        preset=None,
        geometry=QByteArray(),
        splitter=QByteArray(),
    )
    harness = make_window(options=GuiOptions(camera_file=Path("clip.flv"), width=640, height=480))

    harness.window.close()

    state = GuiState()
    assert (state.camera_index(), state.camera_name(), state.resolution()) == (2, "USB", (960, 540))


def test_options_with_a_video_file_ignore_the_remembered_camera(qapp: QApplication) -> None:
    state = GuiState()
    state.save(camera_index=2, resolution=(960, 540), preset=None, geometry=QByteArray(), splitter=QByteArray())

    options = options_from_args(argparse.Namespace(camera_file=Path("clip.flv")), state)

    assert (options.camera_file, options.camera_index, options.camera_name) == (Path("clip.flv"), 0, None)


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
