"""PySide6 settings window with a live preview.

Layout: a top bar (camera, resolution, virtual camera, makeup off, compare),
the preview on the left, and on the right the presets, the background capture
and one page per settings section (generated from ``config.section_fields()``
metadata) behind a grouped list. A status bar runs along the bottom.

Threading: the camera runner calls ``on_frame`` / ``on_status`` on its worker
thread. Those callbacks only hand the newest value to a ``LatestValueRelay``,
which wakes the GUI thread through a queued Qt signal; values that arrive
while the GUI is busy replace older ones, so stale frames are dropped. Widgets
are only touched on the GUI thread. Background captures and photo analyses
answer the same way; both take seconds, and the GUI thread never waits for them.

``MainWindow`` gets its runner from an injected factory and its camera list
from an injected callable, so it can be tested offscreen with fakes; only
``run_gui`` builds the real pipeline, camera runner and camera probe.
"""

from __future__ import annotations

import argparse
import html
import logging
import math
import re
import signal
import sys
import threading
from collections.abc import Callable, Collection, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from PySide6.QtCore import (
    QByteArray,
    QEvent,
    QEventLoop,
    QLibraryInfo,
    QLocale,
    QObject,
    QRect,
    QRegularExpression,
    QSettings,
    QSize,
    Qt,
    QTimer,
    QTranslator,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPalette,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import config
from .config import SECTION_LABELS, SECTIONS, MakeupSettings, section_fields
from .presets import (
    PresetError,
    PresetInfo,
    delete_preset,
    find_preset,
    initial_settings,
    list_presets,
    load_preset,
    sanitize_preset_name,
    save_last_session,
    save_preset,
)

if TYPE_CHECKING:
    from .analyze import AnalysisResult
    from .background import PlateResult
    from .camera import CameraInfo
    from .pipeline import ProcessResult
    from .runner import RunnerStatus

logger = logging.getLogger(__name__)

APP_TITLE = "AI メイクカメラ"
RESOLUTIONS: tuple[tuple[int, int], ...] = ((1280, 720), (960, 540), (640, 480))
SETTINGS_PUSH_INTERVAL_MS = 30
AUTOSAVE_DELAY_MS = 1000
RUNNER_STOP_DEADLINE_S = 2.0
# How often Python gets the chance to run a Ctrl+C handler while Qt waits for events.
SIGNAL_POLL_MS = 200
# How long a window that was closed during a photo analysis waits for it: native code must not outlive Python.
PHOTO_STOP_DEADLINE_S = 15.0
CONSOLE_GONE_WAIT_S = 1.0  # less when the console window was closed: Windows gives the process about 5 s
PLATE_MESSAGE_MS = 10000

# The settings list. A section missing here still gets a page, under SECTION_OTHER_GROUP.
SECTION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("全体", ("general",)),
    ("肌・立体感", ("skin", "contour", "blush")),
    ("目元", ("eyeshadow", "tearbag", "eyeliner", "brows")),
    ("口元", ("lips", "teeth")),
    ("顔の形", ("reshape",)),
    ("髪", ("hair", "hairshape")),
)
SECTION_OTHER_GROUP = "その他"

FACE_HINT = "顔が見つかりません\nカメラの正面に顔が映るようにしてください"
VIRTUAL_CAMERA_HELP = (
    "使い方：Zoom・Microsoft Teams・Google Meet のビデオ設定で、"
    "カメラに『OBS Virtual Camera』を選んでください。"
    "このウィンドウを閉じると仮想カメラも止まります。"
)

BACKGROUND_CAPTURE_HELP = (
    "面接の前に撮影してください。撮影中の映像は相手にもそのまま映ります。\n"
    "押すと数秒後に、誰も映っていない部屋を撮影します（カメラの前から離れてください）。"
    "アホ毛消しと髪型・顔の形の補正で、背景をきれいに残すために使います。\n"
    "カメラの角度を変えたら撮り直してください。"
)
BACKGROUND_CLEAR_HELP = "撮影した背景を、すべての解像度の分まとめて削除します"
# What a saved background plate is called, see background.py.
_PLATE_FILES = "background_*x*.png"

PHOTO_TITLE = "写真から作る"
PHOTO_HELP = (
    "いつものメイクの写真から、同じ見た目になるプリセットを作ります。\n"
    "解析中は映像がカクついたり一瞬止まったりするので、面接中は実行しないでください。写真はどこにも保存しません。"
)
PHOTO_BARE_ADVICE = (
    "すっぴんの写真も、ぜひ入れてください。メイクの写真 1 枚だけでは、アイシャドウをほとんど測れず、"
    "チーク・アイライナー・シェーディング・涙袋も測れないか、ごく一部しか測れません。"
    "すっぴんの写真は、メイクの写真と同じ場所・同じ照明で撮ってください。"
)
PHOTO_TIPS = "きれいに測るコツ：前髪を上げて眉を見せる／メガネを外す／2 枚で構図と服もそろえる／正面を向いて無表情で撮る"
PHOTO_NO_BARE_QUESTION = f"すっぴんの写真なしで作りますか？\n\n{PHOTO_BARE_ADVICE}"
PHOTO_NOT_MEASURED = "測れなかった"
PHOTO_NOTHING_MEASURED_TITLE = "写真から測れませんでした"
PHOTO_NOTHING_MEASURED = (
    "どの項目も測れなかったので、プリセットは保存していません。次のことを確かめて、撮り直してください。\n"
    "・カラーの写真で、顔が白飛びしていないこと\n"
    "・白黒やネガ、強い色のフィルタをかけていないこと\n"
    "・前髪を上げて、メガネを外し、顔が手などで隠れていないこと"
)
_PHOTO_FILTER = "画像ファイル (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;すべてのファイル (*)"

_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{6}")
_ERROR_COLOR = "#D32F2F"
_OK_COLOR = "#2E7D32"
_MODIFIED_COLOR = "#E65100"
_CAMERA_NOT_FOUND = "カメラが見つかりませんでした。接続を確認して ↻ を押してください"
# Camera menu items hold the device index as their data, and the device's real name under this role, if any.
_DEVICE_NAME_ROLE = Qt.ItemDataRole.UserRole + 1

FrameCallback = Callable[["ProcessResult"], None]
StatusCallback = Callable[["RunnerStatus"], None]
# Called with what this app has open, as device names or else indices (listed without re-probing).
CameraLister = Callable[[Collection[int] | Collection[str]], "list[CameraInfo]"]


_SHAPE_HINT_MS = 10000


class Runner(Protocol):
    """What the window needs from ``runner.CameraRunner``."""

    def start(self) -> None: ...
    def stop(self, timeout: float = 2.0) -> None: ...
    def set_settings(self, settings: MakeupSettings) -> None: ...
    def glide_to_settings(self, settings: MakeupSettings) -> None: ...
    def set_bypass(self, bypass: bool) -> None: ...
    def set_camera(
        self, index: int, width: int | None = None, height: int | None = None, name: str | None = None
    ) -> None: ...
    def set_virtual_camera(self, enabled: bool) -> None: ...
    @property
    def has_background(self) -> bool: ...
    def capture_background(self, on_done: Callable[[PlateResult], None]) -> None: ...
    def clear_background(self) -> PlateResult: ...


RunnerFactory = Callable[[FrameCallback, StatusCallback], Runner]
# Called on a worker thread with the makeup photo, the bare photo and the preset to start from.
PhotoAnalyzer = Callable[[Path, "Path | None", "MakeupSettings | None"], "AnalysisResult"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a uint8 BGR HxWx3 frame into a QImage that owns a copy of the pixels."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected a uint8 HxWx3 BGR frame, got {frame.dtype} {frame.shape}")
    pixels = np.ascontiguousarray(frame)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_BGR888)
    return image.copy()


def fit_rect(content: QSize, bounds: QRect) -> QRect:
    """The largest rect with ``content``'s aspect ratio centered in ``bounds``."""
    scaled = content.scaled(bounds.size(), Qt.AspectRatioMode.KeepAspectRatio)
    x = bounds.x() + (bounds.width() - scaled.width()) // 2
    y = bounds.y() + (bounds.height() - scaled.height()) // 2
    return QRect(x, y, scaled.width(), scaled.height())


@contextmanager
def signals_blocked(*objects: QObject) -> Iterator[None]:
    """Temporarily block Qt signals so programmatic updates do not echo back."""
    previous = [obj.blockSignals(True) for obj in objects]
    try:
        yield
    finally:
        for obj, was_blocked in zip(objects, previous, strict=True):
            obj.blockSignals(was_blocked)


def decimals_for_step(step: float) -> int:
    """Digits after the decimal point needed to show multiples of ``step`` (0.01 -> 2)."""
    return min(3, max(0, math.ceil(-math.log10(step) - 1e-9)))


def format_field_value(meta: dict[str, Any], value: Any) -> str:
    """Human-readable value for tooltips (e.g. the reset button's default)."""
    kind = meta.get("kind")
    if kind == "float":
        return f"{value:.{decimals_for_step(meta.get('step', 0.01))}f}"
    if kind == "bool":
        return "ON" if value else "OFF"
    if kind == "choice":
        return dict(meta["choices"]).get(value, str(value))
    return str(value)


def _placeholder_camera_name(index: int) -> str:
    from .camera import default_camera_name

    return default_camera_name(index)


def _selected_row(entries: list[tuple[int, str, int, int]], index: int, name: str | None) -> int | None:
    """The menu row of the selected camera: its name decides, because plugging in a camera renumbers the others."""
    if name is None:
        return next((row for row, entry in enumerate(entries) if entry[0] == index), None)
    same_name = [row for row, entry in enumerate(entries) if entry[1] == name]
    # Two identical cameras have the same name; only there does the index still say which one.
    return next((row for row in same_name if entries[row][0] == index), same_name[0] if same_name else None)


def stop_runner(runner: Runner, deadline_s: float = RUNNER_STOP_DEADLINE_S) -> bool:
    """Stop ``runner`` without blocking the caller for more than ``deadline_s``.

    Returns True if the runner finished stopping in time.
    """

    def stop() -> None:
        try:
            runner.stop(timeout=deadline_s * 0.75)
        except Exception:
            logger.exception("camera runner failed to stop cleanly")

    stopper = threading.Thread(target=stop, name="runner-stop", daemon=True)
    stopper.start()
    stopper.join(deadline_s)
    return not stopper.is_alive()


def grouped_sections(sections: Iterable[str]) -> list[tuple[str, list[str]]]:
    """``SECTION_GROUPS`` limited to ``sections``, plus a last group for any section not listed there."""
    rest = list(sections)
    groups: list[tuple[str, list[str]]] = []
    for title, members in SECTION_GROUPS:
        present = [section for section in members if section in rest]
        if present:
            groups.append((title, present))
            rest = [section for section in rest if section not in present]
    if rest:
        groups.append((SECTION_OTHER_GROUP, rest))
    return groups


def preset_label(info: PresetInfo) -> str:
    """What the preset menu shows: the display name; a built-in preset without one is still marked as such."""
    if info.builtin and info.display_name == info.name:
        return f"{info.name}（標準）"
    return info.display_name


def _plates_on_disk() -> bool:
    """Whether a background plate of any resolution is stored: there is a picture of the room to delete."""
    try:
        return any(config.user_data_dir().glob(_PLATE_FILES))
    except OSError:
        return False


class LatestValueRelay(QObject):
    """Hands the newest value from any thread to the GUI thread, dropping stale ones.

    ``post`` may be called from a worker thread. ``ready`` is emitted once
    until the slot (queued to the relay's thread) calls ``take``; values
    posted in between overwrite each other.
    """

    ready = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._value: Any = None
        self._pending = False

    def post(self, value: Any) -> None:
        with self._lock:
            self._value = value
            if self._pending:
                return
            self._pending = True
        # RuntimeError: the Qt object was deleted while a worker still ran.
        with suppress(RuntimeError):
            self.ready.emit()

    def take(self) -> Any:
        with self._lock:
            value, self._value = self._value, None
            self._pending = False
        return value


class WheelGuard(QObject):
    """Stops unfocused sliders/spin boxes/combos from stealing the mouse wheel.

    Scrolling the settings page must not change a value just because the
    cursor passes over a control (or switch the camera mid-call). The wheel
    event is ignored instead, so it propagates to the scroll area.
    """

    def guard(self, widget: QWidget) -> None:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Wheel and isinstance(watched, QWidget) and not watched.hasFocus():
            event.ignore()
            return True
        return False


# --------------------------------------------------------------------------
# Field editors, one per metadata ``kind``
# --------------------------------------------------------------------------


class FieldEditor(QWidget):
    """Edits one settings field. ``edited`` is emitted for user changes only."""

    edited = Signal(object)
    shows_own_label: ClassVar[bool] = False

    def __init__(self, meta: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.meta = meta
        self.setToolTip(meta.get("help", ""))

    def value(self) -> Any:
        raise NotImplementedError

    def set_value(self, value: Any) -> None:
        """Show ``value`` without emitting ``edited``."""
        raise NotImplementedError


class _NumberEditor(FieldEditor):
    """A slider and a spin box kept in sync; the spin box holds the exact value."""

    def __init__(self, meta: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(meta, parent)
        self._lo = meta["min"]
        self._step = meta.get("step", 1)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, self._to_position(meta["max"]))
        self._slider.setPageStep(10)
        self._spin = self._make_spin_box()
        self._spin.setRange(meta["min"], meta["max"])
        self._spin.setSingleStep(self._step)
        self._spin.setKeyboardTracking(False)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._spin.setFixedWidth(max(76, self._spin.sizeHint().width()))  # the Windows 11 style needs more
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)
        self._slider.valueChanged.connect(self._on_slider_moved)
        self._spin.valueChanged.connect(self._on_spin_changed)

    def _make_spin_box(self) -> QSpinBox | QDoubleSpinBox:
        raise NotImplementedError

    @property
    def slider(self) -> QSlider:
        return self._slider

    @property
    def spin_box(self) -> QSpinBox | QDoubleSpinBox:
        return self._spin

    def value(self) -> Any:
        return self._spin.value()

    def set_value(self, value: Any) -> None:
        with signals_blocked(self._slider, self._spin):
            self._spin.setValue(value)
            self._slider.setValue(self._to_position(value))

    def _to_position(self, value: float) -> int:
        return round((value - self._lo) / self._step)

    def _on_slider_moved(self, position: int) -> None:
        with signals_blocked(self._spin):
            self._spin.setValue(self._lo + position * self._step)
        self.edited.emit(self.value())

    def _on_spin_changed(self, value: float) -> None:
        with signals_blocked(self._slider):
            self._slider.setValue(self._to_position(value))
        self.edited.emit(self.value())


class FloatEditor(_NumberEditor):
    def _make_spin_box(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals_for_step(self.meta.get("step", 0.01)))
        return spin

    def value(self) -> float:
        return round(self._spin.value(), self._spin.decimals())


class IntEditor(_NumberEditor):
    def _make_spin_box(self) -> QSpinBox:
        return QSpinBox()

    def value(self) -> int:
        return int(self._spin.value())


class BoolEditor(FieldEditor):
    shows_own_label = True

    def __init__(self, meta: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(meta, parent)
        self._check = QCheckBox(meta["label"])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._check)
        self._check.toggled.connect(self.edited.emit)

    @property
    def check_box(self) -> QCheckBox:
        return self._check

    def value(self) -> bool:
        return self._check.isChecked()

    def set_value(self, value: Any) -> None:
        with signals_blocked(self._check):
            self._check.setChecked(bool(value))


class ColorEditor(FieldEditor):
    """A color swatch (opens a picker with live preview) plus a "#RRGGBB" text box."""

    def __init__(self, meta: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(meta, parent)
        self._value = ""
        self._swatch = QPushButton()
        self._swatch.setFixedSize(48, 24)
        self._swatch.setToolTip("クリックして色を選ぶ")
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hex = QLineEdit()
        self._hex.setMaxLength(7)
        self._hex.setFixedWidth(84)
        self._hex.setValidator(QRegularExpressionValidator(QRegularExpression("#?[0-9A-Fa-f]{0,6}"), self._hex))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._swatch)
        layout.addWidget(self._hex)
        layout.addStretch(1)
        self._swatch.clicked.connect(self._choose_color)
        self._hex.editingFinished.connect(self._on_hex_edited)

    @property
    def hex_edit(self) -> QLineEdit:
        return self._hex

    def value(self) -> str:
        return self._value

    def set_value(self, value: Any) -> None:
        self._value = str(value).upper()
        self._swatch.setStyleSheet(
            f"QPushButton {{ background-color: {self._value}; border: 1px solid #808080; border-radius: 4px; }}"
        )
        self._hex.setText(self._value)

    def _commit(self, value: str) -> None:
        value = value.upper()
        if value == self._value:
            return
        self.set_value(value)
        self.edited.emit(value)

    def _on_hex_edited(self) -> None:
        text = self._hex.text().strip()
        text = text if text.startswith("#") else f"#{text}"
        if _HEX_COLOR.fullmatch(text):
            self._commit(text)
        else:
            self._hex.setText(self._value)

    def _choose_color(self) -> None:
        """Open a color picker; the makeup follows the picker live, Cancel restores."""
        original = self._value
        dialog = QColorDialog(QColor(original), self)
        dialog.setWindowTitle(self.meta["label"])
        dialog.currentColorChanged.connect(lambda color: self._commit(color.name()))
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        self._commit(dialog.currentColor().name() if accepted else original)


class ChoiceEditor(FieldEditor):
    def __init__(self, meta: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(meta, parent)
        self._combo = QComboBox()
        for value, label in meta["choices"]:
            self._combo.addItem(label, value)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo, 1)
        self._combo.currentIndexChanged.connect(lambda _index: self.edited.emit(self.value()))

    @property
    def combo_box(self) -> QComboBox:
        return self._combo

    def value(self) -> str:
        return self._combo.currentData()

    def set_value(self, value: Any) -> None:
        with signals_blocked(self._combo):
            self._combo.setCurrentIndex(max(0, self._combo.findData(value)))


EDITOR_TYPES: dict[str, type[FieldEditor]] = {
    "float": FloatEditor,
    "int": IntEditor,
    "bool": BoolEditor,
    "color": ColorEditor,
    "choice": ChoiceEditor,
}


# --------------------------------------------------------------------------
# One scrollable tab per settings section
# --------------------------------------------------------------------------


@dataclass
class _FieldRow:
    editor: FieldEditor
    default: Any
    label: QLabel | None
    reset: QToolButton


class SectionPage(QScrollArea):
    """The editors of one settings section, generated from its field metadata.

    Emits ``value_edited(section, field, value)`` for user edits; the
    ``enabled`` checkbox greys out the other rows of the section.
    """

    value_edited = Signal(str, str, object)

    def __init__(self, section: str, section_cls: type, wheel_guard: WheelGuard, intro: str = "") -> None:
        super().__init__()
        self.section = section
        self._rows: dict[str, _FieldRow] = {}
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        if intro:
            layout.addWidget(_help_label(intro))
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        for name, meta, default in section_fields(section_cls):
            self._add_row(grid, name, meta, default)
        layout.addLayout(grid)
        layout.addStretch(1)
        self.setWidget(content)
        for widget_type in (QSlider, QAbstractSpinBox, QComboBox):
            for widget in content.findChildren(widget_type):
                wheel_guard.guard(widget)

    def editor(self, name: str) -> FieldEditor:
        return self._rows[name].editor

    def reset_button(self, name: str) -> QToolButton:
        return self._rows[name].reset

    def set_values(self, values: Any) -> None:
        """Show a section dataclass's values without emitting ``value_edited``."""
        for name, row in self._rows.items():
            row.editor.set_value(getattr(values, name))
            self._update_reset_button(name)
        self._update_enabled_rows()

    def _add_row(self, grid: QGridLayout, name: str, meta: dict[str, Any], default: Any) -> None:
        editor_type = EDITOR_TYPES.get(meta.get("kind", ""))
        if editor_type is None:
            logger.warning("no editor for %s.%s (kind %r)", self.section, name, meta.get("kind"))
            return
        row = grid.rowCount()
        editor = editor_type(meta)
        label = None
        if editor.shows_own_label:
            grid.addWidget(editor, row, 0, 1, 2)
        else:
            label = QLabel(meta["label"])
            label.setToolTip(meta.get("help", ""))
            label.setBuddy(editor)
            grid.addWidget(label, row, 0)
            grid.addWidget(editor, row, 1)
        reset = QToolButton()
        reset.setText("↺")
        reset.setAutoRaise(True)
        reset.setToolTip(f"初期値（{format_field_value(meta, default)}）に戻す")
        grid.addWidget(reset, row, 2)
        editor.set_value(default)
        self._rows[name] = _FieldRow(editor, default, label, reset)
        editor.edited.connect(partial(self._on_edited, name))
        reset.clicked.connect(partial(self._on_reset, name))

    def _on_edited(self, name: str, value: Any) -> None:
        self._update_reset_button(name)
        if name == "enabled":
            self._update_enabled_rows()
        self.value_edited.emit(self.section, name, value)

    def _on_reset(self, name: str) -> None:
        row = self._rows[name]
        row.editor.set_value(row.default)
        self._on_edited(name, row.editor.value())

    def _update_reset_button(self, name: str) -> None:
        row = self._rows[name]
        row.reset.setEnabled(row.editor.value() != row.default)

    def _update_enabled_rows(self) -> None:
        switch = self._rows.get("enabled")
        active = switch is None or bool(switch.editor.value())
        for name, row in self._rows.items():
            if name == "enabled":
                continue
            for widget in (row.editor, row.label, row.reset):
                if widget is not None:
                    widget.setEnabled(active)
            if active:
                self._update_reset_button(name)


def _help_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setStyleSheet(
        "QLabel { background: rgba(25, 118, 210, 0.10); border: 1px solid rgba(25, 118, 210, 0.35);"
        " border-radius: 6px; padding: 8px; }"
    )
    return label


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------


class PreviewWidget(QWidget):
    """Paints the latest frame scaled to fit (aspect kept) plus status overlays."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._hint = ""
        self._badge = ""
        self._notice = "カメラを起動しています…"
        self._banner = ""
        self.setMinimumSize(300, 169)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    @property
    def image(self) -> QImage | None:
        return self._image

    @property
    def hint(self) -> str:
        return self._hint

    @property
    def badge(self) -> str:
        return self._badge

    @property
    def notice(self) -> str:
        return self._notice

    @property
    def banner(self) -> str:
        return self._banner

    def set_image(self, image: QImage) -> None:
        self._image = image
        self.update()

    def set_overlays(self, *, hint: str = "", badge: str = "", notice: str = "", banner: str = "") -> None:
        """``hint``: bottom banner, ``badge``: top-left tag, ``notice``: centered message over a dimmed frame,
        ``banner``: large centered message over the undimmed frame, readable from a few steps away."""
        if (hint, badge, notice, banner) != (self._hint, self._badge, self._notice, self._banner):
            self._hint, self._badge, self._notice, self._banner = hint, badge, notice, banner
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(800, 450)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(17, 17, 17))
        frame_rect = self.rect()
        if self._image is not None and not self._image.isNull():
            frame_rect = fit_rect(self._image.size(), self.rect())
            painter.drawImage(frame_rect, self._image)
        if self._notice:
            painter.fillRect(frame_rect, QColor(0, 0, 0, 160 if self._image is not None else 0))
            self._draw_text(painter, frame_rect.adjusted(24, 24, -24, -24), self._notice, Qt.AlignmentFlag.AlignCenter)
        elif self._banner:
            self._draw_banner(painter, frame_rect, self._banner, scale=1.6, centered=True)
        elif self._hint:
            self._draw_banner(painter, frame_rect, self._hint)
        if self._badge:
            self._draw_badge(painter, frame_rect, self._badge)
        painter.end()

    def _draw_text(self, painter: QPainter, rect: QRect, text: str, align: Qt.AlignmentFlag) -> None:
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(self._overlay_font(1.15))
        painter.drawText(rect, int(align | Qt.TextFlag.TextWordWrap), text)

    def _draw_banner(
        self, painter: QPainter, frame_rect: QRect, text: str, *, scale: float = 1.0, centered: bool = False
    ) -> None:
        painter.setFont(self._overlay_font(scale))
        metrics = painter.fontMetrics()
        flags = int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap)
        text_rect = metrics.boundingRect(QRect(0, 0, frame_rect.width() - 64, 1000), flags, text)
        box = QRect(0, 0, text_rect.width() + 32, text_rect.height() + 16)
        box.moveCenter(frame_rect.center())
        if not centered:
            box.moveBottom(frame_rect.bottom() - 16)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 8, 8)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, flags, text)

    def _draw_badge(self, painter: QPainter, frame_rect: QRect, text: str) -> None:
        painter.setFont(self._overlay_font(0.95, bold=True))
        size = painter.fontMetrics().size(0, text)
        box = QRect(frame_rect.left() + 12, frame_rect.top() + 12, size.width() + 20, size.height() + 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(_MODIFIED_COLOR))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(box, int(Qt.AlignmentFlag.AlignCenter), text)

    def _overlay_font(self, scale: float, *, bold: bool = False) -> QFont:
        font = QFont(self.font())
        font.setPointSizeF(max(9.0, font.pointSizeF() * scale))
        font.setBold(bold)
        return font


# --------------------------------------------------------------------------
# Dialogs and remembered window state (both replaceable in tests)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhotoRequest:
    """What the user chose in the 「写真から作る」 dialog."""

    makeup: Path
    bare: Path | None
    name: str


@dataclass(frozen=True)
class PhotoReport:
    """What a photo analysis found, ready to show.

    ``rows``: one per measured section, (label, confidence or "not measured", measured).
    """

    title: str
    summary: str
    rows: tuple[tuple[str, str, bool], ...]
    notes: tuple[str, ...]


def photo_measured_nothing(result: AnalysisResult) -> bool:
    """True when every measured section has confidence 0: a preset would only copy its base."""
    from .analyze import MEASURED_SECTIONS

    return all(result.confidence.get(section, 0.0) <= 0.0 for section in MEASURED_SECTIONS)


def photo_report(result: AnalysisResult, preset: str | None, base: str, used_bare: bool) -> PhotoReport:
    """The report on ``result``; ``preset`` is the name it was saved under, None if it was not saved."""
    from .analyze import MEASURED_SECTIONS

    if preset is None:
        return PhotoReport(PHOTO_NOTHING_MEASURED_TITLE, PHOTO_NOTHING_MEASURED, (), tuple(result.notes))
    rows = []
    for section in MEASURED_SECTIONS:
        confidence = result.confidence.get(section, 0.0)
        label = SECTION_LABELS.get(section, section)
        if confidence > 0.0:
            rows.append((label, f"{confidence:.2f}", True))
        else:
            rows.append((label, f"{PHOTO_NOT_MEASURED}（{base} の値のまま）", False))
    summary = f"プリセット『{preset}』を保存しました。スライダーで好きなように調整できます。"
    if not used_bare:
        summary += f"\n\n{PHOTO_BARE_ADVICE}"
    return PhotoReport("写真から作りました", summary, tuple(rows), tuple(result.notes))


class PhotoPresetDialog(QDialog):
    """Asks for the photos and the preset name; 「作る」 stays disabled until both are there."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(PHOTO_TITLE)
        self.makeup_edit = QLineEdit()
        self.bare_edit = QLineEdit()
        self.name_edit = QLineEdit()
        self.makeup_edit.setPlaceholderText("必須")
        self.bare_edit.setPlaceholderText("なくても作れますが、入れることを強くおすすめします")
        self.name_edit.setPlaceholderText("例：いつものメイク")
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        for row, (label, edit, hint) in enumerate(
            (
                ("メイクの写真", self.makeup_edit, "いつものメイクをした顔の写真"),
                ("すっぴんの写真", self.bare_edit, "同じ場所・同じ照明で撮ったすっぴんの写真"),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            button = QPushButton("選ぶ…")
            button.setToolTip(hint)
            button.clicked.connect(partial(self._choose_photo, edit, label))
            grid.addWidget(button, row, 2)
        grid.addWidget(QLabel("プリセット名"), 2, 0)
        grid.addWidget(self.name_edit, 2, 1, 1, 2)
        tips = QLabel(PHOTO_TIPS)
        tips.setWordWrap(True)
        self.buttons = QDialogButtonBox()
        self.buttons.addButton("キャンセル", QDialogButtonBox.ButtonRole.RejectRole)
        self.create_button = self.buttons.addButton("作る", QDialogButtonBox.ButtonRole.AcceptRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(PHOTO_HELP))
        layout.addLayout(grid)
        layout.addWidget(_help_label(PHOTO_BARE_ADVICE))
        layout.addWidget(tips)
        layout.addWidget(self.buttons)
        for edit in (self.makeup_edit, self.name_edit):
            edit.textChanged.connect(self._update_create_button)
        self._update_create_button()
        self.setMinimumWidth(min(520, _available_width(self)))

    def request(self) -> PhotoRequest:
        bare = _pasted_path(self.bare_edit.text())
        return PhotoRequest(Path(_pasted_path(self.makeup_edit.text())), Path(bare) if bare else None, self.name_edit.text())

    def _update_create_button(self) -> None:
        self.create_button.setEnabled(bool(self.makeup_edit.text().strip() and self.name_edit.text().strip()))

    def _choose_photo(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), _PHOTO_FILTER)
        if path:
            edit.setText(path)


def _uses_face_shape(settings: MakeupSettings) -> bool:
    """True if the preset narrows the face or moves its features: the room next to it then bends."""
    shape = settings.reshape
    return shape.enabled and any(
        isinstance(value, float) and value != 0.0
        for name, value in vars(shape).items()
        if name != "protect_background"
    )


def _pasted_path(text: str) -> str:
    """A path as typed or pasted: Explorer's "Copy as path" adds double quotes."""
    return text.strip().strip('"').strip()


class PhotoReportDialog(QDialog):
    """Shows a ``PhotoReport``; a long list of notes scrolls."""

    def __init__(self, report: PhotoReport, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(report.title)
        self.body = QLabel(_photo_report_html(report))
        self.body.setTextFormat(Qt.TextFormat.RichText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.body)
        buttons = QDialogButtonBox()
        buttons.addButton("閉じる", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(buttons)
        width = min(560, _available_width(self))
        height = self.body.heightForWidth(width - 40) + 90
        self.resize(width, min(height, 620, _available_height(self)))


def _photo_report_html(report: PhotoReport) -> str:
    parts = [f"<p>{html.escape(line)}</p>" for line in report.summary.split("\n") if line]
    if report.rows:
        parts.append("<p><b>項目ごとの信頼度</b>（1 に近いほど確か）</p><table cellspacing='0' cellpadding='3'>")
        for label, value, measured in report.rows:
            shown = html.escape(value) if measured else f"<b style='color:{_ERROR_COLOR}'>{html.escape(value)}</b>"
            parts.append(f"<tr><td>{html.escape(label)}</td><td>{shown}</td></tr>")
        parts.append("</table>")
    if report.notes:
        items = "".join(f"<li>{html.escape(note)}</li>" for note in report.notes)
        parts.append(f"<p><b>メモ</b></p><ul>{items}</ul>")
    return "".join(parts)


def _available_width(widget: QWidget) -> int:
    screen = widget.screen()
    return screen.availableGeometry().width() if screen is not None else 10000


def _available_height(widget: QWidget) -> int:
    screen = widget.screen()
    return screen.availableGeometry().height() if screen is not None else 10000


class Dialogs:
    """Modal questions the window asks; tests substitute a scripted fake."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.parent = parent

    def confirm(self, title: str, text: str) -> bool:
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        answer = QMessageBox.question(self.parent, title, text, buttons, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def ask_text(self, title: str, label: str, text: str = "") -> str | None:
        value, ok = QInputDialog.getText(self.parent, title, label, QLineEdit.EchoMode.Normal, text)
        return value if ok else None

    def show_error(self, title: str, text: str) -> None:
        QMessageBox.warning(self.parent, title, text)

    def ask_photos(self) -> PhotoRequest | None:
        dialog = PhotoPresetDialog(self.parent)
        return dialog.request() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def show_photo_report(self, report: PhotoReport) -> None:
        PhotoReportDialog(report, self.parent).exec()


class GuiState:
    """Choices remembered between runs (camera, resolution, preset, window layout).

    Stored as an INI file in ``config.user_data_dir()``, next to the presets.
    """

    FILENAME = "gui.ini"

    def __init__(self, path: Path | None = None) -> None:
        location = path or config.user_data_dir() / self.FILENAME
        self._store = QSettings(str(location), QSettings.Format.IniFormat)

    def camera_index(self) -> int | None:
        return _as_int(self._store.value("camera/index"))

    def camera_name(self) -> str | None:
        value = self._store.value("camera/name")
        return value if isinstance(value, str) and value else None

    def resolution(self) -> tuple[int, int] | None:
        width, height = _as_int(self._store.value("camera/width")), _as_int(self._store.value("camera/height"))
        return (width, height) if width and height else None

    def preset(self) -> str | None:
        value = self._store.value("presets/current")
        return value if isinstance(value, str) and value else None

    def geometry(self) -> QByteArray | None:
        return _as_bytes(self._store.value("window/geometry"))

    def splitter(self) -> QByteArray | None:
        return _as_bytes(self._store.value("window/splitter"))

    def save(
        self,
        *,
        camera_index: int | None,
        resolution: tuple[int, int],
        preset: str | None,
        geometry: QByteArray,
        splitter: QByteArray,
        camera_name: str | None = None,
    ) -> None:
        """Remember the choices; ``camera_index=None`` keeps the remembered camera and resolution."""
        if camera_index is not None:
            self._store.setValue("camera/index", camera_index)
            self._store.setValue("camera/name", camera_name or "")
            self._store.setValue("camera/width", resolution[0])
            self._store.setValue("camera/height", resolution[1])
        self._store.setValue("presets/current", preset or "")
        self._store.setValue("window/geometry", geometry)
        self._store.setValue("window/splitter", splitter)
        self._store.sync()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bytes(value: Any) -> QByteArray | None:
    return value if isinstance(value, QByteArray) and not value.isEmpty() else None


@dataclass(frozen=True)
class GuiOptions:
    """How to start: camera, capture size and an explicitly requested preset."""

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    virtual_camera: bool = True
    # A preset named on the command line; it replaces the restored last session.
    preset: str | None = None
    # The device name behind ``camera_index``, when known at start-up.
    camera_name: str | None = None
    # A video file played in place of the webcam (testing without a camera).
    camera_file: Path | None = None

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


def options_from_args(args: argparse.Namespace | None, state: GuiState) -> GuiOptions:
    """Merge command-line arguments (when given) with the remembered state.

    Reads the optional attributes ``camera``, ``camera_file``, ``width``,
    ``height``, ``fps``, ``no_vcam`` and ``preset``; a missing or None attribute
    means "not given". The remembered camera is looked up by name, because its
    index may have changed since the last run.
    """
    from .camera import resolve_camera

    def arg(name: str) -> Any:
        return getattr(args, name, None)

    camera, camera_file = arg("camera"), arg("camera_file")
    width, height = arg("width"), arg("height")
    resolution = (width, height) if width and height else state.resolution() or RESOLUTIONS[0]
    if camera is not None or camera_file is not None:
        camera_index, camera_name = camera or 0, None
    else:
        camera_index, camera_name = resolve_camera(state.camera_index(), state.camera_name())
    return GuiOptions(
        camera_index=camera_index,
        width=resolution[0],
        height=resolution[1],
        fps=arg("fps") or 30,
        virtual_camera=not arg("no_vcam"),
        preset=arg("preset"),
        camera_name=camera_name,
        camera_file=Path(camera_file) if camera_file is not None else None,
    )


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Live preview plus every makeup setting; see the module docstring."""

    def __init__(
        self,
        runner_factory: RunnerFactory,
        *,
        settings: MakeupSettings,
        preset_name: str | None = None,
        camera_lister: CameraLister | None = None,
        options: GuiOptions | None = None,
        dialogs: Dialogs | None = None,
        state: GuiState | None = None,
        photo_analyzer: PhotoAnalyzer | None = None,
    ) -> None:
        super().__init__()
        self._options = options or GuiOptions()
        self._settings = settings.copy()
        self._dialogs = dialogs or Dialogs(self)
        self._state = state or GuiState()
        self._camera_lister = camera_lister
        self._photo_analyzer = photo_analyzer or _analyze_photos
        self._photo_thread: threading.Thread | None = None
        self._photo_request: PhotoRequest | None = None
        self._photo_base = ""
        self._plate_busy = False
        self._plate_message = ""
        self._plates_saved = _plates_on_disk()
        self._presets: dict[str, PresetInfo] = {}
        self._preset_name: str | None = None
        self._preset_baseline: dict[str, Any] | None = None
        # Names of the real cameras as of the latest probe; None until something is known.
        self._camera_names: dict[int, str] | None = (
            {self._options.camera_index: self._options.camera_name} if self._options.camera_name else None
        )
        self._camera_found = True  # whether the latest probe found any device it can vouch for
        self._last_result: ProcessResult | None = None
        self._last_status: RunnerStatus | None = None
        self._comparing = False
        self._probing = False
        self._shut_down = False
        self._runner_stopped = False
        self._wheel_guard = WheelGuard(self)
        self._pages: dict[str, SectionPage] = {}

        self._frame_relay = LatestValueRelay(self)
        self._status_relay = LatestValueRelay(self)
        self._cameras_relay = LatestValueRelay(self)
        self._plate_relay = LatestValueRelay(self)
        self._photo_relay = LatestValueRelay(self)
        self._plate_relay.ready.connect(self._on_plate_done)
        self._photo_relay.ready.connect(self._on_photo_done)
        self._frame_relay.ready.connect(self._on_frame_ready)
        self._status_relay.ready.connect(self._on_status_ready)
        self._cameras_relay.ready.connect(self._on_cameras_ready)
        self._push_timer = _single_shot_timer(self, SETTINGS_PUSH_INTERVAL_MS, self._push_settings)
        self._autosave_timer = _single_shot_timer(self, AUTOSAVE_DELAY_MS, self._persist_session)
        self._plate_message_timer = _single_shot_timer(self, PLATE_MESSAGE_MS, self._clear_plate_message)
        self._runner = runner_factory(self._frame_relay.post, self._status_relay.post)

        self._build_ui()
        for button in self.findChildren(QAbstractButton):
            # Reached with Tab, not by clicking: Space goes on meaning "compare" after a click.
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._camera_choice = self._camera_key()
        self._select_preset(preset_name)
        self._refresh_preset_combo()
        self._show_settings()
        self._update_background_controls()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ---- public API (also used by tests) ---------------------------------

    @property
    def preview(self) -> PreviewWidget:
        return self._preview

    @property
    def preset_name(self) -> str | None:
        return self._preset_name

    def settings(self) -> MakeupSettings:
        return self._settings.copy()

    def section_page(self, section: str) -> SectionPage:
        return self._pages[section]

    def current_section(self) -> str:
        page = self._stack.currentWidget()
        return page.section if isinstance(page, SectionPage) else ""

    def show_section(self, section: str) -> None:
        """Select ``section`` in the settings list, which brings up its page."""
        for row in range(self._nav.count()):
            if self._nav.item(row).data(Qt.ItemDataRole.UserRole) == section:
                self._nav.setCurrentRow(row)
                return

    def is_modified(self) -> bool:
        """Whether the settings differ from the selected preset."""
        return self._preset_baseline is not None and self._settings.to_dict() != self._preset_baseline

    def start(self) -> None:
        """Start the camera worker and look for cameras in the background."""
        self._runner.start()
        self.probe_cameras()

    def probe_cameras(self) -> None:
        """List cameras on a background thread (probing devices can take seconds)."""
        if self._camera_lister is None or self._probing:
            return
        self._probing = True
        self._rescan_button.setEnabled(False)
        self._camera_combo.setToolTip("カメラを検索しています…")
        threading.Thread(
            target=self._probe_worker, args=(self._cameras_in_use(),), name="camera-probe", daemon=True
        ).start()

    def restore_layout(self) -> None:
        """Restore the window geometry and splitter position saved by the last run."""
        geometry, splitter = self._state.geometry(), self._state.splitter()
        if geometry is not None:
            self.restoreGeometry(geometry)
        if splitter is not None:
            self._splitter.restoreState(splitter)

    def shutdown(self) -> bool:
        """Save the session and stop the runner without freezing for more than ~2 s.

        Idempotent. Returns whether the runner stopped within the deadline.
        """
        if self._shut_down:
            return self._runner_stopped
        self._shut_down = True
        self._push_timer.stop()
        self._autosave_timer.stop()
        self._plate_message_timer.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._persist_session()
        self._runner_stopped = stop_runner(self._runner)
        if not self._runner_stopped:
            logger.warning("camera runner did not stop within %.1f s", RUNNER_STOP_DEADLINE_S)
        return self._runner_stopped

    def wait_for_photo_analysis(self, deadline_s: float = PHOTO_STOP_DEADLINE_S) -> bool:
        """Wait (window closed, process about to end) for an analysis under way; True when none is left."""
        thread = self._photo_thread
        if thread is None:
            return True
        thread.join(deadline_s)
        return not thread.is_alive()

    # ---- Qt event handlers -------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        self.shutdown()
        super().closeEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        # A Space release may never arrive once another window has focus.
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow():
            self._set_comparing(False)
        super().changeEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Space anywhere in this window = hold to compare, except on what Space operates (text
        fields, and buttons and lists reached with Tab). Enter presses a button reached with Tab."""
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease) and self._is_compare_key(watched, event):
            if not event.isAutoRepeat():
                self._set_comparing(event.type() == QEvent.Type.KeyPress)
            return True
        if (
            event.type() == QEvent.Type.KeyPress
            and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and isinstance(watched, QAbstractButton)
            and watched.window() is self
        ):
            watched.animateClick()
            return True
        return super().eventFilter(watched, event)

    def _is_compare_key(self, watched: QObject, event: QKeyEvent) -> bool:
        return (
            event.key() == Qt.Key.Key_Space
            and isinstance(watched, QWidget)
            and watched.window() is self
            and not isinstance(watched, (QLineEdit, QAbstractSpinBox, QAbstractButton, QComboBox))
        )

    # ---- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(800, 440)
        self._preview = PreviewWidget()
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._preview)
        self._splitter.addWidget(self._build_settings_panel())
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setSizes([780, 580])
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(8)
        layout.addWidget(self._build_top_bar())
        layout.addWidget(self._splitter, 1)
        self.setCentralWidget(central)
        self._build_status_bar()
        self._fit_to_screen(1360, 800)

    def _fit_to_screen(self, width: int, height: int) -> None:
        """The wanted size, but never more than the screen has (high DPI scaling makes screens small)."""
        screen = self.screen()
        if screen is None:
            self.resize(width, height)
            return
        available = screen.availableGeometry()
        self.resize(min(width, available.width()), min(height, available.height()))

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._camera_combo = QComboBox()
        self._camera_combo.setMinimumContentsLength(14)
        self._camera_combo.setMaximumWidth(360)
        self._camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._camera_combo.activated.connect(self._on_camera_chosen)
        self._rescan_button = QToolButton()
        self._rescan_button.setText("↻")
        self._rescan_button.setToolTip("カメラを再検索")
        self._rescan_button.clicked.connect(self.probe_cameras)
        if self._options.camera_file is not None:
            self._show_camera_file(self._options.camera_file)
        else:
            self._set_camera_items([], self._options.camera_index, self._options.camera_name)

        self._resolutions = list(RESOLUTIONS)
        if self._options.resolution not in self._resolutions:
            self._resolutions.append(self._options.resolution)
        self._resolution_combo = QComboBox()
        for width, height in self._resolutions:
            self._resolution_combo.addItem(f"{width}×{height}")
        self._resolution_combo.setCurrentIndex(self._resolutions.index(self._options.resolution))
        self._resolution_combo.setToolTip(
            "カメラから取り込む解像度です。動作が重いときは下げてください。相手に送る映像の形（16:9）は変わりません"
        )
        self._resolution_combo.activated.connect(self._on_camera_chosen)
        self._wheel_guard.guard(self._camera_combo)
        self._wheel_guard.guard(self._resolution_combo)

        self._vcam_check = QCheckBox("仮想カメラ出力")
        self._vcam_check.setChecked(self._options.virtual_camera)
        self._vcam_check.setToolTip("ONにすると Zoom・Teams・Meet で『OBS Virtual Camera』として映像を使えます")
        self._vcam_check.toggled.connect(self._on_virtual_camera_toggled)

        self._bypass_button = QPushButton("メイクOFF")
        self._bypass_button.setCheckable(True)
        self._bypass_button.setToolTip("押すとメイクを一時的に外します（相手の映像もすっぴんになります）")
        self._bypass_button.toggled.connect(self._on_bypass_toggled)

        self._compare_button = QPushButton("比較（すっぴん）")
        self._compare_button.setToolTip(
            "押している間、メイク前の映像をプレビューに表示します（Space キーでも可）。相手の映像は変わりません"
        )
        self._compare_button.pressed.connect(partial(self._set_comparing, True))
        self._compare_button.released.connect(partial(self._set_comparing, False))

        layout.addWidget(QLabel("カメラ"))
        layout.addWidget(self._camera_combo, 1)  # long device names show in full when there is room
        layout.addWidget(self._rescan_button)
        layout.addSpacing(12)
        layout.addWidget(QLabel("解像度"))
        layout.addWidget(self._resolution_combo)
        layout.addStretch(1)
        layout.addWidget(self._vcam_check)
        layout.addSpacing(8)
        layout.addWidget(self._bypass_button)
        layout.addWidget(self._compare_button)
        return bar

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(470)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._preset_combo = QComboBox()
        self._preset_combo.setPlaceholderText("（プリセット未選択）")
        self._preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._preset_combo.activated.connect(self._on_preset_chosen)
        self._wheel_guard.guard(self._preset_combo)
        self._modified_label = QLabel(f"<span style='color:{_MODIFIED_COLOR}'>● 変更あり</span>")
        self._modified_label.setToolTip(
            "プリセットから変更されています。残すには「保存」か「名前を付けて保存」を押してください"
        )
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("プリセット"))
        preset_row.addWidget(self._preset_combo, 1)
        preset_row.addWidget(self._modified_label)

        self._save_button = QPushButton("保存")
        self._save_as_button = QPushButton("名前を付けて保存…")
        self._delete_button = QPushButton("削除")
        self._defaults_button = QPushButton("初期値に戻す")
        self._photo_button = QPushButton(f"{PHOTO_TITLE}…")
        self._photo_button.setToolTip(PHOTO_HELP)
        self._photo_button.clicked.connect(self._on_from_photo)
        self._save_button.clicked.connect(self._on_save_preset)
        self._save_as_button.clicked.connect(self._on_save_preset_as)
        self._delete_button.clicked.connect(self._on_delete_preset)
        self._defaults_button.clicked.connect(self._on_reset_to_defaults)
        button_row = QHBoxLayout()
        for button in (self._save_button, self._save_as_button, self._delete_button, self._photo_button):
            button_row.addWidget(button)
        button_row.addStretch(1)
        button_row.addWidget(self._defaults_button)

        layout.addLayout(preset_row)
        layout.addLayout(button_row)
        layout.addLayout(self._build_background_row())
        layout.addLayout(self._build_sections(), 1)
        return panel

    def _build_background_row(self) -> QHBoxLayout:
        self._plate_state = QLabel()
        self._capture_button = QPushButton("背景を撮影")
        self._capture_button.setToolTip(BACKGROUND_CAPTURE_HELP)
        self._capture_button.clicked.connect(self._on_capture_background)
        self._clear_plate_button = QPushButton("背景を削除")
        self._clear_plate_button.setToolTip(BACKGROUND_CLEAR_HELP)
        self._clear_plate_button.clicked.connect(self._on_clear_background)
        row = QHBoxLayout()
        row.addWidget(QLabel("背景"))
        row.addWidget(self._plate_state)
        row.addStretch(1)
        row.addWidget(self._capture_button)
        row.addWidget(self._clear_plate_button)
        return row

    def _build_sections(self) -> QHBoxLayout:
        """The grouped list on the left and the page of the selected section on the right."""
        self._nav = QListWidget()
        self._nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._stack = QStackedWidget()
        for title, sections in grouped_sections(SECTIONS):
            single = len(sections) == 1
            if not single:
                header = QListWidgetItem(title)
                header.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable; the arrow keys skip it
                font = QFont(self._nav.font())
                font.setBold(True)
                header.setFont(font)
                color = self._nav.palette().color(QPalette.ColorRole.Text)
                color.setAlphaF(0.65)  # a disabled item would be too faint for a heading
                header.setForeground(color)
                self._nav.addItem(header)
            for section in sections:
                self._add_section_page(section, indent="" if single else "　")
        self._nav.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Row sizes of our own: the Windows 11 style would make the 16 rows taller than many windows.
        metrics = self._nav.fontMetrics()
        items = [self._nav.item(row) for row in range(self._nav.count())]
        widest = max(metrics.horizontalAdvance(item.text()) for item in items) + metrics.height() * 2
        for item in items:
            item.setSizeHint(QSize(widest, metrics.height() + 10))
        self._nav.setFixedWidth(widest + 2 * self._nav.frameWidth() + self._nav.verticalScrollBar().sizeHint().width())
        self._nav.currentRowChanged.connect(self._on_section_chosen)
        self.show_section(next(iter(SECTIONS)))
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._nav)
        row.addWidget(self._stack, 1)
        return row

    def _add_section_page(self, section: str, indent: str) -> None:
        intro = VIRTUAL_CAMERA_HELP if section == "general" else ""
        page = SectionPage(section, SECTIONS[section], self._wheel_guard, intro)
        page.value_edited.connect(self._on_value_edited)
        self._pages[section] = page
        self._stack.addWidget(page)
        item = QListWidgetItem(indent + SECTION_LABELS.get(section, section))
        item.setData(Qt.ItemDataRole.UserRole, section)
        self._nav.addItem(item)

    def _on_section_chosen(self, row: int) -> None:
        item = self._nav.item(row)
        section = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if section in self._pages:
            self._stack.setCurrentWidget(self._pages[section])

    def _build_status_bar(self) -> None:
        status = self.statusBar()
        self._camera_status = QLabel()
        self._processing_status = QLabel()
        self._face_status = QLabel("顔検出 --")
        self._fps_status = QLabel("FPS --")
        self._time_status = QLabel("処理 -- ms")
        self._vcam_status = QLabel()
        self._vcam_status.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self._vcam_status.linkActivated.connect(self._show_virtual_camera_error)
        labels = (
            self._camera_status,
            self._processing_status,
            self._face_status,
            self._fps_status,
            self._time_status,
            self._vcam_status,
        )
        for label in labels:
            label.setContentsMargins(8, 0, 8, 0)
            status.addPermanentWidget(label)
        self._camera_status.hide()
        self._processing_status.hide()
        self._update_virtual_camera_status()

    # ---- settings editing ------------------------------------------------

    def _on_value_edited(self, section: str, name: str, value: Any) -> None:
        setattr(getattr(self._settings, section), name, value)
        self._settings_changed()

    def _settings_changed(self) -> None:
        # Throttle (not debounce): pushes keep flowing while a slider is dragged.
        if not self._push_timer.isActive():
            self._push_timer.start()
        self._autosave_timer.start()
        self._update_preset_controls()

    def _push_settings(self) -> None:
        self._runner.set_settings(self._settings.copy())

    def _apply_settings(self, settings: MakeupSettings) -> None:
        """Replace all settings (preset load / reset) and refresh every widget."""
        self._settings = settings.copy()
        self._show_settings()
        self._push_timer.stop()
        self._runner.glide_to_settings(self._settings.copy())
        self._autosave_timer.start()
        self._update_preset_controls()

    def _show_settings(self) -> None:
        for section, page in self._pages.items():
            page.set_values(getattr(self._settings, section))

    def _persist_session(self) -> None:
        save_last_session(self._settings)
        camera_index, camera_name = self._camera_to_remember()
        self._state.save(
            camera_index=camera_index,
            camera_name=camera_name,
            resolution=self._current_resolution(),
            preset=self._preset_name,
            geometry=self.saveGeometry(),
            splitter=self._splitter.saveState(),
        )

    def _camera_to_remember(self) -> tuple[int | None, str | None]:
        """What the next run should start with; an index of None keeps what is stored.

        A test video, or a selection known not to be a real camera (a virtual
        camera at the default index, an unplugged device), replaces nothing. Nor
        does a bare number replace a stored name, which says more.
        """
        if self._options.camera_file is not None:
            return None, None
        index, name = self._current_camera_index(), self._current_camera_name()
        if self._camera_names is not None and not self._is_listed(index, name):
            return None, None
        if name is None and self._state.camera_name():
            return None, None
        return index, name

    def _is_listed(self, index: int, name: str | None) -> bool:
        names = self._camera_names or {}
        return name in names.values() if name else index in names

    # ---- presets ---------------------------------------------------------

    def _select_preset(self, name: str | None) -> None:
        """Mark ``name`` as the current preset without changing the settings."""
        self._preset_name = name if name and find_preset(name) is not None else None
        self._preset_baseline = None
        if self._preset_name is not None:
            try:
                self._preset_baseline = load_preset(self._preset_name).to_dict()
            except PresetError:
                logger.warning("preset %r is unreadable; not tracking changes against it", self._preset_name)

    def _current_preset(self) -> PresetInfo | None:
        return self._presets.get(self._preset_name) if self._preset_name else None

    def _refresh_preset_combo(self) -> None:
        self._presets = {info.name: info for info in list_presets()}
        with signals_blocked(self._preset_combo):
            self._preset_combo.clear()
            for info in self._presets.values():
                self._preset_combo.addItem(preset_label(info), info.name)
            self._preset_combo.setCurrentIndex(self._preset_combo.findData(self._preset_name))
        self._update_preset_controls()

    def _update_preset_controls(self) -> None:
        current = self._current_preset()
        is_user_preset = current is not None and not current.builtin
        self._save_button.setEnabled(is_user_preset and self.is_modified())
        self._delete_button.setEnabled(is_user_preset)
        self._save_button.setToolTip(
            "標準プリセットは上書きできません。「名前を付けて保存」を使ってください"
            if current is not None and current.builtin
            else "選択中のプリセットに今の設定を上書き保存します"
        )
        self._modified_label.setVisible(self.is_modified())

    def _preset_title(self, name: str) -> str:
        """How the user knows this preset (built-in file names are not what she sees)."""
        info = self._presets.get(name)
        return info.display_name if info is not None else name

    def _on_preset_chosen(self, index: int) -> None:
        name = self._preset_combo.itemData(index)
        if name == self._preset_name and not self.is_modified():
            return
        if self._has_unsaved_changes() and not self._dialogs.confirm(
            "プリセットの読み込み",
            f"保存していない変更があります。変更を破棄してプリセット『{self._preset_title(name)}』を読み込みますか？",
        ):
            self._refresh_preset_combo()
            return
        self.apply_preset(name)

    def _has_unsaved_changes(self) -> bool:
        return self.is_modified() or (self._preset_name is None and self._settings != MakeupSettings())

    def apply_preset(self, name: str) -> bool:
        """Load a preset into every widget and the runner. Returns False on error (shown)."""
        try:
            settings = load_preset(name)
        except PresetError as exc:
            self._dialogs.show_error("プリセット", str(exc))
            self._refresh_preset_combo()
            return False
        self._preset_name = name
        self._preset_baseline = settings.to_dict()
        self._refresh_preset_combo()
        self._apply_settings(settings)
        message = f"プリセット『{self._preset_title(name)}』を読み込みました"
        if _uses_face_shape(settings) and not self._runner.has_background:
            message += "。顔の形の補正は、「背景を撮影」をしておくと顔のまわりの背景がゆがみません"
        self.statusBar().showMessage(message, _SHAPE_HINT_MS if "背景を撮影" in message else 4000)
        return True

    def _on_save_preset(self) -> None:
        current = self._current_preset()
        if current is None or current.builtin:
            return
        self._store_preset(current.name)

    def _on_save_preset_as(self) -> None:
        name = self._dialogs.ask_text("名前を付けて保存", "プリセット名：", self._preset_name or "")
        if name is None:
            return
        try:
            safe_name = sanitize_preset_name(name)
        except PresetError as exc:
            self._dialogs.show_error("プリセット", str(exc))
            return
        existing = find_preset(safe_name)
        if existing is not None and not self._dialogs.confirm("上書きの確認", _overwrite_question(existing)):
            return
        self._store_preset(safe_name)

    def _store_preset(self, name: str) -> None:
        try:
            info = save_preset(name, self._settings)
        except PresetError as exc:
            self._dialogs.show_error("プリセット", str(exc))
            return
        self._preset_name = info.name
        self._preset_baseline = self._settings.to_dict()
        self._refresh_preset_combo()
        self._autosave_timer.start()
        self.statusBar().showMessage(f"プリセット『{info.name}』を保存しました", 4000)

    def _on_delete_preset(self) -> None:
        current = self._current_preset()
        if current is None or current.builtin:
            return
        if not self._dialogs.confirm(
            "プリセットの削除", f"プリセット『{current.name}』を削除しますか？\n（今の設定はそのまま残ります）"
        ):
            return
        try:
            delete_preset(current.name)
        except PresetError as exc:
            self._dialogs.show_error("プリセット", str(exc))
            return
        # A built-in preset of the same name, if any, becomes current again.
        self._select_preset(current.name)
        self._refresh_preset_combo()
        self._autosave_timer.start()
        self.statusBar().showMessage(f"プリセット『{current.name}』を削除しました", 4000)

    def _on_reset_to_defaults(self) -> None:
        if not self._dialogs.confirm("初期値に戻す", "すべての設定を初期値に戻しますか？"):
            return
        self._preset_name = None
        self._preset_baseline = None
        self._refresh_preset_combo()
        self._apply_settings(MakeupSettings())
        self.statusBar().showMessage("すべての設定を初期値に戻しました", 4000)

    # ---- 写真から作る ------------------------------------------------------

    def _on_from_photo(self) -> None:
        if self._photo_thread is not None:
            return
        request = self._dialogs.ask_photos()
        if request is None:
            return
        if request.bare is None and not self._dialogs.confirm(PHOTO_TITLE, PHOTO_NO_BARE_QUESTION):
            return
        base, self._photo_base = self._photo_base_preset()
        self._photo_request = request
        self._photo_button.setEnabled(False)
        self._photo_button.setText("解析中…")
        self.statusBar().showMessage("写真を解析しています…（数秒かかります）")
        self._photo_thread = threading.Thread(
            target=self._photo_worker, args=(request, base), name="photo-analysis", daemon=True
        )
        self._photo_thread.start()

    def _photo_base_preset(self) -> tuple[MakeupSettings | None, str]:
        """What the sections the photos cannot measure start from: the selected preset, or else natural."""
        if self._preset_name is not None:
            try:
                return load_preset(self._preset_name), self._preset_title(self._preset_name)
            except PresetError:
                logger.warning("preset %r is unreadable; starting the analysis from natural", self._preset_name)
        return None, self._preset_title("natural")

    def _photo_worker(self, request: PhotoRequest, base: MakeupSettings | None) -> None:
        """On its own thread. The photos stay in memory here and are not kept or saved anywhere."""
        try:
            self._photo_relay.post(self._photo_analyzer(request.makeup, request.bare, base))
        except Exception as exc:
            self._photo_relay.post(_photo_failure_message(exc))

    def _on_photo_done(self) -> None:
        outcome = self._photo_relay.take()
        request = self._photo_request
        if outcome is None or request is None:
            return
        self._photo_thread, self._photo_request = None, None
        self._photo_button.setEnabled(True)
        self._photo_button.setText(f"{PHOTO_TITLE}…")
        self.statusBar().clearMessage()
        if self._shut_down:
            return
        if isinstance(outcome, str):
            self._dialogs.show_error(PHOTO_TITLE, outcome)
        elif photo_measured_nothing(outcome):
            self._dialogs.show_photo_report(photo_report(outcome, None, self._photo_base, request.bare is not None))
        else:
            self._save_photo_preset(outcome, request)

    def _save_photo_preset(self, result: AnalysisResult, request: PhotoRequest) -> None:
        """Save under the requested name; a refused name costs another name, not another analysis."""
        from .analyze import save_as_preset

        name: str | None = request.name
        while name is not None:
            existing = find_preset(name) if self._is_valid_preset_name(name) else None
            if existing is None or existing.builtin or self._dialogs.confirm("上書きの確認", _overwrite_question(existing)):
                try:
                    info = save_as_preset(result, name)
                except PresetError as exc:
                    self._dialogs.show_error(PHOTO_TITLE, str(exc))
                else:
                    self._use_photo_preset(info)
                    self._dialogs.show_photo_report(
                        photo_report(result, info.name, self._photo_base, request.bare is not None)
                    )
                    return
            name = self._dialogs.ask_text(PHOTO_TITLE, "別のプリセット名：", name)
        self.statusBar().showMessage("プリセットは保存しませんでした", 4000)

    @staticmethod
    def _is_valid_preset_name(name: str) -> bool:
        try:
            sanitize_preset_name(name)
        except PresetError:
            return False
        return True

    def _use_photo_preset(self, info: PresetInfo) -> None:
        if self._has_unsaved_changes() and not self._dialogs.confirm(
            PHOTO_TITLE,
            f"プリセット『{info.name}』を保存しました。今の変更を破棄して、このプリセットに切り替えますか？",
        ):
            self._refresh_preset_combo()
            return
        self.apply_preset(info.name)

    # ---- background capture ----------------------------------------------

    def _on_capture_background(self) -> None:
        if self._plate_busy:
            return
        self._plate_busy = True
        self._clear_plate_message()
        self._update_background_controls()
        self._runner.capture_background(self._plate_relay.post)

    def _on_plate_done(self) -> None:
        result = self._plate_relay.take()
        if result is None:
            return
        self._plate_busy = False
        self._show_plate_result(result)

    def _on_clear_background(self) -> None:
        self._show_plate_result(self._runner.clear_background())

    def _show_plate_result(self, result: PlateResult) -> None:
        self._plate_message = result.message
        self._plate_message_timer.start()
        self.statusBar().showMessage(result.message, PLATE_MESSAGE_MS)
        self._plates_saved = _plates_on_disk()
        self._update_background_controls()
        self._update_preview_overlays()

    def _clear_plate_message(self) -> None:
        self._plate_message = ""
        self._update_preview_overlays()

    def _update_background_controls(self) -> None:
        in_use = self._runner.has_background
        self._capture_button.setEnabled(not self._plate_busy)
        self._capture_button.setText("撮影中…" if self._plate_busy else "背景を撮影")
        self._clear_plate_button.setEnabled(in_use or self._plates_saved)
        self._plate_state.setText("撮影中…" if self._plate_busy else "撮影済み" if in_use else "未撮影")

    # ---- camera / output controls ----------------------------------------

    def _set_camera_items(self, cameras: list[CameraInfo], selected: int, selected_name: str | None = None) -> None:
        """Fill the camera combo; the selected camera stays listed even if the probe missed it."""
        entries = [(camera.index, camera.name, camera.width, camera.height) for camera in cameras]
        row = _selected_row(entries, selected, selected_name)
        if row is None:
            row = 0
            entries.insert(0, (selected, selected_name or _placeholder_camera_name(selected), 0, 0))
        with signals_blocked(self._camera_combo):
            self._camera_combo.clear()
            for index, name, width, height in entries:
                self._camera_combo.addItem(name, index)
                item = self._camera_combo.count() - 1
                tooltip = f"デバイス番号 {index}" + (f"（{width}×{height}）" if width and height else "")
                self._camera_combo.setItemData(item, tooltip, Qt.ItemDataRole.ToolTipRole)
                device_name = None if name == _placeholder_camera_name(index) else name
                self._camera_combo.setItemData(item, device_name, _DEVICE_NAME_ROLE)
            self._camera_combo.setCurrentIndex(row)

    def _show_camera_file(self, path: Path) -> None:
        """The one camera entry while a video file stands in for the webcam."""
        self._camera_combo.addItem(f"動画ファイル: {path.name}", self._options.camera_index)
        self._camera_combo.setToolTip(f"カメラの代わりに動画ファイルを再生しています（カメラなしでのテスト用）\n{path}")
        self._rescan_button.setEnabled(False)
        self._rescan_button.setToolTip("動画ファイルの再生中はカメラを検索しません")

    def _current_camera_index(self) -> int:
        index = self._camera_combo.currentData()
        return index if index is not None else self._options.camera_index

    def _current_camera_name(self) -> str | None:
        """The selected device's real name, if known."""
        return self._camera_combo.currentData(_DEVICE_NAME_ROLE)

    def _current_resolution(self) -> tuple[int, int]:
        return self._resolutions[max(0, self._resolution_combo.currentIndex())]

    def _camera_key(self) -> tuple[int | str, int, int]:
        """What identifies the current choice of camera and size."""
        return (self._current_camera_name() or self._current_camera_index(), *self._current_resolution())

    def _camera_failing(self) -> bool:
        return self._last_status is not None and bool(self._last_status.camera_error)

    def _cameras_in_use(self) -> tuple[int, ...] | tuple[str, ...]:
        """What the probe must leave alone; on DirectShow, opening and closing a live device stops it.

        A device name finds the device wherever it is now. Without a name, or among
        identical cameras, there is only the index, and not while the camera is
        failing: nothing is open then, so the probe can find out what is really there.
        """
        name = self._current_camera_name()
        if name is not None and list((self._camera_names or {}).values()).count(name) < 2:
            return (name,)
        return () if self._camera_failing() else (self._current_camera_index(),)

    def _probe_worker(self, in_use: Collection[int] | Collection[str]) -> None:
        try:
            cameras = list(self._camera_lister(in_use)) if self._camera_lister else []
        except Exception:
            logger.exception("listing cameras failed")
            cameras = []
        self._cameras_relay.post(cameras)

    def _on_cameras_ready(self) -> None:
        cameras = self._cameras_relay.take()
        if cameras is None:
            return
        self._probing = False
        self._rescan_button.setEnabled(True)
        self._camera_names = {camera.index: camera.name for camera in cameras}
        # Listed without being probed and without a name: only the selection itself, which may not exist.
        self._camera_found = any(
            camera.width or camera.height or camera.name != _placeholder_camera_name(camera.index)
            for camera in cameras
        )
        self._update_camera_hint()
        index, name = self._current_camera_index(), self._current_camera_name()
        # The selection failed and is not a real camera (e.g. a virtual one), and one has just appeared: use it.
        if self._camera_failing() and cameras and not self._is_listed(index, name):
            self._set_camera_items(cameras, cameras[0].index, cameras[0].name)
            self._on_camera_chosen(self._camera_combo.currentIndex())
        else:
            self._set_camera_items(cameras, index, name)
            self._camera_choice = self._camera_key()  # the same camera, possibly under a new number

    def _update_camera_hint(self) -> None:
        if self._probing or self._options.camera_file is not None:
            return
        working = self._last_status is not None and self._last_status.camera_ok
        self._camera_combo.setToolTip(_CAMERA_NOT_FOUND if not self._camera_found and not working else "")

    def _on_camera_chosen(self, _index: int) -> None:
        # Re-choosing the working camera would only blank the call while it reconnects.
        if self._camera_key() == self._camera_choice and self._camera_ok():
            return
        self._camera_choice = self._camera_key()
        width, height = self._current_resolution()
        self._runner.set_camera(self._current_camera_index(), width, height, name=self._current_camera_name())
        self._autosave_timer.start()

    def _on_virtual_camera_toggled(self, enabled: bool) -> None:
        self._runner.set_virtual_camera(enabled)
        self._update_virtual_camera_status()

    def _on_bypass_toggled(self, bypass: bool) -> None:
        self._runner.set_bypass(bypass)
        self._show_latest_frame()

    def _set_comparing(self, comparing: bool) -> None:
        if comparing != self._comparing:
            self._comparing = comparing
            self._compare_button.setDown(comparing)
            self._show_latest_frame()

    # ---- frames and status from the worker -------------------------------

    def _on_frame_ready(self) -> None:
        result = self._frame_relay.take()
        if result is not None:
            self._last_result = result
            self._show_latest_frame()

    def _show_latest_frame(self) -> None:
        result = self._last_result
        if result is not None:
            frame = result.original_preview if self._comparing else result.preview
            self._preview.set_image(bgr_to_qimage(frame))
        self._update_preview_overlays()

    def _camera_ok(self) -> bool:
        return self._last_status is None or self._last_status.camera_ok

    def _update_preview_overlays(self) -> None:
        result, status = self._last_result, self._last_status
        badge = "比較中：メイクなし" if self._comparing else "メイクOFF中" if self._bypass_button.isChecked() else ""
        notice = ""
        if status is not None and status.camera_error:
            notice = f"カメラに接続できません\n{status.camera_error}"
        elif result is None or not self._camera_ok():
            notice = "カメラを起動しています…"
        face_missing = result is not None and not result.face_found
        # She is walking out of the picture on purpose: no "face not found" meanwhile.
        banner = self._plate_message or (status.background_message if status is not None else None) or ""
        hint = FACE_HINT if face_missing and not badge and not banner else ""
        self._preview.set_overlays(hint=hint, badge=badge, notice=notice, banner=banner)

    def _on_status_ready(self) -> None:
        status = self._status_relay.take()
        if status is None:
            return
        self._last_status = status
        self._fps_status.setText(f"FPS {status.fps:.1f}")
        self._time_status.setText(f"処理 {status.processing_ms:.1f} ms")
        face_text, face_color = ("OK", _OK_COLOR) if status.face_found else ("NG", _ERROR_COLOR)
        self._face_status.setText(f"顔検出 <b style='color:{face_color}'>{face_text}</b>")
        _show_problem(self._camera_status, "カメラ", "未接続", status.camera_error)
        _show_problem(self._processing_status, "メイク処理", "エラー", status.processing_error)
        self._update_camera_hint()
        self._update_virtual_camera_status()
        self._update_preview_overlays()
        self._update_background_controls()

    def _update_virtual_camera_status(self) -> None:
        status = self._last_status
        error = status.virtual_camera_error if status is not None else None
        label = self._vcam_status
        if not self._vcam_check.isChecked():
            label.setText("仮想カメラ OFF")
            label.setToolTip("")
        elif error:
            label.setText(f"仮想カメラ <b style='color:{_ERROR_COLOR}'>使用できません</b> <a href='details'>詳細</a>")
            label.setToolTip(error)
        elif status is not None and status.virtual_camera:
            label.setText(f"仮想カメラ <b style='color:{_OK_COLOR}'>{html.escape(status.virtual_camera)}</b>")
            label.setToolTip("ビデオ会議アプリのカメラ設定でこのデバイスを選んでください")
        else:
            label.setText("仮想カメラ 起動中…")
            label.setToolTip("")

    def _show_virtual_camera_error(self, _link: str = "") -> None:
        status = self._last_status
        if status is not None and status.virtual_camera_error:
            self._dialogs.show_error("仮想カメラを使用できません", status.virtual_camera_error)


def _analyze_photos(makeup_path: Path, bare_path: Path | None, base: MakeupSettings | None) -> AnalysisResult:
    from .analyze import analyze_makeup, read_photo

    makeup = read_photo(makeup_path)
    bare = read_photo(bare_path) if bare_path is not None else None
    return analyze_makeup(makeup, bare, base=base)


def _photo_failure_message(exc: Exception) -> str:
    """The Japanese message of an expected failure (no face, unreadable photo, no model); else a general one."""
    from .analyze import AnalysisError
    from .models import ModelDownloadError

    if isinstance(exc, (AnalysisError, ModelDownloadError)):
        return str(exc)
    logger.error("photo analysis failed", exc_info=exc)
    return f"写真を解析できませんでした（詳細はログを確認してください）。\n{type(exc).__name__}"


def _show_problem(label: QLabel, subject: str, state: str, detail: str | None) -> None:
    """Show ``label`` as "<subject> <red state>" with ``detail`` as tooltip, or hide it."""
    label.setVisible(bool(detail))
    label.setText(f"{subject} <b style='color:{_ERROR_COLOR}'>{state}</b>")
    label.setToolTip(detail or "")


def _overwrite_question(existing: PresetInfo) -> str:
    if existing.builtin:
        return (
            f"『{existing.name}』は標準プリセットと同じ名前です。今の設定で上書きしますか？\n"
            "（あとで削除すると標準の内容に戻ります）"
        )
    return f"プリセット『{existing.name}』はすでにあります。上書きしますか？"


def _single_shot_timer(parent: QObject, interval_ms: int, slot: Callable[[], None]) -> QTimer:
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(interval_ms)
    timer.timeout.connect(slot)
    return timer


# --------------------------------------------------------------------------
# Application entry point
# --------------------------------------------------------------------------


def run_gui(args: argparse.Namespace | None = None) -> int:
    """Run the settings window until it is closed; returns the process exit code.

    ``args`` may carry ``camera``, ``camera_file``, ``width``, ``height``,
    ``fps``, ``no_vcam`` and ``preset`` (see ``options_from_args``).
    """
    from .camera import CameraSource, list_cameras, video_file_camera_factory, virtual_camera_size
    from .pipeline import MakeupPipeline
    from .runner import CameraRunner, ConsoleCloseWatch

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("ai-makeup-cam")
    _use_japanese_qt_strings(app)
    state = GuiState()
    options = options_from_args(args, state)
    error = prepare_models()
    if error:
        QMessageBox.critical(None, APP_TITLE, error)
        return 1
    settings, preset_name = initial_settings(options.preset)
    try:
        pipeline = MakeupPipeline(settings)
    except Exception as exc:  # model files unreadable, MediaPipe failed to start, ...
        logger.exception("could not start the makeup pipeline")
        QMessageBox.critical(None, APP_TITLE, f"顔認識の準備に失敗しました。\n{exc}")
        return 1
    camera_file = options.camera_file

    def make_runner(on_frame: FrameCallback, on_status: StatusCallback) -> Runner:
        return CameraRunner(
            pipeline,
            camera_index=options.camera_index,
            camera_name=options.camera_name,
            width=options.width,
            height=options.height,
            fps=options.fps,
            virtual_camera=options.virtual_camera,
            output_size=virtual_camera_size(options.width, options.height),
            on_frame=on_frame,
            on_status=on_status,
            camera_factory=video_file_camera_factory(camera_file) if camera_file is not None else CameraSource,
        )

    window: MainWindow | None = None
    console = ConsoleCloseWatch()
    try:
        window = MainWindow(
            make_runner,
            settings=settings,
            preset_name=preset_name or state.preset(),
            camera_lister=None if camera_file is not None else lambda in_use: list_cameras(in_use=in_use),
            options=options,
            state=state,
        )
        window.restore_layout()
        window.show()
        window.start()
        with closed_by_ctrl_c(window, console.closing):
            return app.exec()
    finally:
        # A runner that did not stop may still be inside the pipeline; leave it to process exit.
        if window is None or window.shutdown():
            pipeline.close()
        if window is not None:
            window.wait_for_photo_analysis(CONSOLE_GONE_WAIT_S if console.closing.is_set() else PHOTO_STOP_DEADLINE_S)
            # Now, while QApplication exists: a widget destroyed after it ends the process in an access violation.
            window.deleteLater()
            QApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        console.finish()


@contextmanager
def closed_by_ctrl_c(window: QWidget, console_closing: threading.Event | None = None) -> Iterator[None]:
    """Close the windows on Ctrl+C, or when the terminal goes away, exactly like closing them by hand.

    Python runs signal handlers only between bytecodes, which never happens
    while Qt's event loop sleeps, so a cheap repeating timer wakes it up.
    All of them: a modal dialog left open would keep the application running.
    """

    def close_everything(_signum: int, _frame: object) -> None:
        QApplication.closeAllWindows()

    def poll() -> None:
        if console_closing is not None and console_closing.is_set():
            QApplication.closeAllWindows()

    previous = {
        number: signal.signal(number, close_everything)
        for number in (getattr(signal, name, None) for name in ("SIGINT", "SIGTERM", "SIGBREAK", "SIGHUP"))
        if number is not None
    }
    timer = QTimer(window)
    timer.timeout.connect(poll)
    timer.start(SIGNAL_POLL_MS)
    try:
        yield
    finally:
        timer.stop()
        for number, handler in previous.items():
            signal.signal(number, handler)


def _use_japanese_qt_strings(app: QApplication) -> None:
    """Qt's own buttons (Yes, Cancel, the color dialog) in Japanese, when its translation is installed."""
    translator = QTranslator(app)
    folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(QLocale.Language.Japanese), "qtbase", "_", folder):
        app.installTranslator(translator)


def prepare_models() -> str | None:
    """Download missing models with a progress dialog. Returns a Japanese error or None."""
    from .models import MODEL_SPECS, ModelDownloadError, ensure_all_models, models_dir

    if all((models_dir() / spec.filename).is_file() for spec in MODEL_SPECS.values()):
        return None
    dialog = QProgressDialog("顔認識モデルをダウンロードしています（初回のみ）…", "", 0, 0)
    dialog.setWindowTitle(APP_TITLE)
    dialog.setCancelButton(None)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    progress = LatestValueRelay(dialog)
    finished = LatestValueRelay(dialog)
    progress.ready.connect(lambda: _show_download_progress(dialog, progress.take()))
    loop = QEventLoop()
    finished.ready.connect(loop.quit)
    outcome: dict[str, str] = {}

    def download() -> None:
        try:
            ensure_all_models(lambda name, done, total: progress.post((name, done, total)))
        except ModelDownloadError as exc:
            outcome["error"] = str(exc)
        except Exception as exc:
            logger.exception("model download failed")
            outcome["error"] = f"顔認識モデルを準備できませんでした。\n{exc}"
        finally:
            finished.post(True)

    threading.Thread(target=download, name="model-download", daemon=True).start()
    dialog.show()
    loop.exec()
    dialog.close()
    return outcome.get("error")


def _show_download_progress(dialog: QProgressDialog, progress: tuple[str, int, int] | None) -> None:
    if progress is None:
        return
    name, done, total = progress
    dialog.setMaximum(max(total, 0))
    dialog.setValue(min(done, total) if total else 0)
    size = f"{done / 1e6:.1f} / {total / 1e6:.1f} MB" if total else f"{done / 1e6:.1f} MB"
    dialog.setLabelText(f"顔認識モデルをダウンロードしています（初回のみ）…\n{name}：{size}")
