"""Named makeup presets, the automatically saved last session and the startup settings.

A preset is a ``MakeupSettings`` JSON file whose file stem is its name.
Built-in presets ship in ``BUILTIN_PRESETS_DIR`` and are read-only; user
presets live in ``user_presets_dir()``. A user preset with the same name as a
built-in one overrides it, and deleting that user preset brings the built-in
one back.

Error policy: functions the user triggers explicitly (``load_preset``,
``save_preset``, ``delete_preset``) raise ``PresetError`` with a Japanese
message the GUI can show as is. The last-session helpers are best effort:
a missing, unreadable or corrupt session file is logged and treated as "no
session" so the app always starts.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import config
from .config import MakeupSettings

logger = logging.getLogger(__name__)

PRESET_SUFFIX = ".json"
# Built-in preset used when there is no saved session (就活ナチュラル).
DEFAULT_PRESET_NAME = "natural"
LAST_SESSION_FILENAME = "last_session.json"
MAX_NAME_LENGTH = 64

# Characters that are path separators or invalid in Windows/macOS file names.
_FORBIDDEN_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
# Device names Windows refuses as file stems, with or without an extension.
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE)


class PresetError(Exception):
    """A preset operation failed. The message is Japanese and shown to the user."""


@dataclass(frozen=True)
class PresetInfo:
    name: str
    path: Path
    builtin: bool


def sanitize_preset_name(name: str) -> str:
    """Turn user input into a safe preset (file stem) name.

    Keeps Japanese and other Unicode text, NFC-normalizes it (macOS may hand
    back decomposed names), removes path separators and characters that are
    invalid in file names, trims spaces/dots at both ends and caps the length.
    Raises ``PresetError`` when nothing usable remains.
    """
    cleaned = _FORBIDDEN_CHARS.sub("", unicodedata.normalize("NFC", name))
    cleaned = cleaned.strip(" .\t　")[:MAX_NAME_LENGTH].rstrip(" .　")
    if not cleaned:
        raise PresetError("プリセット名を入力してください（記号だけの名前は使えません）。")
    if _WINDOWS_RESERVED.match(cleaned):
        cleaned += "_"
    return cleaned


def list_presets() -> list[PresetInfo]:
    """All presets: built-in ones first, then user ones, each sorted by name.

    A user preset named like a built-in one replaces it in the list.
    """
    builtin = _scan(config.BUILTIN_PRESETS_DIR, builtin=True)
    user = _scan(config.user_presets_dir(), builtin=False)
    visible_builtin = {name: info for name, info in builtin.items() if name not in user}
    return _sorted_values(visible_builtin) + _sorted_values(user)


def find_preset(name: str) -> PresetInfo | None:
    """The preset ``list_presets`` would show under ``name``, if any."""
    return next((info for info in list_presets() if info.name == name), None)


def load_preset(name: str) -> MakeupSettings:
    """Read a preset. Raises ``PresetError`` if it is missing or unreadable."""
    info = find_preset(name)
    if info is None:
        raise PresetError(f"プリセット『{name}』が見つかりません。")
    try:
        return _read_settings(info.path)
    except (OSError, ValueError) as exc:
        raise PresetError(
            f"プリセット『{name}』を読み込めませんでした。ファイルが壊れている可能性があります。\n{info.path}"
        ) from exc


def save_preset(name: str, settings: MakeupSettings) -> PresetInfo:
    """Write ``settings`` as a user preset and return where it went.

    The stored name is ``sanitize_preset_name(name)``; an existing user
    preset of that name is overwritten.
    """
    safe_name = sanitize_preset_name(name)
    path = config.user_presets_dir() / f"{safe_name}{PRESET_SUFFIX}"
    try:
        settings.save(path)
    except OSError as exc:
        raise PresetError(f"プリセット『{safe_name}』を保存できませんでした。\n{exc}") from exc
    return PresetInfo(safe_name, path, builtin=False)


def delete_preset(name: str) -> None:
    """Delete a user preset. Built-in presets cannot be deleted."""
    info = find_preset(name)
    if info is None:
        raise PresetError(f"プリセット『{name}』が見つかりません。")
    if info.builtin:
        raise PresetError(f"『{name}』は標準プリセットなので削除できません。")
    try:
        info.path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise PresetError(f"プリセット『{name}』を削除できませんでした。\n{exc}") from exc


def last_session_path() -> Path:
    return config.user_data_dir() / LAST_SESSION_FILENAME


def save_last_session(settings: MakeupSettings) -> bool:
    """Save the current settings for the next start. Returns False on failure (logged)."""
    try:
        settings.save(last_session_path())
    except OSError:
        logger.warning("could not save the last session to %s", last_session_path(), exc_info=True)
        return False
    return True


def load_last_session() -> MakeupSettings | None:
    """The settings of the previous run, or None if there is no usable session file."""
    path = last_session_path()
    if not path.is_file():
        return None
    try:
        return _read_settings(path)
    except (OSError, ValueError):
        logger.warning("ignoring unreadable last session file %s", path, exc_info=True)
        return None


def initial_settings(requested_preset: str | None = None) -> tuple[MakeupSettings, str | None]:
    """Settings to start with, and the preset they were loaded from (if any).

    Order: a requested preset, then the last session, then the built-in
    ``DEFAULT_PRESET_NAME`` preset, then plain defaults.
    """
    if requested_preset:
        try:
            return load_preset(requested_preset), requested_preset
        except PresetError:
            logger.warning("requested preset %r is not usable; restoring the last session", requested_preset)
    session = load_last_session()
    if session is not None:
        return session, None
    if find_preset(DEFAULT_PRESET_NAME) is not None:
        try:
            return load_preset(DEFAULT_PRESET_NAME), DEFAULT_PRESET_NAME
        except PresetError:
            logger.warning("built-in preset %r is not usable; using defaults", DEFAULT_PRESET_NAME)
    return MakeupSettings(), None


def _scan(directory: Path, *, builtin: bool) -> dict[str, PresetInfo]:
    if not directory.is_dir():
        return {}
    found = {}
    for path in directory.glob(f"*{PRESET_SUFFIX}"):
        if path.is_file():
            name = unicodedata.normalize("NFC", path.stem)
            found[name] = PresetInfo(name, path, builtin)
    return found


def _sorted_values(presets: dict[str, PresetInfo]) -> list[PresetInfo]:
    return [presets[name] for name in sorted(presets, key=str.casefold)]


def _read_settings(path: Path) -> MakeupSettings:
    """Parse a settings file; ValueError if it is not a JSON object."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return MakeupSettings.from_dict(data)
