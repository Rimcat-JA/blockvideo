"""Preset files and the last session (no Qt needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from makeup_cam import config, presets
from makeup_cam.config import MakeupSettings
from makeup_cam.presets import (
    DEFAULT_PRESET_NAME,
    PresetError,
    delete_preset,
    find_preset,
    initial_settings,
    last_session_path,
    list_presets,
    load_last_session,
    load_preset,
    sanitize_preset_name,
    save_last_session,
    save_preset,
)


@pytest.fixture
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """(builtin dir, user preset dir), both empty and isolated from the real ones."""
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    monkeypatch.setattr(config, "BUILTIN_PRESETS_DIR", builtin)
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))
    return builtin, config.user_presets_dir()


def _write(path: Path, settings: MakeupSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(settings.to_json(), encoding="utf-8")


def _with_lip_opacity(value: float) -> MakeupSettings:
    settings = MakeupSettings()
    settings.lips.opacity = value
    return settings


# ---- name sanitizing -------------------------------------------------------


@pytest.mark.parametrize("name", ["面接用 ナチュラル", "Natural-2", "朝のZoom（明るめ）"])
def test_sanitize_keeps_japanese_and_ordinary_names(name: str) -> None:
    assert sanitize_preset_name(name) == name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "etcpasswd"),
        ("a\\b/c", "abc"),
        ('面接:用*?"<>|', "面接用"),
        ("  .hidden.  ", "hidden"),
        ("　全角スペース　", "全角スペース"),
        ("tab\tname", "tabname"),
    ],
)
def test_sanitize_strips_separators_and_invalid_characters(raw: str, expected: str) -> None:
    assert sanitize_preset_name(raw) == expected


def test_sanitize_normalizes_decomposed_japanese() -> None:
    # macOS file systems may return "が" as "か" + combining voiced mark.
    assert sanitize_preset_name("がお") == "がお"


def test_sanitize_avoids_windows_device_names() -> None:
    assert sanitize_preset_name("CON") == "CON_"
    assert sanitize_preset_name("lpt1") == "lpt1_"
    assert sanitize_preset_name("console") == "console"


def test_sanitize_caps_length() -> None:
    assert sanitize_preset_name("あ" * 200) == "あ" * presets.MAX_NAME_LENGTH


@pytest.mark.parametrize("raw", ["", "   ", "///", "..", "\\:*"])
def test_sanitize_rejects_names_without_usable_characters(raw: str) -> None:
    with pytest.raises(PresetError, match="プリセット名"):
        sanitize_preset_name(raw)


# ---- save / load / list / delete ------------------------------------------


def test_save_and_load_roundtrip_with_japanese_name(dirs: tuple[Path, Path]) -> None:
    _, user_dir = dirs
    settings = _with_lip_opacity(0.8)
    settings.lips.color = "#AA3344"

    info = save_preset("面接用", settings)

    assert info.name == "面接用"
    assert not info.builtin
    assert info.path == user_dir / "面接用.json"
    assert load_preset("面接用") == settings


def test_save_uses_the_sanitized_name_inside_the_user_dir(dirs: tuple[Path, Path]) -> None:
    _, user_dir = dirs
    info = save_preset("../outside/面接", MakeupSettings())
    assert info.name == "outside面接"
    assert info.path.parent == user_dir
    assert find_preset("outside面接") == info


def test_save_overwrites_an_existing_user_preset(dirs: tuple[Path, Path]) -> None:
    save_preset("mine", _with_lip_opacity(0.1))
    save_preset("mine", _with_lip_opacity(0.9))
    assert load_preset("mine").lips.opacity == 0.9
    assert [info.name for info in list_presets()] == ["mine"]


def test_list_puts_builtin_first_and_user_overrides_builtin(dirs: tuple[Path, Path]) -> None:
    builtin_dir, user_dir = dirs
    _write(builtin_dir / "natural.json", _with_lip_opacity(0.2))
    _write(builtin_dir / "office.json", MakeupSettings())
    _write(user_dir / "natural.json", _with_lip_opacity(0.7))
    _write(user_dir / "面接.json", MakeupSettings())
    _write(user_dir / "Audition.json", MakeupSettings())

    listed = [(info.name, info.builtin) for info in list_presets()]

    assert listed == [("office", True), ("Audition", False), ("natural", False), ("面接", False)]
    assert load_preset("natural").lips.opacity == 0.7


def test_list_ignores_other_files_and_missing_dirs(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    (builtin_dir / "readme.txt").write_text("not a preset", encoding="utf-8")
    (builtin_dir / "folder.json").mkdir()
    assert list_presets() == []


def test_delete_user_override_brings_builtin_back(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    _write(builtin_dir / "natural.json", _with_lip_opacity(0.2))
    save_preset("natural", _with_lip_opacity(0.7))

    delete_preset("natural")

    info = find_preset("natural")
    assert info is not None and info.builtin
    assert load_preset("natural").lips.opacity == 0.2


def test_builtin_presets_cannot_be_deleted(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    _write(builtin_dir / "natural.json", MakeupSettings())
    with pytest.raises(PresetError, match="標準プリセット"):
        delete_preset("natural")
    assert (builtin_dir / "natural.json").exists()


def test_missing_presets_raise_clear_errors(dirs: tuple[Path, Path]) -> None:
    with pytest.raises(PresetError, match="見つかりません"):
        load_preset("ない")
    with pytest.raises(PresetError, match="見つかりません"):
        delete_preset("ない")


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]", "\udcff"])
def test_corrupt_preset_raises_a_clear_error(dirs: tuple[Path, Path], content: str) -> None:
    _, user_dir = dirs
    user_dir.mkdir(parents=True)
    (user_dir / "broken.json").write_bytes(content.encode("utf-8", "surrogateescape"))
    with pytest.raises(PresetError, match="『broken』を読み込めませんでした"):
        load_preset("broken")


def test_partial_and_out_of_range_preset_is_completed_with_defaults(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    data = {"lips": {"opacity": 5.0, "color": "#ff0000"}, "unknown": {}}
    (builtin_dir / "partial.json").write_text(json.dumps(data), encoding="utf-8")

    settings = load_preset("partial")

    assert settings.lips.opacity == 1.0  # clamped
    assert settings.lips.color == "#FF0000"
    assert settings.skin == MakeupSettings().skin


# ---- display names ---------------------------------------------------------


def _write_named(path: Path, display_name: object) -> None:
    data = MakeupSettings().to_dict()
    data["display_name"] = display_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_display_name_comes_from_the_file(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    _write_named(builtin_dir / "natural.json", "就活ナチュラル（標準）")
    info = find_preset("natural")
    assert info is not None
    assert (info.name, info.display_name) == ("natural", "就活ナチュラル（標準）")
    assert load_preset("natural") == MakeupSettings()  # not a setting


def test_saved_user_presets_show_their_own_name(dirs: tuple[Path, Path]) -> None:
    info = save_preset("面接用", MakeupSettings())
    assert info.display_name == "面接用"
    assert find_preset("面接用") == info
    assert "display_name" not in json.loads(info.path.read_text(encoding="utf-8"))


def test_user_preset_overriding_a_builtin_does_not_borrow_its_display_name(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    _write_named(builtin_dir / "natural.json", "就活ナチュラル（標準）")
    save_preset("natural", MakeupSettings())
    assert find_preset("natural").display_name == "natural"
    delete_preset("natural")
    assert find_preset("natural").display_name == "就活ナチュラル（標準）"


@pytest.mark.parametrize("value", ["", "   ", 5, None, ["名前"], {"ja": "名前"}])
def test_unusable_display_name_falls_back_to_the_name(dirs: tuple[Path, Path], value: object) -> None:
    builtin_dir, _ = dirs
    _write_named(builtin_dir / "mine.json", value)
    assert find_preset("mine").display_name == "mine"


def test_overlong_display_name_is_cut(dirs: tuple[Path, Path]) -> None:
    builtin_dir, _ = dirs
    _write_named(builtin_dir / "mine.json", "あ" * 200)
    assert find_preset("mine").display_name == "あ" * presets.MAX_NAME_LENGTH


@pytest.mark.parametrize("content", ["{not json", "[1, 2, 3]", "\udcff"])
def test_corrupt_preset_is_still_listed_under_its_name(dirs: tuple[Path, Path], content: str) -> None:
    _, user_dir = dirs
    user_dir.mkdir(parents=True)
    (user_dir / "broken.json").write_bytes(content.encode("utf-8", "surrogateescape"))
    assert find_preset("broken").display_name == "broken"
    with pytest.raises(PresetError):  # choosing it still reports the problem
        load_preset("broken")


def test_builtin_presets_have_japanese_display_names() -> None:
    expected = {
        "natural": "就活ナチュラル（標準）",
        "bare": "すっぴん風補正",
        "polished": "しっかりメイク",
        "strong": "しっかりメイク＋小顔補正",
        "off": "全部オフ",
    }
    files = {path.stem: path for path in config.BUILTIN_PRESETS_DIR.glob("*.json")}
    assert sorted(files) == sorted(expected)
    for name, path in files.items():
        assert name.isascii()
        assert json.loads(path.read_text(encoding="utf-8"))["display_name"] == expected[name]


# ---- what each built-in preset promises ------------------------------------


def _builtin(name: str) -> MakeupSettings:
    return MakeupSettings.load(config.BUILTIN_PRESETS_DIR / f"{name}.json")


_RESHAPE_SLIDERS = (
    "face_slim",
    "jaw",
    "chin_length",
    "eye_size",
    "nose_width",
    "nose_length",
    "mouth_width",
    "mouth_corner",
    "lip_fullness",
    "brow_lift",
)


@pytest.mark.parametrize("name", ["natural", "bare", "polished", "off"])
def test_only_the_strong_preset_changes_the_face_shape(name: str) -> None:
    assert all(getattr(_builtin(name).reshape, slider) == 0.0 for slider in _RESHAPE_SLIDERS)


@pytest.mark.parametrize("name", ["natural", "bare", "polished", "strong"])
def test_no_preset_warps_the_hair_and_all_tidy_flyaways(name: str) -> None:
    hairshape = _builtin(name).hairshape
    assert hairshape.enabled and 0.0 < hairshape.flyaway <= 0.5
    assert (hairshape.volume_top, hairshape.width_sides) == (0.0, 0.0)


def test_bare_has_no_color_makeup() -> None:
    bare = _builtin("bare")
    for section in (bare.lips, bare.blush, bare.eyeshadow, bare.eyeliner, bare.brows, bare.contour, bare.tearbag):
        assert not section.enabled
    for section in (bare.skin, bare.teeth, bare.hair, bare.hairshape):
        assert section.enabled
    assert bare.skin.under_eye > 0.0


def test_strong_is_polished_plus_a_moderate_reshape() -> None:
    strong, polished = _builtin("strong").to_dict(), _builtin("polished").to_dict()
    reshape = strong.pop("reshape")
    polished.pop("reshape")
    assert strong == polished
    assert reshape["enabled"] and reshape["face_slim"] > 0.0
    assert all(abs(reshape[slider]) <= 0.5 for slider in _RESHAPE_SLIDERS)  # still clearly the same person


def test_presets_grow_in_strength_from_natural_to_polished() -> None:
    natural, polished = _builtin("natural"), _builtin("polished")
    assert 0.0 < natural.contour.shading < polished.contour.shading
    assert 0.0 < natural.contour.highlight < polished.contour.highlight
    assert 0.0 < natural.tearbag.strength < polished.tearbag.strength
    assert natural.contour.enabled and natural.tearbag.enabled and natural.teeth.enabled


def test_off_disables_every_section() -> None:
    off = _builtin("off")
    for section in config.SECTIONS:
        if section != "general":
            assert not getattr(off, section).enabled, section


# ---- last session ----------------------------------------------------------


def test_last_session_roundtrip(dirs: tuple[Path, Path]) -> None:
    assert load_last_session() is None
    settings = _with_lip_opacity(0.66)
    assert save_last_session(settings)
    assert last_session_path().parent == config.user_data_dir()
    assert load_last_session() == settings


def test_corrupt_last_session_is_ignored(dirs: tuple[Path, Path]) -> None:
    path = last_session_path()
    path.parent.mkdir(parents=True)
    path.write_text("{oops", encoding="utf-8")
    assert load_last_session() is None


def test_failing_session_save_reports_false_instead_of_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(blocker))
    assert save_last_session(MakeupSettings()) is False


def test_failing_preset_save_raises_preset_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(blocker))
    with pytest.raises(PresetError, match="保存できませんでした"):
        save_preset("mine", MakeupSettings())


# ---- startup settings ------------------------------------------------------


def test_initial_settings_fallback_order(dirs: tuple[Path, Path]) -> None:
    builtin, _user = dirs
    assert initial_settings() == (MakeupSettings(), None)

    natural = _with_lip_opacity(0.11)
    _write(builtin / f"{DEFAULT_PRESET_NAME}.json", natural)
    assert initial_settings() == (natural, DEFAULT_PRESET_NAME)

    session = _with_lip_opacity(0.22)
    save_last_session(session)
    assert initial_settings() == (session, None)
    assert initial_settings(DEFAULT_PRESET_NAME) == (natural, DEFAULT_PRESET_NAME)
    assert initial_settings("missing") == (session, None)


def test_a_preset_saved_with_a_bom_is_read(dirs: tuple[Path, Path]) -> None:
    """Windows Notepad offers "UTF-8 with BOM"."""
    _, user = dirs
    save_preset("mine", MakeupSettings())
    path = user / "mine.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["display_name"] = "ぼむ"
    data["lips"]["opacity"] = 0.33
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
    assert load_preset("mine").lips.opacity == 0.33
    assert find_preset("mine").display_name == "ぼむ"
    assert MakeupSettings.load(path).lips.opacity == 0.33


def test_a_user_preset_cannot_take_the_name_a_builtin_is_shown_under(dirs: tuple[Path, Path]) -> None:
    builtin, _ = dirs
    (builtin / "natural.json").write_text(json.dumps({"display_name": "就活ナチュラル（標準）"}), encoding="utf-8")
    with pytest.raises(PresetError, match="表示名"):
        save_preset("就活ナチュラル（標準）", MakeupSettings())
    assert save_preset("就活ナチュラル", MakeupSettings()).name == "就活ナチュラル"
