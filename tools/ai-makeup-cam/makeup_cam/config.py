"""Makeup settings: the single source of truth for every adjustable value.

Each field carries GUI metadata (Japanese label, range, step, kind) so the
settings window is generated from this file. Adding a field here makes it
appear in the GUI and in saved presets automatically; the effect that owns the
section reads it from ``settings.<section>.<field>``.

Values are plain JSON types (float / int / bool / "#RRGGBB" str / choice str)
so presets are human-editable JSON files.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SETTINGS_VERSION = 1

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# --------------------------------------------------------------------------
# Field helpers. ``kind`` drives which widget the GUI builds.
# --------------------------------------------------------------------------


def num(default: float, lo: float, hi: float, label: str, step: float = 0.01, help: str = "") -> Any:
    """A float slider in [lo, hi]."""
    return field(
        default=default,
        metadata={"kind": "float", "min": lo, "max": hi, "step": step, "label": label, "help": help},
    )


def integer(default: int, lo: int, hi: int, label: str, help: str = "") -> Any:
    """An int slider/spinbox in [lo, hi]."""
    return field(
        default=default,
        metadata={"kind": "int", "min": lo, "max": hi, "step": 1, "label": label, "help": help},
    )


def color(default: str, label: str, help: str = "") -> Any:
    """A "#RRGGBB" color chosen with a color picker."""
    return field(default=default, metadata={"kind": "color", "label": label, "help": help})


def flag(default: bool, label: str, help: str = "") -> Any:
    """A checkbox."""
    return field(default=default, metadata={"kind": "bool", "label": label, "help": help})


def choice(default: str, options: list[tuple[str, str]], label: str, help: str = "") -> Any:
    """A combo box. ``options`` is [(value, Japanese label), ...]."""
    return field(
        default=default,
        metadata={"kind": "choice", "choices": options, "label": label, "help": help},
    )


def hex_to_bgr(value: str) -> tuple[float, float, float]:
    """"#RRGGBB" -> (b, g, r) floats in 0..1, the pipeline's color order."""
    if not _HEX_RE.match(value or ""):
        raise ValueError(f"not a #RRGGBB color: {value!r}")
    r = int(value[1:3], 16) / 255.0
    g = int(value[3:5], 16) / 255.0
    b = int(value[5:7], 16) / 255.0
    return (b, g, r)


# --------------------------------------------------------------------------
# Sections. Every section has ``enabled``; the section name (attribute name on
# MakeupSettings) is also the Effect.name that consumes it.
# --------------------------------------------------------------------------


@dataclass
class GeneralSettings:
    intensity: float = num(1.0, 0.0, 1.0, "全体のメイクの強さ", help="すべてのメイクの濃さをまとめて調整します")
    mirror_preview: bool = flag(True, "プレビューを鏡像にする", help="自分の画面だけ左右反転します（相手には影響しません）")
    mirror_output: bool = flag(False, "相手への映像も左右反転", help="通常はOFF。Zoom等は自分の映像を自動で反転表示します")
    landmark_smoothing: float = num(0.5, 0.0, 1.0, "顔追跡のなめらかさ", help="上げるとメイクの揺れが減り、下げると素早い動きに追従します")
    segmentation_interval: int = integer(2, 1, 6, "髪・肌の検出間隔（フレーム）", help="大きいほど軽くなりますが、髪の追従が遅れます")
    show_landmarks: bool = flag(False, "顔の検出点を表示（確認用）", help="プレビューにだけ表示されます")


@dataclass
class SkinSettings:
    enabled: bool = flag(True, "肌補正を使う")
    smoothing: float = num(0.45, 0.0, 1.0, "なめらかさ", help="肌のキメ・凹凸をなめらかにします")
    texture: float = num(0.35, 0.0, 1.0, "質感を残す", help="上げると毛穴などの細かい質感を残し、塗りすぎ感を防ぎます")
    blemish: float = num(0.5, 0.0, 1.0, "シミ・ニキビ跡カバー", help="小さな色ムラを周りの肌になじませます")
    redness: float = num(0.2, 0.0, 1.0, "赤み補正", help="頬や小鼻の赤みを抑えます")
    under_eye: float = num(0.3, 0.0, 1.0, "クマ補正（コンシーラー）", help="目の下の暗さを明るくします")
    tone_color: str = color("#F1D3C0", "ファンデの色")
    tone_strength: float = num(0.12, 0.0, 1.0, "ファンデの色の強さ")
    brightness: float = num(0.03, -0.3, 0.3, "肌の明るさ")


@dataclass
class LipSettings:
    enabled: bool = flag(True, "リップを使う")
    color: str = color("#C25B62", "リップの色")
    opacity: float = num(0.45, 0.0, 1.0, "濃さ")
    finish: str = choice(
        "satin",
        [("matte", "マット"), ("satin", "サテン（自然なツヤ）"), ("gloss", "グロス（ツヤツヤ）")],
        "質感",
    )
    gloss: float = num(0.3, 0.0, 1.0, "ツヤの強さ", help="質感がサテン・グロスのときのハイライトの強さ")
    feather: float = num(0.4, 0.0, 1.0, "輪郭のぼかし")
    outline: float = num(0.0, -0.2, 0.2, "塗る範囲（オーバー／インライン）", help="プラスで唇の輪郭より少し外側まで、マイナスで内側に塗ります")


@dataclass
class BlushSettings:
    enabled: bool = flag(True, "チークを使う")
    color: str = color("#E8878C", "チークの色")
    opacity: float = num(0.22, 0.0, 1.0, "濃さ")
    shape: str = choice(
        "oval",
        [("round", "丸く（かわいい印象）"), ("oval", "楕円（ナチュラル）"), ("lifted", "斜め上へ（大人っぽい印象）")],
        "形",
    )
    size: float = num(1.0, 0.5, 1.6, "大きさ")
    offset_x: float = num(0.0, -0.5, 0.5, "位置（左右）", help="プラスで外側（耳側）、マイナスで内側（鼻側）")
    offset_y: float = num(0.0, -0.5, 0.5, "位置（上下）", help="プラスで下、マイナスで上")
    softness: float = num(0.7, 0.0, 1.0, "ぼかし")


@dataclass
class EyeshadowSettings:
    enabled: bool = flag(True, "アイシャドウを使う")
    color: str = color("#A0705E", "アイシャドウの色")
    opacity: float = num(0.5, 0.0, 1.0, "濃さ")
    height: float = num(0.5, 0.0, 1.0, "塗る高さ", help="まぶたのどこまで塗るか（上げると眉側まで広がります）")
    softness: float = num(0.6, 0.0, 1.0, "ぼかし")
    shimmer: float = num(0.0, 0.0, 1.0, "ラメ・ツヤ感")


@dataclass
class EyelinerSettings:
    enabled: bool = flag(True, "アイライナーを使う")
    color: str = color("#2E211C", "アイライナーの色")
    opacity: float = num(0.55, 0.0, 1.0, "濃さ")
    thickness: float = num(0.3, 0.0, 1.0, "太さ")
    wing: float = num(0.15, 0.0, 1.0, "目尻の跳ね上げ")
    lower: float = num(0.0, 0.0, 1.0, "下まぶたのライン")
    softness: float = num(0.4, 0.0, 1.0, "ぼかし")


@dataclass
class BrowSettings:
    enabled: bool = flag(True, "眉を使う")
    color: str = color("#5C4334", "眉の色")
    opacity: float = num(0.3, 0.0, 1.0, "濃さ")
    thickness: float = num(0.0, -0.5, 0.5, "太さ", help="プラスで太眉、マイナスで細眉ぎみ")
    softness: float = num(0.5, 0.0, 1.0, "ぼかし")


@dataclass
class HairSettings:
    enabled: bool = flag(True, "髪の補正を使う")
    gloss: float = num(0.3, 0.0, 1.0, "ツヤ出し", help="髪のハイライトを強めて天使の輪のようなツヤを出します")
    smoothness: float = num(0.3, 0.0, 1.0, "まとまり感", help="毛流れに沿ってパサつき・ザラつきを整えます（外にはねたアホ毛は消えません）")
    color: str = color("#3B2A22", "髪色")
    color_strength: float = num(0.0, 0.0, 1.0, "髪色の強さ", help="0で地毛のまま。上げると指定した髪色に近づきます")
    brightness: float = num(0.0, -0.5, 0.5, "髪の明るさ")
    contrast: float = num(0.0, -0.5, 0.5, "髪のコントラスト", help="上げると陰影がはっきりして立体的に見えます")
    edge_softness: float = num(0.5, 0.0, 1.0, "境目のなじませ", help="髪と背景・顔の境目のぼかし")


@dataclass
class ContourSettings:
    enabled: bool = flag(True, "シェーディング・ハイライトを使う")
    shading_color: str = color("#9A7563", "シェーディングの色")
    shading: float = num(0.15, 0.0, 1.0, "シェーディングの濃さ", help="影を入れて小顔・立体的に見せます")
    jawline: float = num(0.6, 0.0, 1.0, "フェイスライン", help="あごから耳にかけての影（シェーディング全体の中での割合）")
    cheek_hollow: float = num(0.5, 0.0, 1.0, "頬骨の下", help="頬骨の下のくぼみの影")
    nose_shadow: float = num(0.4, 0.0, 1.0, "ノーズシャドウ", help="鼻筋の両側の影")
    forehead: float = num(0.3, 0.0, 1.0, "おでこの生え際", help="生え際の影でおでこを小さく見せます")
    highlight_color: str = color("#FFF4EA", "ハイライトの色")
    highlight: float = num(0.15, 0.0, 1.0, "ハイライトの強さ", help="光を入れて立体感とツヤを出します")
    nose_bridge: float = num(0.6, 0.0, 1.0, "鼻筋", help="鼻筋のハイライト（ハイライト全体の中での割合）")
    cheekbone: float = num(0.6, 0.0, 1.0, "頬骨の上")
    chin: float = num(0.4, 0.0, 1.0, "あご先")
    cupids_bow: float = num(0.3, 0.0, 1.0, "唇の山", help="上唇の山の上の小さなハイライト")
    softness: float = num(0.7, 0.0, 1.0, "ぼかし")


@dataclass
class TearbagSettings:
    enabled: bool = flag(True, "涙袋を使う")
    strength: float = num(0.2, 0.0, 1.0, "ぷっくり感", help="下まつげのすぐ下に光を入れて、ふくらみを作ります")
    width: float = num(0.5, 0.0, 1.0, "幅", help="目頭から目尻のどこまで入れるか")
    shadow: float = num(0.3, 0.0, 1.0, "下の影", help="ふくらみの下の細い影（ぷっくり感の中での割合）")
    softness: float = num(0.6, 0.0, 1.0, "ぼかし")


@dataclass
class TeethSettings:
    enabled: bool = flag(True, "歯の補正を使う")
    whitening: float = num(0.3, 0.0, 1.0, "白さ", help="歯を明るくします（真っ白にはならないよう上限があります）")
    yellow_reduction: float = num(0.4, 0.0, 1.0, "黄ばみ補正", help="歯の黄色みを抑えます")
    evenness: float = num(0.3, 0.0, 1.0, "色ムラ・影をなじませる", help="歯ごとの色の差や、すき間・重なりの暗い影を目立たなくします")
    feather: float = num(0.5, 0.0, 1.0, "境目のぼかし", help="歯と唇・歯ぐきの境目のなじませ")


@dataclass
class ReshapeSettings:
    enabled: bool = flag(True, "顔の形の補正を使う")
    face_slim: float = num(0.0, -1.0, 1.0, "小顔（輪郭）", help="プラスで輪郭をすっきり、マイナスでふっくら")
    jaw: float = num(0.0, -1.0, 1.0, "エラ", help="プラスでエラを目立たなくします")
    chin_length: float = num(0.0, -1.0, 1.0, "あごの長さ", help="プラスで長く、マイナスで短く")
    eye_size: float = num(0.0, -1.0, 1.0, "目の大きさ", help="プラスで大きく")
    nose_width: float = num(0.0, -1.0, 1.0, "小鼻の幅", help="プラスで小鼻を細く")
    nose_length: float = num(0.0, -1.0, 1.0, "鼻の長さ", help="プラスで短く")
    mouth_width: float = num(0.0, -1.0, 1.0, "口の大きさ", help="プラスで小さく")
    mouth_corner: float = num(0.0, 0.0, 1.0, "口角アップ")
    lip_fullness: float = num(0.0, -1.0, 1.0, "唇のふっくら感", help="プラスでふっくら")
    brow_lift: float = num(0.0, -1.0, 1.0, "眉の位置", help="プラスで上げる")
    protect_background: float = num(0.8, 0.0, 1.0, "背景のゆがみ防止", help="上げると顔の外（背景）が曲がりにくくなります")


@dataclass
class HairShapeSettings:
    enabled: bool = flag(True, "髪型の補正を使う")
    flyaway: float = num(0.3, 0.0, 1.0, "アホ毛消し", help="髪の輪郭から飛び出した細い毛を消します。「背景を撮影」をしておくときれいに消えます")
    volume_top: float = num(0.0, -1.0, 1.0, "トップのボリューム", help="プラスでふんわり、マイナスでぺたんと")
    width_sides: float = num(0.0, -1.0, 1.0, "横の広がり", help="マイナスで広がりを抑えます")
    use_background_plate: bool = flag(True, "撮影した背景を使う", help="「背景を撮影」で撮った背景で、消した髪の部分を埋めます")


# Order here is the order of tabs in the GUI.
SECTIONS: dict[str, type] = {
    "general": GeneralSettings,
    "skin": SkinSettings,
    "contour": ContourSettings,
    "lips": LipSettings,
    "blush": BlushSettings,
    "eyeshadow": EyeshadowSettings,
    "tearbag": TearbagSettings,
    "eyeliner": EyelinerSettings,
    "brows": BrowSettings,
    "teeth": TeethSettings,
    "reshape": ReshapeSettings,
    "hair": HairSettings,
    "hairshape": HairShapeSettings,
}

SECTION_LABELS: dict[str, str] = {
    "general": "全体",
    "skin": "肌",
    "contour": "立体感",
    "lips": "リップ",
    "blush": "チーク",
    "eyeshadow": "アイシャドウ",
    "tearbag": "涙袋",
    "eyeliner": "アイライナー",
    "brows": "眉",
    "teeth": "歯",
    "reshape": "顔の形",
    "hair": "髪",
    "hairshape": "髪型",
}


@dataclass
class MakeupSettings:
    general: GeneralSettings = field(default_factory=GeneralSettings)
    skin: SkinSettings = field(default_factory=SkinSettings)
    contour: ContourSettings = field(default_factory=ContourSettings)
    lips: LipSettings = field(default_factory=LipSettings)
    blush: BlushSettings = field(default_factory=BlushSettings)
    eyeshadow: EyeshadowSettings = field(default_factory=EyeshadowSettings)
    tearbag: TearbagSettings = field(default_factory=TearbagSettings)
    eyeliner: EyelinerSettings = field(default_factory=EyelinerSettings)
    brows: BrowSettings = field(default_factory=BrowSettings)
    teeth: TeethSettings = field(default_factory=TeethSettings)
    reshape: ReshapeSettings = field(default_factory=ReshapeSettings)
    hair: HairSettings = field(default_factory=HairSettings)
    hairshape: HairShapeSettings = field(default_factory=HairShapeSettings)

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"version": SETTINGS_VERSION}
        data.update(asdict(self))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MakeupSettings":
        """Build settings from (possibly partial / older / hand-edited) data.

        Missing sections or fields fall back to defaults, unknown keys are
        ignored, numbers are clamped to their range, and invalid values
        (bad colors, unknown choices, wrong types) fall back to the default.
        """
        data = data or {}
        kwargs = {}
        for name, section_cls in SECTIONS.items():
            raw = data.get(name)
            kwargs[name] = _section_from_dict(section_cls, raw if isinstance(raw, dict) else {})
        return cls(**kwargs)

    def copy(self) -> "MakeupSettings":
        return MakeupSettings.from_dict(self.to_dict())

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "MakeupSettings":
        return cls.from_dict(json.loads(text))

    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MakeupSettings":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _coerce(value: Any, f) -> Any:
    meta = f.metadata
    default = f.default
    kind = meta.get("kind")
    try:
        if kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            v = float(value)
            if v != v:  # NaN
                return default
            return min(max(v, meta["min"]), meta["max"])
        if kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            return int(min(max(round(value), meta["min"]), meta["max"]))
        if kind == "bool":
            return value if isinstance(value, bool) else default
        if kind == "color":
            return value.upper() if isinstance(value, str) and _HEX_RE.match(value) else default
        if kind == "choice":
            allowed = {opt for opt, _ in meta["choices"]}
            return value if value in allowed else default
    except (TypeError, ValueError):
        return default
    return default


def _section_from_dict(section_cls: type, raw: dict[str, Any]) -> Any:
    values = {}
    for f in fields(section_cls):
        if f.name in raw:
            values[f.name] = _coerce(raw[f.name], f)
    return section_cls(**values)


def section_fields(section_cls: type) -> list[tuple[str, dict[str, Any], Any]]:
    """[(field_name, metadata, default), ...] in declaration order, for the GUI."""
    out = []
    for f in fields(section_cls):
        default = f.default if f.default is not MISSING else None
        out.append((f.name, dict(f.metadata), default))
    return out


# --------------------------------------------------------------------------
# Where presets live.
# --------------------------------------------------------------------------


def user_data_dir() -> Path:
    """Per-user writable directory for presets and downloaded models."""
    override = os.environ.get("AI_MAKEUP_CAM_HOME")
    if override:
        return Path(override)
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "ai-makeup-cam"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ai-makeup-cam"
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "ai-makeup-cam"


def user_presets_dir() -> Path:
    return user_data_dir() / "presets"


BUILTIN_PRESETS_DIR = Path(__file__).resolve().parent / "presets"
