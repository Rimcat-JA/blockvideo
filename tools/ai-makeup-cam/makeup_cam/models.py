"""Model files: where they live and how they are downloaded and verified.

The two MediaPipe models are fetched on first use into ``models_dir()``.
Every file is checked against a pinned SHA-256, downloaded into a temporary
file next to its destination and atomically renamed into place, so an
interrupted download never leaves a half-written model behind.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import ssl
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

from .config import user_data_dir

FACE_LANDMARKER = "face_landmarker"
SELFIE_MULTICLASS = "selfie_multiclass"

MODELS_DIR_ENV = "AI_MAKEUP_CAM_MODELS_DIR"

# (bytes_done, bytes_total); bytes_total is 0 when the server sends no size.
ProgressCallback = Callable[[int, int], None]
# (model_name, bytes_done, bytes_total), used when fetching several models.
NamedProgressCallback = Callable[[str, int, int], None]

_CHUNK_BYTES = 1 << 16
_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ModelSpec:
    url: str
    sha256: str
    filename: str


MODEL_SPECS: dict[str, ModelSpec] = {
    FACE_LANDMARKER: ModelSpec(
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        filename="face_landmarker.task",
    ),
    SELFIE_MULTICLASS: ModelSpec(
        url=(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite"
        ),
        sha256="c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
        filename="selfie_multiclass_256x256.tflite",
    ),
}


class ModelDownloadError(RuntimeError):
    """A model could not be obtained. The message is Japanese and shown to the user."""


def models_dir() -> Path:
    """Directory holding the model files (``$AI_MAKEUP_CAM_MODELS_DIR`` overrides)."""
    override = os.environ.get(MODELS_DIR_ENV)
    return Path(override) if override else user_data_dir() / "models"


def ensure_model(name: str, progress: ProgressCallback | None = None, *, directory: Path | None = None) -> Path:
    """Return the path of a verified model file, downloading it if needed.

    A present file with the right hash is used without touching the network;
    a missing or corrupted one is (re)downloaded. Raises ``ModelDownloadError``
    with a Japanese, actionable message when that fails.
    """
    spec = MODEL_SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown model: {name!r}")
    target = (directory if directory is not None else models_dir()) / spec.filename
    if not _has_hash(target, spec.sha256):
        _download(name, spec, target, progress)
    return target


def ensure_all_models(
    progress: NamedProgressCallback | None = None, *, directory: Path | None = None
) -> dict[str, Path]:
    """Ensure every model in ``MODEL_SPECS``; returns name -> path."""
    return {
        name: ensure_model(name, partial(progress, name) if progress else None, directory=directory)
        for name in MODEL_SPECS
    }


def _has_hash(path: Path, sha256: str) -> bool:
    try:
        return path.is_file() and _sha256_of(path) == sha256
    except OSError:
        return False


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(partial(f.read, 1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(name: str, spec: ModelSpec, target: Path, progress: ProgressCallback | None) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{spec.filename}.", suffix=".part", dir=target.parent)
    except OSError as exc:
        raise ModelDownloadError(_disk_message(name, target.parent, exc)) from exc
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            digest = _fetch(name, spec.url, target.parent, out, progress)
        if digest != spec.sha256:
            raise ModelDownloadError(
                f"モデル「{name}」のダウンロード内容が正しくありません（SHA-256 が一致しません）。"
                "通信が途中で壊れた可能性があります。もう一度お試しください。"
            )
        try:
            os.replace(tmp, target)
        except OSError as exc:
            raise ModelDownloadError(_disk_message(name, target.parent, exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)


def _fetch(name: str, url: str, directory: Path, out: BinaryIO, progress: ProgressCallback | None) -> str:
    """Stream ``url`` into ``out``; return the hex SHA-256 of the bytes written."""
    try:
        response = urllib.request.urlopen(url, timeout=_TIMEOUT_S)
    except (OSError, http.client.HTTPException) as exc:
        raise ModelDownloadError(_network_message(name, url, directory, exc)) from exc
    digest = hashlib.sha256()
    with response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        if progress:
            progress(0, total)
        while True:
            try:
                chunk = response.read(_CHUNK_BYTES)
            except (OSError, http.client.HTTPException) as exc:
                raise ModelDownloadError(_network_message(name, url, directory, exc)) from exc
            if not chunk:
                break
            try:
                out.write(chunk)
            except OSError as exc:
                raise ModelDownloadError(_disk_message(name, directory, exc)) from exc
            digest.update(chunk)
            done += len(chunk)
            if progress:
                progress(done, total)
    return digest.hexdigest()


def _network_message(name: str, url: str, directory: Path, exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        reason = f"サーバーがエラーを返しました（HTTP {exc.code}）。時間をおいてもう一度お試しください。"
    elif isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLError):
        reason = (
            "SSL 証明書を確認できませんでした。macOS で python.org 版の Python をお使いの場合は、"
            "Python のフォルダにある「Install Certificates.command」を実行してください。"
        )
    else:
        reason = "インターネット接続を確認して、もう一度お試しください。"
    return (
        f"モデル「{name}」をダウンロードできませんでした。{reason}\n"
        f"オフラインで使う場合は、次のファイルを手動でダウンロードして「{directory}」に置いてください:\n"
        f"{url}\n（詳細: {exc}）"
    )


def _disk_message(name: str, directory: Path, exc: BaseException) -> str:
    return (
        f"モデル「{name}」を保存できませんでした（保存先: {directory}）。"
        f"ディスクの空き容量と書き込み権限を確認してください。（詳細: {exc}）"
    )
