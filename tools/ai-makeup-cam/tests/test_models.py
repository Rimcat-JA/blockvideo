"""Model download / verification (network is always faked)."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from makeup_cam import models
from makeup_cam.models import MODEL_SPECS, ModelDownloadError, ModelSpec, ensure_all_models, ensure_model

PAYLOAD = bytes(range(256)) * 1000  # 256 kB, several download chunks
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, send_length: bool = True) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))} if send_length else {}


@pytest.fixture
def model_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "models"
    monkeypatch.setenv(models.MODELS_DIR_ENV, str(directory))
    monkeypatch.setitem(MODEL_SPECS, "fake", ModelSpec("https://example.invalid/fake.bin", PAYLOAD_SHA, "fake.bin"))
    return directory


def serve(monkeypatch: pytest.MonkeyPatch, data: bytes, **kwargs: bool) -> list[str]:
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> FakeResponse:
        requested.append(url)
        return FakeResponse(data, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return requested


def fail_network(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def fake_urlopen(url: str, timeout: float) -> FakeResponse:
        raise exc

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != "fake.bin")


def test_download_verifies_and_reports_progress(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested = serve(monkeypatch, PAYLOAD)
    progress: list[tuple[int, int]] = []

    path = ensure_model("fake", lambda done, total: progress.append((done, total)))

    assert path == model_home / "fake.bin"
    assert path.read_bytes() == PAYLOAD
    assert requested == ["https://example.invalid/fake.bin"]
    assert progress[0] == (0, len(PAYLOAD))
    assert progress[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert leftovers(model_home) == []


def test_unknown_size_reports_zero_total(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, PAYLOAD, send_length=False)
    progress: list[tuple[int, int]] = []
    ensure_model("fake", lambda done, total: progress.append((done, total)))
    assert progress[-1] == (len(PAYLOAD), 0)


def test_present_file_is_used_without_network(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_home.mkdir(parents=True)
    (model_home / "fake.bin").write_bytes(PAYLOAD)
    fail_network(monkeypatch, AssertionError("network must not be used"))
    assert ensure_model("fake") == model_home / "fake.bin"


def test_corrupted_file_is_downloaded_again(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_home.mkdir(parents=True)
    (model_home / "fake.bin").write_bytes(b"truncated")
    requested = serve(monkeypatch, PAYLOAD)
    assert ensure_model("fake").read_bytes() == PAYLOAD
    assert len(requested) == 1


def test_hash_mismatch_raises_and_keeps_nothing(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    serve(monkeypatch, PAYLOAD[:-1] + b"!")
    with pytest.raises(ModelDownloadError, match="SHA-256"):
        ensure_model("fake")
    assert not (model_home / "fake.bin").exists()
    assert leftovers(model_home) == []


def test_hash_mismatch_keeps_previous_file(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_home.mkdir(parents=True)
    (model_home / "fake.bin").write_bytes(b"old")
    serve(monkeypatch, b"tampered")
    with pytest.raises(ModelDownloadError):
        ensure_model("fake")
    assert (model_home / "fake.bin").read_bytes() == b"old"


def test_offline_error_is_japanese_and_actionable(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail_network(monkeypatch, urllib.error.URLError(OSError("Network is unreachable")))
    with pytest.raises(ModelDownloadError) as info:
        ensure_model("fake")
    message = str(info.value)
    assert "インターネット接続" in message
    assert "https://example.invalid/fake.bin" in message  # manual download hint
    assert str(model_home) in message
    assert leftovers(model_home) == []


def test_http_error_mentions_status(model_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fail_network(monkeypatch, urllib.error.HTTPError("https://example.invalid", 503, "busy", Message(), None))
    with pytest.raises(ModelDownloadError, match="HTTP 503"):
        ensure_model("fake")


def test_unwritable_directory_is_a_disk_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    monkeypatch.setitem(MODEL_SPECS, "fake", ModelSpec("https://example.invalid/f", PAYLOAD_SHA, "fake.bin"))
    serve(monkeypatch, PAYLOAD)
    with pytest.raises(ModelDownloadError, match="保存できませんでした"):
        ensure_model("fake", directory=blocker / "models")


def test_ensure_all_models_names_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MODEL_SPECS", {"a": ModelSpec("https://x/a", PAYLOAD_SHA, "a.bin")})
    serve(monkeypatch, PAYLOAD)
    seen: list[str] = []
    paths = ensure_all_models(lambda name, done, total: seen.append(name), directory=tmp_path)
    assert paths == {"a": tmp_path / "a.bin"}
    assert set(seen) == {"a"}


def test_models_dir_default_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(models.MODELS_DIR_ENV, raising=False)
    monkeypatch.setenv("AI_MAKEUP_CAM_HOME", str(tmp_path / "home"))
    assert models.models_dir() == tmp_path / "home" / "models"
    monkeypatch.setenv(models.MODELS_DIR_ENV, str(tmp_path / "custom"))
    assert models.models_dir() == tmp_path / "custom"


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError):
        ensure_model("no-such-model")


def test_builtin_specs_are_pinned() -> None:
    assert set(MODEL_SPECS) == {models.FACE_LANDMARKER, models.SELFIE_MULTICLASS}
    for spec in MODEL_SPECS.values():
        assert spec.url.startswith("https://storage.googleapis.com/mediapipe-models/")
        assert spec.url.endswith(spec.filename)
        assert len(spec.sha256) == 64 and set(spec.sha256) <= set("0123456789abcdef")
