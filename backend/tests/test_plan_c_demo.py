"""The demo uses isolated storage and fixed product configuration, never Hard Filter."""
from pathlib import Path
from typing import Any

import pytest

from scripts.plan_c_demo import demo_settings, exclusive_demo, prepare_storage


def test_demo_refuses_existing_unmarked_data(tmp_path: Path) -> None:
    (tmp_path / "user.db").write_bytes(b"untouched")
    with pytest.raises(ValueError):
        prepare_storage(tmp_path)
    assert (tmp_path / "user.db").read_bytes() == b"untouched"
    assert not (tmp_path / "demo.json").exists()


def test_demo_reuses_only_marked_storage_and_prevents_overlap(tmp_path: Path) -> None:
    directory = prepare_storage(tmp_path / "new")
    assert prepare_storage(directory) == directory
    with exclusive_demo(directory):
        with pytest.raises(OSError):
            with exclusive_demo(directory):
                pytest.fail("must not start a second server on the same DB")
    with exclusive_demo(directory):
        pass


def test_demo_settings_do_not_inherit_private_storage_or_modes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "normal"))
    monkeypatch.setenv("LANGUAGE_RETRIEVAL_INDEX", str(tmp_path / "untrusted-index"))
    monkeypatch.setenv("LLM_API_KEY", "must-not-be-used")
    for mode in ("all_tools", "stateful"):
        settings = demo_settings(tmp_path, mode, "chosen", tmp_path)
        assert settings.storage_root == tmp_path and settings.llm_api_key is None
        assert (settings.language_retrieval_index is not None) == (mode == "stateful")
        assert settings.language_model == "chosen" and settings.language_retrieval_readiness
    with pytest.raises(ValueError):
        demo_settings(tmp_path, "B2", "chosen", tmp_path)
    with pytest.raises(ValueError):
        demo_settings(tmp_path, "stateful", "chosen", None)
