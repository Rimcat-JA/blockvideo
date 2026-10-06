from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

CANDIDATE_SUBJECT = "[DONE] Mission 35 Add recovery-oriented operational UI"
CONTROL_KEYS = {
    "git_commit",
    "git_commit_subject",
    "git_tree_clean",
    "schema_version",
}
REQUIRED_CANDIDATE_FILES = {
    ".gitignore": b".pytest_cache/\nrelease-evidence/\n",
    "backend/.env.example": b"BLOCKVIDEO_TEST_SETTING=example\n",
    "backend/app/core/config.py": b"LANGUAGE_RETRIEVAL_ALL_TOOLS = True\n",
    "backend/app/migrations/schema.py": (
        b"def apply_v0_to_v1(connection, metadata):\n"
        b"    connection.execute('PRAGMA user_version=1')\n"
    ),
    "backend/app/operations/definitions.json": b"[]\n",
    "backend/app/operations/search_scope.json": b"{}\n",
    "backend/app/retrieval/e5-profile.json": b'{"profile":"e5"}\n',
    "backend/app/retrieval/nomic-profile.json": b'{"profile":"nomic"}\n',
    "backend/pyproject.toml": b'[project]\nrequires-python = ">=3.12"\n',
    "backend/tests/test_candidate.py": b"def test_candidate(): assert True\n",
    "backend/uv.lock": b"version = 1\n",
    "docs/DTD.md": b"candidate design\n",
    "docs/plan-c/work-unit-36.md": b"candidate freeze contract\n",
    "frontend/package.json": b'{"engines":{"node":">=20"}}\n',
    "frontend/pnpm-lock.yaml": b"lockfileVersion: '9.0'\n",
    "frontend/src/main.tsx": b"export {};\n",
    "specification.md": b"candidate specification\n",
}


def _run(*argv: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _make_candidate(
    root: Path,
    *,
    timestamp: int = 1_700_000_000,
    subject: str = CANDIDATE_SUBJECT,
    omit: set[str] | None = None,
    extras: dict[str, bytes] | None = None,
    detach: bool = True,
) -> tuple[Path, str]:
    candidate = root / "candidate"
    candidate.mkdir()
    _run("git", "init", "-q", cwd=candidate)
    _run("git", "config", "user.email", "d36@example.invalid", cwd=candidate)
    _run("git", "config", "user.name", "D36 Test", cwd=candidate)
    files = {**REQUIRED_CANDIDATE_FILES, **(extras or {})}
    for relative, content in files.items():
        if relative in (omit or set()):
            continue
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _run("git", "add", "-f", ".", cwd=candidate)
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": f"{timestamp} +0000",
        "GIT_COMMITTER_DATE": f"{timestamp} +0000",
    }
    _run("git", "commit", "-q", "-m", subject, cwd=candidate, env=commit_env)
    commit = _run("git", "rev-parse", "HEAD", cwd=candidate)
    if detach:
        _run("git", "checkout", "-q", "--detach", commit, cwd=candidate)
    return candidate, commit


def _control(commit: str, **updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "git_commit": commit,
        "git_commit_subject": CANDIDATE_SUBJECT,
        "git_tree_clean": True,
    }
    value.update(updates)
    return value


def _write_control(root: Path, value: dict[str, Any], *, canonical: bool = True) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "candidate-control.json"
    raw = _canonical_bytes(value) if canonical else json.dumps(value, indent=2).encode("utf-8")
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _freeze_api() -> tuple[Any, Any, Any]:
    try:
        freeze = importlib.import_module("evaluation.release_candidate.freeze")
        fingerprints = importlib.import_module("evaluation.release_candidate.fingerprints")
        attestation = importlib.import_module("evaluation.tool_attestation")
    except ModuleNotFoundError as exc:
        pytest.fail(f"D36 freezer module is missing: {exc}")
    return freeze, fingerprints, attestation


def _freeze(tmp_path: Path, candidate: Path, commit: str, *, output_name: str = "output") -> Any:
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path, _control(commit))
    return freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tmp_path / output_name,
    )


@pytest.fixture(autouse=True)
def _clean_committed_tooling_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    freeze, _, _ = _freeze_api()
    tool_repo = tmp_path / "tool-repo"
    tool_repo.mkdir()
    _run("git", "init", "-q", cwd=tool_repo)
    _run("git", "config", "user.email", "d36@example.invalid", cwd=tool_repo)
    _run("git", "config", "user.name", "D36 Test", cwd=tool_repo)
    for relative in freeze._TOOL_SOURCE_PATHS:
        path = tool_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"committed {relative}\n".encode("utf-8"))
    _run("git", "add", ".", cwd=tool_repo)
    _run("git", "commit", "-q", "-m", "committed tooling", cwd=tool_repo)
    monkeypatch.setattr(freeze, "_tool_repo_root", lambda: tool_repo)
    return tool_repo


def _tree_snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if ".git" in path.relative_to(root).parts or not path.is_file():
            continue
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _rewrite_completion_marker(publication: Path) -> None:
    files = []
    for name in ("d36-tool-attestation.json", "freeze-manifest.json"):
        raw = (publication / name).read_bytes()
        files.append(
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
        )
    marker = {"files": files, "schema_version": 1}
    (publication / ".d36-publication-state").write_bytes(
        _canonical_bytes(marker) + b"\n"
    )


def test_freezer_module_is_available() -> None:
    freeze, fingerprints, attestation = _freeze_api()
    assert callable(freeze.freeze_candidate)
    assert callable(fingerprints.fingerprint_files)
    assert callable(attestation.canonical_json_bytes)


def test_same_inputs_produce_byte_identical_manifest_with_commit_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path, _control(commit))
    monkeypatch.setenv("TZ", "Pacific/Honolulu")
    if hasattr(os, "tzset"):
        os.tzset()
    first = freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tmp_path / "one",
    )
    monkeypatch.setenv("TZ", "Europe/Paris")
    if hasattr(os, "tzset"):
        os.tzset()
    second = freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tmp_path / "two",
    )
    first_bytes = (tmp_path / "one" / first.candidate_id / "freeze-manifest.json").read_bytes()
    second_bytes = (tmp_path / "two" / second.candidate_id / "freeze-manifest.json").read_bytes()
    expected_time = datetime.fromtimestamp(1_700_000_000, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert first.created_at == expected_time
    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert b"+00:00" not in first_bytes and b".000" not in first_bytes
    assert hashlib.sha256(first_bytes).hexdigest() == hashlib.sha256(second_bytes).hexdigest()


@pytest.mark.parametrize(
    "migration_source",
    [
        b"def apply_v0_to_v1(connection, metadata):\n    pass\n",
        b"def apply_v0_to_v1(connection, metadata):\n    connection.execute('PRAGMA user_version=2')\n",
        b"def apply_v0_to_v1(connection, metadata):\n    connection.execute('PRAGMA user_version=1')\n    connection.execute('PRAGMA user_version=1')\n",
        b"def apply_v0_to_v1(:\n    pass\n",
    ],
)
def test_candidate_schema_version_must_be_one_unambiguous_committed_migration(
    tmp_path: Path, migration_source: bytes
) -> None:
    candidate, commit = _make_candidate(
        tmp_path,
        extras={"backend/app/migrations/schema.py": migration_source},
    )
    with pytest.raises(ValueError, match="schema version|migration source"):
        _freeze(tmp_path, candidate, commit)
    assert not (tmp_path / "output").exists()


def test_commit_timestamp_changes_created_at_and_manifest_hash(tmp_path: Path) -> None:
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_candidate, first_commit = _make_candidate(first_root, timestamp=1_700_000_000)
    second_candidate, second_commit = _make_candidate(second_root, timestamp=1_700_000_001)
    first = _freeze(first_root, first_candidate, first_commit)
    second = _freeze(second_root, second_candidate, second_commit)
    first_bytes = (first_root / "output" / first.candidate_id / "freeze-manifest.json").read_bytes()
    second_bytes = (second_root / "output" / second.candidate_id / "freeze-manifest.json").read_bytes()
    assert first.created_at == "2023-11-14T22:13:20Z"
    assert second.created_at == "2023-11-14T22:13:21Z"
    assert hashlib.sha256(first_bytes).digest() != hashlib.sha256(second_bytes).digest()


def test_manifest_binds_control_hash_candidate_id_and_strict_contracts(tmp_path: Path) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path, _control(commit))
    manifest = freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tmp_path / "output",
    )
    assert manifest.git_commit == commit
    assert manifest.git_tree_clean is True
    assert manifest.candidate_control_sha256 == digest
    assert manifest.candidate_id == f"{manifest.aggregate_sha256[:16]}-{commit[:12]}"
    assert manifest.schema_version_number == 1
    assert isinstance(manifest.files, list)
    assert set(manifest.mode_configuration) == {"all_tools", "stateful"}
    assert all("\\" not in item.path and not Path(item.path).is_absolute() for item in manifest.files)
    payload = json.loads((tmp_path / "output" / manifest.candidate_id / "freeze-manifest.json").read_bytes())
    assert payload == manifest.model_dump(mode="json")
    with pytest.raises(ValidationError):
        freeze.FreezeManifest.model_validate({**payload, "unknown": True})


@pytest.mark.parametrize(
    ("expected_digest", "control_transform", "canonical"),
    [
        ("0" * 64, lambda value: value, True),
        ("ABC", lambda value: value, True),
        ("dda5", lambda value: value, True),
        (None, lambda value: {**value, "extra": True}, True),
        (None, lambda value: {key: item for key, item in value.items() if key != "git_commit"}, True),
        (None, lambda value: {**value, "git_tree_clean": False}, True),
        (None, lambda value: value, False),
    ],
)
def test_invalid_candidate_control_fails_without_output(
    tmp_path: Path,
    expected_digest: str | None,
    control_transform: Any,
    canonical: bool,
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    path, actual = _write_control(tmp_path, control_transform(_control(commit)), canonical=canonical)
    with pytest.raises((ValueError, ValidationError)):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=path,
            expected_candidate_control_sha256=expected_digest or actual,
            output_root=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_control_is_read_once_and_hash_checked_before_fingerprinting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, fingerprints, _ = _freeze_api()
    path, _ = _write_control(tmp_path, _control(commit))
    called = False

    def forbidden(_: Path) -> list[Any]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(freeze, "fingerprint_files", forbidden)
    with pytest.raises(ValueError, match="candidate-control SHA-256"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=path,
            expected_candidate_control_sha256="f" * 64,
            output_root=tmp_path / "output",
        )
    assert not called
    assert callable(fingerprints.fingerprint_files)


def test_dirty_untracked_attached_stale_and_subject_mismatch_are_rejected(tmp_path: Path) -> None:
    freeze, _, _ = _freeze_api()
    cases: list[tuple[Path, str, dict[str, Any]]] = []

    dirty_root = tmp_path / "dirty"
    dirty_root.mkdir()
    dirty, dirty_commit = _make_candidate(dirty_root)
    (dirty / "backend/app/core/config.py").write_text("changed\n", encoding="utf-8")
    cases.append((dirty, dirty_commit, _control(dirty_commit)))

    untracked_root = tmp_path / "untracked"
    untracked_root.mkdir()
    untracked, untracked_commit = _make_candidate(untracked_root)
    (untracked / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    cases.append((untracked, untracked_commit, _control(untracked_commit)))

    attached_root = tmp_path / "attached"
    attached_root.mkdir()
    attached, attached_commit = _make_candidate(attached_root, detach=False)
    cases.append((attached, attached_commit, _control(attached_commit)))

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    stale, stale_commit = _make_candidate(stale_root)
    cases.append((stale, stale_commit, _control("0" * 40)))

    subject_root = tmp_path / "subject"
    subject_root.mkdir()
    subject, subject_commit = _make_candidate(subject_root, subject="wrong subject")
    cases.append((subject, subject_commit, _control(subject_commit)))

    for index, (candidate, _, control) in enumerate(cases):
        control_path, digest = _write_control(tmp_path / f"control-{index}", control)
        with pytest.raises(ValueError):
            freeze.freeze_candidate(
                candidate_root=candidate,
                candidate_control_path=control_path,
                expected_candidate_control_sha256=digest,
                output_root=tmp_path / f"output-{index}",
            )
        assert not (tmp_path / f"output-{index}").exists()


def test_candidate_allowlist_is_sorted_and_excludes_tools_secrets_and_runtime(tmp_path: Path) -> None:
    excluded = {
        ".env": b"SECRET=value\n",
        "backend/evaluation/final_protocol.json": b"later protocol",
        "backend/evaluation/release_candidate/freeze.py": b"later freezer",
        "backend/evaluation/scripts/evaluation_trial_host.py": b"later host",
        "backend/evaluation/tool_attestation.py": b"later attestation",
        "backend/node_modules/dependency/index.js": b"generated",
        "backend/storage/runtime.db": b"sqlite",
        "evaluation/d24/held-out.jsonl": b"private held-out text",
        "frontend/dist/bundle.js": b"generated",
        "frontend/node_modules/react/index.js": b"generated",
        "model-weights/model.onnx": b"weights",
        "release-evidence/old.json": b"runtime evidence",
        "storage/video.mp4": b"media",
    }
    candidate, _ = _make_candidate(tmp_path, extras=excluded)
    _, fingerprints, _ = _freeze_api()
    files = fingerprints.fingerprint_files(candidate)
    paths = [item.path for item in files]
    assert paths == sorted(paths)
    assert "backend/.env.example" in paths
    assert "backend/app/core/config.py" in paths
    assert "frontend/src/main.tsx" in paths
    assert "backend/tests/test_candidate.py" in paths
    assert "docs/DTD.md" in paths
    assert not set(excluded).intersection(paths)


def test_candidate_manifest_hashes_committed_bytes_and_reconstructs_from_head(
    tmp_path: Path,
) -> None:
    candidate, _ = _make_candidate(tmp_path)
    _, fingerprints, _ = _freeze_api()
    files = fingerprints.fingerprint_files(candidate)
    by_path = {item.path: item for item in files}
    assert "backend/.env.example" in by_path
    for relative, item in by_path.items():
        committed = subprocess.run(
            ["git", "-C", str(candidate), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert item.size == len(committed)
        assert item.sha256 == hashlib.sha256(committed).hexdigest()

    reconstructed = tmp_path / "reconstructed"
    _run(
        "git",
        "-c",
        "core.autocrlf=false",
        "clone",
        "-q",
        "--no-hardlinks",
        str(candidate),
        str(reconstructed),
        cwd=tmp_path,
    )
    _run("git", "checkout", "-q", "--detach", "HEAD", cwd=reconstructed)
    assert fingerprints.fingerprint_files(reconstructed) == files


def test_allowlisted_working_byte_change_is_rejected_even_when_index_hides_it(
    tmp_path: Path,
) -> None:
    candidate, _ = _make_candidate(tmp_path)
    _, fingerprints, _ = _freeze_api()
    relative = "backend/app/core/config.py"
    _run("git", "update-index", "--assume-unchanged", relative, cwd=candidate)
    (candidate / relative).write_bytes(b"changed but hidden\n")
    with pytest.raises(ValueError, match="index flags|committed blob"):
        fingerprints.fingerprint_files(candidate)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_candidate_special_index_flags_are_rejected(
    tmp_path: Path, flag: str
) -> None:
    candidate, _ = _make_candidate(tmp_path)
    _, fingerprints, _ = _freeze_api()
    _run("git", "update-index", flag, "backend/app/core/config.py", cwd=candidate)
    with pytest.raises(ValueError, match="index flags"):
        fingerprints.fingerprint_files(candidate)


@pytest.mark.parametrize(
    "missing",
    [
        "backend/app/operations/definitions.json",
        "backend/app/operations/search_scope.json",
        "backend/app/retrieval/e5-profile.json",
        "backend/app/retrieval/nomic-profile.json",
        "backend/pyproject.toml",
        "backend/uv.lock",
        "frontend/package.json",
        "frontend/pnpm-lock.yaml",
    ],
)
def test_missing_candidate_inputs_fail_closed(tmp_path: Path, missing: str) -> None:
    candidate, _ = _make_candidate(tmp_path, omit={missing})
    _, fingerprints, _ = _freeze_api()
    with pytest.raises(ValueError, match="required candidate file"):
        fingerprints.fingerprint_files(candidate)


def test_candidate_is_unchanged_including_ignored_inventory(tmp_path: Path) -> None:
    candidate, commit = _make_candidate(tmp_path)
    before_files = _tree_snapshot(candidate)
    before_status = _run("git", "status", "--porcelain=v1", "--ignored", cwd=candidate)
    _freeze(tmp_path, candidate, commit)
    assert _tree_snapshot(candidate) == before_files
    assert _run("git", "status", "--porcelain=v1", "--ignored", cwd=candidate) == before_status


def test_snapshot_does_not_read_excluded_secret_or_held_out_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(
        tmp_path,
        extras={
            ".env": b"SECRET=must-not-be-read\n",
            "evaluation/d24/held-out-private.jsonl": b"held-out body",
        },
    )
    freeze, _, _ = _freeze_api()
    original = freeze._read_regular_once

    def guarded(path: Path, *, maximum: int, label: str) -> bytes:
        relative = path.relative_to(candidate).as_posix() if path.is_relative_to(candidate) else ""
        if relative == ".env" or "held-out" in relative:
            raise AssertionError(f"excluded bytes were read: {relative}")
        return original(path, maximum=maximum, label=label)

    monkeypatch.setattr(freeze, "_read_regular_once", guarded)
    control_path, digest = _write_control(tmp_path, _control(commit))
    freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tmp_path / "output",
    )


def test_candidate_mutation_during_freeze_leaves_empty_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path, _control(commit))
    original = freeze.fingerprint_files

    def mutating(root: Path) -> list[Any]:
        result = original(root)
        ignored = root / ".pytest_cache" / "mutation"
        ignored.parent.mkdir()
        ignored.write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(freeze, "fingerprint_files", mutating)
    with pytest.raises(ValueError, match="candidate changed"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    assert (tmp_path / "output").is_dir()
    assert list((tmp_path / "output").iterdir()) == []


def test_in_repo_output_is_only_allowed_under_release_evidence(
    tmp_path: Path, _clean_committed_tooling_repo: Path
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    for name in ("storage", ".pytest_cache", "other-ignored"):
        forbidden = _clean_committed_tooling_repo / name
        with pytest.raises(ValueError, match="release-evidence"):
            freeze.freeze_candidate(
                candidate_root=candidate,
                candidate_control_path=control_path,
                expected_candidate_control_sha256=digest,
                output_root=forbidden,
            )
        assert not forbidden.exists()

    allowed = _clean_committed_tooling_repo / "release-evidence" / "d36"
    manifest = freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=allowed,
    )
    assert (allowed / manifest.candidate_id / "freeze-manifest.json").is_file()


def test_existing_destination_is_refused_and_external_git_root_is_allowed(tmp_path: Path) -> None:
    candidate, commit = _make_candidate(tmp_path)
    manifest = _freeze(tmp_path, candidate, commit)
    with pytest.raises(ValueError, match="output"):
        _freeze(tmp_path, candidate, commit)
    assert (tmp_path / "output" / manifest.candidate_id / "freeze-manifest.json").is_file()

    tracked = tmp_path / "tracked-output"
    tracked.mkdir()
    _run("git", "init", "-q", cwd=tracked)
    _run("git", "config", "user.email", "d36@example.invalid", cwd=tracked)
    _run("git", "config", "user.name", "D36 Test", cwd=tracked)
    (tracked / "evidence").mkdir()
    (tracked / "evidence" / ".keep").write_text("tracked", encoding="utf-8")
    _run("git", "add", ".", cwd=tracked)
    _run("git", "commit", "-q", "-m", "tracked output", cwd=tracked)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "tracked-control", _control(commit))
    external_manifest = freeze.freeze_candidate(
        candidate_root=candidate,
        candidate_control_path=control_path,
        expected_candidate_control_sha256=digest,
        output_root=tracked / "evidence",
    )
    assert (
        tracked / "evidence" / external_manifest.candidate_id / "freeze-manifest.json"
    ).is_file()


def _make_tool_source_repo(root: Path) -> tuple[Path, str]:
    repository = root / "attestation-repo"
    repository.mkdir()
    _run("git", "init", "-q", cwd=repository)
    _run("git", "config", "user.email", "d36@example.invalid", cwd=repository)
    _run("git", "config", "user.name", "D36 Test", cwd=repository)
    (repository / "a.py").write_bytes(b"A = 1\n")
    (repository / "b.py").write_bytes(b"B = 1\n")
    _run("git", "add", ".", cwd=repository)
    _run("git", "commit", "-q", "-m", "tooling", cwd=repository)
    return repository, _run("git", "rev-parse", "HEAD", cwd=repository)


def test_tool_attestation_binds_clean_working_files_to_declared_head(tmp_path: Path) -> None:
    _, _, attestation = _freeze_api()
    repository, commit = _make_tool_source_repo(tmp_path)
    result = attestation.attest_tool(
        repo_root=repository,
        tool_name="test_tool",
        git_commit=commit,
        source_paths=("a.py", "b.py"),
    )
    assert result.git_commit == commit
    assert result.files[0].sha256 == hashlib.sha256(b"A = 1\n").hexdigest()


@pytest.mark.parametrize("condition", ["dirty", "untracked", "assume", "skip", "mismatch"])
def test_tool_attestation_rejects_uncommitted_or_unbound_source(
    tmp_path: Path, condition: str
) -> None:
    _, _, attestation = _freeze_api()
    repository, commit = _make_tool_source_repo(tmp_path)
    if condition == "dirty":
        (repository / "a.py").write_text("A = 2\n", encoding="utf-8")
    elif condition == "untracked":
        (repository / "untracked.py").write_text("X = 1\n", encoding="utf-8")
    elif condition == "assume":
        _run("git", "update-index", "--assume-unchanged", "a.py", cwd=repository)
    elif condition == "skip":
        _run("git", "update-index", "--skip-worktree", "a.py", cwd=repository)
    else:
        (repository / "later.py").write_text("later\n", encoding="utf-8")
        _run("git", "add", ".", cwd=repository)
        _run("git", "commit", "-q", "-m", "later", cwd=repository)
    with pytest.raises(ValueError, match="clean|index flags|HEAD"):
        attestation.attest_tool(
            repo_root=repository,
            tool_name="test_tool",
            git_commit=commit,
            source_paths=("a.py", "b.py"),
        )


def test_claim_construction_never_opens_final_candidate_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    original_write = freeze._write_owned
    observed_paths: list[Path] = []

    def observe_write(claim: Any, name: str, value: bytes) -> None:
        observed_paths.append(claim.path)
        assert claim.path.name.startswith(freeze._STAGING_PREFIX)
        assert not (claim.path.parent / json.loads(value).get("candidate_id", "missing")).exists()
        original_write(claim, name, value)

    monkeypatch.setattr(freeze, "_write_owned", observe_write)
    manifest = _freeze(tmp_path, candidate, commit)

    assert observed_paths
    assert all(path != tmp_path / "output" / manifest.candidate_id for path in observed_paths)
    assert (tmp_path / "output" / manifest.candidate_id).is_dir()


def test_concurrent_final_creator_wins_without_staging_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original_publish = freeze._publish_staging
    staging: Path | None = None

    def create_winner(claim: Any, final: Path, completion: bytes) -> None:
        nonlocal staging
        staging = claim.path
        final.mkdir()
        (final / "winner.txt").write_text("concurrent owner", encoding="utf-8")
        original_publish(claim, final, completion)

    monkeypatch.setattr(freeze, "_publish_staging", create_winner)
    with pytest.raises(ValueError, match="output destination already exists"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )

    assert staging is not None and staging.is_dir()
    winners = list((tmp_path / "output").glob("*/winner.txt"))
    assert len(winners) == 1
    assert winners[0].read_text(encoding="utf-8") == "concurrent owner"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory handle contract")
def test_windows_directory_anchor_blocks_native_rename(tmp_path: Path) -> None:
    freeze, _, _ = _freeze_api()
    root = tmp_path / "output"
    root.mkdir()
    claim = freeze._create_staging_claim(root)
    moved = root / "moved"
    freeze._close_retained_files(claim)
    try:
        with pytest.raises(PermissionError):
            claim.path.rename(moved)
    finally:
        freeze._close_claim(claim)
    claim.path.rename(moved)
    assert moved.is_dir()


def test_windows_move_uses_no_replace_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze, _, _ = _freeze_api()
    calls: list[tuple[str, str, int]] = []

    def move(source: str, destination: str, flags: int) -> bool:
        calls.append((source, destination, flags))
        return True

    monkeypatch.setattr(freeze, "_move_file_ex_w", move)
    freeze._windows_move_directory_no_replace(Path("stage"), Path("final"))
    assert calls == [(str(Path("stage")), str(Path("final")), 0)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory descriptor contract")
def test_posix_anchor_detects_staging_path_replacement(tmp_path: Path) -> None:
    freeze, _, _ = _freeze_api()
    root = tmp_path / "output"
    root.mkdir()
    claim = freeze._create_staging_claim(root)
    moved = root / "moved"
    try:
        claim.path.rename(moved)
        claim.path.mkdir()
        with pytest.raises(freeze.PublicationOwnershipLost):
            freeze._write_owned(claim, "freeze-manifest.json", b"replacement test")
        assert list(claim.path.iterdir()) == []
    finally:
        freeze._close_claim(claim)


@pytest.mark.skipif(os.name != "posix", reason="Linux renameat2 contract")
def test_linux_rename_noreplace_preserves_existing_destination(tmp_path: Path) -> None:
    freeze, _, _ = _freeze_api()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "winner").write_text("kept", encoding="utf-8")

    with pytest.raises(FileExistsError):
        freeze._linux_rename_directory_no_replace(source, destination)

    assert source.is_dir()
    assert (destination / "winner").read_text(encoding="utf-8") == "kept"


def test_publication_does_not_use_overwrite_capable_os_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()

    def forbidden_rename(*_: object) -> None:
        raise AssertionError("publication must not rename a directory")

    monkeypatch.setattr(freeze.os, "rename", forbidden_rename)
    manifest = _freeze(tmp_path, candidate, commit)
    assert (tmp_path / "output" / manifest.candidate_id / "freeze-manifest.json").is_file()


@pytest.mark.parametrize(
    "ownership_check",
    [2, 4, 6, 8, 10, 12],
    ids=[
        "manifest-write",
        "attestation-write",
        "manifest-readback",
        "attestation-readback",
        "completion-write",
        "completion-readback",
    ],
)
def test_directory_swap_between_ownership_check_and_descriptor_io_leaves_replacement_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ownership_check: int
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original_assert = freeze._assert_claim_owned
    checks = 0
    moved: Path | None = None
    replacement: Path | None = None

    def swap_after_check(
        claim: Any, expected_state: bytes | None = None
    ) -> None:
        nonlocal checks, moved, replacement
        original_assert(claim, expected_state)
        checks += 1
        if checks != ownership_check:
            return
        moved = claim.path
        replacement = claim.path.with_name(f"{claim.path.name}.replacement")
        replacement.mkdir()
        object.__setattr__(claim, "path", replacement)

    monkeypatch.setattr(freeze, "_assert_claim_owned", swap_after_check)
    with pytest.raises(freeze.PublicationOwnershipLost):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )

    assert moved is not None
    assert replacement is not None
    assert list(replacement.iterdir()) == []
    assert {path.name for path in moved.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }


def test_state_path_replacement_at_former_unlink_boundary_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original_ftruncate = freeze.os.ftruncate
    replacement = b"replacement owner"
    raced = False
    replacement_created = False

    def replace_state_before_transition(descriptor: int, length: int) -> None:
        nonlocal raced, replacement_created
        if not raced:
            state = next((tmp_path / "output").glob("*/.d36-publication-state"))
            try:
                state.rename(state.with_name(".moved-publication-state"))
            except PermissionError:
                pass
            else:
                state.write_bytes(replacement)
                replacement_created = True
            raced = True
        original_ftruncate(descriptor, length)

    monkeypatch.setattr(freeze.os, "ftruncate", replace_state_before_transition)
    try:
        manifest = freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    except freeze.PublicationOwnershipLost:
        manifest = None
    state = next((tmp_path / "output").glob("*/.d36-publication-state"))
    assert raced is True
    if replacement_created:
        assert manifest is None
        assert state.read_bytes() == replacement
    else:
        assert manifest is not None
        freeze.read_frozen_candidate(state.parent)


def test_failure_leaves_incomplete_partials_and_closes_state_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original_read = freeze._read_retained

    def fail_readback(file: Any, maximum: int, *, label: str) -> bytes:
        value = original_read(file, maximum, label=label)
        return b"corrupt" if label == "freeze manifest" else value

    monkeypatch.setattr(freeze, "_read_retained", fail_readback)
    with pytest.raises(ValueError, match="published freeze manifest changed"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    incomplete = next((tmp_path / "output").iterdir())
    assert {path.name for path in incomplete.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }
    state = incomplete / freeze._PUBLICATION_STATE_NAME
    moved = state.with_name(".operator-cleanup-state")
    state.rename(moved)
    assert moved.stat().st_size == freeze._CLAIM_TOKEN_BYTES
    with pytest.raises(ValueError, match="completed publication"):
        freeze.read_frozen_candidate(incomplete)


def test_failure_closes_all_three_retained_descriptors_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original_claim = freeze._create_staging_claim
    original_close = freeze.os.close
    original_read = freeze._read_retained
    retained: set[int] = set()
    close_counts: dict[int, int] = {}

    def capture_claim(path: Path) -> Any:
        claim = original_claim(path)
        retained.update(file.descriptor for file in claim.files)
        return claim

    def count_close(descriptor: int) -> None:
        if descriptor in retained:
            close_counts[descriptor] = close_counts.get(descriptor, 0) + 1
        original_close(descriptor)

    def fail_manifest_readback(file: Any, maximum: int, *, label: str) -> bytes:
        if label == "freeze manifest":
            raise ValueError("injected readback failure")
        return original_read(file, maximum, label=label)

    monkeypatch.setattr(freeze, "_create_staging_claim", capture_claim)
    monkeypatch.setattr(freeze.os, "close", count_close)
    monkeypatch.setattr(freeze, "_read_retained", fail_manifest_readback)

    with pytest.raises(ValueError, match="injected readback failure"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )

    assert len(retained) == 3
    assert close_counts == {descriptor: 1 for descriptor in retained}


def test_staging_validation_failure_leaves_random_non_evidence_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path, _control(commit))
    original = freeze._read_retained

    def corrupt_published(file: Any, maximum: int, *, label: str) -> bytes:
        value = original(file, maximum, label=label)
        return b"corrupt" if label == "freeze manifest" else value

    monkeypatch.setattr(freeze, "_read_retained", corrupt_published)
    with pytest.raises(ValueError, match="published freeze manifest changed"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    candidates = list((tmp_path / "output").iterdir())
    assert len(candidates) == 1
    incomplete = candidates[0]
    assert {path.name for path in incomplete.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }
    with pytest.raises(ValueError, match="published freeze manifest changed"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    candidates = list((tmp_path / "output").iterdir())
    assert len(candidates) == 2
    assert incomplete in candidates
    assert all(path.name.startswith(freeze._STAGING_PREFIX) for path in candidates)
    assert {path.name for path in incomplete.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }


def test_published_attestation_mismatch_leaves_incomplete_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    control_path, digest = _write_control(tmp_path / "control", _control(commit))
    original = freeze._read_retained

    def corrupt_attestation(file: Any, maximum: int, *, label: str) -> bytes:
        value = original(file, maximum, label=label)
        return b"{}\n" if label == "tool attestation" else value

    monkeypatch.setattr(freeze, "_read_retained", corrupt_attestation)
    with pytest.raises(ValueError, match="published tool attestation"):
        freeze.freeze_candidate(
            candidate_root=candidate,
            candidate_control_path=control_path,
            expected_candidate_control_sha256=digest,
            output_root=tmp_path / "output",
        )
    incomplete = next((tmp_path / "output").iterdir())
    assert {path.name for path in incomplete.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }


def test_completed_publication_rewrites_one_stable_state_file_canonically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    freeze, _, _ = _freeze_api()
    claim_token = b"x" * freeze._CLAIM_TOKEN_BYTES
    monkeypatch.setattr(freeze.secrets, "token_bytes", lambda _: claim_token)
    manifest = _freeze(tmp_path, candidate, commit)
    publication = tmp_path / "output" / manifest.candidate_id

    loaded_manifest, loaded_attestation = freeze.read_frozen_candidate(publication)

    assert loaded_manifest == manifest
    assert loaded_attestation.tool_name == "d36_candidate_freezer_and_trial_host"
    assert {path.name for path in publication.iterdir()} == {
        freeze._PUBLICATION_STATE_NAME,
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }
    state_raw = (publication / freeze._PUBLICATION_STATE_NAME).read_bytes()
    marker = json.loads(state_raw)
    assert state_raw == _canonical_bytes(marker) + b"\n"
    assert claim_token not in state_raw
    assert marker == {
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256((publication / name).read_bytes()).hexdigest(),
                "size": (publication / name).stat().st_size,
            }
            for name in ("d36-tool-attestation.json", "freeze-manifest.json")
        ],
        "schema_version": 1,
    }


@pytest.mark.parametrize(
    "condition", ["missing", "tampered", "incomplete", "extra", "replaced_during_read"]
)
def test_frozen_candidate_reader_requires_exact_valid_completion_marker(
    tmp_path: Path, condition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    manifest = _freeze(tmp_path, candidate, commit)
    freeze, _, _ = _freeze_api()
    publication = tmp_path / "output" / manifest.candidate_id
    marker = publication / freeze._PUBLICATION_STATE_NAME
    if condition == "missing":
        marker.unlink()
    elif condition == "tampered":
        marker.write_bytes(b"{" + marker.read_bytes()[1:-2] + b"X\n")
    elif condition == "incomplete":
        marker.write_bytes(b"x" * freeze._CLAIM_TOKEN_BYTES)
    elif condition == "extra":
        (publication / "extra").write_bytes(b"extra")
    else:
        original_read = freeze._read_regular_once

        def replace_marker(path: Path, *, maximum: int, label: str) -> bytes:
            if label == "tool attestation":
                saved = marker.read_bytes()
                marker.unlink()
                marker.write_bytes(saved)
            return original_read(path, maximum=maximum, label=label)

        monkeypatch.setattr(freeze, "_read_regular_once", replace_marker)

    with pytest.raises(ValueError, match="completed publication"):
        freeze.read_frozen_candidate(publication)


@pytest.mark.parametrize(
    "tampering",
    [
        "file_fingerprint",
        "aggregate_suffix",
        "candidate_id",
        "duplicate_file_path",
        "unsorted_file_paths",
        "canonical_manifest_bytes",
    ],
)
def test_frozen_candidate_reader_rejects_content_address_tampering(
    tmp_path: Path, tampering: str
) -> None:
    candidate, commit = _make_candidate(tmp_path)
    manifest = _freeze(tmp_path, candidate, commit)
    freeze, _, _ = _freeze_api()
    publication = tmp_path / "output" / manifest.candidate_id
    manifest_path = publication / "freeze-manifest.json"
    payload = json.loads(manifest_path.read_bytes())

    if tampering == "file_fingerprint":
        payload["files"][0]["sha256"] = "f" * 64
    elif tampering == "aggregate_suffix":
        suffix = "0" if payload["aggregate_sha256"][-1] != "0" else "1"
        payload["aggregate_sha256"] = payload["aggregate_sha256"][:-1] + suffix
    elif tampering == "candidate_id":
        payload["candidate_id"] = f"{'f' * 16}-{'e' * 12}"
        publication.rename(publication.with_name(payload["candidate_id"]))
        publication = publication.with_name(payload["candidate_id"])
        manifest_path = publication / "freeze-manifest.json"
    elif tampering == "duplicate_file_path":
        payload["files"][1]["path"] = payload["files"][0]["path"]
    elif tampering == "unsorted_file_paths":
        payload["files"] = list(reversed(payload["files"]))
    else:
        payload["created_at"] = "2023-11-14T22:13:21Z"

    manifest_path.write_bytes(_canonical_bytes(payload) + b"\n")
    if tampering != "canonical_manifest_bytes":
        _rewrite_completion_marker(publication)

    with pytest.raises(ValueError, match="completed publication"):
        freeze.read_frozen_candidate(publication)


def test_d36_tool_attestation_has_exact_allowlist_and_valid_aggregate(tmp_path: Path) -> None:
    candidate, commit = _make_candidate(tmp_path)
    manifest = _freeze(tmp_path, candidate, commit)
    _, _, attestation = _freeze_api()
    path = tmp_path / "output" / manifest.candidate_id / "d36-tool-attestation.json"
    raw = path.read_bytes()
    parsed = attestation.ToolAttestation.model_validate_json(raw)
    expected = {
        "backend/evaluation/blinded_io.py",
        "backend/evaluation/evidence_json.py",
        "backend/evaluation/final_protocol.json",
        "backend/evaluation/release_candidate/__init__.py",
        "backend/evaluation/release_candidate/contracts.py",
        "backend/evaluation/release_candidate/fingerprints.py",
        "backend/evaluation/release_candidate/freeze.py",
        "backend/evaluation/scripts/evaluation_trial_host.py",
        "backend/evaluation/scripts/freeze_candidate.py",
        "backend/evaluation/tool_attestation.py",
        "backend/evaluation/unlabeled_contracts.py",
    }
    assert isinstance(parsed.files, list)
    assert {item.path for item in parsed.files} == expected
    assert parsed.tool_name == "d36_candidate_freezer_and_trial_host"
    assert parsed.aggregate_sha256 == attestation.aggregate_fingerprints(parsed.files)
    assert raw == attestation.canonical_json_bytes(parsed.model_dump(mode="json")) + b"\n"
    with pytest.raises(ValidationError):
        attestation.ToolAttestation.model_validate({**parsed.model_dump(), "extra": True})


def test_cli_writes_completed_publication_contract(tmp_path: Path) -> None:
    candidate, commit = _make_candidate(tmp_path)
    control_path, digest = _write_control(tmp_path, _control(commit))
    backend = Path(__file__).parents[1]
    output = tmp_path / "cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.scripts.freeze_candidate",
            "--candidate-root",
            str(candidate),
            "--candidate-control",
            str(control_path),
            "--expected-candidate-control-sha256",
            digest,
            "--output-root",
            str(output),
        ],
        cwd=backend,
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    candidate_output = output / response["candidate_id"]
    assert {path.name for path in candidate_output.iterdir()} == {
        ".d36-publication-state",
        "d36-tool-attestation.json",
        "freeze-manifest.json",
    }
    assert response == {
        "candidate_id": response["candidate_id"],
        "completion_marker": f"{response['candidate_id']}/.d36-publication-state",
        "freeze_manifest": f"{response['candidate_id']}/freeze-manifest.json",
        "tool_attestation": f"{response['candidate_id']}/d36-tool-attestation.json",
    }


SUCCESSOR_SUBJECT = "[DONE] Mission 35.1 Fix release-candidate verification blockers"


def test_authorized_successor_subject_freezes_and_others_are_refused(tmp_path: Path) -> None:
    freeze, _, _ = _freeze_api()
    successor_root = tmp_path / "successor"
    successor_root.mkdir()
    candidate, commit = _make_candidate(successor_root, subject=SUCCESSOR_SUBJECT)
    control_path, digest = _write_control(successor_root, _control(commit, git_commit_subject=SUCCESSOR_SUBJECT))
    manifest = freeze.freeze_candidate(candidate_root=candidate, candidate_control_path=control_path,
                                       expected_candidate_control_sha256=digest, output_root=successor_root / "output")
    assert manifest.git_commit == commit and manifest.candidate_control_sha256 == digest
    # A control naming the D35 subject cannot vouch for the successor commit, and an
    # unlisted subject is refused by the strict contract itself.
    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    control_path, digest = _write_control(mismatch_root, _control(commit))
    with pytest.raises(ValueError):
        freeze.freeze_candidate(candidate_root=candidate, candidate_control_path=control_path,
                                expected_candidate_control_sha256=digest, output_root=mismatch_root / "output")
    with pytest.raises(ValidationError):
        freeze.CandidateControl.model_validate(_control(commit, git_commit_subject="[DONE] Mission 35.2 Unreviewed"))


def test_readme_referenced_docs_are_inventoried(tmp_path: Path) -> None:
    _, fingerprints, _ = _freeze_api()
    extras = {"docs/plan-c/work-report-22.md": b"# report\n", "docs/frame-subtitles.png": b"\x89PNG\x00\x00\x00\rIHDR",
              "docs/notes.txt": b"not documentation\n"}
    candidate, _ = _make_candidate(tmp_path, extras=extras)
    paths = {item.path for item in fingerprints.fingerprint_files(candidate)}
    assert {"docs/plan-c/work-report-22.md", "docs/frame-subtitles.png"} <= paths
    assert "docs/notes.txt" not in paths
