from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluation.evidence_json import parse_canonical_model
from evaluation.release_candidate import freeze
from evaluation.release_candidate.fingerprints import fingerprint_files
from evaluation.tool_attestation import aggregate_fingerprints
from tests.test_d36_freeze import _make_candidate

SHA = "a" * 64
COMMIT = "b" * 40
CANDIDATE = "aaaaaaaaaaaaaaaa-bbbbbbbbbbbb"


def _module(name: str) -> Any:
    assert importlib.util.find_spec(name) is not None, f"missing Task 1 behavior: {name}"
    return importlib.import_module(name)


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def _fingerprint(path: str = "summary.json") -> dict[str, object]:
    return {"path": path, "size": 3, "sha256": hashlib.sha256(b"{}\n").hexdigest()}


def _binding(role: str = "python") -> dict[str, object]:
    return {"role": role, "version": "3.12.12", "executable": _fingerprint("tools/python_sandbox"), "launcher": None}


def _summary(stage: str) -> dict[str, object]:
    values: dict[str, dict[str, object]] = {
        "legacy_migration": {"from_version": 0, "to_version": 1, "integrity_ok": True, "foreign_keys_enabled": True, "rows_preserved": True, "identities_preserved": True, "backup_size": 3, "backup_sha256": SHA},
        "restore": {"restored_version": 1, "integrity_ok": True, "foreign_keys_enabled": True, "rows_equal": True, "identities_equal": True, "lease_exclusion_passed": True},
        "all_tools_startup": {"mode": "all_tools", "startup_ready": True, "health_ok": True, "request_completed": True, "model_calls": 1, "index_sha256": None},
        "stateful_startup": {"mode": "stateful", "startup_ready": True, "health_ok": True, "request_completed": True, "model_calls": 1, "index_sha256": SHA, "profile_sha256": SHA, "embedding_calls": 1, "retrieval_verified": True},
        "browser": {"narrow_width": 390, "wide_width": 1440, "waiting_ok": True, "safe_retry_ok": True, "unknown_remote_blocked": True, "migration_failed_ok": True, "keyboard_ok": True, "duplicate_post_count": 1, "horizontal_overflow": False, "playback_ok": True, "documentation_checks": dict.fromkeys(("setup_paths", "locked_versions", "mode_commands", "recovery_codes", "migration_restore", "limitation_boundary"), True)},
        "ffmpeg": {"providers_fake": True, "ffmpeg_exit_code": 0, "ffprobe_exit_code": 0, "video_present": True, "subtitle_present": True, "publication_bound": True, "duration_ms": 500},
    }
    return {"stage": stage, **values[stage]}


def _receipt(stage: str = "restore") -> dict[str, object]:
    roles = {
        "node": ("24.11.1", "node", None), "npx": ("11.6.2", "node", "npx_cli"),
        "pnpm": ("10.18.3", "node", "pnpm_cjs"), "python": ("3.12.12", "python_sandbox", None),
        "python_bootstrap": ("3.12.12", "python_bootstrap", None), "uv": ("0.12.15", "python_bootstrap", "uv_module"),
    }
    if stage == "browser":
        roles.update(chrome=("139.0.0.0", "chrome", None), websockets=("16.1.1", "python_sandbox", "websockets_module"))
    if stage == "ffmpeg":
        roles.update(ffmpeg=("7.1", "ffmpeg", None), ffprobe=("7.1", "ffprobe", None))
    tools = [{"role": role, "version": version, "executable": _fingerprint("tools/" + executable),
              "launcher": _fingerprint("tools/" + launcher) if launcher else None}
             for role, (version, executable, launcher) in sorted(roles.items())]
    return {"schema_version": 1, "stage": stage, "candidate_id": CANDIDATE, "git_commit": COMMIT, "freeze_sha256": SHA, "materialization_sha256": SHA, "runtime_instance_id": SHA, "runtime_source_sha256": SHA, "outcome": "passed", "tools": tools, "artifacts": [_fingerprint()], "summary": _summary(stage)}


def _command() -> dict[str, object]:
    bootstrap = {"role": "python_bootstrap", "version": "3.12.12", "executable": _fingerprint("tools/python_bootstrap"), "launcher": None}
    uv = {"role": "uv", "version": "0.12.15", "executable": _fingerprint("tools/python_bootstrap"), "launcher": _fingerprint("tools/uv_module")}
    return {"name": "backend_uv_sync", "argv": ["python", "-m", "uv", "sync", "--locked", "--extra", "dev", "--extra", "retrieval", "--no-python-downloads", "--no-config"], "resolved_argv": ["tools/python_bootstrap", "-m", "uv", "sync", "--locked", "--extra", "dev", "--extra", "retrieval", "--no-python-downloads", "--no-config"], "tool_bindings": [bootstrap, uv], "media_tools": [], "cwd": "backend", "deadline_seconds": 600, "outcome": "completed", "exit_code": 0, "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:01Z", "stdout_size": 0, "stderr_size": 0, "stdout_sha256": hashlib.sha256(b"").hexdigest(), "stderr_sha256": hashlib.sha256(b"").hexdigest()}


def test_schemas_are_pure_frozen_and_json_tuples() -> None:
    contracts = _module("evaluation.smoke_contracts")
    receipt = parse_canonical_model(_bytes(_receipt()), contracts.SmokeStageReceipt, maximum=65536)
    assert receipt.summary.rows_equal is True
    assert type(receipt.tools) is tuple
    with pytest.raises(ValidationError):
        receipt.outcome = "failed"
    source = Path(contracts.__file__).read_text()
    import ast
    dependencies = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)}
    assert not dependencies.intersection({"evaluation.blinded_runtime", "evaluation.runtime_materialization", "evaluation.browser_smoke", "evaluation.release_verification"})


@pytest.mark.parametrize("bad_path", ["C:/outside", "tools/new\nline", "../outside", "/outside"])
def test_tool_and_artifact_aliases_cannot_escape_owned_roots(bad_path: str) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _receipt()
    payload["tools"][0]["executable"]["path"] = bad_path
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.SmokeStageReceipt, maximum=65536)


@pytest.mark.parametrize(("field", "value"), [("schema_version", True), ("schema_version", 1.0), ("stage", 1), ("candidate_id", "x" * 513), ("git_commit", "B" * 40), ("unexpected", 0)])
def test_receipt_rejects_raw_primitives_and_extra_fields(field: str, value: object) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _receipt()
    payload[field] = value
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.SmokeStageReceipt, maximum=65536)


@pytest.mark.parametrize(("stage", "field", "value"), [("legacy_migration", "from_version", False), ("browser", "narrow_width", True), ("browser", "narrow_width", 391), ("restore", "integrity_ok", 1), ("stateful_startup", "embedding_calls", 0), ("stateful_startup", "profile_sha256", None), ("ffmpeg", "ffmpeg_exit_code", None), ("ffmpeg", "duration_ms", 60001), ("browser", "duplicate_post_count", 2)])
def test_passed_summary_requires_actual_bounded_observations(stage: str, field: str, value: object) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _receipt(stage)
    payload["summary"][field] = value
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.SmokeStageReceipt, maximum=65536)


def test_failed_receipt_keeps_nullable_observation_but_no_complete_smoke() -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _receipt("ffmpeg")
    payload["outcome"] = "failed"
    payload["summary"]["ffmpeg_exit_code"] = None
    assert parse_canonical_model(_bytes(payload), contracts.SmokeStageReceipt, maximum=65536).summary.ffmpeg_exit_code is None
    stages = ("legacy_migration", "restore", "all_tools_startup", "stateful_startup", "browser", "ffmpeg")
    receipts = [_receipt(stage) for stage in stages]
    manifest = {"schema_version": 1, "producer_tool_sha256": SHA, "candidate_id": CANDIDATE, "git_commit": COMMIT, "freeze_sha256": SHA, "materialization_sha256": SHA, "runtime_instance_id": SHA, "runtime_source_sha256": SHA, "stage_receipts": receipts, **{stage + "_sha256": hashlib.sha256(_bytes(receipt)).hexdigest() for stage, receipt in zip(stages, receipts, strict=True)}}
    parse_canonical_model(_bytes(manifest), contracts.SmokeManifest, maximum=1024 * 1024)
    manifest["stage_receipts"][-1] = payload
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(manifest), contracts.SmokeManifest, maximum=1024 * 1024)


@pytest.mark.parametrize(("field", "value"), [("exit_code", None), ("deadline_seconds", 601), ("deadline_seconds", True), ("stdout_size", 2097153), ("finished_at", "2026-02-30T00:00:00Z"), ("outcome", "success")])
def test_command_evidence_is_exact_and_bounded(field: str, value: object) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _command()
    payload[field] = value
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)


@pytest.mark.parametrize("kind", ["resolved_argv", "version", "role", "executable", "launcher"])
def test_completed_command_requires_exact_native_role_aliases(kind: str) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _command()
    parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)
    if kind == "resolved_argv":
        payload["resolved_argv"][0] = "tools/unbound"
    elif kind == "version":
        payload["tool_bindings"][0]["version"] = "3.13.0"
    elif kind == "role":
        payload["tool_bindings"][0]["role"] = "python"
    elif kind == "executable":
        payload["tool_bindings"][0]["executable"]["path"] = "tools/other"
    else:
        payload["tool_bindings"][1]["launcher"] = None
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)


@pytest.mark.parametrize("outcome", ["launch_failed", "timeout", "output_limit", "memory_limit", "teardown_failed"])
@pytest.mark.parametrize("kind", ["raw_path", "resolved_tail", "version", "role", "executable", "launcher", "blob_size", "shared_hash", "shared_size", "duplicate_role"])
def test_failed_command_rejects_untrusted_populated_native_proof(outcome: str, kind: str) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _command()
    payload.update(outcome=outcome, exit_code=None)
    parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)
    if kind == "raw_path":
        payload["resolved_argv"][0] = "C:/synthetic-private/python.exe"
    elif kind == "resolved_tail":
        payload["resolved_argv"][-1] = "--untrusted"
    elif kind == "version":
        payload["tool_bindings"][0]["version"] = "3.13.0"
    elif kind == "role":
        payload["tool_bindings"][0]["role"] = "python"
    elif kind == "executable":
        payload["tool_bindings"][0]["executable"]["path"] = "tools/other"
    elif kind == "launcher":
        payload["tool_bindings"][1]["launcher"] = None
    elif kind == "blob_size":
        payload["tool_bindings"][1]["launcher"]["size"] = 256 * 1024 * 1024 + 1
    elif kind == "shared_hash":
        payload["tool_bindings"][1]["executable"]["sha256"] = "f" * 64
    elif kind == "shared_size":
        payload["tool_bindings"][1]["executable"]["size"] = 4
    else:
        payload["tool_bindings"].append(payload["tool_bindings"][1])
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)


def test_unavailable_failed_launch_keeps_empty_native_observations() -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _command()
    payload.update(outcome="launch_failed", exit_code=None, resolved_argv=[], tool_bindings=[])
    result = parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)
    assert result.resolved_argv == () and result.tool_bindings == ()
    assert result.exit_code is None
    payload["outcome"] = "completed"
    payload["exit_code"] = 0
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)


@pytest.mark.parametrize("role", ["python_bootstrap", "uv"])
def test_partial_failed_launch_still_validates_available_binding(role: str) -> None:
    contracts = _module("evaluation.smoke_contracts")
    payload = _command()
    payload.update(outcome="launch_failed", exit_code=None, resolved_argv=[])
    payload["tool_bindings"] = [item for item in payload["tool_bindings"] if item["role"] == role]
    result = parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)
    assert tuple(item.role for item in result.tool_bindings) == (role,)
    payload["tool_bindings"][0]["version"] = "unverified"
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), contracts.CommandEvidence, maximum=65536)


def test_failed_verification_accepts_only_attempted_prefix_and_truthful_nulls() -> None:
    release = _module("evaluation.release_verification")
    payload = {"schema_version": 1, "candidate_id": CANDIDATE, "git_commit": COMMIT, "freeze_sha256": SHA, "verifier_tool_sha256": SHA, "status": "failed", "commands": [], "smoke_manifest_sha256": None, "smoke_manifest": None, "secret_scan_passed": False, "candidate_clean_before": True, "candidate_clean_after": False, "candidate_snapshot_before_sha256": SHA, "candidate_snapshot_after_sha256": None, "materialization_sha256": SHA, "runtime_instance_id": SHA, "runtime_source_sha256": SHA, "runtime_snapshot_after_sha256": None, "cleanup_status": "failed"}
    manifest = parse_canonical_model(_bytes(payload), release.VerificationManifest, maximum=16 * 1024 * 1024)
    assert manifest.commands == ()
    payload["status"] = "passed"
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), release.VerificationManifest, maximum=16 * 1024 * 1024)
    payload["status"] = "failed"
    bad = _command()
    bad["name"] = "backend_import"
    payload["commands"] = [bad]
    with pytest.raises(ValueError):
        parse_canonical_model(_bytes(payload), release.VerificationManifest, maximum=16 * 1024 * 1024)


@pytest.fixture
def publication(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    candidate, commit = _make_candidate(tmp_path, extras={".gitignore": b"ignored/\n"})
    (candidate / "ignored").mkdir()
    (candidate / "ignored" / "public-fixture.txt").write_bytes(b"ignored fixture\n")
    files = fingerprint_files(candidate)
    aggregate = aggregate_fingerprints(files)
    manifest = {"schema_version": 1, "candidate_id": f"{aggregate[:16]}-{commit[:12]}", "git_commit": commit, "git_tree_clean": True, "candidate_control_sha256": SHA, "created_at": "2026-01-01T00:00:00Z", "runtime": {"python": "3.12.12"}, "schema_version_number": 1, "mode_configuration": {}, "files": [item.model_dump(mode="json") for item in files], "aggregate_sha256": aggregate}
    directory = tmp_path / "publication" / manifest["candidate_id"]
    directory.mkdir(parents=True)
    raw = _bytes(manifest)
    tool = {"schema_version": 1, "tool_name": "d36_candidate_freezer_and_trial_host", "git_commit": commit, "files": [files[0].model_dump(mode="json")], "aggregate_sha256": aggregate_fingerprints([files[0]])}
    tool_raw = _bytes(tool)
    (directory / "freeze-manifest.json").write_bytes(raw)
    (directory / "d36-tool-attestation.json").write_bytes(tool_raw)
    marker = {"schema_version": 1, "files": [{"path": name, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()} for name, blob in [("d36-tool-attestation.json", tool_raw), ("freeze-manifest.json", raw)]]}
    (directory / ".d36-publication-state").write_bytes(_bytes(marker))
    freeze.read_frozen_candidate(directory)
    work = tmp_path / "work"
    evidence = tmp_path / "evidence"
    work.mkdir()
    evidence.mkdir()
    return candidate, directory / "freeze-manifest.json", work, evidence / "runtime-materialization.json"


def _materialize(paths: tuple[Path, Path, Path, Path]) -> tuple[Any, Path, str]:
    module = _module("evaluation.runtime_materialization")
    candidate, frozen, work, output = paths
    result = module.materialize_candidate_runtime(candidate_root=candidate, freeze_manifest_path=frozen, work_root=work, output_path=output)
    return result, work / ("runtime-" + result.runtime_instance_id), hashlib.sha256(output.read_bytes()).hexdigest()


def _cleanup(paths: tuple[Path, Path, Path, Path], runtime: Path, digest: str) -> None:
    _module("evaluation.runtime_materialization").cleanup_candidate_runtime(runtime_root=runtime, work_root=paths[2], materialization_path=paths[3], expected_materialization_sha256=digest)


def test_materializes_only_exact_tracked_bytes_and_cleans_marker_bound_idempotently(publication: tuple[Path, Path, Path, Path]) -> None:
    candidate, frozen, work, output = publication
    before = freeze._snapshot_tree(candidate)
    result, runtime, digest = _materialize(publication)
    expected = json.loads(frozen.read_bytes())["files"]
    assert [item.model_dump(mode="json") for item in result.files] == expected
    assert result.runtime_source_sha256 == json.loads(frozen.read_bytes())["aggregate_sha256"]
    actual_files = sorted(path.relative_to(runtime).as_posix() for path in runtime.rglob("*") if path.is_file())
    assert actual_files == [item["path"] for item in expected]
    assert not (runtime / "ignored").exists()
    for item in expected:
        path = runtime / item["path"]
        assert path.read_bytes() == (candidate / item["path"]).read_bytes()
        assert not path.stat().st_mode & stat.S_IWRITE
    marker_path = output.with_name(output.name + ".ownership.json")
    marker_inode = marker_path.stat().st_ino
    assert marker_inode == result.marker_inode
    assert marker_path.stat().st_size <= 4096
    assert json.loads(marker_path.read_bytes())["state"] == "active"
    _cleanup(publication, runtime, digest)
    _cleanup(publication, runtime, digest)
    assert not runtime.exists()
    assert marker_path.stat().st_ino == marker_inode
    assert json.loads(marker_path.read_bytes())["state"] == "cleaned"
    assert json.loads(output.with_name(output.name + ".cleanup.json").read_bytes())["status"] == "completed"
    assert freeze._snapshot_tree(candidate) == before
    assert work.exists()


@pytest.mark.parametrize("kind", ["source_change", "output_candidate", "output_work", "incomplete_publication"])
def test_materialization_rejects_unsafe_inputs_without_creating_runtime(publication: tuple[Path, Path, Path, Path], kind: str) -> None:
    module = _module("evaluation.runtime_materialization")
    candidate, frozen, work, output = publication
    if kind == "source_change":
        (candidate / "backend/pyproject.toml").write_bytes(b"changed committed source\n")
    elif kind == "output_candidate":
        output = candidate / "materialization.json"
    elif kind == "output_work":
        output = work / "runtime-predicted" / "materialization.json"
        output.parent.mkdir()
    else:
        (frozen.parent / ".d36-publication-state").unlink()
    with pytest.raises((ValueError, OSError)):
        module.materialize_candidate_runtime(candidate_root=candidate, freeze_manifest_path=frozen, work_root=work, output_path=output)
    assert not list(work.glob("runtime-*")) if kind != "output_work" else not list(work.glob("runtime-" + "[0-9a-f]" * 64))
    assert not output.exists()


@pytest.mark.parametrize("kind", ["extra", "writable", "replaced_file", "replaced_directory", "replaced_root", "replaced_marker", "missing_root", "missing_marker"])
def test_cleanup_refuses_drift_and_replacements_without_deleting_them(publication: tuple[Path, Path, Path, Path], kind: str) -> None:
    result, runtime, digest = _materialize(publication)
    output = publication[3]
    target = runtime / result.files[0].path
    marker = output.with_name(output.name + ".ownership.json")
    if kind == "extra":
        os.chmod(runtime, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        target = runtime / "extra"
        target.write_bytes(b"do not delete\n")
    elif kind == "writable":
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
    elif kind == "replaced_file":
        data = target.read_bytes()
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        target.rename(target.with_name(target.name + ".old"))
        target.write_bytes(data)
    elif kind == "replaced_directory":
        target = runtime / "backend"
        target.rename(runtime / "backend.old")
        target.mkdir()
    elif kind == "replaced_root":
        runtime.rename(runtime.with_name(runtime.name + ".old"))
        runtime.mkdir()
        target = runtime / "unrelated"
        target.write_bytes(b"replacement\n")
    elif kind == "replaced_marker":
        raw = marker.read_bytes()
        marker.rename(marker.with_name(marker.name + ".old"))
        marker.write_bytes(raw)
        target = marker
    elif kind == "missing_root":
        runtime.rename(runtime.with_name(runtime.name + ".old"))
        target = runtime.with_name(runtime.name + ".old")
    else:
        marker.unlink()
    identity = target.stat().st_ino
    mode = target.stat().st_mode
    with pytest.raises((ValueError, OSError)):
        _cleanup(publication, runtime, digest)
    assert target.stat().st_ino == identity
    assert target.stat().st_mode == mode


def test_cleanup_refuses_hardlink_alias_without_chmod_or_unlink(publication: tuple[Path, Path, Path, Path]) -> None:
    result, runtime, digest = _materialize(publication)
    path = runtime / result.files[0].path
    alias = publication[3].parent / "external-alias"
    os.link(path, alias)
    before = alias.stat()
    with pytest.raises(ValueError):
        _cleanup(publication, runtime, digest)
    assert path.exists() and alias.exists()
    assert alias.stat().st_mode == before.st_mode
    assert alias.stat().st_ino == before.st_ino


def test_wrong_detached_hash_precedes_parsing(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime, digest = _materialize(publication)
    module = _module("evaluation.runtime_materialization")
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("untrusted model parsed before detached hash")
    monkeypatch.setattr(module, "parse_canonical_model", forbidden)
    with pytest.raises(ValueError, match="digest"):
        _cleanup(publication, runtime, "0" * 64)
    assert runtime.exists() and digest != "0" * 64


@pytest.mark.parametrize("marker", [False, True])
def test_metadata_cap_plus_one_fails_closed(publication: tuple[Path, Path, Path, Path], marker: bool) -> None:
    _, runtime, digest = _materialize(publication)
    path = publication[3]
    if marker:
        path = path.with_name(path.name + ".ownership.json")
    with path.open("r+b") as stream:
        stream.truncate((4096 if marker else 16 * 1024 * 1024) + 1)
    with pytest.raises((ValueError, OSError)):
        _cleanup(publication, runtime, digest)
    assert runtime.exists()


def test_cleanup_interruption_resumes_only_recorded_remaining_subset(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime, digest = _materialize(publication)
    output = publication[3]
    module = _module('evaluation.runtime_materialization')
    original = module._delete_owned_path
    count = 0
    def interrupted(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal count
        if str(path).endswith(".json") and ".cleanup." in str(path):
            return original(path, *args, **kwargs)
        count += 1
        if count == 2:
            raise OSError("synthetic cleanup interruption")
        original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_delete_owned_path", interrupted)
        with pytest.raises((ValueError, OSError)):
            _cleanup(publication, runtime, digest)
    assert json.loads(output.with_name(output.name + ".ownership.json").read_bytes())["state"] == "cleaning"
    assert json.loads(output.with_name(output.name + ".cleanup.json").read_bytes())["status"] == "failed"
    _cleanup(publication, runtime, digest)
    assert not runtime.exists()


@pytest.mark.parametrize("replace_remaining", [False, True])
def test_published_failure_cleanup_resumes_without_deleting_replacements(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch, replace_remaining: bool) -> None:
    module = _module("evaluation.runtime_materialization")
    candidate, frozen, work, output = publication
    original_update = module._marker_update
    original_unlink = module._delete_owned_path
    deleted: list[Path] = []
    runtime: Path | None = None
    attempts = 0

    def change_after_activation(*args: Any, **kwargs: Any) -> Any:
        nonlocal runtime
        marker = original_update(*args, **kwargs)
        if kwargs.get("state") == "active":
            runtime = work / ("runtime-" + marker.runtime_instance_id)
            (candidate / "backend/pyproject.toml").write_bytes(b"synthetic post-activation source drift\n")
        return marker

    def interrupted_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        if runtime is not None and (kwargs.get("dir_fd") is not None or runtime in Path(path).parents):
            attempts += 1
            if attempts == 2:
                raise OSError("synthetic published cleanup interruption")
            deleted.append(Path(path))
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_marker_update", change_after_activation)
        patch.setattr(module, "_delete_owned_path", interrupted_unlink)
        with pytest.raises(ValueError, match="cleanup_failed"):
            module.materialize_candidate_runtime(candidate_root=candidate, freeze_manifest_path=frozen, work_root=work, output_path=output)
    assert runtime is not None and runtime.exists() and len(deleted) == 1 and attempts == 2
    record = json.loads(output.read_bytes())
    assert sum((runtime / item["path"]).exists() for item in record["files"]) == len(record["files"]) - 1
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    marker_path = output.with_name(output.name + ".ownership.json")
    assert json.loads(marker_path.read_bytes())["state"] == "cleaning"
    assert json.loads(output.with_name(output.name + ".cleanup.json").read_bytes())["status"] == "failed"
    if replace_remaining:
        target = next(runtime / item["path"] for item in record["files"] if (runtime / item["path"]).exists())
        data = target.read_bytes()
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        target.rename(output.parent / "retained-old-source")
        target.write_bytes(data)
        os.chmod(target, stat.S_IREAD)
        before = target.stat()
        with pytest.raises(ValueError):
            _cleanup(publication, runtime, digest)
        assert target.read_bytes() == data
        assert target.stat().st_ino == before.st_ino and target.stat().st_mode == before.st_mode
        assert json.loads(marker_path.read_bytes())["state"] == "cleaning"
    else:
        _cleanup(publication, runtime, digest)
        _cleanup(publication, runtime, digest)
        assert not runtime.exists()
        assert json.loads(marker_path.read_bytes())["state"] == "cleaned"
        assert json.loads(output.with_name(output.name + ".cleanup.json").read_bytes())["status"] == "completed"


@pytest.mark.skipif(os.name != "posix", reason="POSIX special-file/symlink behavior; Windows reparse test separate")
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_materializer_rejects_special_ignored_files(publication: tuple[Path, Path, Path, Path], kind: str) -> None:
    module = _module("evaluation.runtime_materialization")
    candidate, frozen, work, output = publication
    path = candidate / "ignored" / "unsafe"
    if kind == "symlink":
        path.symlink_to(candidate / "backend")
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError):
        module.materialize_candidate_runtime(candidate_root=candidate, freeze_manifest_path=frozen, work_root=work, output_path=output)
    assert not list(work.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="native Windows junction")
def test_materializer_rejects_junction(publication: tuple[Path, Path, Path, Path]) -> None:
    module = _module("evaluation.runtime_materialization")
    candidate, frozen, work, output = publication
    junction = candidate / "ignored" / "junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(candidate / "backend")], check=True, stdout=subprocess.DEVNULL)
    try:
        with pytest.raises(ValueError):
            module.materialize_candidate_runtime(candidate_root=candidate, freeze_manifest_path=frozen, work_root=work, output_path=output)
        assert not list(work.iterdir())
    finally:
        junction.rmdir()


def test_materialization_cli_redacts_argument_errors(capsys: pytest.CaptureFixture[str]) -> None:
    cli = _module("evaluation.scripts.materialize_candidate_runtime")
    assert cli.main(["cleanup", "--unexpected", "private-synthetic-path"]) == 2
    captured = capsys.readouterr()
    assert "private-synthetic-path" not in captured.out + captured.err
    assert "Traceback" not in captured.out + captured.err


def test_materialization_cli_runs_and_cleanup_is_detached_bound(publication: tuple[Path, Path, Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    cli = _module("evaluation.scripts.materialize_candidate_runtime")
    candidate, frozen, work, output = publication
    assert cli.main(["--candidate-root", str(candidate), "--freeze-manifest", str(frozen), "--work-root", str(work), "--output", str(output)]) == 0
    record = json.loads(output.read_bytes())
    runtime = work / ("runtime-" + record["runtime_instance_id"])
    args = ["cleanup", "--runtime-root", str(runtime), "--work-root", str(work), "--materialization", str(output), "--expected-materialization-sha256"]
    assert cli.main([*args, "0" * 64]) == 2
    assert runtime.exists()
    assert cli.main([*args, hashlib.sha256(output.read_bytes()).hexdigest()]) == 0
    assert not runtime.exists()
    text = capsys.readouterr().out
    assert str(candidate) not in text and str(runtime) not in text


def test_interruption_after_root_removal_resumes_cleaning_empty_subset(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    _, runtime, digest = _materialize(publication)
    module = _module("evaluation.runtime_materialization")
    original = module._marker_update
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("state") == "cleaned":
            raise OSError("synthetic publication interruption")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_marker_update", interrupt)
        with pytest.raises(OSError):
            _cleanup(publication, runtime, digest)
    assert not runtime.exists()
    _cleanup(publication, runtime, digest)
    assert json.loads(publication[3].with_name(publication[3].name + ".ownership.json").read_bytes())["state"] == "cleaned"


def test_marker_creation_failure_removes_proven_empty_runtime(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("evaluation.runtime_materialization")
    original = module._file_descriptor
    def fail(path: Path, **kwargs: Any) -> int:
        if kwargs.get("create") and path.name.endswith(".ownership.json"):
            raise OSError("synthetic marker creation refusal")
        return original(path, **kwargs)
    monkeypatch.setattr(module, "_file_descriptor", fail)
    with pytest.raises(OSError):
        module.materialize_candidate_runtime(candidate_root=publication[0], freeze_manifest_path=publication[1], work_root=publication[2], output_path=publication[3])
    assert not list(publication[2].iterdir())
    assert json.loads(publication[3].with_name(publication[3].name + ".cleanup.json").read_bytes())["status"] == "completed"


def test_build_interruption_removes_only_retained_partial_tree(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module("evaluation.runtime_materialization")
    original = module._OwnedTree.copy
    roots: list[Path] = []
    def change(tree: Any, candidate: Path, expected: Any) -> None:
        original(tree, candidate, expected)
        roots.append(tree.root)
        raise ValueError("synthetic build interruption")
    with monkeypatch.context() as patch:
        patch.setattr(module._OwnedTree, "copy", change)
        with pytest.raises(ValueError, match="interruption"):
            module.materialize_candidate_runtime(candidate_root=publication[0], freeze_manifest_path=publication[1], work_root=publication[2], output_path=publication[3])
    assert roots and not roots[0].exists()
    assert json.loads(publication[3].with_name(publication[3].name + ".cleanup.json").read_bytes())["status"] == "completed"


@pytest.mark.parametrize("kind", ["extra", "write", "content"])
def test_active_runtime_reverification_rejects_source_or_permission_drift(publication: tuple[Path, Path, Path, Path], kind: str) -> None:
    module = _module("evaluation.runtime_materialization")
    result, runtime, digest = _materialize(publication)
    arguments = {"runtime_root": runtime, "work_root": publication[2], "materialization_path": publication[3], "expected_materialization_sha256": digest}
    assert module.read_materialized_runtime(**arguments) == result
    if kind == "extra":
        os.chmod(runtime, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        (runtime / "extra").write_bytes(b"extra\n")
    else:
        path = runtime / result.files[0].path
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        if kind == "content":
            path.write_bytes(b"content drift\n")
            os.chmod(path, stat.S_IREAD)
    with pytest.raises(ValueError):
        module.read_materialized_runtime(**arguments)


def _run_owned(tmp_path: Path, code: str, *, deadline: int = 5) -> Any:
    runtime = _module("evaluation.blinded_runtime")
    assert hasattr(runtime, "run_owned_command"), "missing gated owned-command API"
    return asyncio.run(runtime.run_owned_command(argv=(sys.executable, "-B", "-c", code), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "stdout", stderr_path=tmp_path / "stderr", deadline_seconds=deadline))


def test_scope_reuses_group_for_commands_without_stopping_server(tmp_path: Path) -> None:
    runtime = _module("evaluation.blinded_runtime")
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            server = await runtime.start_owned_process(scope=scope, argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "server.out", stderr_path=tmp_path / "server.err", deadline_seconds=60)
            for index in range(2):
                result = await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "print('command')"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / f"out{index}", stderr_path=tmp_path / f"err{index}", deadline_seconds=5)
                assert result.outcome == "completed" and result.exit_code == 0
                assert server.outcome is None
        assert scope.teardown_confirmed
    asyncio.run(exercise())


def test_failed_command_prevents_further_group_execution(tmp_path: Path) -> None:
    runtime = _module("evaluation.blinded_runtime")
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            result = await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "raise SystemExit(7)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5)
            assert result.exit_code == 7
            with pytest.raises(ValueError, match="failed"):
                await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "open('payload-ran','w').write('unsafe')"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out2", stderr_path=tmp_path / "err2", deadline_seconds=5)
        assert not (tmp_path / "payload-ran").exists()
    asyncio.run(exercise())


def test_owned_command_records_real_bounded_outputs_and_nonzero_exit(tmp_path: Path) -> None:
    result = _run_owned(tmp_path, "import sys; print('ok'); sys.stderr.write('err'); sys.exit(7)")
    assert result.outcome == "completed" and result.exit_code == 7
    stdout = (tmp_path / "stdout").read_bytes()
    assert result.stdout_size == len(stdout)
    assert result.stdout_sha256 == hashlib.sha256(stdout).hexdigest()
    assert result.stderr_size == 3 and result.stderr_sha256 == hashlib.sha256(b"err").hexdigest()
    assert result.started_at.endswith("Z") and result.finished_at.endswith("Z")


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job accounting loss")
def test_unconfirmed_teardown_returns_failed_actual_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _module("evaluation.blinded_runtime")
    original = runtime._terminate_job_confirmed
    async def lost(handle: int, *, deadline: float | None = None) -> None:
        await original(handle, deadline=deadline)
        raise ValueError("synthetic terminal accounting loss")
    monkeypatch.setattr(runtime, "_terminate_job_confirmed", lost)
    result = _run_owned(tmp_path, "print('actual')")
    assert result.outcome == "teardown_failed"
    assert result.stdout_size == len((tmp_path / "stdout").read_bytes())
    assert result.stdout_sha256 == hashlib.sha256((tmp_path / "stdout").read_bytes()).hexdigest()


@pytest.mark.parametrize("kind", ["timeout", "output_limit", "parent_exit"])
def test_native_owned_command_confirms_descendants_and_readers(tmp_path: Path, kind: str) -> None:
    child = "import time; time.sleep(60)"
    spawn = f"import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',{child!r}]); open('child.pid','w').write(str(p.pid)); "
    code = spawn + ("time.sleep(60)" if kind == "timeout" else "sys.stdout.buffer.write(b'x'*2200000); sys.stdout.flush(); time.sleep(60)" if kind == "output_limit" else "sys.exit(0)")
    started = time.monotonic()
    result = _run_owned(tmp_path, code, deadline=1 if kind == "timeout" else 5)
    assert result.outcome == ("completed" if kind == "parent_exit" else kind)
    assert time.monotonic() - started < 12.5
    assert result.stdout_size + result.stderr_size <= 2 * 1024 * 1024
    pid = int((tmp_path / "child.pid").read_text())
    if os.name == "nt":
        import ctypes
        handle = ctypes.WinDLL("kernel32", use_last_error=True).OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.WinDLL("kernel32").CloseHandle(handle)
        assert not handle
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_scope_shares_ownership_with_long_lived_children_and_cancellation(tmp_path: Path) -> None:
    runtime = _module("evaluation.blinded_runtime")
    assert hasattr(runtime, "OwnedProcessScope"), "missing reusable owned scope"
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            server = await runtime.start_owned_process(scope=scope, argv=(sys.executable, "-c", "import time; print('server',flush=True); time.sleep(60)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "server.out", stderr_path=tmp_path / "server.err", deadline_seconds=60)
            command = asyncio.create_task(runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "cmd.out", stderr_path=tmp_path / "cmd.err", deadline_seconds=60))
            await asyncio.sleep(0.3)
            assert server.pid > 0
            command.cancel()
            with pytest.raises(asyncio.CancelledError):
                await command
        assert scope.teardown_confirmed is True
        assert server.outcome is not None
    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "nt", reason="Windows Job assignment failure")
def test_job_assignment_failure_never_releases_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _module("evaluation.blinded_runtime")
    assert hasattr(runtime, "OwnedProcessScope"), "missing reusable owned scope"
    def fail(*args: object) -> None:
        raise OSError("synthetic assignment failure")
    monkeypatch.setattr(runtime.OwnedProcessScope, "_assign_job", fail)
    result = _run_owned(tmp_path, "open('payload-ran','w').write('unsafe')")
    assert result.outcome == "launch_failed"
    assert not (tmp_path / "payload-ran").exists()


@pytest.mark.skipif(os.name != "nt", reason="native Windows pre-assignment startup hooks")
@pytest.mark.parametrize("hook", ["sitecustomize", "subprocess"])
@pytest.mark.parametrize("refuse_assignment", [False, True])
def test_isolated_supervisor_blocks_startup_hook_descendants_before_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook: str, refuse_assignment: bool) -> None:
    runtime = _module("evaluation.blinded_runtime")
    base_python = str(getattr(sys, "_base_executable", sys.executable))
    marker = tmp_path / "startup-descendant-ran"
    payload = f"from pathlib import Path; Path({str(marker)!r}).write_text('synthetic hook descendant')"
    (tmp_path / "hook-descendant.py").write_text(payload)
    hook_code = f"import os\nos.spawnv(os.P_WAIT, {base_python!r}, ('python', '-I', '-S', '-B', 'hook-descendant.py'))\n"
    hook_root = tmp_path / "bootstrap-dependencies"
    hook_root.mkdir()
    (hook_root / (hook + ".py")).write_text(hook_code)
    if hook == "subprocess":
        (tmp_path / "subprocess.py").write_text(hook_code)
    original_launch = asyncio.create_subprocess_exec
    original_assign = runtime.OwnedProcessScope._assign_job
    checked: list[int] = []

    async def observe_startup(*args: Any, **kwargs: Any) -> Any:
        process = await original_launch(*args, **kwargs)
        until = time.monotonic() + 1
        while not marker.exists() and process.returncode is None and time.monotonic() < until:
            await asyncio.sleep(0.01)
        return process

    def assign(scope: Any, process: Any, child: Any) -> None:
        checked.append(process.pid)
        if refuse_assignment:
            raise OSError("synthetic assignment refusal")
        original_assign(scope, process, child)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_startup)
    monkeypatch.setattr(runtime.OwnedProcessScope, "_assign_job", assign)
    environment = runtime.clean_subprocess_environment(Path(__file__).parents[2])
    environment["PYTHONPATH"] = str(hook_root)

    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            result = await runtime.run_owned_command(scope=scope, argv=(base_python, "-I", "-S", "-B", "-c", "open('target-ran','w').write('assigned target')"), cwd=tmp_path, env=environment, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5)
            assert result.outcome == ("launch_failed" if refuse_assignment else "completed")
            assert (tmp_path / "target-ran").exists() is not refuse_assignment
        assert scope.teardown_confirmed and checked
        assert not marker.exists()
    asyncio.run(exercise())


def test_isolated_supervisor_preserves_assigned_target_environment_and_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _module("evaluation.blinded_runtime")
    dependency_root = tmp_path / "bootstrap-dependencies"
    dependency_root.mkdir()
    assigned = tmp_path / "assigned"
    target_hook = tmp_path / "target-hook"
    (dependency_root / "sitecustomize.py").write_text(f"from pathlib import Path\nPath({str(target_hook)!r}).write_text('assigned' if Path({str(assigned)!r}).exists() else 'premature')\n")
    (dependency_root / "bootstrap_dependency.py").write_text("value = 'dependency-loaded'\n")
    original_assign = runtime.OwnedProcessScope._assign_job

    def assign(scope: Any, process: Any, child: Any) -> None:
        original_assign(scope, process, child)
        assert not target_hook.exists()
        assigned.write_text("owned")

    monkeypatch.setattr(runtime.OwnedProcessScope, "_assign_job", assign)
    environment = runtime.clean_subprocess_environment(Path(__file__).parents[2])
    environment.update(PYTHONPATH=str(dependency_root), D39_BOOTSTRAP="original-environment")
    code = "import bootstrap_dependency,os; print(bootstrap_dependency.value); print(os.environ['D39_BOOTSTRAP']); print(os.getcwd())"
    result = asyncio.run(runtime.run_owned_command(argv=(sys.executable, "-B", "-c", code), cwd=tmp_path, env=environment, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5))
    assert result.outcome == "completed" and result.exit_code == 0
    assert (tmp_path / "out").read_text().splitlines() == ["dependency-loaded", "original-environment", str(tmp_path)]
    assert target_hook.read_text() == "assigned"


@pytest.mark.parametrize("reason", ["pressure", "accounting"])
def test_scope_stops_owned_work_on_memory_or_accounting_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    runtime = _module("evaluation.blinded_runtime")
    original = runtime._owned_memory_sample
    armed = False
    def sample(scope: Any = None) -> tuple[int, int]:
        if armed:
            if reason == "accounting":
                raise ValueError("synthetic accounting loss")
            return 0, runtime.OWNED_MEMORY_START_LIMIT_BYTES
        return original(scope)
    monkeypatch.setattr(runtime, "_owned_memory_sample", sample)
    async def exercise() -> None:
        nonlocal armed
        async with runtime.OwnedProcessScope() as scope:
            child = await runtime.start_owned_process(scope=scope, argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=60)
            armed = True
            result = await child.wait()
            assert result.outcome == "memory_limit"
        assert scope.teardown_confirmed
    asyncio.run(exercise())


def _pipe_holding_parent(tmp_path: Path, *, flood: bool = False) -> str:
    output = "sys.stdout.buffer.write(b'x' * 2200000); sys.stdout.flush()" if flood else "print('late', flush=True)"
    (tmp_path / "pipe-child.py").write_text("import pathlib,sys,time\nwhile not pathlib.Path('release-descendant').exists(): time.sleep(0.01)\n" + output + "\ntime.sleep(60)\n")
    base_python = str(getattr(sys, "_base_executable", sys.executable))
    return f"import subprocess; p=subprocess.Popen([{base_python!r},'-I','-S','-B','pipe-child.py']); open('child.pid','w').write(str(p.pid))"


@pytest.mark.parametrize("reason", ["memory_limit", "timeout"])
def test_parent_drain_latches_late_failures_and_refuses_next_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    runtime = _module("evaluation.blinded_runtime")
    original_wait = asyncio.wait
    original_read = runtime.OwnedProcess._read
    injected: list[int] = []
    parent_code = _pipe_holding_parent(tmp_path)

    async def drain(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timeout") == runtime.HOST_PIPE_DRAIN_GRACE_SECONDS:
            (tmp_path / "release-descendant").write_text("parent exited")
        return await original_wait(*args, **kwargs)

    async def read(child: Any, stream: Any, descriptor: int, index: int) -> None:
        original_chunk = stream.read
        async def chunk(size: int = -1) -> bytes:
            value = await original_chunk(size)
            if index == 0 and value and not injected:
                assert child._process.returncode == 0
                injected.append(child.pid)
                if reason == "memory_limit":
                    child.scope._memory_lost.set()
                else:
                    child._stop_reason = "timeout"
            return value
        monkeypatch.setattr(stream, "read", chunk)
        await original_read(child, stream, descriptor, index)

    monkeypatch.setattr(asyncio, "wait", drain)
    monkeypatch.setattr(runtime.OwnedProcess, "_read", read)
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            result = await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-B", "-c", parent_code), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5)
            assert injected and result.exit_code == 0
            assert result.outcome == reason
            assert (tmp_path / "out").read_text() == "late\n"
            with pytest.raises(ValueError, match="failed"):
                await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "open('next-command','w').write('unsafe')"), cwd=tmp_path, env={}, stdout_path=tmp_path / "next.out", stderr_path=tmp_path / "next.err", deadline_seconds=5)
        assert scope.teardown_confirmed
        assert not (tmp_path / "next-command").exists()
    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "nt", reason="native Windows Job final-reader settlement seam")
@pytest.mark.parametrize("reason", ["output_limit", "memory_limit", "memory_over_output", "timeout", "teardown_failed"])
def test_final_readers_latch_failures_after_real_descendant_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    runtime = _module("evaluation.blinded_runtime")
    original_wait = asyncio.wait
    original_read = runtime.OwnedProcess._read
    original_terminate = runtime._terminate_job_confirmed
    tree_settled = asyncio.Event()
    observed: list[int] = []
    flood = reason in {"output_limit", "memory_over_output"}
    parent_code = _pipe_holding_parent(tmp_path, flood=flood)

    async def drain(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timeout") == runtime.HOST_PIPE_DRAIN_GRACE_SECONDS:
            (tmp_path / "release-descendant").write_text("parent exited")
        return await original_wait(*args, **kwargs)

    async def terminate(handle: int, *, deadline: float | None = None) -> None:
        await original_terminate(handle, deadline=deadline)
        tree_settled.set()

    async def read(child: Any, stream: Any, descriptor: int, index: int) -> None:
        if index == 0 and flood:
            original_chunk = stream.read
            count = 0
            async def chunk(size: int = -1) -> bytes:
                nonlocal count
                value = await original_chunk(size)
                count += len(value)
                if count > 2 * 1024 * 1024 and not tree_settled.is_set():
                    assert child._process.returncode == 0
                    await tree_settled.wait()
                return value
            monkeypatch.setattr(stream, "read", chunk)
        await original_read(child, stream, descriptor, index)
        if index == 0:
            assert tree_settled.is_set() and child._process.returncode == 0
            observed.append(child.pid)
            if reason in {"memory_limit", "memory_over_output", "teardown_failed"}:
                child.scope._memory_lost.set()
            if reason == "timeout":
                child._stop_reason = "timeout"
            if reason == "teardown_failed":
                raise ValueError("synthetic final reader failure after real drain")

    monkeypatch.setattr(asyncio, "wait", drain)
    monkeypatch.setattr(runtime.OwnedProcess, "_read", read)
    monkeypatch.setattr(runtime, "_terminate_job_confirmed", terminate)
    async def exercise() -> None:
        scope = runtime.OwnedProcessScope()
        await scope.__aenter__()
        try:
            result = await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-B", "-c", parent_code), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5)
            assert observed and result.exit_code == 0
            assert result.outcome == ("memory_limit" if reason == "memory_over_output" else reason)
            raw = (tmp_path / "out").read_bytes()
            assert result.stdout_size == len(raw) == (2 * 1024 * 1024 if flood else len(b"late\r\n"))
            assert result.stdout_sha256 == hashlib.sha256(raw).hexdigest()
            with pytest.raises(ValueError, match="failed"):
                await runtime.run_owned_command(scope=scope, argv=(sys.executable, "-c", "open('next-command','w').write('unsafe')"), cwd=tmp_path, env={}, stdout_path=tmp_path / "next.out", stderr_path=tmp_path / "next.err", deadline_seconds=5)
        finally:
            if reason == "teardown_failed":
                with pytest.raises(ValueError, match="teardown_failed"):
                    await scope.close()
            else:
                await scope.close()
                assert scope.teardown_confirmed
        assert not (tmp_path / "next-command").exists()
    asyncio.run(exercise())


@pytest.mark.parametrize("reason", ["memory_limit", "output_limit", "timeout"])
def test_explicit_stop_cannot_clear_an_already_latched_failure(tmp_path: Path, reason: str) -> None:
    runtime = _module("evaluation.blinded_runtime")
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            child = await runtime.start_owned_process(scope=scope, argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path, env=runtime.clean_subprocess_environment(Path(__file__).parents[2]), stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", deadline_seconds=5)
            child._stop_reason = reason
            result = await child.stop()
            assert result.outcome == reason
            assert scope._failed
        assert scope.teardown_confirmed
    asyncio.run(exercise())


def test_scope_refuses_unsafe_memory_headroom_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _module("evaluation.blinded_runtime")
    assert hasattr(runtime, "OwnedProcessScope"), "missing memory-gated scope"
    monkeypatch.setattr(runtime, "_owned_memory_sample", lambda *args: (0, runtime.OWNED_MEMORY_START_LIMIT_BYTES))
    with pytest.raises(ValueError, match="memory"):
        _run_owned(tmp_path, "open('payload-ran','w').write('unsafe')")
    assert not (tmp_path / "payload-ran").exists()
