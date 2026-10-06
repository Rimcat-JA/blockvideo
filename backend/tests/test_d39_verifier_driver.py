from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from evaluation import blinded_runtime, release_verification as release
from evaluation.evidence_json import parse_canonical_model
from evaluation.tool_attestation import canonical_json_bytes
from tests.test_d39_release_verification import _bytes, _fingerprint, _materialize, _receipt
from tests.test_d39_release_verification import publication as publication


PLANNED_PATHS = (
    "backend/evaluation/__init__.py", "backend/evaluation/blinded_io.py",
    "backend/evaluation/blinded_runtime.py", "backend/evaluation/browser_smoke.py",
    "backend/evaluation/evidence_json.py", "backend/evaluation/release_candidate/__init__.py",
    "backend/evaluation/release_candidate/contracts.py", "backend/evaluation/release_candidate/fingerprints.py",
    "backend/evaluation/release_candidate/freeze.py", "backend/evaluation/release_verification.py",
    "backend/evaluation/runtime_materialization.py", "backend/evaluation/scripts/d39_candidate_smoke.py",
    "backend/evaluation/scripts/d39_smoke.py", "backend/evaluation/scripts/materialize_candidate_runtime.py",
    "backend/evaluation/scripts/verify_release_candidate.py", "backend/evaluation/smoke_contracts.py",
    "backend/evaluation/tool_attestation.py", "backend/pyproject.toml", "backend/uv.lock",
)


def _api(name: str) -> Any:
    assert hasattr(release, name), f"missing Task 2 behavior: {name}"
    return getattr(release, name)


def _synthetic_smoke(record: Any, digest: str, *, tool_hash: str = "a" * 64) -> dict[str, object]:
    stages = ("legacy_migration", "restore", "all_tools_startup", "stateful_startup", "browser", "ffmpeg")
    binding = {key: getattr(record, key) for key in ("candidate_id", "git_commit", "freeze_sha256", "runtime_instance_id", "runtime_source_sha256")}
    binding["materialization_sha256"] = digest
    roles = {
        "python_bootstrap": ("3.12.12", "python_bootstrap", None), "uv": ("0.12.15", "python_bootstrap", "uv_module"),
        "python": ("3.12.12", "python_sandbox", None), "node": ("24.11.1", "node", None),
        "npx": ("11.6.2", "node", "npx_cli"), "pnpm": ("10.18.3", "node", "pnpm_cjs"),
        "chrome": ("139.0.0.0", "chrome", None), "websockets": ("16.1.1", "python_sandbox", "websockets_module"),
        "ffmpeg": ("7.1", "ffmpeg", None), "ffprobe": ("7.1", "ffprobe", None),
    }
    receipts = []
    for stage in stages:
        receipt = _receipt(stage)
        receipt.update(binding)
        selected = {"python_bootstrap", "uv", "python", "node", "npx", "pnpm"}
        selected.update({"chrome", "websockets"} if stage == "browser" else {"ffmpeg", "ffprobe"} if stage == "ffmpeg" else set())
        receipt["tools"] = [{"role": role, "version": roles[role][0], "executable": _fingerprint("tools/" + roles[role][1]), "launcher": None if roles[role][2] is None else _fingerprint("tools/" + roles[role][2])} for role in sorted(selected)]
        receipts.append(receipt)
    return {"schema_version": 1, "producer_tool_sha256": tool_hash, **binding, "stage_receipts": receipts, **{stage + "_sha256": hashlib.sha256(_bytes(receipt)).hexdigest() for stage, receipt in zip(stages, receipts, strict=True)}}


def _git(root: Path, *args: str) -> bytes:
    value = subprocess.run(("git", "-C", str(root), *args), capture_output=True, check=True)
    return value.stdout


@pytest.fixture
def tools_repo(tmp_path: Path) -> Path:
    root = tmp_path / "tools"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Synthetic fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")
    for name in PLANNED_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pass\n" if name.endswith(".py") else b"fixture\n")
    (root / ".gitignore").write_text("evidence/\n", encoding="ascii")
    (root / "evidence").mkdir()
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "Synthetic full source closure")
    return root


def test_source_attestation_never_substitutes_a_partial_closure(tools_repo: Path) -> None:
    attest = _api("_attest_verifier")
    assert _api("VERIFIER_SOURCE_PATHS") == PLANNED_PATHS
    value = attest(tools_repo)
    assert len(value.files) == 19 and value.tool_name == "d39_release_verifier"
    _git(tools_repo, "rm", "backend/evaluation/browser_smoke.py")
    _git(tools_repo, "commit", "-qm", "Missing planned Task 3 file")
    with pytest.raises((ValueError, OSError)):
        attest(tools_repo)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_source_attestation_rejects_hidden_index_changes(tools_repo: Path, flag: str) -> None:
    attest = _api("_attest_verifier")
    _git(tools_repo, "update-index", flag, PLANNED_PATHS[0])
    with pytest.raises(ValueError):
        attest(tools_repo)


def test_source_attestation_rejects_dirty_committed_bytes(tools_repo: Path) -> None:
    attest = _api("_attest_verifier")
    (tools_repo / PLANNED_PATHS[0]).write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        attest(tools_repo)


def test_actual_tool_import_closure_is_full_planned_inventory_not_app() -> None:
    paths = _api("VERIFIER_SOURCE_PATHS")
    root = Path(release.__file__).resolve().parents[2]
    missing = {name for name in paths if not (root / name).exists()}
    assert missing <= {"backend/evaluation/browser_smoke.py", "backend/evaluation/scripts/d39_smoke.py", "backend/evaluation/scripts/d39_candidate_smoke.py"}
    for name in paths:
        if not name.endswith(".py") or name in missing:
            continue
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [module, *(module + "." + alias.name for alias in node.names)]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for imported in names:
                assert not imported.startswith(("evaluation.corpus", "evaluation.blinded_runner", "evaluation.result_import"))
                if imported.startswith("app"):
                    assert name == "backend/evaluation/scripts/d39_candidate_smoke.py"
                if imported.startswith("evaluation."):
                    source = "backend/" + imported.replace(".", "/") + ".py"
                    package = "backend/" + imported.replace(".", "/") + "/__init__.py"
                    if (root / source).exists() or (root / package).exists():
                        assert source in paths or package in paths


def test_environment_discards_ambient_secrets_hooks_and_production(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build = _api("build_group_environment")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PYTHONPATH", "PYTHONHOME", "PYTEST_PLUGINS", "NODE_ENV", "NODE_OPTIONS", "NPM_CONFIG_USERCONFIG", "UV_INDEX_URL"):
        monkeypatch.setenv(name, "UNTRUSTED")
    env = build(tmp_path, python_executable=Path(sys._base_executable), node_executable=None)
    assert all("UNTRUSTED" not in value for value in env.values())
    assert env["NODE_OPTIONS"] == "--max-old-space-size=512"
    assert env["UV_CONCURRENT_DOWNLOADS"] == "2"
    assert env["UV_CONCURRENT_BUILDS"] == env["UV_CONCURRENT_INSTALLS"] == "1"
    assert env["UV_PYTHON"] == str(Path(sys._base_executable).absolute())
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "env")
    assert "PYTHONPATH" not in env and "PYTEST_PLUGINS" not in env and "NODE_ENV" not in env
    for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "TEMP", "TMP", "TMPDIR", "NPM_CONFIG_CACHE", "NPM_CONFIG_USERCONFIG", "NPM_CONFIG_GLOBALCONFIG", "RUFF_CACHE_DIR"):
        assert Path(env[key]).is_relative_to(tmp_path)
    import shlex
    assert shlex.split(env["PYTEST_ADDOPTS"]) == ["-o", "cache_dir=" + (tmp_path / "pytest-cache").as_posix()]


@pytest.mark.parametrize(("path", "blob", "rule"), [
    ("backend/.env.prod", b"safe", "tracked_private_state"),
    ("private/safe.txt", b"safe", "tracked_private_state"),
    ("frontend/node_modules/tool.js", b"safe", "tracked_generated_state"),
    ("backend/tests/secret.py", b"sk-" + b"a" * 25, "credential_token"),
    ("public.txt", b"ghp_" + b"a" * 30, "credential_token"),
    ("public.txt", b"ASIA" + b"A" * 16, "credential_token"),
    ("public.txt", b"-----BEGIN OPENSSH PRIVATE KEY-----", "private_key_block"),
])
def test_bounded_scan_reports_rule_counts_not_paths_values(tmp_path: Path, path: str, blob: bytes, rule: str) -> None:
    scan = _api("scan_public_files")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)
    counts = scan(tmp_path, (path,), ())
    assert counts[rule] == 1
    assert set(counts) == {"tracked_private_state", "tracked_generated_state", "credential_token", "private_key_block"}
    assert path not in repr(counts) and blob.decode() not in repr(counts)


def test_scan_overlap_counts_once_and_examples_are_not_test_exemptions(tmp_path: Path) -> None:
    scan = _api("scan_public_files")
    name = "test_fixture.txt"
    (tmp_path / name).write_bytes(b" " * 65530 + b"sk-" + b"a" * 25 + b" " * 70000)
    assert scan(tmp_path, (name,), ())["credential_token"] == 1
    (tmp_path / name).write_bytes(b"sk-" + b"x" * 32 + b"\nAKIAIOSFODNN7EXAMPLE\n")
    assert not any(scan(tmp_path, (name,), ()).values())
    assert scan(tmp_path, (name,), (b"sk-" + b"b" * 30,))["credential_token"] == 1


def test_exact_public_env_example_and_synthetic_path_exception(tmp_path: Path) -> None:
    scan = _api("scan_public_files")
    names = ("backend/.env.example", "backend/tests/fixtures/blinded/synthetic-human-review.json")
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"safe\n")
    assert not any(scan(tmp_path, names, ()).values())
    (tmp_path / names[1]).write_bytes(b"sk-" + b"a" * 25)
    assert scan(tmp_path, names, ())["credential_token"] == 1


def test_scan_refuses_oversize_or_link_without_reading_target(tmp_path: Path) -> None:
    scan = _api("scan_public_files")
    target = tmp_path / "large"
    with target.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        scan(tmp_path, ("large",), ())


def test_group_source_drift_is_detected_and_generated_state_is_removed(publication: tuple[Path, Path, Path, Path]) -> None:
    group_type = _api("_ExecutionGroup")
    materialized, runtime, digest = _materialize(publication)
    group = group_type(publication[2], runtime, materialized)
    root = group.root
    assert group.source_sha256() == materialized.runtime_source_sha256
    assert not (group.source / "ignored").exists()
    generated = root / "cache" / "nested" / "binary"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"generated")
    (group.source / materialized.files[0].path).write_bytes(b"source mutation")
    with pytest.raises(ValueError):
        group.assert_source()
    group.cleanup()
    assert not root.exists() and runtime.exists()
    from evaluation.runtime_materialization import cleanup_candidate_runtime
    cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_cleanup_inventory_lstat_matches_full_native_opened_identity(tmp_path: Path, kind: str) -> None:
    materialization = release.materialization
    target = tmp_path / "owned-child"
    if kind == "directory":
        target.mkdir()
        anchor = materialization._open_anchor(target)
        try:
            identity = materialization._anchor_identity(anchor)
            if os.name == "nt":
                assert anchor.handle is not None
                assert identity == materialization._native_identity(anchor.handle)
            assert materialization._identity(target.lstat()) == identity
            assert identity[1] > 0
        finally:
            release.freeze._close_directory_anchor(anchor)
    else:
        target.write_bytes(b"owned file")
        descriptor = materialization._file_descriptor(target)
        try:
            identity = materialization._identity(materialization._descriptor_stat(descriptor))
            if os.name == "nt":
                import msvcrt
                assert identity == materialization._native_identity(msvcrt.get_osfhandle(descriptor))
            assert materialization._identity(target.lstat()) == identity
            assert identity[1] > 0
        finally:
            os.close(descriptor)


def test_generated_child_anchor_replacement_is_not_adopted(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    materialization = release.materialization
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    child = group.root / "generated-child"
    moved = group.work / (group.root.name + "-original-child")
    child.mkdir()
    (child / "original").write_bytes(b"owned original")
    original_id = materialization._directory_identity(child)
    open_anchor = materialization._open_anchor
    replacement_id: tuple[int, int] | None = None

    def replace_before_anchor(path: Path) -> release.freeze._DirectoryAnchor:
        nonlocal replacement_id
        if path == child and replacement_id is None:
            assert materialization._identity(path.lstat()) == original_id
            path.rename(moved)
            path.mkdir()
            (path / "sentinel").write_bytes(b"replacement sentinel")
            replacement_id = materialization._identity(path.lstat())
            assert replacement_id != original_id
        return open_anchor(path)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(materialization, "_open_anchor", replace_before_anchor)
            with pytest.raises(ValueError, match="identity"):
                group.cleanup()
            assert replacement_id is not None
            assert (child / "sentinel").read_bytes() == b"replacement sentinel"
            assert (moved / "original").read_bytes() == b"owned original"
            group.assert_owned()
            assert group.anchor is not None and not group.anchor.closed
    finally:
        try:
            assert materialization._directory_identity(group.work) == group.work_anchor.identity
            for path, identity in ((child, replacement_id if replacement_id is not None else original_id), (moved, original_id)):
                if path.exists():
                    retained = open_anchor(path)
                    try:
                        assert retained.identity == identity
                        materialization._assert_directory(path, retained)
                        release._remove_group_directory(path, retained)
                    finally:
                        release.freeze._close_directory_anchor(retained)
            if group.root.exists():
                group.cleanup()
        finally:
            if group.anchor is not None:
                release.freeze._close_directory_anchor(group.anchor)
            release.freeze._close_directory_anchor(group.work_anchor)
            materialization.cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_generated_leaf_inventory_replacement_is_not_adopted(publication: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    materialization = release.materialization
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    child = group.root / "generated-leaf"
    moved = group.work / (group.root.name + "-original-leaf")
    child.write_bytes(b"owned original")
    original_id = materialization._identity(child.lstat())
    scandir = os.scandir
    replacement_id: tuple[int, int] | None = None

    class ReplacingInventory:
        def __enter__(self) -> ReplacingInventory:
            self.entries = scandir(group.root)
            return self

        def __exit__(self, *args: object) -> None:
            self.entries.close()

        def __iter__(self) -> ReplacingInventory:
            return self

        def __next__(self) -> os.DirEntry[str]:
            nonlocal replacement_id
            try:
                return next(self.entries)
            except StopIteration:
                if replacement_id is None:
                    assert materialization._identity(child.lstat()) == original_id
                    child.rename(moved)
                    child.write_bytes(b"replacement sentinel")
                    replacement_id = materialization._identity(child.lstat())
                    assert replacement_id != original_id
                raise

    def replace_after_inventory(path: Path) -> Any:
        return ReplacingInventory() if path in (group.root, materialization._fs_path(group.root)) and replacement_id is None else scandir(path)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", replace_after_inventory)
            with pytest.raises(ValueError, match="identity|changed"):
                group.cleanup()
            assert replacement_id is not None
            assert child.read_bytes() == b"replacement sentinel"
            assert moved.read_bytes() == b"owned original"
            group.assert_owned()
            assert group.anchor is not None and not group.anchor.closed
    finally:
        try:
            assert materialization._directory_identity(group.work) == group.work_anchor.identity
            for path, identity in ((child, replacement_id if replacement_id is not None else original_id), (moved, original_id)):
                if path.exists():
                    descriptor = materialization._file_descriptor(path)
                    try:
                        assert materialization._identity(materialization._descriptor_stat(descriptor)) == identity
                        assert materialization._identity(path.lstat()) == identity
                    finally:
                        os.close(descriptor)
                    assert materialization._identity(path.lstat()) == identity
                    path.unlink()
            if group.root.exists():
                group.cleanup()
        finally:
            if group.anchor is not None:
                release.freeze._close_directory_anchor(group.anchor)
            release.freeze._close_directory_anchor(group.work_anchor)
            materialization.cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_smoke_reader_rejects_arbitrary_hashes_and_wrong_runtime(publication: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    read = _api("_read_smoke")
    materialized, runtime, digest = _materialize(publication)
    path = tmp_path / "smoke.json"
    payload = _synthetic_smoke(materialized, digest)
    path.write_bytes(_bytes(payload))
    result, observed = read(path, materialized, digest)
    assert observed == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.runtime_instance_id == materialized.runtime_instance_id
    payload["runtime_instance_id"] = "f" * 64
    path.write_bytes(_bytes(payload))
    with pytest.raises(ValueError):
        read(path, materialized, digest)
    del payload["stage_receipts"]
    path.write_bytes(_bytes(payload))
    with pytest.raises(ValueError):
        read(path, materialized, digest)
    from evaluation.runtime_materialization import cleanup_candidate_runtime
    cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_atomic_pair_publication_preserves_previous_evidence(tmp_path: Path) -> None:
    publish = _api("_publish_verification")
    output = tmp_path / "run"
    publish(output, b"{}\n", b"{\"source\":1}\n")
    assert sorted(path.name for path in output.iterdir()) == ["d39-verifier-attestation.json", "verification-manifest.json"]
    assert (output / "verification-manifest.json").read_bytes() == b"{}\n"
    with pytest.raises(ValueError):
        publish(output, b"{\"replace\":true}\n", b"{}\n")
    assert (output / "verification-manifest.json").read_bytes() == b"{}\n"


def test_missing_closure_still_cleans_real_materialization(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verify = _api("verify_release_candidate")
    materialized, runtime, digest = _materialize(publication)
    _git(tools_repo, "rm", "backend/evaluation/browser_smoke.py")
    _git(tools_repo, "commit", "-qm", "Task 3 absent")
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    output = tools_repo / "evidence" / "new-run"
    with pytest.raises((ValueError, OSError)):
        verify(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=output, smoke_manifest_path=tools_repo / "evidence" / "smoke" / "absent-smoke.json")
    assert not runtime.exists() and not output.exists()
    receipt = json.loads(publication[3].with_name(publication[3].name + ".cleanup.json").read_bytes())
    assert receipt["status"] == "completed" and receipt["materialization_sha256"] == digest


def test_missing_smoke_returns_truthful_failed_nulls_and_cleanup(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verify = _api("verify_release_candidate")
    materialized, runtime, digest = _materialize(publication)
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    output = tools_repo / "evidence" / "new-run"
    result = verify(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=output, smoke_manifest_path=tools_repo / "evidence" / "smoke" / "absent-smoke.json")
    assert result.status == "failed" and result.commands == ()
    assert result.smoke_manifest is None and result.smoke_manifest_sha256 is None
    assert result.cleanup_status == "completed" and not runtime.exists()
    assert result.candidate_clean_before and result.candidate_clean_after
    assert result.runtime_snapshot_after_sha256 == materialized.runtime_source_sha256
    raw = (output / "verification-manifest.json").read_bytes()
    assert parse_canonical_model(raw, release.VerificationManifest, maximum=16 * 1024 * 1024) == result
    assert raw == canonical_json_bytes(result) + b"\n"


@pytest.mark.parametrize("suffix", [".cmd", ".bat", ".py"])
def test_native_resolver_rejects_scripts_before_execution(tmp_path: Path, suffix: str) -> None:
    native = _api("_native_file")
    path = tmp_path / ("pretend" + suffix)
    path.write_bytes(b"not a native executable")
    with pytest.raises(ValueError):
        native(path, "tools/node")


def test_node_argv_expansion_binds_real_native_and_js_bytes(tmp_path: Path) -> None:
    resolve = _api("_command_target")
    tool_type = _api("_ToolSet")
    native = _api("_native_file")
    fingerprint = _api("_tool_file")
    exe = Path(sys._base_executable)
    npx = tmp_path / "npx-cli.js"
    pnpm = tmp_path / "pnpm.cjs"
    npx.write_bytes(b"synthetic npx fixture")
    pnpm.write_bytes(b"synthetic pnpm fixture")
    executable = native(exe, "tools/node")
    bindings = tuple(release.ToolExecutionBinding(role=role, version=version, executable=executable, launcher=launcher) for role, version, launcher in (
        ("node", "24.11.1", None),
        ("npx", "11.6.2", fingerprint(npx, "tools/npx_cli")),
        ("pnpm", "10.18.3", fingerprint(pnpm, "tools/pnpm_cjs")),
    ))
    tools = tool_type(exe, npx, bindings, ((exe, executable), (npx, bindings[1].launcher), (pnpm, bindings[2].launcher)))
    argv, aliases = resolve(6, tools)
    assert argv == (str(exe), str(npx), "-y", "pnpm@10.18.3", "test", "--maxWorkers=1", "--minWorkers=1", "--no-file-parallelism")
    assert aliases == ("tools/node", "tools/npx_cli", "-y", "pnpm@10.18.3", "test", "--maxWorkers=1", "--minWorkers=1", "--no-file-parallelism")
    tools.verify()
    pnpm.write_bytes(b"changed pnpm fixture")
    with pytest.raises(ValueError):
        tools.verify()


def test_python_origin_check_rejects_global_optional_packages(tmp_path: Path) -> None:
    check = _api("_validate_python_probe")
    base = tmp_path / "base"
    sandbox = tmp_path / "env"
    base.mkdir()
    sandbox.mkdir()
    exe = sandbox / "python.exe"
    baseexe = base / "python.exe"
    data = {"version": "3.12.12", "executable": str(exe), "prefix": str(sandbox), "base_prefix": str(base), "base_executable": str(baseexe), "origins": {"numpy": str(base / "Lib/site-packages/numpy/__init__.py")}}
    with pytest.raises(ValueError):
        check(data, executable=exe, base_executable=baseexe, environment=sandbox, required=("numpy",))
    data["origins"]["numpy"] = str(sandbox / "Lib/site-packages/numpy/__init__.py")
    data.update(locations={'numpy': []}, versions={'numpy': '2.0.0'})
    check(data, executable=exe, base_executable=baseexe, environment=sandbox, required=("numpy",))
    data["base_executable"] = str(base / "other.exe")
    with pytest.raises(ValueError):
        check(data, executable=exe, base_executable=baseexe, environment=sandbox, required=("numpy",))


def test_real_probe_uses_native_gate_and_drains_bounded_results(publication: tuple[Path, Path, Path, Path]) -> None:
    probe = _api("_probe")
    group_type = _api("_ExecutionGroup")
    materialized, runtime, digest = _materialize(publication)
    group = group_type(publication[2], runtime, materialized)
    env = release.build_group_environment(group.root, python_executable=Path(sys._base_executable), node_executable=None)
    async def perform() -> None:
        async with blinded_runtime.OwnedProcessScope() as scope:
            observed = await probe(scope, group, (sys._base_executable, "-I", "-S", "-B", "-c", "print('native fixture')"), env)
            assert observed == b"native fixture\r\n" if os.name == "nt" else observed == b"native fixture\n"
        assert scope.teardown_confirmed
    try:
        asyncio.run(perform())
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_actual_failed_import_retains_evidence_before_stopping_source_drift(publication: tuple[Path, Path, Path, Path]) -> None:
    execute = _api("_execute_command")
    tool_type = _api("_ToolSet")
    native = _api("_native_file")
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    exe = Path(sys._base_executable)
    executable = native(exe, "tools/python_sandbox")
    binding = release.ToolExecutionBinding(role="python", version="3.12.12", executable=executable, launcher=None)
    tools = tool_type(exe, None, (binding,), ((exe, executable),))
    env = release.build_group_environment(group.root, python_executable=exe, node_executable=None)
    commands: list[release.CommandEvidence] = []
    async def perform() -> None:
        async with blinded_runtime.OwnedProcessScope() as scope:
            result = await execute(index=1, group=group, scope=scope, tools=tools, env=env, commands=commands)
            assert result.exit_code != 0 and result.outcome == "completed"
            assert result.argv == ("python", "-B", "-c", "import app.main")
            assert result.resolved_argv[0] == "tools/python_sandbox"
            assert result.stderr_size > 0 and len(commands) == 1
            changed = group.source / materialized.files[0].path
            changed.write_bytes(b"drift")
            with pytest.raises(ValueError):
                await execute(index=2, group=group, scope=scope, tools=tools, env=env, commands=commands)
            assert len(commands) == 1
    try:
        asyncio.run(perform())
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


@pytest.mark.parametrize("arguments", [[], ["--command", "PRIVATE-EXAMPLE"], ["--repo-root", "PRIVATE-EXAMPLE"], ["--candidate-root"]])
def test_verifier_cli_refuses_extensible_arguments_without_echo(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    from evaluation.scripts import verify_release_candidate as cli
    assert hasattr(cli, "main"), "missing Task 2 fixed verifier CLI"
    assert cli.main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == "verification rejected\n" and captured.err == ""


def test_generated_group_root_replacement_is_not_deleted(publication: tuple[Path, Path, Path, Path]) -> None:
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    from evaluation.runtime_materialization import _open_anchor, cleanup_candidate_runtime
    freeze = __import__("evaluation.release_candidate.freeze", fromlist=["freeze"])
    freeze._close_directory_anchor(group.anchor)
    moved = group.root.with_name(group.root.name + "-old")
    group.root.rename(moved)
    group.root.mkdir()
    replacement = group.root / "replacement"
    replacement.write_bytes(b"never delete a replacement")
    with pytest.raises(ValueError):
        group.cleanup()
    assert replacement.read_bytes() == b"never delete a replacement"
    replacement.unlink()
    group.root.rmdir()
    moved.rename(group.root)
    group.anchor = _open_anchor(group.root)
    group.cleanup()
    cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_atomic_pair_failure_before_rename_leaves_no_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.release_candidate import freeze
    def fail(*args: object) -> None:
        raise ValueError("injected pre-rename failure")
    name = "_windows_move_directory_no_replace" if os.name == "nt" else "_linux_rename_directory_no_replace"
    monkeypatch.setattr(freeze, name, fail)
    with pytest.raises(ValueError):
        release._publish_verification(tmp_path / "output", b"{}\n", b"{}\n")
    assert not list(tmp_path.iterdir())


def test_toolset_cannot_execute_a_path_other_than_its_bound_native(tmp_path: Path) -> None:
    exe = Path(sys._base_executable)
    fingerprint = release._native_file(exe, "tools/python_sandbox")
    wrong = tmp_path / "wrong.cmd"
    wrong.write_bytes(b"not native")
    binding = release.ToolExecutionBinding(role="python", version="3.12.12", executable=fingerprint, launcher=None)
    tools = release._ToolSet(wrong, None, (binding,), ((exe, fingerprint),))
    with pytest.raises(ValueError):
        tools.verify()


def test_origin_under_environment_but_outside_site_packages_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    env = tmp_path / "env"
    data = {"version": "3.12.12", "executable": str(env / "python.exe"), "prefix": str(env), "base_prefix": str(base), "base_executable": str(base / "python.exe"), "origins": {"numpy": str(env / "injected/numpy.py")}}
    with pytest.raises(ValueError):
        release._validate_python_probe(data, executable=env / "python.exe", base_executable=base / "python.exe", environment=env, required=("numpy",))


def test_pnpm_cache_requires_one_exact_version_under_group(publication: tuple[Path, Path, Path, Path]) -> None:
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    try:
        for slot, version in (("first", "10.18.3"), ("second", "12.0.0")):
            package = group.root / "npm-cache/_npx" / slot / "node_modules/pnpm/package.json"
            package.parent.mkdir(parents=True)
            package.write_bytes(_bytes({"name": "pnpm", "version": version}))
        package, launcher = release._pnpm_package(group)
        assert package == group.root / "npm-cache/_npx/first/node_modules/pnpm/package.json"
        assert launcher == package.parent / "bin/pnpm.cjs"
        wrong = group.root / "npm-cache/_npx/second/node_modules/pnpm/package.json"
        wrong.write_bytes(_bytes({"name": "pnpm", "version": "10.18.3"}))
        with pytest.raises(ValueError):
            release._pnpm_package(group)
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


@pytest.mark.parametrize("kind", ["node_version", "browser_tools", "shared_hash", "launcher"])
def test_smoke_reader_rejects_unverified_tool_proof(publication: tuple[Path, Path, Path, Path], tmp_path: Path, kind: str) -> None:
    materialized, runtime, digest = _materialize(publication)
    try:
        payload = _synthetic_smoke(materialized, digest)
        receipt = payload["stage_receipts"][4]
        if kind == "node_version":
            next(item for item in receipt["tools"] if item["role"] == "node")["version"] = "28.0.0"
        elif kind == "browser_tools":
            receipt["tools"] = [item for item in receipt["tools"] if item["role"] != "chrome"]
        elif kind == "shared_hash":
            next(item for item in receipt["tools"] if item["role"] == "npx")["executable"]["sha256"] = "f" * 64
        else:
            next(item for item in receipt["tools"] if item["role"] == "pnpm")["launcher"] = None
        payload["browser_sha256"] = hashlib.sha256(_bytes(receipt)).hexdigest()
        path = tmp_path / "smoke.json"
        path.write_bytes(_bytes(payload))
        with pytest.raises(ValueError):
            release._read_smoke(path, materialized, digest)
    finally:
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_wrong_tool_binding_is_rejected_before_target_instruction(publication: tuple[Path, Path, Path, Path]) -> None:
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    exe = Path(sys._base_executable)
    fingerprint = release._native_file(exe, "tools/python_sandbox")
    binding = release.ToolExecutionBinding.model_construct(role="python", version="3.12.11", executable=fingerprint, launcher=None)
    tools = release._ToolSet(exe, None, (binding,), ((exe, fingerprint),))
    env = release.build_group_environment(group.root, python_executable=exe, node_executable=None)
    main = group.source / "backend/app/main.py"
    main.write_text("from pathlib import Path\nPath('../../ran-target').write_text('unexpected')\n", encoding="ascii")
    commands: list[release.CommandEvidence] = []
    async def perform() -> None:
        async with blinded_runtime.OwnedProcessScope() as scope:
            with pytest.raises(ValueError):
                await release._execute_command(index=1, group=group, scope=scope, tools=tools, env=env, commands=commands)
        assert commands == [] and not (group.root / "ran-target").exists()
    try:
        asyncio.run(perform())
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def _inventory_fixture(monkeypatch: pytest.MonkeyPatch, *, fail_index: int | None = None, mutate_source: int | None = None, mutate_freeze: Path | None = None) -> list[Path]:
    original = blinded_runtime.run_owned_command
    directories: list[Path] = []
    async def backend(scope: Any, group: Any, env: dict[str, str]) -> Any:
        exe = Path(sys._base_executable)
        fp = release._native_file(exe, "tools/python_bootstrap")
        module = group.root / "uv-fixture.py"
        module.write_bytes(b"synthetic resolver boundary\n")
        launcher = release._tool_file(module, "tools/uv_module")
        bindings = (release.ToolExecutionBinding(role="python_bootstrap", version="3.12.12", executable=fp, launcher=None), release.ToolExecutionBinding(role="uv", version="0.12.15", executable=fp, launcher=launcher))
        return release._ToolSet(exe, None, bindings, ((exe, fp), (module, launcher)))
    async def sandbox(scope: Any, group: Any, env: dict[str, str], bootstrap: Any) -> Any:
        exe = Path(sys._base_executable)
        fp = release._native_file(exe, "tools/python_sandbox")
        return release._ToolSet(exe, None, (release.ToolExecutionBinding(role="python", version="3.12.12", executable=fp, launcher=None),), ((exe, fp),))
    async def frontend(scope: Any, group: Any, env: dict[str, str], executable: Path) -> Any:
        fp = release._native_file(executable, "tools/node")
        npx = group.root / "npx-cli.js"
        pnpm = group.root / "pnpm.cjs"
        npx.write_bytes(b"synthetic npx resolver boundary\n")
        pnpm.write_bytes(b"synthetic pnpm resolver boundary\n")
        npx_fp = release._tool_file(npx, "tools/npx_cli")
        pnpm_fp = release._tool_file(pnpm, "tools/pnpm_cjs")
        bindings = tuple(release.ToolExecutionBinding(role=role, version=version, executable=fp, launcher=launcher) for role, version, launcher in (("node", "24.11.1", None), ("npx", "11.6.2", npx_fp), ("pnpm", "10.18.3", pnpm_fp)))
        return release._ToolSet(executable, npx, bindings, ((executable, fp), (npx, npx_fp), (pnpm, pnpm_fp)))
    async def native_fixture(**arguments: Any) -> Any:
        name = arguments["stdout_path"].stem
        index = tuple(item[0] for item in release.D39_REQUIRED_COMMANDS).index(name)
        logical = release.D39_REQUIRED_COMMANDS[index][1]
        argv = arguments["argv"]
        assert argv[0] == sys._base_executable
        assert argv[1:] == logical[1:] if index < 5 else argv[2:] == logical[1:]
        assert arguments["deadline_seconds"] == (600, 30, 1800, 120, 600, 600, 600, 300, 180)[index]
        cwd = arguments["cwd"]
        directories.append(cwd)
        script = "import sys; print('bounded native fixture'); sys.exit(" + str(7 if index == fail_index else 0) + ")"
        if index == mutate_source:
            script = "from pathlib import Path; Path('pyproject.toml').write_text('drift'); " + script
        result = await original(**{**arguments, "argv": (sys._base_executable, "-I", "-S", "-B", "-c", script)})
        if index == 8 and mutate_freeze is not None:
            payload = json.loads(mutate_freeze.read_bytes())
            payload["created_at"] = "2026-01-02T00:00:00Z"
            raw = _bytes(payload)
            mutate_freeze.write_bytes(raw)
            marker_path = mutate_freeze.parent / ".d36-publication-state"
            marker = json.loads(marker_path.read_bytes())
            item = next(entry for entry in marker["files"] if entry["path"] == "freeze-manifest.json")
            item.update(size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            marker_path.write_bytes(_bytes(marker))
        return result
    async def esbuild(scope: Any, group: Any, tools: Any, env: dict[str, str]) -> None:
        if fail_index == 9:
            raise ValueError("synthetic missing native esbuild")
    async def media(scope: Any, group: Any, tools: Any, env: dict[str, str]) -> Any:
        # This driver fixture already substitutes tool/version resolution. The
        # native target, stdout/stderr, process ownership and cleanup stay real.
        from dataclasses import replace
        bindings = tuple(release.ToolExecutionBinding(role=role, version='synthetic-1', executable=release._native_file(tools.executable, 'tools/' + role), launcher=None) for role in ('ffmpeg', 'ffprobe'))
        return replace(tools, media_bindings=bindings), dict(env)
    monkeypatch.setattr(release, "_bootstrap_python", backend)
    monkeypatch.setattr(release, "_sandbox_python", sandbox)
    monkeypatch.setattr(release, "_frontend_tools", frontend)
    monkeypatch.setattr(release, "_esbuild_preflight", esbuild)
    monkeypatch.setattr(release, "_backend_media", media)
    monkeypatch.setattr(release, "_installed_node", lambda: Path(sys._base_executable))
    monkeypatch.setattr(blinded_runtime, "run_owned_command", native_fixture)
    return directories


@pytest.mark.parametrize(("fail", "mutation", "length", "status"), [(None, None, 9, "passed"), (0, None, 1, "failed"), (2, None, 3, "failed"), (6, None, 7, "failed"), (9, None, 6, "failed"), (None, 1, 2, "failed")])
def test_inventory_driver_real_native_outcomes_prefix_and_two_fresh_groups(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch, fail: int | None, mutation: int | None, length: int, status: str) -> None:
    materialized, runtime, digest = _materialize(publication)
    directories = _inventory_fixture(monkeypatch, fail_index=fail, mutate_source=mutation)
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    smoke_path = tools_repo / "evidence/smoke/smoke-manifest.json"
    smoke_path.parent.mkdir()
    smoke_path.write_bytes(_bytes(_synthetic_smoke(materialized, digest, tool_hash=release._attest_verifier(tools_repo).aggregate_sha256)))
    output = tools_repo / "evidence/verification"
    result = release.verify_release_candidate(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=output, smoke_manifest_path=smoke_path)
    assert len(result.commands) == length and result.status == status
    assert tuple((item.name, item.argv) for item in result.commands) == release.D39_REQUIRED_COMMANDS[:length]
    assert all(item.stdout_size > 0 and item.stdout_sha256 != hashlib.sha256(b"").hexdigest() for item in result.commands)
    assert result.cleanup_status == "completed" and not list(publication[2].iterdir())
    assert result.candidate_snapshot_before_sha256 == result.candidate_snapshot_after_sha256
    assert len(set(directories[:5])) == 1
    if length > 5:
        assert len(set(directories[5:])) == 1 and directories[0].parent != directories[5].parent
    assert not any(directory.exists() for directory in directories)
    if fail is not None and fail < 9:
        assert result.commands[-1].exit_code == 7
    else:
        assert result.commands[-1].exit_code == 0


def test_freeze_metadata_changed_after_commands_cannot_pass(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    materialized, runtime, digest = _materialize(publication)
    _inventory_fixture(monkeypatch, mutate_freeze=publication[1])
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    smoke_path = tools_repo / "evidence/smoke/smoke-manifest.json"
    smoke_path.parent.mkdir()
    smoke_path.write_bytes(_bytes(_synthetic_smoke(materialized, digest, tool_hash=release._attest_verifier(tools_repo).aggregate_sha256)))
    result = release.verify_release_candidate(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=tools_repo / "evidence/verification", smoke_manifest_path=smoke_path)
    assert result.status == "failed" and len(result.commands) == 9
    assert not result.candidate_clean_after
    assert result.cleanup_status == "completed" and not runtime.exists()


def test_git_failure_cannot_interrupt_bound_runtime_cleanup(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    materialized, runtime, digest = _materialize(publication)
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    (publication[0] / ".git").rename(tmp_path / "saved-git")
    result = release.verify_release_candidate(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=tools_repo / "evidence/verification", smoke_manifest_path=tools_repo / "evidence/smoke/missing.json")
    assert result.status == "failed" and not result.candidate_clean_before and not result.candidate_clean_after
    assert result.commands == () and result.smoke_manifest is None
    assert result.cleanup_status == "completed" and not runtime.exists()


def test_atomic_publication_lost_file_ownership_preserves_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.release_candidate import freeze
    def replace_then_fail(stage: Path, final: Path) -> None:
        target = stage / "verification-manifest.json"
        target.rename(stage / "original.json")
        target.write_bytes(b"foreign replacement")
        raise ValueError("lost staged file ownership")
    name = "_windows_move_directory_no_replace" if os.name == "nt" else "_linux_rename_directory_no_replace"
    monkeypatch.setattr(freeze, name, replace_then_fail)
    with pytest.raises(ValueError):
        release._publish_verification(tmp_path / "output", b"{}\n", b"{}\n")
    assert not (tmp_path / "output").exists()
    stages = list(tmp_path.glob(".d39-stage-*"))
    assert len(stages) == 1
    assert (stages[0] / "verification-manifest.json").read_bytes() == b"foreign replacement"


def test_group_cleanup_failure_is_not_erased_by_successful_runtime_cleanup(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    materialized, runtime, digest = _materialize(publication)
    _inventory_fixture(monkeypatch)
    original = release._ExecutionGroup.cleanup
    def cleanup_then_fail(group: Any) -> None:
        original(group)
        raise ValueError("injected late cleanup failure")
    monkeypatch.setattr(release._ExecutionGroup, "cleanup", cleanup_then_fail)
    monkeypatch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
    smoke_path = tools_repo / "evidence/smoke/smoke-manifest.json"
    smoke_path.parent.mkdir()
    smoke_path.write_bytes(_bytes(_synthetic_smoke(materialized, digest, tool_hash=release._attest_verifier(tools_repo).aggregate_sha256)))
    result = release.verify_release_candidate(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=tools_repo / "evidence/verification", smoke_manifest_path=smoke_path)
    assert result.status == "failed" and result.cleanup_status == "failed"
    assert len(result.commands) == 5 and not runtime.exists()
    assert json.loads(publication[3].with_name(publication[3].name + ".cleanup.json").read_bytes())["status"] == "completed"


def test_generated_child_replacement_is_reported_as_cleanup_failed(publication: tuple[Path, Path, Path, Path], tools_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    materialization = release.materialization
    materialized, runtime, digest = _materialize(publication)
    cleanup = release._ExecutionGroup.cleanup
    open_anchor = materialization._open_anchor
    group: release._ExecutionGroup | None = None
    child: Path | None = None
    moved: Path | None = None
    original_id: tuple[int, int] | None = None
    replacement_id: tuple[int, int] | None = None

    def prepare_cleanup(value: release._ExecutionGroup) -> None:
        nonlocal group, child, moved, original_id
        if group is None:
            group = value
            child = value.root / "generated-child"
            moved = value.work / (value.root.name + "-original-child")
            child.mkdir()
            (child / "original").write_bytes(b"owned original")
            original_id = materialization._identity(child.lstat())
        cleanup(value)

    def replace_before_anchor(path: Path) -> release.freeze._DirectoryAnchor:
        nonlocal replacement_id
        if path == child and replacement_id is None:
            assert moved is not None and materialization._identity(path.lstat()) == original_id
            path.rename(moved)
            path.mkdir()
            (path / "sentinel").write_bytes(b"replacement sentinel")
            replacement_id = materialization._identity(path.lstat())
            assert replacement_id != original_id
        return open_anchor(path)

    try:
        with monkeypatch.context() as patch:
            _inventory_fixture(patch)
            patch.setattr(release._ExecutionGroup, "cleanup", prepare_cleanup)
            patch.setattr(materialization, "_open_anchor", replace_before_anchor)
            patch.setattr(release, "_trusted_tool_root", lambda: tools_repo)
            smoke_path = tools_repo / "evidence/smoke/smoke-manifest.json"
            smoke_path.parent.mkdir()
            smoke_path.write_bytes(_bytes(_synthetic_smoke(materialized, digest, tool_hash=release._attest_verifier(tools_repo).aggregate_sha256)))
            output = tools_repo / "evidence/verification"
            result = release.verify_release_candidate(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime, materialization_path=publication[3], expected_materialization_sha256=digest, work_root=publication[2], output_dir=output, smoke_manifest_path=smoke_path)
            assert result.status == "failed" and result.cleanup_status == "failed"
            # A real memory preflight may refuse command launch before cleanup.
            # This regression asserts cleanup truth, not successful command execution.
            assert len(result.commands) <= 5 and not runtime.exists()
            assert tuple((item.name, item.argv) for item in result.commands) == release.D39_REQUIRED_COMMANDS[:len(result.commands)]
            assert group is not None and child is not None and moved is not None and replacement_id is not None
            assert group.anchor is not None and materialization._directory_identity(group.root) == group.anchor.identity
            assert (child / "sentinel").read_bytes() == b"replacement sentinel"
            assert (moved / "original").read_bytes() == b"owned original"
            assert json.loads((output / "verification-manifest.json").read_bytes())["cleanup_status"] == "failed"
            assert json.loads(publication[3].with_name(publication[3].name + ".cleanup.json").read_bytes())["status"] == "completed"
    finally:
        if group is not None:
            assert materialization._directory_identity(group.work) == group.work_anchor.identity
            assert child is not None and moved is not None and group.anchor is not None
            if child.exists():
                assert materialization._directory_identity(child) == replacement_id
            for path, identity in ((moved, original_id), (group.root, group.anchor.identity)):
                if path.exists():
                    retained = open_anchor(path)
                    try:
                        assert retained.identity == identity
                        materialization._assert_directory(path, retained)
                        release._remove_group_directory(path, retained)
                    finally:
                        release.freeze._close_directory_anchor(retained)
        materialization.cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_installed_native_node_and_npx_origins_without_package_fetch(publication: tuple[Path, Path, Path, Path]) -> None:
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    node = release._installed_node()
    npm = node.parent / "node_modules/npm" if os.name == "nt" else node.parent.parent / "lib/node_modules/npm"
    npx = npm / "bin/npx-cli.js"
    env = release.build_group_environment(group.root, python_executable=Path(sys._base_executable), node_executable=node)
    native_before = release._native_file(node, "tools/node")
    script_before = release._tool_file(npx, "tools/npx_cli")
    async def perform() -> None:
        async with blinded_runtime.OwnedProcessScope() as scope:
            data = release._probe_json(await release._probe(scope, group, (str(node), "-e", "console.log(JSON.stringify({version:process.versions.node,executable:process.execPath}))"), env))
            assert data["version"] == "24.11.1" and Path(data["executable"]) == node
            version = (await release._probe(scope, group, (str(node), str(npx), "--version"), env)).strip().decode("ascii")
            assert version == json.loads((npm / "package.json").read_bytes())["version"]
        assert scope.teardown_confirmed
    try:
        asyncio.run(perform())
        assert release._native_file(node, "tools/node") == native_before
        assert release._tool_file(npx, "tools/npx_cli") == script_before
        assert not (group.root / "npm-cache/_npx").exists()
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_missing_native_esbuild_fails_owned_preflight_without_rebuild(publication: tuple[Path, Path, Path, Path]) -> None:
    materialized, runtime, digest = _materialize(publication)
    group = release._ExecutionGroup(publication[2], runtime, materialized)
    node = release._installed_node()
    fingerprint = release._native_file(node, "tools/node")
    tools = release._ToolSet(node, None, (release.ToolExecutionBinding(role="node", version="24.11.1", executable=fingerprint, launcher=None),), ((node, fingerprint),))
    env = release.build_group_environment(group.root, python_executable=Path(sys._base_executable), node_executable=node)
    lock = group.source / "frontend/pnpm-lock.yaml"
    lock.write_bytes(b"packages:\n  esbuild@0.21.5:\n")
    async def perform() -> None:
        async with blinded_runtime.OwnedProcessScope() as scope:
            with pytest.raises(ValueError, match="native tool probe failed"):
                await release._esbuild_preflight(scope, group, tools, env)
        assert scope.teardown_confirmed
    try:
        asyncio.run(perform())
        assert not (group.source / "frontend/node_modules").exists()
        assert not (group.root / "npm-cache/_npx").exists()
        assert lock.read_bytes() == b"packages:\n  esbuild@0.21.5:\n"
    finally:
        group.cleanup()
        from evaluation.runtime_materialization import cleanup_candidate_runtime
        cleanup_candidate_runtime(runtime_root=runtime, work_root=publication[2], materialization_path=publication[3], expected_materialization_sha256=digest)


def test_env_example_exception_is_an_exact_lexical_filename(tmp_path: Path) -> None:
    target = tmp_path / ".ENV.EXAMPLE"
    target.write_bytes(b"safe example bytes")
    counts = release.scan_public_files(tmp_path, (".ENV.EXAMPLE",), ())
    assert counts["tracked_private_state"] == 1


def test_native_metadata_depth_failure_is_redacted_value_error() -> None:
    raw = b'{"extra":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}"
    with pytest.raises(ValueError):
        release._probe_json(raw)
