"""External exact-candidate verification; no application or private evaluator imports."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from pydantic import RootModel

from evaluation import blinded_io, blinded_runtime, runtime_materialization as materialization
from evaluation.evidence_json import parse_canonical_model
from evaluation.release_candidate import freeze
from evaluation.release_candidate.contracts import FreezeManifest
from evaluation.smoke_contracts import (
    D39_COMMAND_DEADLINES,
    D39_REQUIRED_COMMANDS,
    CommandEvidence,
    RuntimeMaterialization,
    SmokeManifest,
    ToolExecutionBinding,
    VerificationManifest,
    TOOL_RULES,
    validate_tool_consistency,
)
from evaluation.tool_attestation import (
    FileFingerprint,
    ToolAttestation,
    aggregate_fingerprints,
    attest_tool,
    canonical_json_bytes,
    validate_git_repository,
)

VERIFIER_SOURCE_PATHS: tuple[str, ...] = (
    "backend/evaluation/__init__.py",
    "backend/evaluation/blinded_io.py",
    "backend/evaluation/blinded_runtime.py",
    "backend/evaluation/browser_smoke.py",
    "backend/evaluation/evidence_json.py",
    "backend/evaluation/release_candidate/__init__.py",
    "backend/evaluation/release_candidate/contracts.py",
    "backend/evaluation/release_candidate/fingerprints.py",
    "backend/evaluation/release_candidate/freeze.py",
    "backend/evaluation/release_verification.py",
    "backend/evaluation/runtime_materialization.py",
    "backend/evaluation/scripts/d39_candidate_smoke.py",
    "backend/evaluation/scripts/d39_smoke.py",
    "backend/evaluation/scripts/materialize_candidate_runtime.py",
    "backend/evaluation/scripts/verify_release_candidate.py",
    "backend/evaluation/smoke_contracts.py",
    "backend/evaluation/tool_attestation.py",
    "backend/pyproject.toml",
    "backend/uv.lock",
)
SCAN_RULE_IDS: tuple[str, ...] = ("tracked_private_state", "tracked_generated_state", "credential_token", "private_key_block")
PRIVATE_COMPONENTS: frozenset[str] = frozenset({"private", "corpus", "keys", "secrets", "reviews", "evidence", "storage"})
PRIVATE_FILENAMES: frozenset[str] = frozenset({"held-out.jsonl", "human-review.json", "independent-review.json"})
PUBLIC_SYNTHETIC_PATHS: tuple[str, ...] = (
    "backend/tests/fixtures/blinded/synthetic-held-out.jsonl",
    "backend/tests/fixtures/blinded/synthetic-human-review.json",
    "backend/tests/fixtures/blinded/synthetic-independent-review.json",
    "backend/tests/fixtures/blinded/synthetic-result-bundle.json",
)
GENERATED_COMPONENTS: frozenset[str] = frozenset({".venv", "venv", "node_modules", "dist", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache", "media", "generated"})
GENERATED_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".mp4", ".webm", ".wav"})
CREDENTIAL_PATTERN: bytes = rb"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,200}|gh[pousr]_[A-Za-z0-9_]{20,200}|github_pat_[A-Za-z0-9_]{20,200}|(?:AKIA|ASIA)[A-Z0-9]{16})(?![A-Za-z0-9_-])"
PRIVATE_KEY_PATTERN: bytes = rb"-----BEGIN (?:PRIVATE|RSA PRIVATE|EC PRIVATE|OPENSSH PRIVATE) KEY-----"
_EXAMPLE_PATTERN = re.compile(rb"(?:sk-|gh[pousr]_|github_pat_)[xX]{20,200}\Z")
_SCAN_PATTERNS = (("credential_token", re.compile(CREDENTIAL_PATTERN)), ("private_key_block", re.compile(PRIVATE_KEY_PATTERN)))
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_TOOL_BYTES = 256 * 1024 * 1024
_MAX_SCAN_COUNT = 8192


def _trusted_tool_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _attest_verifier(root: Path) -> ToolAttestation:
    commit = validate_git_repository(root)
    try:
        return attest_tool(repo_root=root, tool_name="d39_release_verifier", git_commit=commit, source_paths=VERIFIER_SOURCE_PATHS)
    except ValueError as error:
        if str(error).startswith('working file does not match committed blob'):
            raise ValueError('D39 committed bytes mismatch; require exact LF checkout') from None
        raise


def _scan_stream(stream: BinaryIO, counts: dict[str, int], maximum: int) -> int:
    size = 0
    overlap = b""
    counted_before = 0
    chunk = stream.read(65536)
    while chunk:
        size += len(chunk)
        if size > maximum:
            raise ValueError("public scan byte limit exceeded")
        window = overlap + chunk
        offset = size - len(window)
        following = stream.read(65536)
        stable = size if not following else max(0, size - 4096)
        for rule, pattern in _SCAN_PATTERNS:
            for match in pattern.finditer(window):
                start = offset + match.start()
                if start < counted_before or start >= stable:
                    continue
                token = match.group()
                if rule == "credential_token" and (token == b"AKIAIOSFODNN7EXAMPLE" or _EXAMPLE_PATTERN.fullmatch(token)):
                    continue
                counts[rule] = min(_MAX_SCAN_COUNT, counts[rule] + 1)
        counted_before = stable
        # Include the byte before the next uncounted start for the lookbehind.
        overlap = window[-4097:]
        chunk = following
    return size


def _count_path_rules(relative: str, counts: dict[str, int]) -> None:
    FileFingerprint(path=relative, size=0, sha256="0" * 64)
    parsed = PurePosixPath(relative.lower())
    name = parsed.name
    public_example = PurePosixPath(relative).name == ".env.example"
    private = bool(PRIVATE_COMPONENTS.intersection(parsed.parts) or name in PRIVATE_FILENAMES or (not public_example and (name == ".env" or name.startswith(".env."))))
    if private and relative not in PUBLIC_SYNTHETIC_PATHS:
        counts["tracked_private_state"] = min(_MAX_SCAN_COUNT, counts["tracked_private_state"] + 1)
    if GENERATED_COMPONENTS.intersection(parsed.parts) or parsed.suffix in GENERATED_SUFFIXES:
        counts["tracked_generated_state"] = min(_MAX_SCAN_COUNT, counts["tracked_generated_state"] + 1)


def scan_public_files(root: Path, paths: tuple[str, ...], metadata: tuple[bytes, ...]) -> dict[str, int]:
    """Count four bounded rules in explicit tracked files and bounded public metadata."""
    if len(paths) > 8192 or len(set(paths)) != len(paths) or len(metadata) > 16:
        raise ValueError("public scan inventory limit exceeded")
    counts = dict.fromkeys(SCAN_RULE_IDS, 0)
    total = 0
    for relative in paths:
        _count_path_rules(relative, counts)
        path = root / relative
        materialization._directory(path.parent)
        descriptor = materialization._file_descriptor(path)
        try:
            before = materialization._descriptor_stat(descriptor)
            identity = materialization._identity(before)
            materialization._assert_file(path, descriptor, identity)
            if before.st_size > _MAX_SOURCE_BYTES:
                raise ValueError("public scan file limit exceeded")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                size = _scan_stream(stream, counts, _MAX_SOURCE_BYTES)
            after = materialization._assert_file(path, descriptor, identity)
            if size != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("public scan source changed")
            total += size
        finally:
            os.close(descriptor)
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("public scan total limit exceeded")
    return _scan_metadata(metadata, counts, total)


def _scan_metadata(metadata: tuple[bytes, ...], counts: dict[str, int], total: int) -> dict[str, int]:
    if len(metadata) > 16:
        raise ValueError("public scan inventory limit exceeded")
    for blob in metadata:
        if type(blob) is not bytes or len(blob) > _MAX_METADATA_BYTES:
            raise ValueError("public scan metadata limit exceeded")
        total += _scan_stream(io.BufferedReader(io.BytesIO(blob)), counts, _MAX_METADATA_BYTES)
        if total > _MAX_TOTAL_BYTES:
            raise ValueError("public scan total limit exceeded")
    return counts


class _BlobReader:
    """Expose exactly one `cat-file --batch` blob body as a bounded stream."""

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self._stream = stream
        self._remaining = size

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        wanted = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._stream.read(wanted)
        if len(chunk) != wanted:
            raise ValueError("public scan blob truncated")
        self._remaining -= len(chunk)
        return chunk


def scan_candidate_commit(root: Path, commit: str, metadata: tuple[bytes, ...]) -> dict[str, int]:
    """Scan the bound commit's committed blobs, including freeze exclusions.

    Bytes come from Git objects, not the working tree, so attributes such as
    working-tree-encoding or filters cannot hide committed content, and
    replace refs are ignored for both the tree listing and the blobs.
    """
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("public scan commit invalid")
    prefix = ("git", "--no-replace-objects", "-c", "core.longpaths=true", "-C", str(root))
    process = subprocess.Popen((*prefix, "ls-tree", "-r", "-z", "--full-tree", commit),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(30, process.kill)
    timer.daemon = True
    timer.start()
    try:
        assert process.stdout is not None
        raw = process.stdout.read(_MAX_METADATA_BYTES + 1)
        if len(raw) > _MAX_METADATA_BYTES:
            raise ValueError("public scan inventory byte limit exceeded")
        if process.wait(timeout=30) != 0:
            raise ValueError("public scan inventory unavailable")
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
    entries: list[tuple[str, bytes]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        fields, name = entry.split(b"\t", 1)
        mode, kind, oid = fields.split()
        if mode not in {b"100644", b"100755"} or kind != b"blob" or re.fullmatch(rb"[0-9a-f]{40}", oid) is None:
            raise ValueError("public tracked file must be regular")
        entries.append((name.decode("utf-8", errors="strict"), oid))
        if len(entries) > 8192:
            raise ValueError("public scan inventory limit exceeded")
    if len({path for path, _ in entries}) != len(entries):
        raise ValueError("public scan inventory duplicated")
    counts = dict.fromkeys(SCAN_RULE_IDS, 0)
    total = 0
    blobs = subprocess.Popen((*prefix, "cat-file", "--batch"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(300, blobs.kill)
    timer.daemon = True
    timer.start()
    try:
        assert blobs.stdin is not None and blobs.stdout is not None
        for relative, oid in entries:
            _count_path_rules(relative, counts)
            blobs.stdin.write(oid + b"\n")
            blobs.stdin.flush()
            header = blobs.stdout.readline(128).split()
            if len(header) != 3 or header[:2] != [oid, b"blob"] or not header[2].isdigit():
                raise ValueError("public scan blob header invalid")
            size = int(header[2])
            total += size
            if size > _MAX_SOURCE_BYTES or total > _MAX_TOTAL_BYTES:
                raise ValueError("public scan file limit exceeded")
            if _scan_stream(_BlobReader(blobs.stdout, size), counts, _MAX_SOURCE_BYTES) != size:  # type: ignore[arg-type]
                raise ValueError("public scan blob truncated")
            if blobs.stdout.read(1) != b"\n":
                raise ValueError("public scan blob framing invalid")
        blobs.stdin.close()
        if blobs.wait(timeout=30) != 0:
            raise ValueError("public scan blobs unavailable")
    finally:
        timer.cancel()
        if blobs.poll() is None:
            blobs.kill()
        blobs.wait(timeout=10)
        for stream in (blobs.stdin, blobs.stdout):
            if stream is not None and not stream.closed:
                stream.close()
    return _scan_metadata(metadata, counts, total)


def build_group_environment(root: Path, *, python_executable: Path, node_executable: Path | None) -> dict[str, str]:
    """Construct a clean, group-contained low-concurrency environment from scratch."""
    root = materialization._directory(root)
    env: dict[str, str] = {}
    system_paths: list[str] = []
    if os.name == "nt":
        system = materialization._directory(Path(os.environ.get("SystemRoot", "C:/Windows")))
        env.update(SystemRoot=str(system), WINDIR=str(system), COMSPEC=str(system / "System32" / "cmd.exe"))
        system_paths = [str(system / "System32"), str(system)]
    else:
        system_paths = ["/usr/bin", "/bin"]
    writable = {
        # Windows-shaped AppData: programs such as Chrome resolve known folders from
        # %USERPROFILE%\AppData\{Local,Roaming}, not from APPDATA/LOCALAPPDATA.
        "HOME": "home", "USERPROFILE": "home", "APPDATA": "home/AppData/Roaming", "LOCALAPPDATA": "home/AppData/Local",
        "XDG_CONFIG_HOME": "home/config", "XDG_CACHE_HOME": "home/cache", "XDG_DATA_HOME": "home/data", "XDG_STATE_HOME": "home/state",
        "TEMP": "temp", "TMP": "temp", "TMPDIR": "temp", "UV_PROJECT_ENVIRONMENT": "env", "UV_CACHE_DIR": "uv-cache",
        "NPM_CONFIG_CACHE": "npm-cache", "RUFF_CACHE_DIR": "ruff-cache", "PNPM_HOME": "pnpm-home",
        "BLOCKVIDEO_STORAGE_DIR": "storage",
    }
    for key, suffix in writable.items():
        directory = blinded_io.create_directory_tree(root / suffix, "group state")
        env[key] = str(directory)
    env.update({
        "NPM_CONFIG_USERCONFIG": str(root / "npm-user.conf"), "NPM_CONFIG_GLOBALCONFIG": str(root / "npm-global.conf"),
        "UV_PYTHON": str(python_executable.absolute()), "UV_CONCURRENT_DOWNLOADS": "2", "UV_CONCURRENT_BUILDS": "1", "UV_CONCURRENT_INSTALLS": "1",
        "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1",
        "NODE_OPTIONS": "--max-old-space-size=512", "NODE_DISABLE_COMPILE_CACHE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "PYTEST_ADDOPTS": "-o " + shlex.quote("cache_dir=" + (root / "pytest-cache").as_posix()),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
        "NPM_CONFIG_UPDATE_NOTIFIER": "false", "NPM_CONFIG_AUDIT": "false", "NPM_CONFIG_FUND": "false",
        "NPM_CONFIG_PRODUCTION": "false", "NPM_CONFIG_IGNORE_SCRIPTS": "true", "NPM_CONFIG_ENGINE_STRICT": "true",
        "CI": "true", "NO_COLOR": "1", "LANG": "C", "LC_ALL": "C",
    })
    native_paths = [str(python_executable.absolute().parent)]
    if node_executable is not None:
        native_paths.insert(0, str(node_executable.absolute().parent))
    git = shutil.which("git")
    if git is not None:
        native_paths.append(str(_native_path(Path(git)).parent))
    env["PATH"] = os.pathsep.join([str(root / "env" / ("Scripts" if os.name == "nt" else "bin")), *native_paths, *system_paths])
    for name in ("npm-user.conf", "npm-global.conf"):
        blinded_io.write_exclusive(root / name, b"")
    return env


class _ExecutionGroup:
    def __init__(self, work: Path, runtime: Path, record: RuntimeMaterialization) -> None:
        self.work = materialization._directory(work)
        self.work_anchor = materialization._open_anchor(self.work)
        self.root = self.work / ("group-" + secrets.token_hex(32))
        self.anchor: freeze._DirectoryAnchor | None = None
        self.source = self.root / "source"
        self.record = record
        created: tuple[int, int] | None = None
        replaced = False
        try:
            materialization._assert_directory(self.work, self.work_anchor)
            materialization._fs_path(self.root).mkdir(mode=0o700)
            created = materialization._identity(materialization._fs_path(self.root).lstat())
            anchor = materialization._open_anchor(self.root)
            if anchor.identity != created:
                # The created root was swapped before anchoring: never adopt or
                # delete the replacement; the original is reported as lost.
                freeze._close_directory_anchor(anchor)
                replaced = True
                raise ValueError("group root replaced before anchoring")
            self.anchor = anchor
            materialization._fs_path(self.source).mkdir(mode=0o700)
            tree = materialization._OwnedTree(self.source)
            try:
                tree.guard = self.assert_owned
                for item in record.files:
                    tree.copy(runtime, item)
                self.assert_source()
            finally:
                tree.close()
        except BaseException:
            try:
                if replaced:
                    raise ValueError("group root ownership lost")
                if self.anchor is not None:
                    self.cleanup()
                elif created is not None:
                    materialization._delete_owned_path(self.root, created, directory=True, parent_descriptor=self.work_anchor.descriptor)
            except (OSError, ValueError):
                raise _GroupCleanupFailed("owned group construction cleanup failed") from None
            finally:
                if self.anchor is not None:
                    freeze._close_directory_anchor(self.anchor)
                freeze._close_directory_anchor(self.work_anchor)
            raise

    def assert_owned(self) -> None:
        materialization._assert_directory(self.work, self.work_anchor)
        if self.anchor is None:
            raise ValueError("group ownership unavailable")
        materialization._assert_directory(self.root, self.anchor)

    def source_sha256(self) -> str:
        self.assert_owned()
        files: list[FileFingerprint] = []
        for expected in self.record.files:
            path = self.source / expected.path
            materialization._directory(path.parent)
            size, digest = blinded_io.fingerprint_regular(materialization._fs_path(path), maximum=_MAX_SOURCE_BYTES)
            files.append(FileFingerprint(path=expected.path, size=size, sha256=digest))
        return aggregate_fingerprints(files)

    def assert_source(self) -> None:
        if self.source_sha256() != self.record.runtime_source_sha256:
            raise ValueError("group tracked source drift")

    def cleanup(self) -> None:
        self.assert_owned()
        assert self.anchor is not None
        _remove_group_directory(self.root, self.anchor)
        freeze._close_directory_anchor(self.work_anchor)


def _remove_group_directory(path: Path, anchor: freeze._DirectoryAnchor, *, _parents_verified: bool = False) -> None:
    if not _parents_verified:
        materialization._assert_directory(path, anchor)
    def check() -> None:
        before = materialization._fs_path(path).lstat()
        if (not stat.S_ISDIR(before.st_mode) or blinded_io.is_reparse(before)
                or materialization._identity(before) != anchor.identity
                or materialization._anchor_identity(anchor) != anchor.identity):
            raise ValueError("group directory identity lost")
    check()
    inventory: list[tuple[str, os.stat_result]] = []
    with os.scandir(materialization._fs_path(path)) as entries:
        for entry in entries:
            if len(inventory) >= 262144:
                raise ValueError("group cleanup inventory exceeded")
            observed = materialization._fs_path(path / entry.name).lstat()
            if observed.st_ino <= 0:
                raise ValueError("group child identity unavailable")
            inventory.append((entry.name, observed))
    for name, before in inventory:
        check()
        child = path / name
        expected = (materialization._identity(before), stat.S_IFMT(before.st_mode), blinded_io.is_reparse(before))
        named = materialization._fs_path(child).lstat()
        if (materialization._identity(named), stat.S_IFMT(named.st_mode), blinded_io.is_reparse(named)) != expected:
            raise ValueError("group child identity or type lost")
        if stat.S_ISDIR(before.st_mode) and not blinded_io.is_reparse(before):
            nested = materialization._open_anchor(child)
            try:
                if nested.identity != expected[0] or materialization._directory_identity(child) != expected[0]:
                    raise ValueError("group child identity lost")
                _remove_group_directory(child, nested, _parents_verified=True)
            finally:
                freeze._close_directory_anchor(nested)
        else:
            descriptor = -1
            try:
                if stat.S_ISREG(before.st_mode) and not blinded_io.is_reparse(before):
                    descriptor = materialization._file_descriptor(child)
                    opened = materialization._descriptor_stat(descriptor)
                    if (materialization._identity(opened), stat.S_IFMT(opened.st_mode), blinded_io.is_reparse(opened)) != expected:
                        raise ValueError("group leaf changed before open")
                named = materialization._fs_path(child).lstat()
                if (materialization._identity(named), stat.S_IFMT(named.st_mode), blinded_io.is_reparse(named)) != expected:
                    raise ValueError("group leaf identity or type lost")
                # uv/pnpm hardlink group-local caches into the group. Deleting this
                # verified name through its handle never changes the shared file's
                # attributes or another link, so additional links are not refused.
                check()
                if descriptor >= 0:
                    os.close(descriptor)
                    descriptor = -1
                materialization._delete_owned_path(child, expected[0], directory=stat.S_ISDIR(before.st_mode),
                                                  reparse=blinded_io.is_reparse(before), parent_descriptor=anchor.descriptor)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    check()
    if os.listdir(materialization._fs_path(path)):
        raise ValueError("group cleanup has unexpected entries")
    freeze._close_directory_anchor(anchor)
    materialization._delete_owned_path(path, anchor.identity, directory=True)


def _read_smoke(path: Path, record: RuntimeMaterialization, digest: str) -> tuple[SmokeManifest, str]:
    raw = blinded_io.read_regular(path, maximum=1024 * 1024)
    result = parse_canonical_model(raw, SmokeManifest, maximum=1024 * 1024)
    for name in ("candidate_id", "git_commit", "freeze_sha256", "runtime_instance_id", "runtime_source_sha256"):
        if getattr(result, name) != getattr(record, name):
            raise ValueError("smoke runtime binding mismatch")
    if result.materialization_sha256 != digest:
        raise ValueError("smoke detached binding mismatch")
    _validate_smoke_tools(result)
    return result, hashlib.sha256(raw).hexdigest()


def _validate_smoke_tools(smoke: SmokeManifest) -> None:
    # The public contracts enforce the same rules for the offline D40 reader.
    smoke.complete_smoke()
    validate_tool_consistency(tuple(tool for receipt in smoke.stage_receipts for tool in receipt.tools))


def _publish_verification(output: Path, manifest_bytes: bytes, attestation_bytes: bytes) -> None:
    if os.name != "nt" and not sys.platform.startswith("linux"):
        raise ValueError("native verification publication unavailable")
    values = {"verification-manifest.json": manifest_bytes, "d39-verifier-attestation.json": attestation_bytes}
    if any(type(raw) is not bytes or not raw or len(raw) > _MAX_METADATA_BYTES for raw in values.values()):
        raise ValueError("verification publication size invalid")
    final = output.absolute()
    parent = materialization._directory(final.parent)
    if final.exists() or final.is_symlink() or final.name in {"", ".", ".."}:
        raise ValueError("verification destination already exists or is invalid")
    parent_anchor = materialization._open_anchor(parent)
    stage = parent / (".d39-stage-" + secrets.token_hex(32))
    stage_anchor = None
    retained: dict[str, tuple[int, tuple[int, int]]] = {}
    closed = False
    published = False

    def check(directory: Path) -> None:
        materialization._assert_directory(parent, parent_anchor)
        assert stage_anchor is not None
        if materialization._directory_identity(directory) != stage_anchor.identity:
            raise ValueError("verification stage identity lost")
        if set(os.listdir(directory)) != set(values):
            raise ValueError("verification stage inventory invalid")
        for name, (descriptor, identity) in retained.items():
            path = directory / name
            if not closed:
                materialization._assert_file(path, descriptor, identity)
                raw = materialization._read_descriptor(descriptor, maximum=_MAX_METADATA_BYTES)
            else:
                if materialization._identity(path.lstat()) != identity:
                    raise ValueError("verification file identity lost")
                raw = blinded_io.read_regular(path, maximum=_MAX_METADATA_BYTES)
            if raw != values[name]:
                raise ValueError("verification readback mismatch")

    try:
        materialization._assert_directory(parent, parent_anchor)
        stage.mkdir(mode=0o700)
        stage_anchor = materialization._open_anchor(stage)
        for name, raw in values.items():
            materialization._assert_directory(parent, parent_anchor)
            materialization._assert_directory(stage, stage_anchor)
            descriptor = materialization._file_descriptor(stage / name, create=True, writable=True)
            retained[name] = descriptor, materialization._identity(materialization._descriptor_stat(descriptor))
            materialization._write_descriptor(descriptor, raw)
        check(stage)
        if stage_anchor.descriptor is not None:
            assert parent_anchor.descriptor is not None
            os.fsync(stage_anchor.descriptor)
            alias = Path(f"/proc/self/fd/{parent_anchor.descriptor}")
            freeze._linux_rename_directory_no_replace(alias / stage.name, alias / final.name)
            published = True
            check(final)
            os.fsync(parent_anchor.descriptor)
        else:
            for descriptor, _ in retained.values():
                os.close(descriptor)
            closed = True
            check(stage)
            freeze._close_directory_anchor(stage_anchor)
            materialization._assert_directory(parent, parent_anchor)
            freeze._windows_move_directory_no_replace(stage, final)
            published = True
            check(final)
    finally:
        if not closed:
            for descriptor, _ in retained.values():
                os.close(descriptor)
        try:
            if stage_anchor is not None:
                freeze._close_directory_anchor(stage_anchor)
                if not published and stage.exists() and materialization._directory_identity(stage) == stage_anchor.identity:
                    _discard_publication_stage(stage, parent, parent_anchor, stage_anchor.identity, retained, values)
        finally:
            freeze._close_directory_anchor(parent_anchor)


def _discard_publication_stage(stage: Path, parent: Path, parent_anchor: freeze._DirectoryAnchor, stage_identity: tuple[int, int], retained: dict[str, tuple[int, tuple[int, int]]], values: dict[str, bytes]) -> None:
    materialization._assert_directory(parent, parent_anchor)
    anchor = materialization._open_anchor(stage)
    try:
        if anchor.identity != stage_identity or set(os.listdir(stage)) != set(retained):
            raise ValueError("verification stage cleanup ownership lost")
        for name, (_, identity) in retained.items():
            path = stage / name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or blinded_io.is_reparse(metadata) or metadata.st_nlink != 1 or materialization._identity(metadata) != identity:
                raise ValueError("verification stage cleanup file replaced")
            raw = blinded_io.read_regular(path, maximum=len(values[name]))
            if not values[name].startswith(raw):
                raise ValueError("verification stage cleanup content changed")
        for name, (_, identity) in retained.items():
            materialization._assert_directory(parent, parent_anchor)
            materialization._assert_directory(stage, anchor)
            path = stage / name
            descriptor = materialization._file_descriptor(path)
            try:
                materialization._assert_file(path, descriptor, identity)
            finally:
                os.close(descriptor)
            if materialization._identity(path.lstat()) != identity:
                raise ValueError("verification stage file lost before unlink")
            if anchor.descriptor is not None:
                os.unlink(name, dir_fd=anchor.descriptor)
            else:
                path.unlink()
        materialization._assert_directory(stage, anchor)
        if os.listdir(stage):
            raise ValueError("verification stage has unexpected cleanup entries")
        freeze._close_directory_anchor(anchor)
        if materialization._directory_identity(stage) != stage_identity:
            raise ValueError("verification stage lost before removal")
        stage.rmdir()
    finally:
        freeze._close_directory_anchor(anchor)


def _validate_output(output: Path, tool_root: Path, candidate: Path, runtime: Path, work: Path, materialization_path: Path, smoke_path: Path, freeze_path: Path) -> None:
    parent = materialization._directory(output.absolute().parent)
    final = parent / output.name
    if not final.is_relative_to(tool_root) or final == tool_root or final.exists() or final.is_symlink():
        raise ValueError("verification output must be new tooling evidence")
    for path in (candidate, runtime, work, materialization_path.parent, smoke_path.parent, freeze_path.parent):
        if final == path or final in path.parents or path in final.parents:
            raise ValueError("verification output aliases overlap")
    if freeze._git(tool_root, "check-ignore", "--no-index", "--", str(final), check=False).returncode != 0:
        raise ValueError("verification output must be Git ignored")
    if freeze._git(tool_root, "ls-files", "--", str(final)).stdout:
        raise ValueError("verification output must be untracked")


def verify_release_candidate(*, candidate_root: Path, freeze_manifest_path: Path, runtime_root: Path, materialization_path: Path, expected_materialization_sha256: str, work_root: Path, output_dir: Path, smoke_manifest_path: Path) -> VerificationManifest:
    """Verify fixed inventories, preserve actual failed observations, always clean bound runtime."""
    record = materialization._bound_materialization(materialization_path.absolute(), expected_materialization_sha256)
    commands: list[CommandEvidence] = []
    attestation: ToolAttestation | None = None
    before: str | None = None
    after: str | None = None
    runtime_after: str | None = None
    clean_before = clean_after = scan_passed = False
    smoke: SmokeManifest | None = None
    smoke_digest: str | None = None
    failed = True
    cleanup = "completed"
    root = _trusted_tool_root()
    candidate = candidate_root.absolute()
    runtime = runtime_root.absolute()
    work = work_root.absolute()
    try:
        attestation = _attest_verifier(root)
        _validate_output(output_dir, root, candidate, runtime, work, materialization_path.absolute(), smoke_manifest_path.absolute(), freeze_manifest_path.absolute())
        before = freeze._snapshot_tree(candidate)
        frozen, digest, snapshot = materialization._freeze_inputs(candidate, freeze_manifest_path.absolute())
        _candidate_binding(frozen, digest, snapshot, record)
        clean_before = True
        materialization.read_materialized_runtime(runtime_root=runtime, work_root=work, materialization_path=materialization_path, expected_materialization_sha256=expected_materialization_sha256)
        smoke, smoke_digest = _read_smoke(smoke_manifest_path, record, expected_materialization_sha256)
        if smoke.producer_tool_sha256 != attestation.aggregate_sha256:
            smoke = None
            smoke_digest = None
            raise ValueError("smoke producer attestation mismatch")
        counts = scan_candidate_commit(candidate, record.git_commit, (canonical_json_bytes(record) + b"\n", canonical_json_bytes(smoke) + b"\n", canonical_json_bytes(attestation) + b"\n"))
        for rule in SCAN_RULE_IDS:
            print(rule + "=" + str(counts[rule]))
        scan_passed = not any(counts.values())
        if not scan_passed:
            raise ValueError("public scan failed")
        asyncio.run(_run_inventory(candidate=candidate, runtime=runtime, work=work, path=materialization_path, digest=expected_materialization_sha256, record=record, tool_root=root, attestation=attestation, commands=commands, frozen_path=freeze_manifest_path.absolute()))
        failed = False
    except _GroupCleanupFailed:
        cleanup = "failed"
        failed = True
    except (OSError, ValueError, subprocess.SubprocessError):
        failed = True
    finally:
        try:
            after = freeze._snapshot_tree(candidate)
            frozen, freeze_digest, snapshot = materialization._freeze_inputs(candidate, freeze_manifest_path.absolute())
            _candidate_binding(frozen, freeze_digest, snapshot, record)
            clean_after = True
        except (OSError, ValueError, subprocess.SubprocessError):
            failed = True
        try:
            materialization.read_materialized_runtime(runtime_root=runtime, work_root=work, materialization_path=materialization_path, expected_materialization_sha256=expected_materialization_sha256)
            runtime_after = record.runtime_source_sha256
        except (OSError, ValueError):
            failed = True
        try:
            materialization.cleanup_candidate_runtime(runtime_root=runtime, work_root=work, materialization_path=materialization_path, expected_materialization_sha256=expected_materialization_sha256)
        except (OSError, ValueError):
            cleanup = "failed"
            failed = True
    if attestation is None or before is None:
        raise ValueError("verification preflight rejected; runtime cleanup attempted")
    if after != before or before != record.candidate_snapshot_sha256:
        failed = True
    payload = dict(schema_version=1, candidate_id=record.candidate_id, git_commit=record.git_commit, freeze_sha256=record.freeze_sha256, verifier_tool_sha256=attestation.aggregate_sha256, status="failed" if failed else "passed", commands=tuple(commands), smoke_manifest_sha256=smoke_digest, smoke_manifest=smoke, secret_scan_passed=scan_passed, candidate_clean_before=clean_before, candidate_clean_after=clean_after, candidate_snapshot_before_sha256=before, candidate_snapshot_after_sha256=after, materialization_sha256=expected_materialization_sha256, runtime_instance_id=record.runtime_instance_id, runtime_source_sha256=record.runtime_source_sha256, runtime_snapshot_after_sha256=runtime_after, cleanup_status=cleanup)
    result = VerificationManifest.model_validate(payload, strict=True)
    if _attest_verifier(root) != attestation:
        raise ValueError("verifier source changed during execution")
    _validate_output(output_dir, root, candidate, runtime, work, materialization_path.absolute(), smoke_manifest_path.absolute(), freeze_manifest_path.absolute())
    _publish_verification(output_dir, canonical_json_bytes(result) + b"\n", canonical_json_bytes(attestation) + b"\n")
    return result


_BACKEND_IMPORTS: tuple[str, ...] = (
    "fastapi", "uvicorn", "pydantic", "pydantic_settings", "sqlalchemy", "aiosqlite",
    "httpx", "python_multipart", "anyio", "PIL", "cairosvg", "jinja2", "yaml", "loguru",
    "pytest", "pytest_asyncio", "mypy", "ruff", "numpy", "onnxruntime", "tokenizers",
)
_PYTHON_PROBE = (
    "import importlib,importlib.util,importlib.metadata,json,sys; "
    "names=json.loads(sys.argv[1]); modules={n:importlib.util.find_spec(n) for n in names}; "
    "distributions={'PIL':'Pillow','yaml':'PyYAML'}; "
    "data={'version':'.'.join(map(str,sys.version_info[:3])),'executable':sys.executable,"
    "'prefix':sys.prefix,'base_prefix':sys.base_prefix,'base_executable':sys._base_executable,"
    "'origins':{n:m.origin if m else None for n,m in modules.items()},"
    "'locations':{n:list(m.submodule_search_locations or ()) if m else [] for n,m in modules.items()},"
    "'versions':{n:importlib.metadata.version(distributions.get(n,n)) for n in names}}; "
    "data.update({'uv_version':importlib.metadata.version('uv'),'uv_main':importlib.util.find_spec('uv.__main__').origin,"
    "'uv_binary':importlib.import_module('uv').find_uv_bin()}) if 'uv' in modules else None; print(json.dumps(data))"
)


class _GroupCleanupFailed(ValueError):
    pass


@dataclass(frozen=True)
class _ToolSet:
    executable: Path
    launcher: Path | None
    bindings: tuple[ToolExecutionBinding, ...]
    files: tuple[tuple[Path, FileFingerprint], ...]
    base_prefix: Path | None = None
    media_bindings: tuple[ToolExecutionBinding, ...] = ()

    def verify(self) -> None:
        if not self.files or not self.bindings or self.executable.suffix.lower() in {".cmd", ".bat", ".py", ".ps1", ".sh"}:
            raise ValueError("native tool binding unavailable")
        native = self.executable.resolve(strict=True)
        if native != self.files[0][0].absolute() or any(item.executable != self.files[0][1] for item in self.bindings):
            raise ValueError("command native path differs from bound executable")
        if self.launcher is not None and not any(path.absolute() == self.launcher.absolute() and expected.path == "tools/npx_cli" for path, expected in self.files):
            raise ValueError("command launcher differs from bound npx")
        total = 0
        for path, expected in self.files:
            actual = _tool_file(path, expected.path)
            if actual != expected:
                raise ValueError("native tool binding drift")
            total += actual.size
        if total > 1024 * 1024 * 1024:
            raise ValueError("native tool aggregate byte limit exceeded")


def _native_path(path: Path) -> Path:
    if path.suffix.lower() in {".cmd", ".bat", ".py", ".ps1", ".sh"}:
        raise ValueError("native executable required")
    resolved = path.absolute().resolve(strict=True)
    _native_file(resolved, "tools/probe")
    return resolved


def _tool_file(path: Path, alias: str) -> FileFingerprint:
    materialization._directory(path.absolute().parent)
    size, digest = blinded_io.fingerprint_regular(materialization._fs_path(path.absolute()), maximum=_MAX_TOOL_BYTES)
    return FileFingerprint(path=alias, size=size, sha256=digest)


def _native_file(path: Path, alias: str) -> FileFingerprint:
    if path.suffix.lower() in {".cmd", ".bat", ".py", ".ps1", ".sh"}:
        raise ValueError("native executable required")
    result = _tool_file(path, alias)
    descriptor = materialization._file_descriptor(path)
    try:
        header = os.read(descriptor, 4)
    finally:
        os.close(descriptor)
    if (os.name == "nt" and not header.startswith(b"MZ")) or (os.name != "nt" and header not in (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")):
        raise ValueError("native executable format required")
    return result


async def _probe(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, argv: tuple[str, ...], env: dict[str, str]) -> bytes:
    group.assert_owned()
    token = secrets.token_hex(32)
    stdout = group.root / ("probe-" + token + ".out")
    stderr = group.root / ("probe-" + token + ".err")
    outcome = await blinded_runtime.run_owned_command(scope=scope, argv=argv, cwd=group.root, env=env, stdout_path=stdout, stderr_path=stderr, deadline_seconds=30)
    if outcome.outcome != "completed" or outcome.exit_code != 0:
        raise ValueError("native tool probe failed")
    raw = blinded_io.read_regular(stdout, maximum=65536)
    if (len(raw), hashlib.sha256(raw).hexdigest()) != (outcome.stdout_size, outcome.stdout_sha256):
        raise ValueError("native tool probe output changed")
    return raw


def _probe_json(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or len(raw) > 65536:
        raise ValueError("native probe byte limit exceeded")
    try:
        text = raw.decode("utf-8", errors="strict")
        try:
            parsed = parse_canonical_model(raw, RootModel[dict[str, object]], maximum=65536)
        except ValueError as error:
            # This public-parser error is emitted only after lexical, duplicate,
            # finite-value and UTF-8 validation. Other refusals stay terminal.
            if str(error) != "evidence is not canonical":
                raise
            normalized = canonical_json_bytes(json.loads(text)) + b"\n"
            parsed = parse_canonical_model(normalized, RootModel[dict[str, object]], maximum=65536)
        result = parsed.root
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError("native probe malformed output") from None
    if type(result) is not dict:
        raise ValueError("native probe object required")
    return result


def _validate_python_probe(data: dict[str, object], *, executable: Path, base_executable: Path, environment: Path | None, required: tuple[str, ...]) -> None:
    if data.get("version") != TOOL_RULES["python"][0]:
        raise ValueError("Python version mismatch")
    for key, expected in (("executable", executable), ("base_executable", base_executable)):
        value = data.get(key)
        if type(value) is not str or Path(value).absolute() != expected.absolute():
            raise ValueError("Python native origin mismatch")
    prefix = data.get("prefix")
    base = data.get("base_prefix")
    if type(prefix) is not str or type(base) is not str:
        raise ValueError("Python prefix observation missing")
    if not base_executable.absolute().is_relative_to(Path(base).absolute()):
        raise ValueError("Python base origin mismatch")
    if environment is None:
        if Path(prefix).absolute() != Path(base).absolute():
            raise ValueError("bootstrap Python must be native base")
    elif Path(prefix).absolute() != environment.absolute() or Path(base).absolute() == environment.absolute():
        raise ValueError("sandbox interpreter prefix mismatch")
    origins = data.get("origins")
    if type(origins) is not dict or set(origins) != set(required):
        raise ValueError("Python import inventory missing")
    allowed = Path(prefix).absolute()
    if environment is not None:
        allowed = environment.absolute() / ("Lib/site-packages" if os.name == "nt" else "lib/python3.12/site-packages")
    locations = data.get("locations")
    versions = data.get("versions")
    if type(locations) is not dict or set(locations) != set(required) or type(versions) is not dict or set(versions) != set(required) or any(type(value) is not str or not value or len(value) > 128 for value in versions.values()):
        raise ValueError("Python package metadata missing")
    for name in required:
        origin = origins[name]
        search = locations.get(name, [])
        if type(search) is not list or any(type(value) is not str or not Path(value).absolute().is_relative_to(allowed) for value in search):
            raise ValueError("Python package outside verified environment")
        if (origin is None and not search) or (origin is not None and (type(origin) is not str or not Path(origin).absolute().is_relative_to(allowed))):
            raise ValueError("Python package outside verified environment")


def _parse_uv_version(raw: bytes) -> str:
    triple = rb"[A-Za-z0-9_]{1,24}(?:-[A-Za-z0-9_.]{1,24}){2,4}"
    revision = rb"[a-f0-9]{7,40} [0-9]{4}-[0-9]{2}-[0-9]{2}"
    if len(raw) > 256 or re.fullmatch(rb"uv " + re.escape(TOOL_RULES['uv'][0].encode('ascii')) + rb"(?: \((?:" + revision + rb"(?: " + triple + rb")?|" + triple + rb")\))?", raw.strip()) is None:
        raise ValueError("uv native version mismatch")
    return TOOL_RULES["uv"][0]


async def _bootstrap_python(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, env: dict[str, str]) -> _ToolSet:
    executable = await asyncio.to_thread(_native_path, Path(getattr(sys, "_base_executable", sys.executable)))
    fingerprint = await asyncio.to_thread(_native_file, executable, "tools/python_bootstrap")
    data = _probe_json(await _probe(scope, group, (str(executable), "-I", "-B", "-c", _PYTHON_PROBE, '["uv"]'), env))
    _validate_python_probe(data, executable=executable, base_executable=executable, environment=None, required=("uv",))
    if data.get("uv_version") != TOOL_RULES["uv"][0]:
        raise ValueError("uv version mismatch")
    files: list[tuple[Path, FileFingerprint]] = [(executable, fingerprint)]
    origins = data["origins"]
    assert isinstance(origins, dict)
    for value in (origins["uv"], data.get("uv_main"), data.get("uv_binary")):
        if type(value) is not str or not Path(value).absolute().is_relative_to(Path(str(data["prefix"])).absolute()):
            raise ValueError("uv origin outside native bootstrap")
    main = Path(str(data["uv_main"]))
    if main != Path(str(origins["uv"])).parent / "__main__.py":
        raise ValueError("uv module origin mismatch")
    module = await asyncio.to_thread(_tool_file, main, "tools/uv_module")
    binary = await asyncio.to_thread(_native_path, Path(str(data["uv_binary"])))
    init = Path(str(origins["uv"]))
    files.extend(((main, module), (init, await asyncio.to_thread(_tool_file, init, "tools/uv_init")), (binary, await asyncio.to_thread(_native_file, binary, "tools/uv_binary"))))
    version = await _probe(scope, group, (str(binary), "--version"), env)
    _parse_uv_version(version)
    bindings = (
        ToolExecutionBinding(role="python_bootstrap", version=TOOL_RULES["python"][0], executable=fingerprint, launcher=None),
        ToolExecutionBinding(role="uv", version=TOOL_RULES["uv"][0], executable=fingerprint, launcher=module),
    )
    result = _ToolSet(executable, None, bindings, tuple(files), Path(str(data["base_prefix"])))
    await asyncio.to_thread(result.verify)
    return result


async def _sandbox_python(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, env: dict[str, str], bootstrap: _ToolSet) -> _ToolSet:
    environment = group.root / "env"
    config_fp = await asyncio.to_thread(_tool_file, environment / "pyvenv.cfg", "tools/python_environment_config")
    base_fp = await asyncio.to_thread(_native_file, bootstrap.executable, "tools/python_base")
    config = blinded_io.read_regular(environment / "pyvenv.cfg", maximum=4096)
    if (len(config), hashlib.sha256(config).hexdigest()) != (config_fp.size, config_fp.sha256):
        raise ValueError("sandbox configuration changed before probe")
    if re.search(rb"(?m)^include-system-site-packages\s*=\s*false\s*$", config) is None:
        raise ValueError("sandbox Python must exclude global site packages")
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if os.name != "nt":
        target = executable.resolve(strict=True)
        if target != bootstrap.executable:
            raise ValueError("sandbox interpreter native target mismatch")
        fingerprint = await asyncio.to_thread(_native_file, target, "tools/python_sandbox")
    else:
        fingerprint = await asyncio.to_thread(_native_file, executable, "tools/python_sandbox")
    data = _probe_json(await _probe(scope, group, (str(executable), "-I", "-B", "-c", _PYTHON_PROBE, json.dumps(_BACKEND_IMPORTS)), env))
    _validate_python_probe(data, executable=executable, base_executable=bootstrap.executable, environment=environment, required=_BACKEND_IMPORTS)
    if bootstrap.base_prefix is None or Path(str(data["base_prefix"])) != bootstrap.base_prefix:
        raise ValueError("sandbox base prefix differs from verified bootstrap")
    origins = data["origins"]
    assert isinstance(origins, dict)
    for origin in origins.values():
        if origin is None:
            continue  # Namespace search locations were checked by the probe validator.
        materialization._directory(Path(str(origin)).parent)
        await asyncio.to_thread(blinded_io.fingerprint_regular, Path(str(origin)), maximum=_MAX_TOOL_BYTES)
    native_target = executable.resolve(strict=True) if os.name != "nt" else executable
    result = _ToolSet(executable, None, (ToolExecutionBinding(role="python", version=TOOL_RULES["python"][0], executable=fingerprint, launcher=None),),
                      ((native_target, fingerprint), (environment / "pyvenv.cfg", config_fp), (bootstrap.executable, base_fp)))
    await asyncio.to_thread(result.verify)
    await asyncio.to_thread(bootstrap.verify)
    return result


def _installed_node() -> Path:
    value = shutil.which("node.exe" if os.name == "nt" else "node")
    if value is None:
        raise ValueError("installed native Node unavailable")
    return _native_path(Path(value))


def _pnpm_package(group: _ExecutionGroup) -> tuple[Path, Path]:
    root = group.root / "npm-cache" / "_npx"
    materialization._directory(root)
    selected: list[tuple[Path, Path]] = []
    with os.scandir(root) as entries:
        for index, entry in enumerate(entries):
            if index >= 64:
                raise ValueError("npm cache inventory exceeded")
            directory = Path(entry.path)
            materialization._directory(directory)
            package = directory / "node_modules" / "pnpm" / "package.json"
            if not package.exists():
                continue
            materialization._directory(package.parent)
            data = _probe_json(blinded_io.read_regular(package, maximum=65536))
            if data.get("name") == "pnpm" and data.get("version") == TOOL_RULES["pnpm"][0]:
                selected.append((package, package.parent / "bin" / "pnpm.cjs"))
    if len(selected) != 1:
        raise ValueError("pinned pnpm launcher origin ambiguous or absent")
    return selected[0]


async def _frontend_tools(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, env: dict[str, str], executable: Path) -> _ToolSet:
    node = await asyncio.to_thread(_native_file, executable, "tools/node")
    npm_root = executable.parent / "node_modules" / "npm" if os.name == "nt" else executable.parent.parent / "lib" / "node_modules" / "npm"
    npx = npm_root / "bin" / "npx-cli.js"
    package = npm_root / "package.json"
    npm = _probe_json(blinded_io.read_regular(package, maximum=65536))
    if npm.get("name") != "npm" or type(npm.get("version")) is not str or re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(npm["version"])) is None:
        raise ValueError("installed npm metadata invalid")
    launcher = await asyncio.to_thread(_tool_file, npx, "tools/npx_cli")
    initial = ((executable, node), (npx, launcher), (package, await asyncio.to_thread(_tool_file, package, "tools/npm_package")))
    data = _probe_json(await _probe(scope, group, (str(executable), "-e", "console.log(JSON.stringify({version:process.versions.node,executable:process.execPath}))"), env))
    if data.get("version") != TOOL_RULES["node"][0] or type(data.get("executable")) is not str or Path(str(data["executable"])) != executable:
        raise ValueError("Node native origin or version mismatch")
    if (await _probe(scope, group, (str(executable), str(npx), "--version"), env)).strip().decode("ascii") != npm["version"]:
        raise ValueError("npx version mismatch")
    npmrc = (
        "store-dir=" + (group.root / "pnpm-store").as_posix() + "\n"
        "cache-dir=" + (group.root / "pnpm-cache").as_posix() + "\n"
        "state-dir=" + (group.root / "pnpm-state").as_posix() + "\n"
        "child-concurrency=1\nnetwork-concurrency=2\nengine-strict=true\nignore-scripts=true\n"
    ).encode("utf-8")
    blinded_io.write_exclusive(group.source / "frontend" / ".npmrc", npmrc)
    if (await _probe(scope, group, (str(executable), str(npx), "-y", "pnpm@" + TOOL_RULES["pnpm"][0], "--version"), env)).strip() != TOOL_RULES["pnpm"][0].encode("ascii"):
        raise ValueError("pinned pnpm bootstrap version mismatch")
    pnpm_package, pnpm = await asyncio.to_thread(_pnpm_package, group)
    pnpm_launcher = await asyncio.to_thread(_tool_file, pnpm, "tools/pnpm_cjs")
    implementation = pnpm_package.parent / "dist" / "pnpm.cjs"
    implementation_fp = await asyncio.to_thread(_tool_file, implementation, "tools/pnpm_implementation")
    if (await _probe(scope, group, (str(executable), str(pnpm), "--version"), env)).strip() != TOOL_RULES["pnpm"][0].encode("ascii"):
        raise ValueError("pinned pnpm native launcher failed")
    # Other executed/read parts of pinned pnpm live in the group-writable npm
    # cache: the install worker (tarball extraction/integrity) and the builtin
    # rc file loaded on every run. They are rehashed around every command too.
    extras = []
    for name, alias in (("worker.js", "tools/pnpm_worker"), ("pnpmrc", "tools/pnpm_builtin_rc")):
        path = pnpm_package.parent / "dist" / name
        extras.append((path, await asyncio.to_thread(_tool_file, path, alias)))
    result = _ToolSet(executable, npx, (
        ToolExecutionBinding(role="node", version=TOOL_RULES["node"][0], executable=node, launcher=None),
        ToolExecutionBinding(role="npx", version=str(npm["version"]), executable=node, launcher=launcher),
        ToolExecutionBinding(role="pnpm", version=TOOL_RULES["pnpm"][0], executable=node, launcher=pnpm_launcher),
    ), (*initial, (pnpm, pnpm_launcher), (implementation, implementation_fp), *extras,
        (pnpm_package, await asyncio.to_thread(_tool_file, pnpm_package, "tools/pnpm_package"))))
    await asyncio.to_thread(result.verify)
    return result


async def _esbuild_preflight(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, tools: _ToolSet, env: dict[str, str]) -> None:
    lock = blinded_io.read_regular(group.source / "frontend" / "pnpm-lock.yaml", maximum=_MAX_SOURCE_BYTES)
    versions = set(re.findall(rb"(?m)^  esbuild@([0-9]+\.[0-9]+\.[0-9]+):", lock))
    if len(versions) != 1:
        raise ValueError("locked esbuild version ambiguous or absent")
    script = (
        "const p=require('node:path'),fs=require('node:fs'),{createRequire}=require('node:module');"
        "const root=fs.realpathSync(p.join(process.argv[1],'node_modules'));"
        "const inside=x=>{const r=p.relative(root,fs.realpathSync(x));if(r==='..'||r.startsWith('..'+p.sep)||p.isAbsolute(r))throw Error('origin');};"
        "const vite=require.resolve('vite/package.json',{paths:[process.argv[1]]});inside(vite);"
        "const req=createRequire(vite);inside(req.resolve('esbuild'));"
        "const e=req('esbuild');const r=e.transformSync('const d39 = 1;', {loader:'js'});"
        "if(!r.code.includes('d39'))process.exit(2);console.log(e.version)"
    )
    raw = await _probe(scope, group, (str(tools.executable), "-e", script, str(group.source / "frontend")), env)
    if raw.strip() != next(iter(versions)):
        raise ValueError("native esbuild does not match frozen lock")


async def _frontend_preflight(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, tools: _ToolSet, env: dict[str, str]) -> None:
    for name in ('pnpm', 'pnpm.cmd', 'pnpm.exe'):
        if os.path.lexists(materialization._fs_path(group.source / 'frontend/node_modules/.bin' / name)):
            raise ValueError('local pnpm could shadow pinned launcher')
    await _esbuild_preflight(scope, group, tools, env)


async def _close_group(group: _ExecutionGroup, scope: blinded_runtime.OwnedProcessScope, entered: bool) -> None:
    failed = False
    source_error: Exception | None = None
    try:
        try:
            if entered:
                await asyncio.shield(scope.close())
        except (OSError, ValueError):
            failed = True
        if failed:
            # DTD: generated entries are removed only after confirmed scope
            # teardown. An unconfirmed descendant may still use the group, so
            # the group is retained and cleanup is reported as failed.
            raise ValueError('owned scope cleanup failed')
        try:
            await asyncio.to_thread(group.assert_source)
        except (OSError, ValueError) as error:
            source_error = error
        finally:
            await asyncio.to_thread(group.cleanup)
    except (OSError, ValueError):
        raise _GroupCleanupFailed('owned group cleanup failed') from None
    finally:
        if group.anchor is not None:
            freeze._close_directory_anchor(group.anchor)
        freeze._close_directory_anchor(group.work_anchor)
    if source_error is not None:
        raise source_error


_MEDIA_PROGRAMS: frozenset[str] = frozenset({"ffmpeg", "ffprobe", "ffplay"})
_EXECUTABLE_SUFFIXES: frozenset[str] = frozenset({".exe", ".com", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".msc"})


def _dedicated_media_directory(directory: Path) -> None:
    """Refuse shim/system directories that would expose other host programs."""
    with os.scandir(directory) as entries:
        for count, entry in enumerate(entries):
            if count >= 4096:
                raise ValueError("backend media directory inventory exceeded")
            if entry.is_dir(follow_symlinks=False):
                continue
            stem, suffix = os.path.splitext(entry.name.lower())
            if os.name == "nt":
                executable = suffix in _EXECUTABLE_SUFFIXES
            else:
                executable = bool(entry.stat(follow_symlinks=True).st_mode & 0o111)
            if executable and (stem if os.name == "nt" else entry.name) not in _MEDIA_PROGRAMS:
                raise ValueError("backend media directory exposes unbound programs")


async def _backend_media(scope: blinded_runtime.OwnedProcessScope, group: _ExecutionGroup, tools: _ToolSet, env: dict[str, str]) -> tuple[_ToolSet, dict[str, str]]:
    """Bind real FFmpeg for backend_pytest only, in a per-command environment copy."""
    files = list(tools.files)
    bindings = []
    directories = []
    for role in ('ffmpeg', 'ffprobe'):
        found = shutil.which(role)
        if found is None:
            raise ValueError('backend media coverage prerequisite missing')
        executable = await asyncio.to_thread(_native_path, Path(found))
        await asyncio.to_thread(_dedicated_media_directory, executable.parent)
        fingerprint = await asyncio.to_thread(_native_file, executable, 'tools/' + role)
        lines = (await _probe(scope, group, (str(executable), '-version'), env)).decode('utf-8').splitlines()
        if not lines:
            raise ValueError('backend media version unavailable')
        bindings.append(ToolExecutionBinding(role=role, version=lines[0][:512], executable=fingerprint, launcher=None))
        files.append((executable, fingerprint))
        directories.append(str(executable.parent))
    # The sandbox interpreter directory stays first; media directories follow it.
    first, *rest = env['PATH'].split(os.pathsep)
    command_env = {**env, 'PATH': os.pathsep.join((first, *dict.fromkeys(directories), *rest))}
    for role, (executable, _) in zip(('ffmpeg', 'ffprobe'), files[-2:], strict=True):
        selected = shutil.which(role, path=command_env['PATH'])
        if selected is None or _native_path(Path(selected)) != executable:
            raise ValueError('backend media PATH binding mismatch')
    return replace(tools, files=tuple(files), media_bindings=tuple(bindings)), command_env


def _command_target(index: int, tools: _ToolSet) -> tuple[tuple[str, ...], tuple[str, ...]]:
    logical = D39_REQUIRED_COMMANDS[index][1]
    if index < 5:
        alias = "tools/python_bootstrap" if index == 0 else "tools/python_sandbox"
        return (str(tools.executable), *logical[1:]), (alias, *logical[1:])
    if tools.launcher is None:
        raise ValueError("native npx launcher unavailable")
    return (str(tools.executable), str(tools.launcher), *logical[1:]), ("tools/node", "tools/npx_cli", *logical[1:])


async def _execute_command(*, index: int, group: _ExecutionGroup, scope: blinded_runtime.OwnedProcessScope, tools: _ToolSet, env: dict[str, str], commands: list[CommandEvidence]) -> CommandEvidence:
    await asyncio.to_thread(group.assert_source)
    await asyncio.to_thread(tools.verify)
    argv, aliases = _command_target(index, tools)
    name, canonical = D39_REQUIRED_COMMANDS[index]
    cwd = "backend" if index < 5 else "frontend"
    expected = ("python_bootstrap", "uv") if index == 0 else ("python",) if index < 5 else ("node", "npx", "pnpm")
    if tuple(item.role for item in tools.bindings) != expected:
        raise ValueError("command native binding inventory incomplete")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    CommandEvidence(name=name, argv=canonical, resolved_argv=aliases, tool_bindings=tools.bindings, cwd=cwd, deadline_seconds=D39_COMMAND_DEADLINES[index], outcome="launch_failed", exit_code=None, started_at=timestamp, finished_at=timestamp, stdout_size=0, stderr_size=0, stdout_sha256=hashlib.sha256(b"").hexdigest(), stderr_sha256=hashlib.sha256(b"").hexdigest())
    environment = {**env, "PYTHONSAFEPATH": "1"} if index == 0 else env
    observed = await blinded_runtime.run_owned_command(scope=scope, argv=argv, cwd=group.source / cwd, env=environment, stdout_path=group.root / (name + ".out"), stderr_path=group.root / (name + ".err"), deadline_seconds=D39_COMMAND_DEADLINES[index])
    fields = dict(observed.__dict__)
    fields['finished_at'] = max(fields['started_at'], fields['finished_at'])
    result = CommandEvidence(name=name, argv=canonical, resolved_argv=aliases, tool_bindings=tools.bindings, media_tools=tools.media_bindings if index == 2 else (), cwd=cwd, deadline_seconds=D39_COMMAND_DEADLINES[index], **fields)
    commands.append(result)
    await asyncio.to_thread(group.assert_source)
    await asyncio.to_thread(tools.verify)
    return result


def _candidate_binding(frozen: FreezeManifest, digest: str, snapshot: str, record: RuntimeMaterialization) -> None:
    if (frozen.candidate_id, frozen.git_commit, digest, snapshot, tuple(frozen.files)) != (record.candidate_id, record.git_commit, record.freeze_sha256, record.candidate_snapshot_sha256, record.files):
        raise ValueError("candidate materialization binding mismatch")


def _inventory_boundary(candidate: Path, runtime: Path, work: Path, path: Path, digest: str, record: RuntimeMaterialization, tool_root: Path, attestation: ToolAttestation, frozen_path: Path) -> None:
    actual = materialization.read_materialized_runtime(runtime_root=runtime, work_root=work, materialization_path=path, expected_materialization_sha256=digest)
    if actual != record:
        raise ValueError("runtime group boundary drift")
    frozen, freeze_digest, snapshot = materialization._freeze_inputs(candidate, frozen_path)
    _candidate_binding(frozen, freeze_digest, snapshot, record)
    if _attest_verifier(tool_root) != attestation:
        raise ValueError("verifier source boundary drift")


async def _run_inventory(*, candidate: Path, runtime: Path, work: Path, path: Path, digest: str, record: RuntimeMaterialization, tool_root: Path, attestation: ToolAttestation, commands: list[CommandEvidence], frozen_path: Path) -> None:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix="d39-check"))
    async def boundary() -> None:
        await asyncio.to_thread(_inventory_boundary, candidate, runtime, work, path, digest, record, tool_root, attestation, frozen_path)
    for indices in (range(0, 5), range(5, 9)):
        await boundary()
        # Constructed before the group: an invalid D39_AGENT_PID refuses without
        # leaving a group behind.
        scope = blinded_runtime.OwnedProcessScope()
        group = await asyncio.to_thread(_ExecutionGroup, work, runtime, record)
        entered = False
        try:
            python = await asyncio.to_thread(_native_path, Path(getattr(sys, "_base_executable", sys.executable)))
            node = await asyncio.to_thread(_installed_node) if indices.start == 5 else None
            env = await asyncio.to_thread(build_group_environment, group.root, python_executable=python, node_executable=node)
            await scope.__aenter__()
            entered = True
            if indices.start == 0:
                tools = await _bootstrap_python(scope, group, env)
            else:
                assert node is not None
                tools = await _frontend_tools(scope, group, env, node)
            for index in indices:
                selected, command_env = await _backend_media(scope, group, tools, env) if index == 2 else (tools, env)
                result = await _execute_command(index=index, group=group, scope=scope, tools=selected, env=command_env, commands=commands)
                if result.outcome != "completed" or result.exit_code != 0:
                    raise ValueError("verification command failed")
                if index == 0:
                    tools = await _sandbox_python(scope, group, env, tools)
                elif index == 5:
                    await _frontend_preflight(scope, group, tools, env)
            await asyncio.to_thread(group.assert_source)
        finally:
            await _close_group(group, scope, entered)
        await boundary()


__all__ = ["D39_COMMAND_DEADLINES", "D39_REQUIRED_COMMANDS", "CommandEvidence", "SmokeManifest", "ToolExecutionBinding", "VerificationManifest", "VERIFIER_SOURCE_PATHS", "verify_release_candidate"]
