"""Strict canonical source attestations for external evaluation tools."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REPARSE_POINT = 0x400
_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_TOTAL_BYTES = 512 * 1024 * 1024
_MAX_GIT_METADATA_BYTES = 16 * 1024 * 1024
_BLINDED_REQUIRED_PATHS = (
    "backend/app/operations/definitions.json",
    "backend/pyproject.toml",
    "backend/scripts/run_blinded_evaluation.py",
    "backend/uv.lock",
)


class FileFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    size: Annotated[int, Field(ge=0)]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if not value or "\\" in value or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("fingerprint path must be a lexical relative POSIX path")
        if parsed.as_posix() != value or any(part in ("", ".") for part in parsed.parts):
            raise ValueError("fingerprint path must be normalized")
        return value


class ToolAttestation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)]
    tool_name: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9_]+$")]
    git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    files: Annotated[list[FileFingerprint], Field(min_length=1, max_length=8192)]
    aggregate_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[FileFingerprint]) -> list[FileFingerprint]:
        paths = [item.path for item in value]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("attestation files must be non-empty, unique, and sorted")
        return value


def canonical_json_bytes(value: object) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def fingerprint_file(repo_root: Path, relative_path: str) -> FileFingerprint:
    FileFingerprint(path=relative_path, sha256="0" * 64, size=0)
    path = repo_root / Path(*PurePosixPath(relative_path).parts)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"tool source is not a regular file: {relative_path}")
    if metadata.st_size > _MAX_SOURCE_BYTES:
        raise ValueError("tool source exceeds its size limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse(opened):
            raise ValueError(f"tool source is not a regular file: {relative_path}")
        if ((metadata.st_dev, metadata.st_ino, metadata.st_size)
                != (opened.st_dev, opened.st_ino, opened.st_size)):
            raise ValueError("tool source changed while opening")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            size += len(chunk)
            if size > _MAX_SOURCE_BYTES:
                raise ValueError("tool source exceeds its size limit")
            digest.update(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
    opened_identity = (opened.st_dev, opened.st_ino, opened.st_size)
    final_identity = (final.st_dev, final.st_ino, final.st_size)
    after = path.lstat()
    after_identity = (after.st_dev, after.st_ino, after.st_size)
    if (identity != opened_identity or opened_identity != final_identity
            or final_identity != after_identity or size != opened.st_size
            or stat.S_ISLNK(after.st_mode) or _is_reparse(after) or not stat.S_ISREG(after.st_mode)):
        raise ValueError(f"tool source changed while hashing: {relative_path}")
    return FileFingerprint(path=relative_path, sha256=digest.hexdigest(), size=size)


def aggregate_fingerprints(files: tuple[FileFingerprint, ...] | list[FileFingerprint]) -> str:
    payload = [item.model_dump(mode="json") for item in files]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=False,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    if completed.returncode != 0:
        raise ValueError("Git repository validation failed")
    return completed.stdout


def validate_git_repository(
    repo_root: Path, *, expected_commit: str | None = None, require_clean: bool = True
) -> str:
    root = repo_root.resolve(strict=True)
    reported = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve(
        strict=True
    )
    if reported != root:
        raise ValueError("repository root does not match the declared root")
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("repository HEAD is invalid")
    if expected_commit is not None and head != expected_commit:
        raise ValueError("declared Git commit does not match repository HEAD")
    if require_clean and _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("repository must be clean at attestation time")

    tagged = _git(root, "ls-files", "-v", "-z", "--cached").split(b"\0")
    tracked_count = 0
    for entry in tagged:
        if not entry:
            continue
        tracked_count += 1
        if not entry.startswith(b"H "):
            raise ValueError("repository contains forbidden index flags")
    debug = _git(root, "ls-files", "--debug", "-z", "--cached")
    index_flags = re.findall(rb"\tflags: ([0-9]+)\n", debug)
    if len(index_flags) != tracked_count or any(flag != b"0" for flag in index_flags):
        raise ValueError("repository contains forbidden index flags")
    staged = _git(root, "ls-files", "--stage", "-z", "--cached").split(b"\0")
    for entry in staged:
        if not entry:
            continue
        metadata, separator, _ = entry.partition(b"\t")
        fields = metadata.split()
        if (
            separator != b"\t"
            or len(fields) != 3
            or fields[0] not in {b"100644", b"100755"}
            or fields[2] != b"0"
        ):
            raise ValueError("repository contains sparse or special index entries")
    return head


def fingerprint_committed_file(
    repo_root: Path, git_commit: str, relative_path: str
) -> FileFingerprint:
    working = fingerprint_file(repo_root, relative_path)
    committed = _git(repo_root, "show", f"{git_commit}:{relative_path}")
    committed_fingerprint = FileFingerprint(
        path=relative_path,
        sha256=hashlib.sha256(committed).hexdigest(),
        size=len(committed),
    )
    if working != committed_fingerprint:
        raise ValueError(f"working file does not match committed blob: {relative_path}")
    return committed_fingerprint


def _bounded_git(repo_root: Path, *arguments: str, maximum: int) -> bytes:
    pathspec_flags = ["--literal-pathspecs"] if arguments[0] in {"ls-tree", "ls-files"} else []
    with subprocess.Popen(
        ["git", "--no-replace-objects", *pathspec_flags, "-C", str(repo_root), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    ) as process:
        try:
            assert process.stdout is not None
            raw = process.stdout.read(maximum + 1)
            if len(raw) > maximum:
                raise ValueError("historical Git output exceeds its size limit")
            if process.wait(timeout=60) != 0:
                raise ValueError("historical Git objects are unavailable")
            return raw
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def _historical_tree(
    repo_root: Path, git_commit: str, paths: tuple[str, ...],
) -> dict[str, tuple[bytes, bytes, str]]:
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ValueError("historical commit is invalid")
    root = repo_root.resolve(strict=True)
    reported = _bounded_git(root, "rev-parse", "--show-toplevel",
                            maximum=_MAX_GIT_METADATA_BYTES)
    if Path(reported.decode("utf-8").strip()).resolve(strict=True) != root:
        raise ValueError("repository root does not match the declared root")
    if _bounded_git(root, "cat-file", "-t", git_commit, maximum=128) != b"commit\n":
        raise ValueError("historical object is not a commit")
    raw = _bounded_git(root, "ls-tree", "-r", "-z", "--full-tree", git_commit, "--", *paths,
                       maximum=_MAX_GIT_METADATA_BYTES)
    if raw and not raw.endswith(b"\0"):
        raise ValueError("historical tree metadata is invalid")
    entries: dict[str, tuple[bytes, bytes, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, separator, raw_path = entry.partition(b"\t")
        fields = metadata.split(b" ")
        if separator != b"\t" or len(fields) != 3 or not re.fullmatch(rb"[0-9a-f]{40}", fields[2]):
            raise ValueError("historical tree metadata is invalid")
        path = raw_path.decode("utf-8", errors="strict")
        FileFingerprint(path=path, size=0, sha256="0" * 64)
        if path in entries:
            raise ValueError("historical tree metadata contains duplicate paths")
        entries[path] = (fields[0], fields[1], fields[2].decode("ascii"))
    return entries


def _require_regular_blob(entry: tuple[bytes, bytes, str]) -> str:
    mode, kind, object_id = entry
    if mode not in {b"100644", b"100755"} or kind != b"blob":
        raise ValueError("historical source must be a regular Git blob")
    return object_id


def historical_blinded_source_paths(*, repo_root: Path, git_commit: str) -> tuple[str, ...]:
    """Derive D37's exact inventory from its recorded tree without checking it out."""
    entries = _historical_tree(repo_root, git_commit, (
        "backend/evaluation", "backend/app", *_BLINDED_REQUIRED_PATHS,
    ))
    selected = {path for path in entries if path.endswith(".py") and path.startswith(
        ("backend/evaluation/", "backend/app/")
    )} | set(_BLINDED_REQUIRED_PATHS)
    for path in selected:
        if path not in entries:
            raise ValueError("historical inventory is missing a required source")
        _require_regular_blob(entries[path])
    return tuple(sorted(selected))


def verify_historical_attestation(
    *, repo_root: Path, attestation: ToolAttestation,
    expected_tool_name: str, source_paths: tuple[str, ...],
) -> None:
    """Verify source identities at the recorded commit, independent of live bytes/HEAD."""
    if (not source_paths or source_paths != tuple(sorted(set(source_paths)))
            or attestation.tool_name != expected_tool_name
            or tuple(item.path for item in attestation.files) != source_paths):
        raise ValueError("historical source inventory or tool identity mismatch")
    for path in source_paths:
        FileFingerprint(path=path, size=0, sha256="0" * 64)
    entries = _historical_tree(repo_root, attestation.git_commit, source_paths)
    if set(entries) != set(source_paths):
        raise ValueError("historical source inventory mismatch")
    files: list[FileFingerprint] = []
    total = 0
    for path in source_paths:
        object_id = _require_regular_blob(entries[path])
        size_raw = _bounded_git(repo_root, "cat-file", "-s", object_id, maximum=128)
        if not re.fullmatch(rb"[0-9]+\n", size_raw):
            raise ValueError("historical blob size is invalid")
        size = int(size_raw)
        total += size
        if size > _MAX_SOURCE_BYTES or total > _MAX_SOURCE_TOTAL_BYTES:
            raise ValueError("historical source exceeds its size limit")
        raw = _bounded_git(repo_root, "cat-file", "blob", object_id, maximum=size)
        if len(raw) != size:
            raise ValueError("historical blob size changed")
        files.append(FileFingerprint(path=path, size=size, sha256=hashlib.sha256(raw).hexdigest()))
    if files != attestation.files or aggregate_fingerprints(files) != attestation.aggregate_sha256:
        raise ValueError("historical source fingerprint mismatch")


def attest_tool(
    *, repo_root: Path, tool_name: str, git_commit: str, source_paths: tuple[str, ...]
) -> ToolAttestation:
    if tuple(sorted(source_paths)) != source_paths or len(source_paths) != len(set(source_paths)):
        raise ValueError("tool source allowlist must be unique and sorted")
    root = repo_root.resolve(strict=True)
    head = validate_git_repository(root, expected_commit=git_commit)
    files: list[FileFingerprint] = []
    total = 0
    for path in source_paths:
        fingerprint = fingerprint_committed_file(root, head, path)
        total += fingerprint.size
        if total > _MAX_SOURCE_TOTAL_BYTES:
            raise ValueError("tool source inventory exceeds its size limit")
        files.append(fingerprint)
    validate_git_repository(root, expected_commit=git_commit)
    if [fingerprint_committed_file(root, head, path) for path in source_paths] != files:
        raise ValueError("tool source changed while attesting")
    return ToolAttestation(
        schema_version=1,
        tool_name=tool_name,
        git_commit=head,
        files=files,
        aggregate_sha256=aggregate_fingerprints(files),
    )
