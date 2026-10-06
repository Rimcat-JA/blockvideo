"""Deterministically fingerprint private D37 detailed evidence without logging contents."""
from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath

from evaluation.tool_attestation import FileFingerprint, aggregate_fingerprints

_REPARSE_POINT = 0x400
_PUBLICATION_TEMP = re.compile(r"^\.[^/\\]+\.tmp-[0-9a-f]{64}$")
_EXCLUDED_OUTPUT_NAMES = frozenset(
    {
        "aggregate.json",
        "evaluation-result.json",
        "partial-result.json",
        "protocol.json",
        "result-bundle.json",
        "tool-attestation.json",
    }
)


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _require_directory(path: Path, description: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"sealed evidence {description} must not be a symlink")
    if _is_reparse(metadata):
        raise ValueError(f"sealed evidence {description} must not be a reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"sealed evidence {description} must be a directory")
    return metadata


def _fingerprint(root: Path, root_resolved: Path, relative: str) -> FileFingerprint:
    FileFingerprint(path=relative, sha256="0" * 64, size=0)
    path = root / Path(*PurePosixPath(relative).parts)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"sealed evidence entry is not a regular file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError("sealed evidence entry escapes its root") from error

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse(opened):
            raise ValueError(f"sealed evidence entry is not a regular file: {relative}")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identities = {
        (metadata.st_dev, metadata.st_ino, metadata.st_size),
        (opened.st_dev, opened.st_ino, opened.st_size),
        (final.st_dev, final.st_ino, final.st_size),
    }
    if len(identities) != 1 or size != opened.st_size:
        raise ValueError(f"sealed evidence entry changed while hashing: {relative}")
    return FileFingerprint(path=relative, sha256=digest.hexdigest(), size=size)


def seal_evidence(root: Path) -> tuple[list[FileFingerprint], str]:
    """Return sorted detailed-file fingerprints and their canonical aggregate SHA-256."""
    _require_directory(root, "root")
    root_resolved = root.resolve(strict=True)
    relative_files: list[str] = []
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        _require_directory(current_path, "directory")
        for directory_name in sorted(directory_names):
            _require_directory(current_path / directory_name, "directory")
        directory_names.sort()
        for file_name in sorted(file_names):
            path = current_path / file_name
            relative = path.relative_to(root).as_posix()
            FileFingerprint(path=relative, sha256="0" * 64, size=0)
            metadata = path.lstat()
            if current_path == root and file_name in _EXCLUDED_OUTPUT_NAMES:
                if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISREG(
                    metadata.st_mode
                ):
                    raise ValueError(f"excluded sealed output is not a regular file: {relative}")
                continue
            if _PUBLICATION_TEMP.fullmatch(file_name):
                if not stat.S_ISREG(metadata.st_mode) or _is_reparse(metadata):
                    raise ValueError(f"publication temporary is not regular: {relative}")
                continue
            relative_files.append(relative)
    if len(relative_files) != len(set(relative_files)):
        raise ValueError("sealed evidence paths must be unique")
    files = [_fingerprint(root, root_resolved, relative) for relative in sorted(relative_files)]
    return files, aggregate_fingerprints(files)
