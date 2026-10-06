"""Filesystem validation and crash-safe publication for the blinded evaluator."""
from __future__ import annotations

import hashlib
import os
import secrets
import stat
import sys
from pathlib import Path

DEFAULT_JSON_BYTES = 16 * 1024 * 1024
_REPARSE_POINT = 0x400


def is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def validate_directory(path: Path, description: str) -> Path:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or is_reparse(metadata) or not stat.S_ISDIR(
        metadata.st_mode
    ):
        raise ValueError(f"{description} must be a non-symlink, non-reparse directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise ValueError(f"{description} has an unsafe path component")
    return resolved


def create_directory_tree(path: Path, description: str) -> Path:
    absolute = path.absolute()
    missing: list[str] = []
    current = absolute
    while True:
        try:
            validate_directory(current, description)
            break
        except FileNotFoundError:
            if current.parent == current:
                raise ValueError(f"{description} has no existing directory ancestor") from None
            missing.append(current.name)
            current = current.parent
    for name in reversed(missing):
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"{description} contains an unsafe component")
        child = current / name
        try:
            child.mkdir()
        except FileExistsError:
            pass
        current = validate_directory(child, description)
    return validate_directory(absolute, description)


def require_directory(path: Path, description: str, *, create: bool = False) -> Path:
    return (
        create_directory_tree(path, description)
        if create
        else validate_directory(path.absolute(), description)
    )


def ensure_writable_directory(root: Path, *parts: str) -> Path:
    current = validate_directory(root.absolute(), "evaluation output")
    for part in parts:
        if not part or part in {".", ".."} or Path(part).name != part:
            raise ValueError("evaluation output directory component is invalid")
        child = current / part
        try:
            child.mkdir()
        except FileExistsError:
            pass
        current = validate_directory(child, "evaluation output directory")
    return current


def read_regular(path: Path, *, maximum: int = DEFAULT_JSON_BYTES) -> bytes:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or is_reparse(metadata) or not stat.S_ISREG(
        metadata.st_mode
    ):
        raise ValueError("input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or is_reparse(opened)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        ):
            raise ValueError("input identity changed")
        value = os.read(descriptor, maximum + 1)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identities = {
        (metadata.st_dev, metadata.st_ino, metadata.st_size),
        (opened.st_dev, opened.st_ino, opened.st_size),
        (final.st_dev, final.st_ino, final.st_size),
        (after.st_dev, after.st_ino, after.st_size),
    }
    if (
        len(value) > maximum
        or len(value) != opened.st_size
        or len(identities) != 1
        or stat.S_ISLNK(after.st_mode)
        or is_reparse(after)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ValueError("input changed or exceeds its size limit")
    return value


def fingerprint_regular(path: Path, *, maximum: int) -> tuple[int, str]:
    """Stream a bounded blob, rejecting link, identity, size, or timestamp changes."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError("blob must be a regular non-link file")
    if before.st_size > maximum:
        raise ValueError("blob exceeds its size limit")

    def identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or is_reparse(opened)
                or identity(before) != identity(opened)):
            raise ValueError("blob identity changed")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ValueError("blob exceeds its size limit")
            digest.update(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if (stat.S_ISLNK(after.st_mode) or is_reparse(after) or not stat.S_ISREG(after.st_mode)
            or len({identity(before), identity(opened), identity(final), identity(after)}) != 1
            or before.st_ctime_ns != after.st_ctime_ns
            or opened.st_ctime_ns != final.st_ctime_ns
            or size != before.st_size):
        raise ValueError("blob changed while hashing")
    return size, digest.hexdigest()


def fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path: Path, value: bytes) -> None:
    validate_directory(path.parent.absolute(), "exclusive output parent")
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_atomic(path: Path, value: bytes) -> None:
    parent = validate_directory(path.parent.absolute(), "output parent")
    try:
        current = path.lstat()
    except FileNotFoundError:
        current = None
    if current is not None and (
        stat.S_ISLNK(current.st_mode)
        or is_reparse(current)
        or not stat.S_ISREG(current.st_mode)
    ):
        raise ValueError("output must be a regular non-link file")
    temporary = parent / f".{path.name}.tmp-{secrets.token_hex(32)}"
    descriptor = os.open(
        temporary,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, path)
        fsync_directory(parent)
        if read_regular(path, maximum=len(value)) != value:
            raise ValueError("atomic output changed after publication")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def publish_immutable(
    path: Path,
    value: bytes,
    description: str,
    *,
    maximum: int = DEFAULT_JSON_BYTES,
) -> Path:
    if len(value) > maximum:
        raise ValueError(f"{description} exceeds its maximum canonical size")
    parent = validate_directory(path.parent.absolute(), f"{description} parent")
    if path.parent.absolute() != parent:
        raise ValueError(f"{description} parent identity changed")
    temporary = parent / f".{path.name}.tmp-{secrets.token_hex(32)}"
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or is_reparse(opened):
            raise ValueError(f"{description} temporary is not a regular file")
        offset = 0
        while offset < len(value):
            offset += os.write(descriptor, value[offset:])
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (final.st_dev, final.st_ino)
            or final.st_size != len(value)
        ):
            raise ValueError(f"{description} temporary identity changed")
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            try:
                existing = read_regular(path, maximum=maximum)
            except (OSError, ValueError) as error:
                raise ValueError(f"existing {description} is invalid") from error
            if existing != value:
                raise ValueError(
                    f"existing {description} bytes do not match this run"
                ) from None
        else:
            temporary_stat = temporary.lstat()
            published_stat = path.lstat()
            if (
                stat.S_ISLNK(published_stat.st_mode)
                or is_reparse(published_stat)
                or not stat.S_ISREG(published_stat.st_mode)
                or (temporary_stat.st_dev, temporary_stat.st_ino)
                != (published_stat.st_dev, published_stat.st_ino)
            ):
                raise ValueError(f"{description} publication identity changed")
            fsync_directory(parent)
        if read_regular(path, maximum=maximum) != value:
            raise ValueError(f"{description} bytes changed after publication")
        return path
    finally:
        try:
            metadata = temporary.lstat()
            if stat.S_ISREG(metadata.st_mode) and not is_reparse(metadata):
                temporary.unlink()
                fsync_directory(parent)
        except FileNotFoundError:
            pass


def write_or_validate_immutable(
    path: Path,
    expected: bytes,
    description: str,
    *,
    maximum: int = DEFAULT_JSON_BYTES,
) -> Path:
    return publish_immutable(path, expected, description, maximum=maximum)


def publish_accepted_triplet(
    *, output_dir: Path, accepted_result_bytes: bytes,
    validation_bytes: bytes, tool_attestation_bytes: bytes,
) -> None:
    """Publish exactly three retained, byte-verified files by native no-replace rename."""
    from evaluation.release_candidate import freeze

    if os.name != "nt" and not sys.platform.startswith("linux"):
        raise ValueError("native no-replace publication is unavailable")
    values = {
        "accepted-result.json": accepted_result_bytes,
        "validation.json": validation_bytes,
        "d38-tool-attestation.json": tool_attestation_bytes,
    }
    for name, value in values.items():
        maximum = 128 * 1024 * 1024 if name == "accepted-result.json" else DEFAULT_JSON_BYTES
        if type(value) is not bytes or not value or len(value) > maximum:
            raise ValueError("accepted triplet bytes are invalid")
    final = output_dir.absolute()
    parent = validate_directory(final.parent, "accepted triplet parent")
    if final.name in {"", ".", ".."}:
        raise ValueError("accepted triplet destination is invalid")
    try:
        final.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("accepted triplet destination already exists")
    parent_anchor = freeze._open_directory_anchor(parent)
    stage = parent / f".d38-stage-{secrets.token_hex(32)}"
    stage_anchor = None
    retained: dict[str, tuple[int, tuple[int, int]]] = {}
    closed_files = False

    def assert_parent() -> None:
        if (freeze._anchor_identity(parent_anchor) != parent_anchor.identity
                or freeze._path_directory_identity(parent) != parent_anchor.identity
                or validate_directory(parent, "accepted triplet parent") != parent):
            raise ValueError("accepted triplet parent ownership lost")

    def assert_files(directory: Path) -> None:
        assert_parent()
        assert stage_anchor is not None
        if freeze._path_directory_identity(directory) != stage_anchor.identity:
            raise ValueError("accepted triplet stage ownership lost")
        descriptor = stage_anchor.descriptor
        if not stage_anchor.closed and freeze._anchor_identity(stage_anchor) != stage_anchor.identity:
            raise ValueError("accepted triplet stage ownership lost")
        entries = set(os.listdir(descriptor if descriptor is not None else directory))
        if entries != set(values):
            raise ValueError("accepted triplet entries are invalid")
        for name, (file_descriptor, identity) in retained.items():
            metadata = (os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                        if descriptor is not None else (directory / name).lstat())
            if (not stat.S_ISREG(metadata.st_mode) or is_reparse(metadata)
                    or (metadata.st_dev, metadata.st_ino) != identity
                    or metadata.st_size != len(values[name])):
                raise ValueError("accepted triplet file ownership lost")
            if not closed_files:
                opened = os.fstat(file_descriptor)
                if (not stat.S_ISREG(opened.st_mode) or is_reparse(opened)
                        or (opened.st_dev, opened.st_ino) != identity
                        or opened.st_size != len(values[name])):
                    raise ValueError("accepted triplet file ownership lost")
                os.lseek(file_descriptor, 0, os.SEEK_SET)
                remaining = memoryview(values[name])
                while remaining:
                    chunk = os.read(file_descriptor, min(len(remaining), 65536))
                    if not chunk or chunk != remaining[:len(chunk)]:
                        raise ValueError("accepted triplet readback mismatch")
                    remaining = remaining[len(chunk):]
                if os.read(file_descriptor, 1):
                    raise ValueError("accepted triplet readback mismatch")
            elif read_regular(directory / name, maximum=len(values[name])) != values[name]:
                raise ValueError("accepted triplet readback mismatch")

    try:
        assert_parent()
        if parent_anchor.descriptor is not None:
            os.mkdir(stage.name, mode=0o700, dir_fd=parent_anchor.descriptor)
        else:
            stage.mkdir(mode=0o700)
        assert_parent()
        stage_anchor = freeze._open_directory_anchor(stage)
        if freeze._path_directory_identity(stage) != stage_anchor.identity:
            raise ValueError("accepted triplet stage ownership lost")
        for name, value in values.items():
            assert_parent()
            if freeze._path_directory_identity(stage) != stage_anchor.identity:
                raise ValueError("accepted triplet stage ownership lost")
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = (os.open(name, flags, 0o600, dir_fd=stage_anchor.descriptor)
                          if stage_anchor.descriptor is not None else os.open(stage / name, flags, 0o600))
            metadata = os.fstat(descriptor)
            retained[name] = (descriptor, (metadata.st_dev, metadata.st_ino))
            if not stat.S_ISREG(metadata.st_mode) or is_reparse(metadata):
                raise ValueError("accepted triplet file must be regular")
            remaining = memoryview(value)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise ValueError("accepted triplet write failed")
                remaining = remaining[written:]
            os.fsync(descriptor)
        assert_files(stage)
        if stage_anchor.descriptor is not None:
            assert parent_anchor.descriptor is not None
            os.fsync(stage_anchor.descriptor)
            parent_alias = Path(f"/proc/self/fd/{parent_anchor.descriptor}")
            stage_alias = Path(f"/proc/self/fd/{stage_anchor.descriptor}")
            for alias, anchor in ((parent_alias, parent_anchor), (stage_alias, stage_anchor)):
                metadata = alias.stat()
                if (not stat.S_ISDIR(metadata.st_mode)
                        or (metadata.st_dev, metadata.st_ino) != anchor.identity):
                    raise ValueError("accepted triplet descriptor aliases are unavailable")
            assert_files(stage)
            freeze._linux_rename_directory_no_replace(parent_alias / stage.name, parent_alias / final.name)
            assert_files(final)
            os.fsync(parent_anchor.descriptor)
        else:
            for descriptor, _ in retained.values():
                os.close(descriptor)
            closed_files = True
            assert_files(stage)
            freeze._close_directory_anchor(stage_anchor)
            assert_parent()
            freeze._windows_move_directory_no_replace(stage, final)
            assert_files(final)
    finally:
        if not closed_files:
            for descriptor, _ in retained.values():
                os.close(descriptor)
        if stage_anchor is not None:
            freeze._close_directory_anchor(stage_anchor)
        freeze._close_directory_anchor(parent_anchor)


def open_exclusive_regular(path: Path, description: str) -> int:
    validate_directory(path.parent.absolute(), f"{description} parent")
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or is_reparse(metadata):
        os.close(descriptor)
        raise ValueError(f"{description} must be a regular file")
    return descriptor
