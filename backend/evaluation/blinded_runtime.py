"""Candidate anchoring and supervised trial-host execution for D37."""
from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

from evaluation import blinded_io

HOST_WATCHDOG_SECONDS = 180 * 4 + 30
HOST_OUTPUT_CAP_BYTES = 2 * 1024 * 1024
HOST_PIPE_DRAIN_GRACE_SECONDS = 1.0
HOST_TEARDOWN_SECONDS = 10.0
_REPARSE_POINT = 0x400
_FILE_SHARE_READ = 0x1
_FILE_SHARE_WRITE = 0x2
_GENERIC_READ = 0x80000000
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


@dataclass
class CandidateAnchor:
    canonical_path: Path
    execution_path: Path
    identity: tuple[int, int]
    descriptor: int | None = None
    handle: int | None = None
    closed: bool = False


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _BasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = [
        ("file_attributes", ctypes.c_uint32),
        ("creation_time_low", ctypes.c_uint32),
        ("creation_time_high", ctypes.c_uint32),
        ("last_access_time_low", ctypes.c_uint32),
        ("last_access_time_high", ctypes.c_uint32),
        ("last_write_time_low", ctypes.c_uint32),
        ("last_write_time_high", ctypes.c_uint32),
        ("volume_serial_number", ctypes.c_uint32),
        ("file_size_high", ctypes.c_uint32),
        ("file_size_low", ctypes.c_uint32),
        ("number_of_links", ctypes.c_uint32),
        ("file_index_high", ctypes.c_uint32),
        ("file_index_low", ctypes.c_uint32),
    ]


def _windows_directory_identity(handle: int) -> tuple[int, int]:
    information = _ByHandleFileInformation()
    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
    get_information.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ByHandleFileInformation)]
    get_information.restype = ctypes.c_int
    if not get_information(ctypes.c_void_p(handle), ctypes.byref(information)):
        error = ctypes.get_last_error()
        raise OSError(error, "GetFileInformationByHandle failed for candidate directory")
    if (
        not information.file_attributes & _FILE_ATTRIBUTE_DIRECTORY
        or information.file_attributes & _REPARSE_POINT
    ):
        raise ValueError("candidate anchor must be a non-reparse directory")
    file_index = (information.file_index_high << 32) | information.file_index_low
    return information.volume_serial_number, file_index


def _windows_open_candidate_directory(path: Path) -> int:
    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    handle = create_file(
        str(path),
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (None, invalid_handle):
        error = ctypes.get_last_error()
        raise OSError(error, "CreateFileW failed for candidate directory")
    return int(handle)


def _windows_close_handle(handle: int) -> None:
    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    if not close_handle(ctypes.c_void_p(handle)):
        error = ctypes.get_last_error()
        raise OSError(error, "CloseHandle failed for candidate directory")


def canonical_candidate_root(candidate_root: Path) -> Path:
    supplied = candidate_root.absolute()
    metadata = supplied.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or blinded_io.is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValueError("candidate root must be a non-link directory")
    resolved = supplied.resolve(strict=True)
    if resolved != supplied:
        raise ValueError("candidate root contains an unsafe path component")
    blinded_io.validate_directory(resolved / "backend", "candidate backend")
    return resolved


def open_candidate_anchor(candidate_root: Path) -> CandidateAnchor:
    canonical = canonical_candidate_root(candidate_root)
    if os.name == "nt":
        handle = _windows_open_candidate_directory(canonical)
        try:
            identity = _windows_directory_identity(handle)
        except BaseException:
            _windows_close_handle(handle)
            raise
        return CandidateAnchor(canonical, canonical, identity, handle=handle)
    if os.name != "posix" or not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("candidate directory anchoring is unsupported on this platform")
    descriptor = os.open(canonical, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or blinded_io.is_reparse(metadata):
            raise ValueError("candidate anchor must be a non-reparse directory")
        execution = Path(f"/proc/{os.getpid()}/fd/{descriptor}")
        if not execution.exists() or execution.resolve(strict=True) != canonical:
            raise ValueError("stable candidate fd path is unavailable on this POSIX host")
        return CandidateAnchor(
            canonical,
            execution,
            (metadata.st_dev, metadata.st_ino),
            descriptor=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise


def close_candidate_anchor(anchor: CandidateAnchor) -> None:
    if anchor.closed:
        return
    anchor.closed = True
    if anchor.handle is not None:
        handle = anchor.handle
        anchor.handle = None
        _windows_close_handle(handle)
    if anchor.descriptor is not None:
        descriptor = anchor.descriptor
        anchor.descriptor = None
        os.close(descriptor)


def _candidate_path_identity(path: Path) -> tuple[int, int]:
    if os.name == "nt":
        handle = _windows_open_candidate_directory(path)
        try:
            return _windows_directory_identity(handle)
        finally:
            _windows_close_handle(handle)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or blinded_io.is_reparse(metadata)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise ValueError("candidate root must remain a non-link directory")
    return metadata.st_dev, metadata.st_ino


def assert_candidate_anchor(anchor: CandidateAnchor) -> None:
    if anchor.closed:
        raise ValueError("candidate directory anchor is closed")
    if anchor.handle is not None:
        retained = _windows_directory_identity(anchor.handle)
    elif anchor.descriptor is not None:
        metadata = os.fstat(anchor.descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or blinded_io.is_reparse(metadata):
            raise ValueError("candidate directory anchor changed type")
        retained = metadata.st_dev, metadata.st_ino
    else:
        raise ValueError("candidate directory anchor is unavailable")
    if retained != anchor.identity or _candidate_path_identity(anchor.canonical_path) != anchor.identity:
        raise ValueError("candidate directory identity changed during evaluation")


def clean_subprocess_environment(tool_root: Path) -> dict[str, str]:
    retained = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
    environment = {name: os.environ[name] for name in retained if name in os.environ}
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(tool_root / "backend"),
        }
    )
    return environment


async def _bounded_stream(
    stream: asyncio.StreamReader,
    descriptor: int,
    overflow: asyncio.Event,
    budget: list[int],
    budget_lock: asyncio.Lock,
) -> None:
    try:
        while chunk := await stream.read(65536):
            async with budget_lock:
                remaining = HOST_OUTPUT_CAP_BYTES - budget[0]
                written = chunk[: max(0, remaining)]
                budget[0] += len(written)
                if len(chunk) > remaining:
                    overflow.set()
            offset = 0
            while offset < len(written):
                offset += os.write(descriptor, written[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _attach_windows_job(process: asyncio.subprocess.Process) -> None:
    if os.name != "nt":
        return
    create_job = ctypes.WinDLL("kernel32", use_last_error=True).CreateJobObjectW
    create_job.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    create_job.restype = ctypes.c_void_p
    job = create_job(None, None)
    invalid_handle = ctypes.c_void_p(-1).value
    if job in (None, invalid_handle):
        error = ctypes.get_last_error()
        raise OSError(error, "CreateJobObjectW failed for trial host")
    job_handle = int(job)
    try:
        information = _ExtendedLimitInformation()
        information.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        set_information = ctypes.WinDLL("kernel32", use_last_error=True).SetInformationJobObject
        set_information.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        set_information.restype = ctypes.c_int
        if not set_information(
            ctypes.c_void_p(job_handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            raise OSError(error, "SetInformationJobObject failed for trial host")
        open_process = ctypes.WinDLL("kernel32", use_last_error=True).OpenProcess
        open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        open_process.restype = ctypes.c_void_p
        process_handle = open_process(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, 0, process.pid
        )
        if process_handle in (None, invalid_handle):
            error = ctypes.get_last_error()
            raise OSError(error, "OpenProcess failed for trial host")
        try:
            assign = ctypes.WinDLL("kernel32", use_last_error=True).AssignProcessToJobObject
            assign.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            assign.restype = ctypes.c_int
            if not assign(ctypes.c_void_p(job_handle), ctypes.c_void_p(process_handle)):
                error = ctypes.get_last_error()
                raise OSError(error, "AssignProcessToJobObject failed for trial host")
        finally:
            _windows_close_handle(int(process_handle))
        setattr(process, "_d37_job_handle", job_handle)
    except BaseException:
        _windows_close_handle(job_handle)
        raise


async def _release_windows_bootstrap(process: asyncio.subprocess.Process) -> None:
    if os.name != "nt":
        return
    if process.stdin is None:
        raise ValueError("trial host bootstrap pipe is unavailable")
    process.stdin.write(b"1")
    await process.stdin.drain()
    process.stdin.close()
    await process.stdin.wait_closed()


def _close_windows_job(process: asyncio.subprocess.Process) -> None:
    handle = getattr(process, "_d37_job_handle", None)
    if handle is None:
        return
    setattr(process, "_d37_job_handle", None)
    _windows_close_handle(handle)


async def _wait_for_parent_exit(process: asyncio.subprocess.Process) -> int:
    while process.returncode is None:
        await asyncio.sleep(0.01)
    return process.returncode


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "nt" and getattr(process, "_d37_job_handle", None) is not None:
        _close_windows_job(process)
    elif os.name == "nt" and hasattr(process, "pid") and process.returncode is None:
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    elif os.name == "posix" and hasattr(process, "pid"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=HOST_TEARDOWN_SECONDS)
    except TimeoutError:
        if process.returncode is None:
            process.kill()
        raise ValueError("trial host process teardown could not be confirmed") from None


async def invoke_trial_host(
    *,
    tool_root: Path,
    candidate_root: Path,
    mode: str,
    input_path: Path,
    output_path: Path,
    storage: Path,
    model: str,
    index: Path,
    stdout_path: Path,
    stderr_path: Path,
    candidate_identity: tuple[int, int] | None = None,
    embedding_profile: Path | None = None,
    embedding_base_url: str | None = None,
) -> str:
    arguments = [
        sys.executable,
        "-B",
        "-m",
        "evaluation.scripts.evaluation_trial_host",
        "--candidate-root",
        str(candidate_root),
        "--mode",
        mode,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--storage",
        str(storage),
        "--model",
        model,
    ]
    if os.name == "posix" and candidate_identity is not None:
        arguments.extend(
            (
                "--expected-candidate-dev",
                str(candidate_identity[0]),
                "--expected-candidate-ino",
                str(candidate_identity[1]),
            )
        )
    if (embedding_profile is None) != (embedding_base_url is None):
        raise ValueError("invalid embedding configuration")
    if mode == "all_tools" and embedding_profile is not None:
        raise ValueError("invalid embedding configuration")
    if mode == "stateful":
        arguments.extend(("--index", str(index)))
        if embedding_profile is not None and embedding_base_url is not None:
            arguments.extend(("--embedding-profile", str(embedding_profile),
                              "--embedding-base-url", embedding_base_url))
    process_options: dict[str, object] = {}
    launch_arguments = arguments
    process_stdin: int = asyncio.subprocess.DEVNULL
    if os.name == "posix":
        process_options["start_new_session"] = True
    elif os.name == "nt":
        process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        bootstrap = (
            "import subprocess,sys; "
            "ready=sys.stdin.buffer.read(1); "
            "raise SystemExit(125) if ready != b'1' else "
            "SystemExit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL))"
        )
        launch_arguments = [sys.executable, "-B", "-c", bootstrap, *arguments]
        process_stdin = asyncio.subprocess.PIPE
    stdout_descriptor = blinded_io.open_exclusive_regular(stdout_path, "trial stdout")
    try:
        stderr_descriptor = blinded_io.open_exclusive_regular(stderr_path, "trial stderr")
    except BaseException:
        os.close(stdout_descriptor)
        raise
    try:
        process = await asyncio.create_subprocess_exec(
            *launch_arguments,
            cwd=tool_root / "backend",
            env=clean_subprocess_environment(tool_root),
            stdin=process_stdin,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_options,
        )
        _attach_windows_job(process)
        await _release_windows_bootstrap(process)
    except BaseException:
        if "process" in locals():
            if getattr(process, "_d37_job_handle", None) is not None:
                _close_windows_job(process)
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)
        raise
    if process.stdout is None or process.stderr is None:
        os.close(stdout_descriptor)
        os.close(stderr_descriptor)
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise ValueError("trial host output pipes are unavailable")
    overflow = asyncio.Event()
    budget = [0]
    budget_lock = asyncio.Lock()
    readers = (
        asyncio.create_task(
            _bounded_stream(process.stdout, stdout_descriptor, overflow, budget, budget_lock)
        ),
        asyncio.create_task(
            _bounded_stream(process.stderr, stderr_descriptor, overflow, budget, budget_lock)
        ),
    )
    wait_task = asyncio.create_task(_wait_for_parent_exit(process))
    overflow_task = asyncio.create_task(overflow.wait())
    outcome: str
    try:
        done, _ = await asyncio.wait(
            (wait_task, overflow_task),
            timeout=HOST_WATCHDOG_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            outcome = "deadline_failure"
        elif overflow_task in done and overflow.is_set():
            outcome = "transport_failure"
        else:
            await wait_task
            outcome = "completed" if process.returncode == 0 else "transport_failure"
            _, pending_readers = await asyncio.wait(
                readers, timeout=HOST_PIPE_DRAIN_GRACE_SECONDS
            )
            if not pending_readers:
                await asyncio.gather(*readers, return_exceptions=False)
        await _terminate_process_tree(process)
        try:
            await asyncio.wait_for(
                asyncio.gather(*readers, return_exceptions=False),
                timeout=HOST_TEARDOWN_SECONDS,
            )
        except TimeoutError:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            raise ValueError("trial host pipe teardown could not be confirmed") from None
        return outcome
    except BaseException:
        await _terminate_process_tree(process)
        try:
            await asyncio.wait_for(
                asyncio.gather(*readers, return_exceptions=True),
                timeout=HOST_TEARDOWN_SECONDS,
            )
        except TimeoutError:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            raise ValueError("trial host teardown could not be confirmed") from None
        raise
    finally:
        overflow_task.cancel()
        await asyncio.gather(overflow_task, return_exceptions=True)


OwnedOutcome = Literal["completed", "launch_failed", "timeout", "output_limit", "memory_limit", "teardown_failed"]
_MIB = 1024 * 1024
OWNED_MEMORY_HARD_LIMIT_BYTES: int = 16384 * _MIB
OWNED_MEMORY_START_LIMIT_BYTES: int = 14336 * _MIB
OWNED_MEMORY_TARGET_BYTES: int = 12288 * _MIB
_OWNED_GATE = r'''
import ctypes,json,os,subprocess,sys,threading,time
if os.read(0,1) != b'1':
    raise SystemExit(125)
control=os.environ.pop('_D39_TARGET_OBSERVATION',None)
if control is None:
    raise SystemExit(subprocess.call(sys.argv[1:],stdin=subprocess.DEVNULL))
def record(value):
    raw=json.dumps(value,separators=(',',':')).encode('ascii')
    with open(control,'r+b',buffering=0) as output:
        output.write(raw)
        output.truncate()
try:
    target=subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL)
except (OSError,ValueError):
    record({'state':'launch_failed'})
    raise SystemExit(125)
record({'state':'running','pid':target.pid})
requested=threading.Event()
def read_stop():
    if os.read(0,1) == b'2':
        requested.set()
threading.Thread(target=read_stop,daemon=True).start()
stopped=False
while True:
    code=target.poll()
    if code is not None:
        break
    if requested.is_set():
        # Popen.poll uses the retained native handle / waitpid(WNOHANG).
        # An exit racing our kill is administrative only for our kill's code.
        if target.poll() is None:
            if os.name == 'nt':
                terminate=ctypes.WinDLL('kernel32',use_last_error=True).TerminateProcess
                terminate.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
                terminate.restype=ctypes.c_int
                sent=bool(terminate(int(target._handle),125))
                code=target.wait()
                stopped=sent and code == 125
            else:
                target.kill()
                code=target.wait()
                stopped=code == -9
        else:
            code=target.returncode
        break
    time.sleep(0.01)
record({'state':'exited','pid':target.pid,'code':code,'stopped':stopped})
raise SystemExit(code)
'''
_AGENT_NAMES: frozenset[str] = frozenset({"codex", "claude", "node", "chatgpt"})
_WIN_FUNCTIONS: dict[tuple[object, ...], object] = {}


@dataclass(frozen=True)
class OwnedCommandOutcome:
    outcome: OwnedOutcome
    exit_code: int | None
    started_at: str
    finished_at: str
    stdout_size: int
    stderr_size: int
    stdout_sha256: str
    stderr_sha256: str


class _JobAccounting(ctypes.Structure):
    _fields_ = [
        ("user_time", ctypes.c_int64), ("kernel_time", ctypes.c_int64),
        ("period_user_time", ctypes.c_int64), ("period_kernel_time", ctypes.c_int64),
        ("faults", ctypes.c_uint32), ("total", ctypes.c_uint32),
        ("active", ctypes.c_uint32), ("terminated", ctypes.c_uint32),
    ]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32), ("usage", ctypes.c_uint32), ("pid", ctypes.c_uint32),
        ("heap", ctypes.c_size_t), ("module", ctypes.c_uint32), ("threads", ctypes.c_uint32),
        ("parent", ctypes.c_uint32), ("priority", ctypes.c_long), ("flags", ctypes.c_uint32),
        ("name", ctypes.c_wchar * 260),
    ]


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32), ("faults", ctypes.c_uint32),
        ("peak_ws", ctypes.c_size_t), ("ws", ctypes.c_size_t),
        ("peak_paged", ctypes.c_size_t), ("paged", ctypes.c_size_t),
        ("peak_nonpaged", ctypes.c_size_t), ("nonpaged", ctypes.c_size_t),
        ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t),
        ("private", ctypes.c_size_t),
    ]


def _win_function(name: str, args: list[object], result: object = ctypes.c_int, *, library: str = "kernel32") -> object:
    key = (library, name, tuple(args), result)
    if key not in _WIN_FUNCTIONS:
        function = getattr(ctypes.WinDLL(library, use_last_error=True), name)
        function.argtypes = args
        function.restype = result
        _WIN_FUNCTIONS[key] = function
    return _WIN_FUNCTIONS[key]


def _new_owned_job(limit: int | None) -> int:
    create = _win_function("CreateJobObjectW", [ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p)
    job = create(None, None)
    if not job:
        raise OSError("owned Job creation failed")
    information = _ExtendedLimitInformation()
    information.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if limit is not None:
        information.basic_limit_information.limit_flags |= 0x200
        information.job_memory_limit = limit
    try:
        setter = _win_function("SetInformationJobObject", [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32])
        if not setter(job, 9, ctypes.byref(information), ctypes.sizeof(information)):
            raise OSError("owned Job limits failed")
        actual = _ExtendedLimitInformation()
        _query_job(int(job), 9, actual)
        if actual.basic_limit_information.limit_flags != information.basic_limit_information.limit_flags or actual.job_memory_limit != information.job_memory_limit:
            raise ValueError("owned Job limits not verified")
        return int(job)
    except BaseException:
        _windows_close_handle(int(job))
        raise


def _query_job(handle: int, kind: int, result: object) -> None:
    query = _win_function("QueryInformationJobObject", [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p])
    if not query(handle, kind, ctypes.byref(result), ctypes.sizeof(result), None):
        raise OSError("owned Job accounting unavailable")


def _job_pids(handle: int) -> set[int]:
    class ProcessIds(ctypes.Structure):
        _fields_ = [("assigned", ctypes.c_uint32), ("count", ctypes.c_uint32), ("ids", ctypes.c_size_t * 8192)]
    information = ProcessIds()
    _query_job(handle, 3, information)
    if information.count > 8192:
        raise ValueError("owned process inventory limit exceeded")
    return set(information.ids[:information.count])


def _assign_owned_job(handle: int, pid: int) -> None:
    opener = _win_function("OpenProcess", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p)
    process = opener(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, 0, pid)
    if not process:
        raise OSError("owned process handle unavailable")
    try:
        assign = _win_function("AssignProcessToJobObject", [ctypes.c_void_p, ctypes.c_void_p])
        if not assign(handle, process):
            raise OSError("owned Job assignment failed")
    finally:
        _windows_close_handle(int(process))


def _windows_process_table() -> dict[int, tuple[int, str]]:
    snapshot = _win_function("CreateToolhelp32Snapshot", [ctypes.c_uint32, ctypes.c_uint32], ctypes.c_void_p)(2, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        raise OSError("resident process inventory unavailable")
    table: dict[int, tuple[int, str]] = {}
    try:
        first = _win_function("Process32FirstW", [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)])
        next_entry = _win_function("Process32NextW", [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)])
        entry = _ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        if not first(snapshot, ctypes.byref(entry)):
            raise OSError("resident process inventory unavailable")
        while True:
            table[entry.pid] = (entry.parent, entry.name.lower().removesuffix(".exe"))
            if len(table) > 32768:
                raise ValueError("resident inventory limit exceeded")
            if not next_entry(snapshot, ctypes.byref(entry)):
                if ctypes.get_last_error() != 18:
                    raise OSError("resident process inventory lost")
                break
        return table
    finally:
        _windows_close_handle(int(snapshot))


def _windows_resident(pid: int) -> int | None:
    opener = _win_function("OpenProcess", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p)
    handle = opener(0x1000, 0, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return None
        raise OSError("resident accounting unavailable")
    try:
        counters = _MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        query = _win_function("GetProcessMemoryInfo", [ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.c_uint32], library="psapi")
        if not query(handle, ctypes.byref(counters), counters.size):
            raise OSError("resident accounting unavailable")
        return counters.ws
    finally:
        _windows_close_handle(int(handle))


def _windows_creation_time(pid: int) -> int | None:
    """Process creation FILETIME; None when the process is gone and
    `UNKNOWN_CREATION` when it exists but cannot be queried (for example access
    denied), so callers never mistake an unqueryable process for a gone one."""
    opener = _win_function("OpenProcess", [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p)
    handle = opener(0x1000, 0, pid)
    if not handle:
        return None if ctypes.get_last_error() == 87 else UNKNOWN_CREATION
    try:
        times = [ctypes.c_uint64() for _ in range(4)]
        query = _win_function("GetProcessTimes", [ctypes.c_void_p, *(ctypes.POINTER(ctypes.c_uint64),) * 4])
        if not query(handle, *(ctypes.byref(value) for value in times)):
            return UNKNOWN_CREATION
        return times[0].value
    finally:
        _windows_close_handle(int(handle))


def _posix_process_table() -> dict[int, tuple[int, str, int, int, int]]:
    table: dict[int, tuple[int, str, int, int, int]] = {}
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                with open(f"/proc/{entry.name}/stat", "rb") as stream:
                    raw = stream.read(4097)
                if len(raw) > 4096:
                    raise ValueError("resident accounting unavailable")
                end = raw.rindex(b")")
                fields = raw[end + 2:].split()
                # comm is truncated to 15 bytes and may split a UTF-8 sequence.
                name = raw[raw.index(b"(") + 1:end].decode("utf-8", "replace")
                table[int(entry.name)] = (int(fields[1]), name, int(fields[3]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), int(fields[19]))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if len(table) > 32768:
                raise ValueError("resident inventory limit exceeded")
    return table


CreationTime = Callable[[int], "int | None"]
# Creation times are non-negative; this marks a live process whose time is unknown.
UNKNOWN_CREATION: int = -1


def _process_descendants(table: dict[int, tuple[int, str]], root: int, created: CreationTime) -> set[int]:
    """Descendants by recorded parent PID, excluding proven PID-reuse orphans.

    A recorded parent PID may belong to a newer process that reused it; a real
    child is never created before its parent. Only a relation proven to be
    reuse (both times known, child older) is excluded: a live child whose time
    cannot be queried stays counted, so its memory is measured or the sample
    fails closed instead of silently dropping its whole subtree.
    """
    selected = {root}
    children: dict[int, list[int]] = {}
    for pid, (parent, _) in table.items():
        if pid != parent:
            children.setdefault(parent, []).append(pid)
    pending = [root]
    while pending:
        parent = pending.pop()
        parent_time = created(parent)
        if parent_time is None:
            continue
        for pid in children.get(parent, ()):
            if pid in selected:
                continue
            child_time = created(pid)
            if child_time is None or (child_time >= 0 and parent_time >= 0 and child_time < parent_time):
                continue
            selected.add(pid)
            pending.append(pid)
    return selected


def _declared_agent_pid() -> int | None:
    """Operator-declared agent process tree (validated as live at sampling time)."""
    value = os.environ.get("D39_AGENT_PID")
    if value is None:
        return None
    if not value.isdecimal() or not 0 < int(value) < 2 ** 32:
        raise ValueError("D39_AGENT_PID must be a positive process id")
    return int(value)


def _agent_root(table: dict[int, tuple[int, str]], controller: int,
                created: CreationTime | None = None) -> int | None:
    ancestors: list[int] = []
    pid = controller
    while pid and pid not in ancestors and pid in table:
        if ancestors and created is not None:
            # Stop at a reused parent PID: an ancestor is never newer than its child.
            child_time, parent_time = created(ancestors[-1]), created(pid)
            if child_time is None or parent_time is None or child_time < 0 or parent_time < 0 or parent_time > child_time:
                break
        ancestors.append(pid)
        pid = table[pid][0]
    for index, pid in enumerate(ancestors):
        if table[pid][1] in _AGENT_NAMES:
            root = pid
            for ancestor in ancestors[index + 1:]:
                if table[ancestor][1] not in _AGENT_NAMES:
                    break
                root = ancestor
            return root
    return None


def _owned_memory_sample(scope: OwnedProcessScope | None = None) -> tuple[int, int]:
    cache: dict[int, int | None] = {}
    if os.name == "nt":
        table = _windows_process_table()
        owned = _job_pids(scope._job) if scope is not None and scope._job is not None else set()
        def created(pid: int) -> int | None:
            if pid not in cache:
                cache[pid] = _windows_creation_time(pid)
            return cache[pid]
    elif os.name == "posix" and Path("/proc").is_dir():
        posix = _posix_process_table()
        table = {pid: (record[0], record[1]) for pid, record in posix.items()}
        sessions = scope._sessions if scope is not None else set()
        owned = {pid for pid, record in posix.items() if record[2] in sessions}
        def created(pid: int) -> int | None:
            return posix[pid][4] if pid in posix else None
    else:
        raise ValueError("resident accounting unavailable")
    controller = os.getpid()
    if controller not in table:
        raise ValueError("controller resident accounting lost")
    root = _agent_root(table, controller, created)
    agent = _process_descendants(table, root, created) if root is not None else set()
    declared = scope._agent_pid if scope is not None else None
    if declared is not None:
        # A declaration only adds a live tree; it never replaces the detected
        # agent or removes the unknown-agent reserve, so it cannot undercount.
        if declared not in table or created(declared) is None:
            raise ValueError("D39_AGENT_PID must identify a live process")
        agent |= _process_descendants(table, declared, created)
    controller_tree = _process_descendants(table, controller, created)
    selected = owned | agent | controller_tree
    group = 0
    outside = 0
    unknown_agent = root is None
    for pid in selected:
        try:
            size = _windows_resident(pid) if os.name == "nt" else posix[pid][3]
        except OSError:
            # A live process in any accounted tree whose memory cannot be read
            # fails closed: a reserve could undercount an arbitrarily large tree.
            raise ValueError("owned resident accounting lost") from None
        if size is None:
            if pid == root:
                unknown_agent = True
            continue
        if pid in owned:
            group += size
        else:
            outside += size
    if unknown_agent:
        outside += 1024 * _MIB
    return group, outside


class OwnedProcessScope:
    """One memory-gated Job/session registry for commands and long-lived children."""
    def __init__(self, *, agent_pid: int | None = None) -> None:
        self._agent_pid = agent_pid if agent_pid is not None else _declared_agent_pid()
        # Dedicated sampler thread: resident samples never queue behind the
        # drivers' single-worker hashing executor (DTD: monitor stays runnable).
        self._sampler: ThreadPoolExecutor | None = None
        self._job: int | None = None
        self._sessions: set[int] = set()
        self._children: list[OwnedProcess] = []
        self._monitor: asyncio.Task[None] | None = None
        self._memory_lost = asyncio.Event()
        self._entered = False
        self._failed = False
        self._closing = False
        self._closed = False
        self._launch_lock = asyncio.Lock()
        self._teardown_deadline: float | None = None
        self.committed_memory_limit: int = 0
        self.peak_aggregate_resident: int = 0
        self.teardown_confirmed: bool = False
        self.teardown_details: tuple[str, ...] = ()

    async def _sample(self) -> tuple[int, int]:
        if self._sampler is None:
            self._sampler = ThreadPoolExecutor(max_workers=1, thread_name_prefix="d39-memory")
        return await asyncio.get_running_loop().run_in_executor(self._sampler, _owned_memory_sample, self)

    def _stop_sampler(self) -> None:
        if self._sampler is not None:
            self._sampler.shutdown(wait=False, cancel_futures=True)
            self._sampler = None

    async def __aenter__(self) -> OwnedProcessScope:
        if self._entered or self._closed:
            raise ValueError("owned scope is single-use")
        try:
            group, outside = await self._sample()
        except BaseException:
            self._stop_sampler()
            raise
        limit = min(1536 * _MIB, OWNED_MEMORY_START_LIMIT_BYTES - outside - 512 * _MIB)
        if group + outside >= OWNED_MEMORY_START_LIMIT_BYTES or limit < 768 * _MIB or outside + limit + 512 * _MIB > OWNED_MEMORY_START_LIMIT_BYTES:
            self._stop_sampler()
            raise ValueError("insufficient aggregate memory headroom")
        self.committed_memory_limit = limit
        self.peak_aggregate_resident = group + outside
        if os.name == "nt":
            self._job = _new_owned_job(limit)
        self._entered = True
        self._monitor = asyncio.create_task(self._monitor_memory())
        return self

    async def __aexit__(self, *exception: object) -> None:
        await self.close()

    def _assign_job(self, process: asyncio.subprocess.Process, child: OwnedProcess) -> None:
        if os.name == "nt":
            if self._job is None:
                raise ValueError("owned group Job unavailable")
            _assign_owned_job(self._job, process.pid)
            child._job = _new_owned_job(None)
            try:
                _assign_owned_job(child._job, process.pid)
            except BaseException:
                _windows_close_handle(child._job)
                child._job = None
                raise
        else:
            self._sessions.add(process.pid)

    async def _monitor_memory(self) -> None:
        while not self._closing:
            try:
                group, outside = await self._sample()
                self.peak_aggregate_resident = max(self.peak_aggregate_resident, group + outside)
                if group + outside >= OWNED_MEMORY_START_LIMIT_BYTES or group >= self.committed_memory_limit:
                    self._memory_lost.set()
                self._sample_job_peak()
            except (OSError, ValueError):
                self._memory_lost.set()
            if self._memory_lost.is_set():
                self._failed = True
                if self._job is not None:
                    terminate = _win_function("TerminateJobObject", [ctypes.c_void_p, ctypes.c_uint32])
                    terminate(self._job, 125)
                else:
                    for session in self._sessions:
                        _kill_owned_session(session)
                for child in tuple(self._children):
                    child._latch("memory_limit")
                return
            await asyncio.sleep(0.1)

    def _sample_job_peak(self) -> None:
        if self._job is not None:
            limits = _ExtendedLimitInformation()
            _query_job(self._job, 9, limits)
            if limits.peak_job_memory_used >= self.committed_memory_limit:
                self._memory_lost.set()
                self._failed = True

    async def close(self) -> None:
        if self._closed:
            if not self.teardown_confirmed:
                raise ValueError("owned scope teardown_failed")
            return
        self._closing = True
        self._teardown_deadline = time.monotonic() + HOST_TEARDOWN_SECONDS
        try:
            self._sample_job_peak()
        except (OSError, ValueError):
            self._memory_lost.set()
            self._failed = True
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
        self._stop_sampler()
        confirmed = True
        details: list[str] = []
        children = tuple(self._children)
        # Stop children concurrently so they share, rather than serially exhaust,
        # the one confirmed teardown budget. Any stop failure is unconfirmed.
        results = await asyncio.gather(*(child.stop() for child in children), return_exceptions=True)
        for child, result in zip(children, results, strict=True):
            if isinstance(result, BaseException):
                confirmed = False
                details.append("child stop raised " + type(result).__name__)
            if child.outcome is not None and child.outcome.outcome == "teardown_failed":
                confirmed = False
                details.append("child: " + (child.teardown_detail or "teardown_failed"))
        if self._job is not None:
            try:
                self._sample_job_peak()
                await _terminate_job_confirmed(self._job, deadline=self._teardown_deadline)
            except (OSError, ValueError):
                confirmed = False
                details.append("scope Job termination unconfirmed")
            finally:
                _windows_close_handle(self._job)
                self._job = None
        if os.name == "posix":
            for session in self._sessions:
                try:
                    await _confirm_session_gone(session, self._teardown_deadline)
                except (OSError, ValueError):
                    confirmed = False
                    details.append("session teardown unconfirmed")
        self.teardown_confirmed = confirmed
        self.teardown_details = tuple(details)
        self._closed = True
        if not confirmed:
            raise ValueError("owned scope teardown_failed (" + "; ".join(details or ["unconfirmed"]) + ")")


async def _terminate_job_confirmed(handle: int, *, deadline: float | None = None) -> None:
    terminate = _win_function("TerminateJobObject", [ctypes.c_void_p, ctypes.c_uint32])
    if not terminate(handle, 125):
        raise OSError("owned Job termination failed")
    until = deadline if deadline is not None else time.monotonic() + HOST_TEARDOWN_SECONDS
    while True:
        information = _JobAccounting()
        _query_job(handle, 1, information)
        if information.active == 0:
            return
        if time.monotonic() >= until:
            raise ValueError(f"owned Job descendant teardown_failed (active={information.active})")
        await asyncio.sleep(0.02)


def _kill_owned_session(session: int) -> set[int]:
    pids = {pid for pid, entry in _posix_process_table().items() if entry[2] == session}
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # Gone, or the PID was reused outside our session before the kill.
            pass
    return pids


async def _confirm_session_gone(session: int, deadline: float) -> None:
    while _kill_owned_session(session):
        if time.monotonic() >= deadline:
            raise ValueError("owned session descendant teardown_failed")
        await asyncio.sleep(0.02)


class OwnedProcess:
    """A gated native child with incremental receipts and confirmed terminal teardown."""
    def __init__(self, scope: OwnedProcessScope, deadline_seconds: int) -> None:
        self.scope = scope
        self.pid: int = 0
        self.outcome: OwnedCommandOutcome | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._job: int | None = None
        self._readers: tuple[asyncio.Task[None], ...] = ()
        self._task: asyncio.Task[OwnedCommandOutcome] | None = None
        self._overflow = asyncio.Event()
        self._budget = 0
        self._sizes = [0, 0]
        self._hashes = [hashlib.sha256(), hashlib.sha256()]
        self._stop_reason: OwnedOutcome | None = None
        self._administrative_stop = False
        self._stop_requested = False
        self._control_path: Path | None = None
        self._started_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._deadline = time.monotonic() + deadline_seconds
        self._terminate_lock = asyncio.Lock()
        self._terminated = False
        self._tree_confirmed = False
        # Diagnostic only (never evidence): why this child ended teardown_failed.
        self.teardown_detail: str | None = None

    def _observation(self) -> tuple[str, int | None, bool]:
        if self._control_path is None:
            return "unavailable", None, False
        try:
            raw = blinded_io.read_regular(self._control_path, maximum=256)
            value = json.loads(raw.decode("utf-8"))
            if value == {"state": "launch_failed"}:
                return "launch_failed", None, False
            if type(value) is dict and value.get("state") == "exited" and set(value) == {"state", "pid", "code", "stopped"} and type(value["pid"]) is int and value["pid"] > 0 and type(value["code"]) is int and type(value["stopped"]) is bool:
                return "exited", value["code"], value["stopped"] and self._stop_requested
            if type(value) is dict and value.get("state") == "running" and set(value) == {"state", "pid"} and type(value["pid"]) is int:
                return "running", None, False
        except (OSError, ValueError, UnicodeError):
            pass
        return "unavailable", None, False

    def _latch(self, reason: OwnedOutcome) -> None:
        if self.outcome is not None or self._observation()[0] in {"exited", "launch_failed"}:
            return
        priority = {None: -1, "completed": 0, "launch_failed": 1, "timeout": 2, "output_limit": 3, "memory_limit": 4, "teardown_failed": 5}
        if priority[reason] > priority[self._stop_reason]:
            self._stop_reason = reason

    def is_running(self) -> bool:
        """Require a live target parented by this owned gate, not a cached exit."""
        if self.outcome is not None or self._process is None or self._process.returncode is not None or self._control_path is None:
            return False
        try:
            value = json.loads(blinded_io.read_regular(self._control_path, maximum=256).decode('utf-8'))
            if type(value) is not dict or set(value) != {'state', 'pid'} or value['state'] != 'running' or type(value['pid']) is not int:
                return False
            table = _windows_process_table() if os.name == 'nt' else _posix_process_table()
            entry = table.get(value['pid'])
            return entry is not None and entry[0] == self.pid
        except (OSError, ValueError, UnicodeError):
            return False

    async def _read(self, stream: asyncio.StreamReader, descriptor: int, index: int) -> None:
        try:
            while chunk := await stream.read(65536):
                remaining = max(0, HOST_OUTPUT_CAP_BYTES - self._budget)
                value = chunk[:remaining]
                self._budget += len(value)
                if len(chunk) > remaining:
                    self._overflow.set()
                offset = 0
                while offset < len(value):
                    written = os.write(descriptor, value[offset:])
                    if written < 1:
                        raise ValueError("owned output write failed")
                    self._sizes[index] += written
                    self._hashes[index].update(value[offset:offset + written])
                    offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    async def _terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                if not self._tree_confirmed:
                    raise ValueError("owned child teardown_failed")
                return
            deadline = time.monotonic() + HOST_TEARDOWN_SECONDS
            if self.scope._teardown_deadline is not None:
                deadline = min(deadline, self.scope._teardown_deadline)
            confirmed = True
            process = self._process
            step = "Job termination"
            try:
                if self._job is not None:
                    await _terminate_job_confirmed(self._job, deadline=deadline)
                elif process is not None and os.name == "posix":
                    _kill_owned_session(process.pid)
                elif process is not None and process.returncode is None:
                    process.kill()
                if process is not None:
                    step = "gate process wait"
                    await asyncio.wait_for(process.wait(), timeout=max(0.01, deadline - time.monotonic()))
                if os.name == "posix" and process is not None:
                    step = "session confirmation"
                    await _confirm_session_gone(process.pid, deadline)
                    self.scope._sessions.discard(process.pid)
                if self._readers:
                    step = "pipe drain"
                    await asyncio.wait_for(asyncio.gather(*self._readers), timeout=max(0.01, deadline - time.monotonic()))
            except BaseException as error:
                confirmed = False
                # Diagnostic only: which step could not be confirmed, with the budget left.
                left = round(deadline - time.monotonic(), 2)
                self.teardown_detail = f"process tree termination unconfirmed at {step}: {error or type(error).__name__} (budget left {left}s)"
                for task in self._readers:
                    task.cancel()
                await asyncio.gather(*self._readers, return_exceptions=True)
                if not isinstance(error, (OSError, ValueError, TimeoutError)):
                    raise
            finally:
                if process is not None and process.stdin is not None:
                    process.stdin.close()
                if self._job is not None:
                    _windows_close_handle(self._job)
                    self._job = None
                self._terminated = True
                self._tree_confirmed = confirmed
            if not confirmed:
                raise ValueError("owned child teardown_failed")

    def _result(self, reason: OwnedOutcome) -> OwnedCommandOutcome:
        _, code, _ = self._observation()
        self.outcome = OwnedCommandOutcome(
            outcome=reason, exit_code=None if reason == "launch_failed" else code,
            started_at=self._started_at, finished_at=max(self._started_at, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")),
            stdout_size=self._sizes[0], stderr_size=self._sizes[1],
            stdout_sha256=self._hashes[0].hexdigest(), stderr_sha256=self._hashes[1].hexdigest(),
        )
        return self.outcome

    async def _watch(self) -> OwnedCommandOutcome:
        reason: OwnedOutcome = "completed"
        assert self._process is not None
        while True:
            if self.scope._memory_lost.is_set():
                reason = "memory_limit"
                break
            if self._overflow.is_set():
                reason = "output_limit"
                break
            if self._stop_reason is not None:
                reason = self._stop_reason
                break
            if time.monotonic() >= self._deadline:
                reason = "timeout"
                break
            if self._process.returncode is not None:
                await asyncio.wait(self._readers, timeout=HOST_PIPE_DRAIN_GRACE_SECONDS)
                if self._overflow.is_set():
                    reason = "output_limit"
                break
            if any(task.done() and task.exception() is not None for task in self._readers):
                reason = "teardown_failed"
                self.teardown_detail = "pipe reader failed"
                break
            await asyncio.sleep(0.02)
        try:
            self.scope._sample_job_peak()
            await self._terminate()
        except (OSError, ValueError):
            reason = "teardown_failed"
            self.teardown_detail = self.teardown_detail or "process tree termination unconfirmed"
        latched = (reason, self._stop_reason)
        if "teardown_failed" in latched:
            reason = "teardown_failed"
        elif self.scope._memory_lost.is_set() or "memory_limit" in latched:
            reason = "memory_limit"
        elif self._overflow.is_set() or "output_limit" in latched:
            reason = "output_limit"
        elif "timeout" in latched:
            reason = "timeout"
        elif self._stop_reason is not None:
            reason = self._stop_reason
        state, code, stopped = self._observation()
        self._administrative_stop = stopped and not self.scope._failed
        if reason == "completed" and state != "exited":
            reason = "launch_failed" if state == "launch_failed" else "teardown_failed"
            if reason == "teardown_failed":
                self.teardown_detail = self.teardown_detail or (
                    f"gate exit record missing (state={state}, tree_confirmed={self._tree_confirmed})")
        if reason != "completed" or (code != 0 and not self._administrative_stop):
            self.scope._failed = True
            for child in self.scope._children:
                if child is not self and child.outcome is None:
                    child._latch(reason if reason != "completed" else "teardown_failed")
        return self._result(reason)

    async def wait(self) -> OwnedCommandOutcome:
        if self.outcome is not None:
            return self.outcome
        if self._task is None:
            raise ValueError("owned process has no terminal observer")
        return await asyncio.shield(self._task)

    async def stop(self) -> OwnedCommandOutcome:
        if self.outcome is None:
            if self._stop_reason is None:
                if time.monotonic() >= self._deadline:
                    self._stop_reason = "timeout"
                else:
                    self._stop_requested = True
                    process = self._process
                    if process is not None and process.stdin is not None and not process.stdin.is_closing():
                        try:
                            process.stdin.write(b"2")
                            await process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                    # Wait for the gate's own terminal record (bounded by the confirmed
                    # teardown budget, not a fixed 1 s): a slow but successful stop
                    # under load must not become teardown_failed.
                    until = min(self._deadline, time.monotonic() + HOST_TEARDOWN_SECONDS)
                    if self.scope._teardown_deadline is not None:
                        # Inside scope close, keep at least half of the shared budget
                        # for the confirmed Job/session termination that follows (a
                        # browser tree holds the pipes until every process is gone).
                        now = time.monotonic()
                        until = min(until, now + max(0.0, self.scope._teardown_deadline - now) / 2)
                    while self._observation()[0] not in {"exited", "launch_failed"} and time.monotonic() < until:
                        if process is not None and process.returncode is not None:
                            break
                        await asyncio.sleep(0.01)
                    self._latch("completed")
            await self._terminate()
        return await self.wait()


async def start_owned_process(*, scope: OwnedProcessScope, argv: tuple[str, ...], cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path, deadline_seconds: int) -> OwnedProcess:
    """Register a stdin-gated child before allowing any target instruction to run."""
    if scope._failed:
        raise ValueError("owned scope failed; further commands refused")
    if not scope._entered or scope._closing or scope._closed:
        raise ValueError("owned scope is not active")
    if type(deadline_seconds) is not int or deadline_seconds < 1 or type(argv) is not tuple or not argv or any(type(arg) is not str or not arg or "\0" in arg for arg in argv):
        raise ValueError("invalid owned command contract")
    blinded_io.validate_directory(cwd.absolute(), "owned cwd")
    if type(env) is not dict or any(type(key) is not str or type(value) is not str for key, value in env.items()):
        raise ValueError("invalid owned environment")
    async with scope._launch_lock:
        group, outside = _owned_memory_sample(scope)
        if group + outside >= OWNED_MEMORY_START_LIMIT_BYTES or scope._memory_lost.is_set():
            raise ValueError("aggregate memory headroom lost")
        child = OwnedProcess(scope, deadline_seconds)
        stdout_descriptor = blinded_io.open_exclusive_regular(stdout_path, "owned stdout")
        try:
            stderr_descriptor = blinded_io.open_exclusive_regular(stderr_path, "owned stderr")
        except BaseException:
            os.close(stdout_descriptor)
            raise
        descriptors_owned = True
        scope._children.append(child)
        try:
            child._control_path = stdout_path.with_name(stdout_path.name + ".target.json")
            control = blinded_io.open_exclusive_regular(child._control_path, "owned target observation")
            os.close(control)
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            child._process = await asyncio.create_subprocess_exec(getattr(sys, "_base_executable", sys.executable), "-I", "-S", "-B", "-c", _OWNED_GATE, *argv, cwd=cwd, env={**env, "_D39_TARGET_OBSERVATION": str(child._control_path)}, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **options)
            process = child._process
            child.pid = process.pid
            if process.stdout is None or process.stderr is None or process.stdin is None:
                raise ValueError("owned pipes unavailable")
            scope._assign_job(process, child)
            child._readers = (asyncio.create_task(child._read(process.stdout, stdout_descriptor, 0)), asyncio.create_task(child._read(process.stderr, stderr_descriptor, 1)))
            descriptors_owned = False
            if time.monotonic() >= child._deadline:
                child._stop_reason = "timeout"
            elif scope._memory_lost.is_set():
                child._stop_reason = "memory_limit"
            elif scope._failed or scope._closing or child._stop_reason is not None:
                child._latch("launch_failed")
            else:
                process.stdin.write(b"1")
                await process.stdin.drain()
            if child._stop_reason is not None:
                process.stdin.close()
            child._task = asyncio.create_task(child._watch())
            return child
        except BaseException as error:
            reason: OwnedOutcome = "launch_failed"
            try:
                await child._terminate()
            except (OSError, ValueError):
                reason = "teardown_failed"
            if descriptors_owned:
                os.close(stdout_descriptor)
                os.close(stderr_descriptor)
            scope._failed = True
            child._result(reason)
            if isinstance(error, asyncio.CancelledError):
                raise
            return child


async def run_owned_command(*, argv: tuple[str, ...], cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path, deadline_seconds: int, scope: OwnedProcessScope | None = None) -> OwnedCommandOutcome:
    """Return only bounded observations after native descendant and reader proof."""
    if scope is None:
        temporary = OwnedProcessScope()
        await temporary.__aenter__()
        result: OwnedCommandOutcome | None = None
        cleanup_failed = False
        try:
            result = await run_owned_command(scope=temporary, argv=argv, cwd=cwd, env=env, stdout_path=stdout_path, stderr_path=stderr_path, deadline_seconds=deadline_seconds)
        finally:
            try:
                await asyncio.shield(temporary.close())
            except (OSError, ValueError):
                cleanup_failed = True
        assert result is not None
        return replace(result, outcome="teardown_failed") if cleanup_failed else result
    child = await start_owned_process(scope=scope, argv=argv, cwd=cwd, env=env, stdout_path=stdout_path, stderr_path=stderr_path, deadline_seconds=deadline_seconds)
    try:
        return await child.wait()
    except asyncio.CancelledError:
        await asyncio.shield(scope.close())
        raise
