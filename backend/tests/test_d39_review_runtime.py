"""Native lifecycle tests with an explicit synthetic agent inventory.

Only the agent inventory is synthetic. Targets, Job assignment, exit observation,
pipe readers and teardown run through the real D39 implementation. These tests
are not an acceptance claim about this desktop's actual operational headroom.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from pathlib import Path
import sys
import time

import pytest

from evaluation import blinded_runtime as runtime


def test_user_authorized_16gib_scope_admits_4gib_agent_and_keeps_job_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, '_owned_memory_sample', lambda scope=None: (0, 4096 * 1024 ** 2))
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            assert scope.committed_memory_limit == 1536 * 1024 ** 2
    asyncio.run(exercise())


def test_16gib_scope_still_refuses_14gib_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, '_owned_memory_sample', lambda scope=None: (0, 14336 * 1024 ** 2))
    with pytest.raises(ValueError, match='headroom'):
        asyncio.run(runtime.OwnedProcessScope().__aenter__())


def test_16gib_scope_reserves_realistic_headroom_for_5gib_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, '_owned_memory_sample', lambda scope=None: (0, 5376 * 1024 ** 2))
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            assert scope.committed_memory_limit == 1536 * 1024 ** 2
    asyncio.run(exercise())


@pytest.fixture
def synthetic_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.name == 'nt':
        original = runtime._windows_process_table
        def inventory() -> dict[int, tuple[int, str]]:
            table = original()
            return {pid: entry for pid, entry in table.items() if pid == os.getpid() or entry[0] == os.getpid()}
        monkeypatch.setattr(runtime, '_windows_process_table', inventory)
    else:
        original_posix = runtime._posix_process_table
        def posix_inventory() -> dict[int, tuple[int, str, int, int]]:
            table = original_posix()
            return {pid: entry for pid, entry in table.items()
                    if pid == os.getpid() or entry[2] != os.getsid(0)}
        monkeypatch.setattr(runtime, '_posix_process_table', posix_inventory)


def _arguments(tmp_path: Path, *argv: str) -> dict[str, object]:
    return dict(argv=tuple(argv), cwd=tmp_path, env=dict(os.environ),
                stdout_path=tmp_path / 'stdout', stderr_path=tmp_path / 'stderr', deadline_seconds=15)


@pytest.mark.parametrize('delay', [0.004, 0.006, 0.02])
def test_i2_nonzero_exit_before_stop_cannot_be_forgiven(tmp_path: Path, synthetic_agent: None, delay: float) -> None:
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            child = await runtime.start_owned_process(scope=scope, **_arguments(
                tmp_path, sys.executable, '-c', "from pathlib import Path; Path('exiting').write_text('1'); raise SystemExit(3)"))
            until = time.monotonic() + 5
            # Block the asyncio callback deliberately, reproducing its stale
            # returncode while the native target/gate have already exited. The
            # marker is written before exit, so wait for the gate's own exited
            # record: under load the target may otherwise still be alive and the
            # stop would legitimately kill it (exit 125), which is not this case.
            while child._observation()[0] != 'exited' and time.monotonic() < until:
                time.sleep(0.001)
            assert (tmp_path / 'exiting').exists() and child._observation()[0] == 'exited'
            time.sleep(delay)
            result = await child.stop()
            assert result.exit_code == 3
            assert scope._failed
    asyncio.run(exercise())


def test_i2_real_administrative_stop_permits_next_command(tmp_path: Path, synthetic_agent: None) -> None:
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            child = await runtime.start_owned_process(scope=scope, **_arguments(tmp_path, sys.executable, '-c', 'import time; time.sleep(10)'))
            await asyncio.sleep(0.15)
            outcome = await child.stop()
            assert outcome.exit_code != 0 and outcome.exit_code is not None
            assert not scope._failed
            second = tmp_path / 'second'
            second.mkdir()
            result = await runtime.run_owned_command(scope=scope, **_arguments(second, sys.executable, '-c', 'pass'))
            assert result.outcome == 'completed' and result.exit_code == 0
    asyncio.run(exercise())


def test_m8_missing_target_has_no_fabricated_exit_or_gate_traceback(tmp_path: Path, synthetic_agent: None) -> None:
    result = asyncio.run(runtime.run_owned_command(**_arguments(tmp_path, str(tmp_path / 'absent.exe'))))
    assert result.outcome == 'launch_failed'
    assert result.exit_code is None
    assert result.stderr_size == 0
    assert result.stderr_sha256 == hashlib.sha256(b'').hexdigest()


def test_l1_failed_scope_during_spawn_never_releases_gate(tmp_path: Path, synthetic_agent: None,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            original = asyncio.create_subprocess_exec
            async def spawn(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
                child = await original(*args, **kwargs)
                scope._failed = True
                return child
            monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
            child = await runtime.start_owned_process(scope=scope, **_arguments(
                tmp_path, sys.executable, '-c', "from pathlib import Path; Path('released').write_text('bad')"))
            await child.wait()
            assert not (tmp_path / 'released').exists()
    asyncio.run(exercise())


def test_l2_cancellation_never_confirms_teardown(monkeypatch: pytest.MonkeyPatch) -> None:
    async def cancel(*args: object, **kwargs: object) -> None:
        raise asyncio.CancelledError
    monkeypatch.setattr(runtime, '_terminate_job_confirmed', cancel)
    monkeypatch.setattr(runtime, '_windows_close_handle', lambda handle: None)
    child = runtime.OwnedProcess(runtime.OwnedProcessScope(), 10)
    child._job = 123
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(child._terminate())
    assert not child._tree_confirmed


def test_l6_backwards_clock_does_not_discard_observation() -> None:
    child = runtime.OwnedProcess(runtime.OwnedProcessScope(), 10)
    child._started_at = '9999-12-31T23:59:59Z'
    result = child._result('launch_failed')
    assert result.finished_at == result.started_at


def test_l3_latch_preserves_already_exited_child(tmp_path: Path) -> None:
    import json
    child = runtime.OwnedProcess(runtime.OwnedProcessScope(), 10)
    child._control_path = tmp_path / 'observation'
    child._control_path.write_text(json.dumps(dict(state='exited', pid=123, code=3, stopped=False)))
    child._latch('memory_limit')
    assert child._stop_reason is None
    assert child._result('completed').exit_code == 3


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job peak settlement')
def test_l4_settlement_checks_job_peak_after_quiet_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    scope = runtime.OwnedProcessScope()
    scope._job = 123
    scope.committed_memory_limit = 768 * 1024 ** 2
    def peak(handle: int, kind: int, limits: object) -> None:
        limits.peak_job_memory_used = scope.committed_memory_limit
    monkeypatch.setattr(runtime, '_query_job', peak)
    scope._sample_job_peak()
    assert scope._memory_lost.is_set() and scope._failed


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job peak settlement')
def test_l4_quick_command_settlement_samples_job_peak(tmp_path: Path, synthetic_agent: None,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    # The resident monitor is silenced, so only the settlement-time Job peak
    # sample (not the 100 ms monitor or close()) can latch this command.
    async def quiet(self: runtime.OwnedProcessScope) -> None:
        return None
    monkeypatch.setattr(runtime.OwnedProcessScope, '_monitor_memory', quiet)
    real_query = runtime._query_job
    def query(handle: int, kind: int, result: object) -> None:
        real_query(handle, kind, result)
        if kind == 9:
            result.peak_job_memory_used = 1 << 40  # type: ignore[attr-defined]
    monkeypatch.setattr(runtime, '_query_job', query)
    result = asyncio.run(runtime.run_owned_command(**_arguments(tmp_path, sys.executable, '-c', 'pass')))
    assert result.outcome == 'memory_limit'


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='installed native media required')
def test_m7_backend_path_selects_real_bound_media(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from evaluation import release_verification as release
    executable = Path(sys._base_executable)
    fingerprint = release._native_file(executable, 'tools/python_sandbox')
    tools = release._ToolSet(executable, None, (release.ToolExecutionBinding(role='python', version='3.12.12', executable=fingerprint, launcher=None),), ((executable, fingerprint),))
    env = release.build_group_environment(tmp_path, python_executable=executable, node_executable=None)
    shared_path = env['PATH']
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            bound, command_env = await release._backend_media(scope, SimpleNamespace(root=tmp_path, assert_owned=lambda: None), tools, env)
            assert tuple(binding.role for binding in bound.media_bindings) == ('ffmpeg', 'ffprobe')
            bound.verify()
            # Only backend_pytest's copy sees media; the sandbox directory stays first.
            assert env['PATH'] == shared_path
            assert command_env['PATH'].split(os.pathsep)[0] == shared_path.split(os.pathsep)[0]
            for role, (path, _) in zip(('ffmpeg', 'ffprobe'), bound.files[-2:], strict=True):
                assert Path(shutil.which(role, path=command_env['PATH'])).resolve() == path
    asyncio.run(exercise())


def test_m7_shim_directory_with_other_programs_is_refused(tmp_path: Path) -> None:
    from evaluation import release_verification as release
    suffix = '.exe' if os.name == 'nt' else ''
    for name in ('ffmpeg', 'ffprobe', 'python', 'npx'):
        path = tmp_path / (name + suffix)
        path.write_bytes(b'MZ shim' if os.name == 'nt' else b'#!/bin/sh\n')
        path.chmod(0o755)
    with pytest.raises(ValueError, match='unbound programs'):
        release._dedicated_media_directory(tmp_path)
    for name in ('python', 'npx'):
        (tmp_path / (name + suffix)).unlink()
    release._dedicated_media_directory(tmp_path)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX session teardown')
def test_m6_descendant_new_process_group_cannot_survive(tmp_path: Path, synthetic_agent: None) -> None:
    code = "import os,time; pid=os.fork(); os.setpgid(0,0) if pid==0 else None; time.sleep(30)"
    result = asyncio.run(runtime.run_owned_command(**(_arguments(tmp_path, sys.executable, '-c', code) | {'deadline_seconds': 1})))
    assert result.outcome == 'timeout'


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job teardown diagnostics')
def test_late_gate_exit_record_names_the_failed_teardown_step(tmp_path: Path, synthetic_agent: None,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    # Reviewer hypothesis for the cold-start Chrome flake: the tree is gone but the
    # gate's exit record never arrives. The outcome stays fail-closed; the message
    # now says which step failed so a real occurrence can be told apart.
    marker = "record({'state':'exited','pid':target.pid,'code':code,'stopped':stopped})"
    assert runtime._OWNED_GATE.count(marker) == 1
    monkeypatch.setattr(runtime, '_OWNED_GATE', runtime._OWNED_GATE.replace(marker, 'time.sleep(30)\n' + marker))
    async def exercise() -> None:
        scope = runtime.OwnedProcessScope()
        await scope.__aenter__()
        await runtime.start_owned_process(scope=scope, **_arguments(tmp_path, sys.executable, '-c', 'import time; time.sleep(30)'))
        await asyncio.sleep(0.3)
        with pytest.raises(ValueError, match=r'teardown_failed \(child: gate exit record missing \(state=running, tree_confirmed=True\)\)'):
            await scope.close()
        assert scope.teardown_details == ('child: gate exit record missing (state=running, tree_confirmed=True)',)
    asyncio.run(exercise())
