"""Regression seams for the 2026-10-02 independent D39 review."""
from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from evaluation import blinded_runtime as runtime, release_verification as release
from evaluation.smoke_contracts import BrowserSummary, SmokeStageReceipt
from tests.test_d39_release_verification import _binding, _bytes, _receipt, _summary


def test_l6_command_evidence_survives_backwards_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = runtime.OwnedCommandOutcome(
        outcome='completed', exit_code=0, started_at='2026-10-02T00:00:02Z',
        finished_at='2026-10-02T00:00:01Z', stdout_size=0, stderr_size=0,
        stdout_sha256='0' * 64, stderr_sha256='0' * 64,
    )
    async def run(**kwargs: Any) -> runtime.OwnedCommandOutcome:
        return observed
    monkeypatch.setattr(runtime, 'run_owned_command', run)
    group = SimpleNamespace(source=tmp_path, root=tmp_path, assert_source=lambda: None)
    tools = SimpleNamespace(executable=Path(sys.executable), bindings=(release.ToolExecutionBinding.model_validate(_binding('python')),), verify=lambda: None)
    commands = []
    result = asyncio.run(release._execute_command(index=1, group=group, scope=None, tools=tools, env={}, commands=commands))
    assert commands == [result] and result.exit_code == 0
    assert result.finished_at == result.started_at


def test_m9_shared_frontend_preflight_refuses_local_pnpm(tmp_path: Path) -> None:
    bin_path = tmp_path / 'frontend/node_modules/.bin'
    bin_path.mkdir(parents=True)
    (bin_path / ('pnpm.cmd' if os.name == 'nt' else 'pnpm')).write_bytes(b'shadow')
    with pytest.raises(ValueError, match='shadow'):
        asyncio.run(release._frontend_preflight(None, SimpleNamespace(source=tmp_path), None, {}))


def test_m7_backend_pytest_cannot_hide_unbound_media_coverage() -> None:
    from evaluation.smoke_contracts import CommandEvidence, D39_COMMAND_DEADLINES, D39_REQUIRED_COMMANDS
    from pydantic import ValidationError
    name, argv = D39_REQUIRED_COMMANDS[2]
    with pytest.raises(ValidationError, match='media'):
        CommandEvidence(name=name, argv=argv, resolved_argv=('tools/python_sandbox', *argv[1:]),
                        tool_bindings=(_binding('python'),), cwd='backend', deadline_seconds=D39_COMMAND_DEADLINES[2],
                        outcome='completed', exit_code=0, started_at='2026-10-02T00:00:00Z', finished_at='2026-10-02T00:00:00Z',
                        stdout_size=0, stderr_size=0, stdout_sha256='0'*64, stderr_sha256='0'*64)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process table seam')
@pytest.mark.parametrize('agent', ['codex', 'claude', 'node', 'chatgpt', None])
def test_b1_only_controller_agent_and_owned_memory(monkeypatch: pytest.MonkeyPatch, agent: str | None) -> None:
    table = {1: (0, 'explorer'), 10: (1, agent or 'unrecognized'), 11: (10, 'renderer'),
             20: (10, 'pwsh'), 30: (20, 'python'), 40: (1, 'node'),
             41: (1, 'git'), 42: (1, 'powershell'), 43: (0, 'wslservice')}
    created = {1: 1, 10: 2, 11: 3, 20: 3, 30: 4, 40: 2, 41: 2, 42: 2, 43: 0}
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime, '_windows_creation_time', created.get)
    monkeypatch.delenv('D39_AGENT_PID', raising=False)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    sampled = []
    def resident(pid: int) -> int:
        sampled.append(pid)
        if pid == 43:
            raise OSError('SYSTEM access denied')
        return 10
    monkeypatch.setattr(runtime, '_windows_resident', resident)
    group, outside = runtime._owned_memory_sample()
    assert group == 0
    assert outside == (40 if agent else 10 + 1024 ** 3)
    assert not {1, 40, 41, 42, 43}.intersection(sampled)


@pytest.mark.skipif(os.name != 'nt', reason='Windows process table seam')
def test_b1_pid_reuse_orphans_are_never_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    # 50 and 51 are old orphans whose dead parents' PIDs were reused by the
    # controller (30) and its agent ancestor (10); 31 is the controller's real child.
    table = {1: (0, 'explorer'), 10: (1, 'claude'), 30: (10, 'python'), 31: (30, 'git'),
             50: (30, 'svchost'), 51: (10, 'wininit')}
    created = {1: 1, 10: 5, 30: 10, 31: 11, 50: 3, 51: 4}
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime, '_windows_creation_time', created.get)
    monkeypatch.delenv('D39_AGENT_PID', raising=False)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    sampled: list[int] = []
    def resident(pid: int) -> int:
        sampled.append(pid)
        if pid in (50, 51):
            raise OSError('SYSTEM access denied')
        return 10
    monkeypatch.setattr(runtime, '_windows_resident', resident)
    group, outside = runtime._owned_memory_sample()
    assert (group, outside) == (0, 30)
    assert sorted(sampled) == [10, 30, 31]


@pytest.mark.skipif(os.name != 'nt', reason='Windows process table seam')
def test_b1_declared_agent_tree_only_adds_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    # 10 is an unrecognized agent; 11 its worker; 40 a detached agent helper outside the chain.
    table = {1: (0, 'explorer'), 10: (1, 'custom-agent'), 11: (10, 'worker'), 20: (10, 'pwsh'), 30: (20, 'python'),
             40: (1, 'helper')}
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime, '_windows_creation_time', {1: 1, 10: 2, 11: 3, 20: 3, 30: 4, 40: 2}.get)
    monkeypatch.setattr(runtime, '_windows_resident', lambda pid: 10)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    reserve = 1024 ** 3
    monkeypatch.setenv('D39_AGENT_PID', '10')
    assert runtime._owned_memory_sample(runtime.OwnedProcessScope()) == (0, 40 + reserve)
    monkeypatch.setenv('D39_AGENT_PID', '40')  # broken chain: not an ancestor, still counted
    assert runtime._owned_memory_sample(runtime.OwnedProcessScope()) == (0, 20 + reserve)
    monkeypatch.setenv('D39_AGENT_PID', '99')
    with pytest.raises(ValueError, match='live process'):
        runtime._owned_memory_sample(runtime.OwnedProcessScope())


@pytest.mark.skipif(os.name != 'nt', reason='Windows process table seam')
@pytest.mark.parametrize('readable', [True, False])
def test_b1_unqueryable_live_agent_child_is_counted_or_fails_closed(monkeypatch: pytest.MonkeyPatch, readable: bool) -> None:
    # 12 is a live child of the detected agent whose creation time is access-denied.
    table = {1: (0, 'explorer'), 10: (1, 'claude'), 12: (10, 'worker'), 13: (12, 'worker'), 20: (10, 'pwsh'), 30: (20, 'python')}
    times = {1: 1, 10: 2, 12: runtime.UNKNOWN_CREATION, 13: 5, 20: 3, 30: 4}
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime, '_windows_creation_time', times.get)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    monkeypatch.delenv('D39_AGENT_PID', raising=False)
    large = 15 * 1024 ** 3
    def resident(pid: int) -> int:
        if pid == 12 and not readable:
            raise OSError('access denied')
        return large if pid in (12, 13) else 10
    monkeypatch.setattr(runtime, '_windows_resident', resident)
    if readable:
        assert runtime._owned_memory_sample() == (0, 2 * large + 30)
    else:
        with pytest.raises(ValueError, match='accounting lost'):
            runtime._owned_memory_sample()


@pytest.mark.skipif(os.name != 'nt', reason='Windows process API seam')
@pytest.mark.parametrize('readable', [True, False])
def test_b1_win32_creation_errors_are_classified_and_counted(monkeypatch: pytest.MonkeyPatch, readable: bool) -> None:
    # The real _windows_creation_time runs against a fake Win32 layer: OpenProcess
    # access denied (5) is unknown, invalid parameter (87) is gone, and a failed
    # GetProcessTimes is unknown. The unknown live child must then be counted.
    created = {1: 1, 10: 2, 20: 3, 30: 4, 13: 6}
    denied, gone, times_fail = {12}, {14}, {15}
    table = {1: (0, 'explorer'), 10: (1, 'claude'), 12: (10, 'worker'), 13: (12, 'worker'), 14: (10, 'exited'),
             15: (10, 'worker'), 20: (10, 'pwsh'), 30: (20, 'python')}
    state = {'error': 0}
    def open_process(access: int, inherit: int, pid: int) -> int:
        if pid in denied | gone:
            state['error'] = 5 if pid in denied else 87
            return 0
        return 100000 + pid
    def process_times(handle: int, creation: Any, *others: Any) -> int:
        pid = handle - 100000
        if pid in times_fail:
            return 0
        creation._obj.value = created[pid]
        return 1
    def win_function(name: str, args: list[object], result: object = None, *, library: str = 'kernel32') -> object:
        return {'OpenProcess': open_process, 'GetProcessTimes': process_times}[name]
    monkeypatch.setattr(runtime, '_win_function', win_function)
    monkeypatch.setattr(runtime, '_windows_close_handle', lambda handle: None)
    monkeypatch.setattr(runtime.ctypes, 'get_last_error', lambda: state['error'])
    assert runtime._windows_creation_time(10) == 2
    assert runtime._windows_creation_time(12) == runtime.UNKNOWN_CREATION
    assert runtime._windows_creation_time(14) is None
    assert runtime._windows_creation_time(15) == runtime.UNKNOWN_CREATION
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    monkeypatch.delenv('D39_AGENT_PID', raising=False)
    large = 15 * 1024 ** 3
    def resident(pid: int) -> int | None:
        if pid == 14:
            return None
        if pid in (12, 15) and not readable:
            raise OSError('access denied')
        return large if pid in (12, 13, 15) else 10
    monkeypatch.setattr(runtime, '_windows_resident', resident)
    if readable:
        assert runtime._owned_memory_sample() == (0, 3 * large + 30)
    else:
        with pytest.raises(ValueError, match='accounting lost'):
            runtime._owned_memory_sample()


@pytest.mark.parametrize('value', ['abc', '0', '-1'])
def test_b1_invalid_agent_pid_refuses_before_any_group_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str) -> None:
    monkeypatch.setenv('D39_AGENT_PID', value)
    monkeypatch.setattr(release, '_inventory_boundary', lambda *args: None)
    created: list[object] = []
    monkeypatch.setattr(release, '_ExecutionGroup', lambda *args: created.append(args))
    with pytest.raises(ValueError, match='D39_AGENT_PID'):
        asyncio.run(release._run_inventory(candidate=tmp_path, runtime=tmp_path, work=tmp_path, path=tmp_path, digest='0' * 64,
                                           record=None, tool_root=tmp_path, attestation=None, commands=[], frozen_path=tmp_path))
    assert created == []


@pytest.mark.skipif(os.name != 'nt', reason='Windows process table seam')
def test_b1_declared_agent_below_detected_root_cannot_undercount(monkeypatch: pytest.MonkeyPatch) -> None:
    table = {1: (0, 'explorer'), 10: (1, 'claude'), 12: (10, 'subagent'), 20: (10, 'pwsh'), 30: (20, 'python')}
    monkeypatch.setattr(runtime, '_windows_process_table', lambda: table)
    monkeypatch.setattr(runtime, '_windows_creation_time', {1: 1, 10: 2, 12: 3, 20: 3, 30: 4}.get)
    monkeypatch.setattr(runtime, '_windows_resident', lambda pid: 10)
    monkeypatch.setattr(runtime.os, 'getpid', lambda: 30)
    monkeypatch.delenv('D39_AGENT_PID', raising=False)
    baseline = runtime._owned_memory_sample(runtime.OwnedProcessScope())
    monkeypatch.setenv('D39_AGENT_PID', '20')
    assert runtime._owned_memory_sample(runtime.OwnedProcessScope()) == baseline == (0, 40)


@pytest.mark.parametrize('text', [
    b'uv 0.12.15\n', b'uv 0.12.15 (x86_64-pc-windows-msvc)\n',
    b'uv 0.12.15 (abcdef012 2026-09-30 x86_64-unknown-linux-gnu)\n',
    b'uv 0.12.15 (abcdef0 2026-09-30 aarch64-apple-darwin)\n',
])
def test_b2_real_uv_version_formats(text: bytes) -> None:
    assert release._parse_uv_version(text) == '0.12.15'


@pytest.mark.parametrize('text', [b'uv 0.12.14', b'uv 0.12.15 (arbitrary words)',
                                     b'uv 0.12.15\nextra', b'uv 0.12.15 (' + b'x' * 200 + b')'])
def test_b2_uv_version_is_strict(text: bytes) -> None:
    with pytest.raises(ValueError):
        release._parse_uv_version(text)


def test_b3_isolated_vite_resolves_sibling_esbuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = tmp_path / 'source' / 'frontend'
    packages = frontend / 'node_modules' / '.pnpm' / 'fixture' / 'node_modules'
    vite = packages / 'vite'
    esbuild = packages / 'esbuild'
    for package in (vite, esbuild):
        package.mkdir(parents=True)
    (vite / 'package.json').write_text('{"name":"vite","version":"5.4.21"}')
    (esbuild / 'package.json').write_text('{"name":"esbuild","version":"0.21.5","main":"index.js"}')
    (esbuild / 'index.js').write_text("exports.version='0.21.5';exports.transformSync=s=>({code:s});")
    # A directory junction needs no elevated symlink privilege on Windows.
    link = frontend / 'node_modules' / 'vite'
    if os.name == 'nt':
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(vite)], check=True, capture_output=True)
    else:
        link.symlink_to(vite, target_is_directory=True)
    (frontend / 'pnpm-lock.yaml').write_text('packages:\n  esbuild@0.21.5:\n')
    async def probe(scope: object, group: object, argv: tuple[str, ...], env: dict[str, str]) -> bytes:
        return subprocess.check_output(argv, cwd=tmp_path, timeout=10)
    monkeypatch.setattr(release, '_probe', probe)
    node = shutil.which('node')
    assert node is not None
    asyncio.run(release._esbuild_preflight(None, SimpleNamespace(source=tmp_path / 'source'), SimpleNamespace(executable=Path(node)), {}))


def test_b4_origin_probe_does_not_execute_packages(tmp_path: Path) -> None:
    (tmp_path / 'explosive.py').write_text("raise RuntimeError('must not import')\n")
    metadata = tmp_path / 'explosive-1.0.dist-info'
    metadata.mkdir()
    (metadata / 'METADATA').write_text('Name: explosive\nVersion: 1.0\n')
    result = subprocess.run([sys.executable, '-B', '-c', release._PYTHON_PROBE, '["explosive"]'],
                            cwd=tmp_path, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = release._probe_json(result.stdout)
    assert data['origins']['explosive'] == str(tmp_path / 'explosive.py')
    assert data['versions']['explosive'] == '1.0'


@pytest.mark.parametrize('offset', [61440, 126976, 61439, 61441])
def test_i1_stream_preserves_left_boundary(offset: int) -> None:
    raw = b'.' * (offset - 1) + b'xsk-' + b'a' * 24 + b'.' * 65536
    counts = dict.fromkeys(release.SCAN_RULE_IDS, 0)
    release._scan_stream(io.BufferedReader(io.BytesIO(raw)), counts, len(raw) + 1)
    assert counts['credential_token'] == 0


def test_i6_shared_receipt_rejects_incomplete_tools() -> None:
    with pytest.raises(ValueError):
        SmokeStageReceipt.model_validate_json(_bytes(_receipt() | {'tools': [_binding()]}))


def test_i6_documentation_is_immutable_and_hashable() -> None:
    summary = BrowserSummary.model_validate_json(_bytes(_summary('browser')))
    hash(summary)
    with pytest.raises((ValueError, TypeError)):
        summary.documentation_checks['setup_paths'] = False


def test_l7_pytest_cache_path_preserves_japanese(tmp_path: Path) -> None:
    root = tmp_path / '日本語 workspace'
    root.mkdir()
    env = release.build_group_environment(root, python_executable=Path(sys._base_executable), node_executable=None)
    assert shlex.split(env['PYTEST_ADDOPTS']) == ['-o', 'cache_dir=' + (root / 'pytest-cache').as_posix()]


@pytest.mark.parametrize('encoding', ['utf-16', 'utf-32'])
def test_l8_probe_requires_utf8(encoding: str) -> None:
    with pytest.raises(ValueError):
        release._probe_json(json.dumps({'a': 1}).encode(encoding))


def test_step0_archive_ignores_local_eol_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests import d37_pinned_support as support
    repo = tmp_path / 'repo'
    repo.mkdir()
    def git(*args: str) -> bytes:
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)
    git('init', '-q')
    git('config', 'user.name', 'Synthetic')
    git('config', 'user.email', 'synthetic@example.invalid')
    git('config', 'core.autocrlf', 'false')
    (repo / 'source.txt').write_bytes(b'first\nsecond\n')
    git('add', '.')
    git('commit', '-qm', 'Synthetic archive')
    monkeypatch.setattr(support, 'PINNED_D35', git('rev-parse', 'HEAD').decode().strip())
    git('config', 'core.autocrlf', 'true')
    support.verified_archive(repo, tmp_path / 'unpacked')
    assert (tmp_path / 'unpacked/source.txt').read_bytes() == b'first\nsecond\n'


@pytest.mark.parametrize('entrypoint', ['materialize_candidate_runtime', 'd39_smoke'])
def test_m5_cli_redacts_git_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str], entrypoint: str) -> None:
    import importlib
    cli = importlib.import_module('evaluation.scripts.' + entrypoint)
    def non_git(**kwargs: object) -> None:
        subprocess.run(['git', '-C', str(tmp_path), 'rev-parse', '--show-toplevel'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, 'GIT_CEILING_DIRECTORIES': str(tmp_path.parent)})
    function = 'run_candidate_smokes' if entrypoint == 'd39_smoke' else 'materialize_candidate_runtime'
    monkeypatch.setattr(cli, function, non_git)
    args = [item for flag in ('candidate-root', 'freeze-manifest', 'work-root', 'output')
            for item in ('--' + flag, str(tmp_path / flag))]
    if entrypoint == 'd39_smoke':
        args += ['--runtime-root', str(tmp_path / 'runtime'), '--materialization', str(tmp_path / 'record'),
                 '--expected-materialization-sha256', 'a' * 64]
    assert cli.main(args) == 2
    output = capsys.readouterr()
    assert str(tmp_path) not in output.err + output.out
    assert 'Traceback' not in output.err + output.out


def test_i1_scan_includes_tracked_excluded_paths(tmp_path: Path) -> None:
    from tests.test_d36_freeze import _make_candidate
    root, commit = _make_candidate(tmp_path, extras={'.env.prod': b'example setting',
                                                  'frontend/node_modules/example.js': b'fixture'})
    counts = release.scan_candidate_commit(root, commit, ())
    assert counts['tracked_private_state'] >= 1
    assert counts['tracked_generated_state'] >= 1


def _scan_repo(tmp_path: Path) -> tuple[Path, Callable[..., str]]:
    repo = tmp_path / 'scan-repo'
    repo.mkdir()
    def git(*args: str, data: bytes | None = None) -> str:
        return subprocess.run(['git', '-C', str(repo), *args], check=True, input=data,
                              capture_output=True).stdout.decode('utf-8').strip()
    git('init', '-q')
    git('config', 'user.email', 'fixture@example.invalid')
    git('config', 'user.name', 'Fixture')
    git('config', 'core.autocrlf', 'false')
    return repo, git


def test_i1_scan_reads_committed_blobs_not_working_tree_encoding(tmp_path: Path) -> None:
    repo, git = _scan_repo(tmp_path)
    token = 'sk-' + 'A1b2' * 6  # synthetic credential shape, assembled at runtime
    (repo / '.gitattributes').write_bytes(b'*.txt text working-tree-encoding=UTF-16LE eol=lf\n')
    (repo / 'notes.txt').write_bytes(('note ' + token + '\n').encode('utf-16-le'))
    git('add', '.')
    git('commit', '-q', '-m', 'fixture')
    assert token.encode('ascii') in subprocess.run(['git', '-C', str(repo), 'cat-file', '-p', 'HEAD:notes.txt'],
                                                   check=True, capture_output=True).stdout
    assert release.scan_candidate_commit(repo, git('rev-parse', 'HEAD'), ())['credential_token'] == 1


def test_i1_scan_ignores_replace_refs_that_hide_tracked_files(tmp_path: Path) -> None:
    repo, git = _scan_repo(tmp_path)
    (repo / 'readme.txt').write_bytes(b'public\n')
    (repo / '.env.prod').write_bytes(b'setting\n')
    git('add', '.')
    git('commit', '-q', '-m', 'fixture')
    tree = git('rev-parse', 'HEAD^{tree}')
    readme = git('rev-parse', 'HEAD:readme.txt')
    hidden = git('mktree', data=('100644 blob ' + readme + '\treadme.txt\n').encode('ascii'))
    git('replace', tree, hidden)
    assert '.env.prod' not in git('ls-tree', '-r', '--name-only', 'HEAD')
    assert release.scan_candidate_commit(repo, git('rev-parse', 'HEAD'), ())['tracked_private_state'] == 1


@pytest.mark.skipif(os.name != 'nt', reason='PE (MZ) native fixtures; POSIX requires ELF/Mach-O binaries')
@pytest.mark.parametrize('changed', ['pyvenv.cfg', 'base.exe'])
def test_m1_sandbox_binds_config_and_native_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    base = tmp_path / 'base.exe'
    base.write_bytes(b'MZbase fixture')
    environment = tmp_path / 'env'
    executable = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    executable.parent.mkdir(parents=True)
    if os.name == 'nt':
        executable.write_bytes(b'MZvenv redirector')
    else:
        executable.symlink_to(base)
    config = environment / 'pyvenv.cfg'
    config.write_bytes(b'include-system-site-packages = false\n')
    site = environment / ('Lib/site-packages' if os.name == 'nt' else 'lib/python3.12/site-packages')
    site.mkdir(parents=True)
    package = site / 'fixture.py'
    package.write_bytes(b'pass\n')
    data = dict(version='3.12.12', executable=str(executable), prefix=str(environment),
                base_prefix=str(tmp_path), base_executable=str(base), origins={'fixture': str(package)},
                locations={'fixture': []}, versions={'fixture': '1.0'})
    async def probe(*args: object) -> bytes:
        return json.dumps(data).encode()
    monkeypatch.setattr(release, '_probe', probe)
    monkeypatch.setattr(release, '_BACKEND_IMPORTS', ('fixture',))
    base_fp = release._tool_file(base, 'tools/python_bootstrap')
    bootstrap = release._ToolSet(base, None, (release.ToolExecutionBinding(role='python_bootstrap', version='3.12.12', executable=base_fp, launcher=None),), ((base, base_fp),), tmp_path)
    result = asyncio.run(release._sandbox_python(None, SimpleNamespace(root=tmp_path), {}, bootstrap))
    (config if changed == 'pyvenv.cfg' else base).write_bytes(b'MZchanged fixture')
    with pytest.raises(ValueError):
        result.verify()


@pytest.mark.skipif(os.name != 'nt', reason='PE (MZ) native fixtures; POSIX requires ELF/Mach-O binaries')
@pytest.mark.parametrize('changed', ['pnpm.cjs', 'worker.js', 'pnpmrc'])
def test_m10_pnpm_implementation_is_rehashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    node = tmp_path / 'native/node.exe'
    node.parent.mkdir()
    node.write_bytes(b'MZnative fixture')
    npm = node.parent / 'node_modules/npm' if os.name == 'nt' else node.parent.parent / 'lib/node_modules/npm'
    (npm / 'bin').mkdir(parents=True)
    (npm / 'package.json').write_text('{"name":"npm","version":"11.6.2"}')
    (npm / 'bin/npx-cli.js').write_text('fixture')
    pnpm = tmp_path / 'npm-cache/_npx/test/node_modules/pnpm'
    (pnpm / 'bin').mkdir(parents=True)
    (pnpm / 'dist').mkdir()
    (pnpm / 'package.json').write_text('{"name":"pnpm","version":"10.18.3"}')
    (pnpm / 'bin/pnpm.cjs').write_text("require('../dist/pnpm.cjs')")
    implementation = pnpm / 'dist/pnpm.cjs'
    implementation.write_text('fixture')
    (pnpm / 'dist/worker.js').write_text('worker fixture')
    (pnpm / 'dist/pnpmrc').write_text('builtin fixture')
    (tmp_path / 'source/frontend').mkdir(parents=True)
    async def probe(scope: Any, group: Any, argv: tuple[str, ...], env: Any) -> bytes:
        if '-e' in argv:
            return json.dumps({'version': '24.11.1', 'executable': str(node)}).encode()
        return b'10.18.3' if '-y' in argv or 'pnpm.cjs' in argv[1] else b'11.6.2'
    monkeypatch.setattr(release, '_probe', probe)
    tools = asyncio.run(release._frontend_tools(None, SimpleNamespace(root=tmp_path, source=tmp_path / 'source'), {}, node))
    tools.verify()
    (pnpm / 'dist' / changed).write_text('changed')
    with pytest.raises(ValueError):
        tools.verify()


def test_m11_smoke_requires_producer_attestation() -> None:
    from evaluation.smoke_contracts import SmokeManifest
    from tests.test_d39_verifier_driver import _synthetic_smoke
    record = SimpleNamespace(**{key: value for key, value in _receipt().items()
                                if key in ('candidate_id', 'git_commit', 'freeze_sha256', 'runtime_instance_id', 'runtime_source_sha256')})
    payload = _synthetic_smoke(record, 'a' * 64)
    payload.pop('producer_tool_sha256', None)
    with pytest.raises(ValueError):
        SmokeManifest.model_validate_json(_bytes(payload))


def test_m11_shared_verification_rejects_different_smoke_producer() -> None:
    import hashlib
    from evaluation.smoke_contracts import VerificationManifest
    from tests.test_d39_verifier_driver import _synthetic_smoke
    record = SimpleNamespace(candidate_id='a' * 16 + '-' + 'b' * 12, git_commit='b' * 40, freeze_sha256='a' * 64,
                             runtime_instance_id='a' * 64, runtime_source_sha256='a' * 64)
    smoke = _synthetic_smoke(record, 'a' * 64)
    payload = dict(schema_version=1, **record.__dict__, materialization_sha256='a' * 64,
                   verifier_tool_sha256=smoke['producer_tool_sha256'], status='failed', commands=[], smoke_manifest=smoke,
                   smoke_manifest_sha256=hashlib.sha256(_bytes(smoke)).hexdigest(), secret_scan_passed=False,
                   candidate_clean_before=True, candidate_clean_after=True, candidate_snapshot_before_sha256='a' * 64,
                   candidate_snapshot_after_sha256='a' * 64, runtime_snapshot_after_sha256='a' * 64, cleanup_status='completed')
    # Positive control: the same smoke from the same tool revision is accepted.
    VerificationManifest.model_validate_json(_bytes(payload))
    assert smoke['producer_tool_sha256'] != 'c' * 64
    with pytest.raises(ValueError, match='producer'):
        VerificationManifest.model_validate_json(_bytes(payload | {'verifier_tool_sha256': 'c' * 64}))


@pytest.mark.skipif(os.name != 'nt', reason='Windows ancestor junction rejection')
def test_l9_browser_ancestor_junction_is_rejected(tmp_path: Path) -> None:
    from evaluation.scripts import d39_smoke as producer
    real = tmp_path / 'real'
    real.mkdir()
    (real / 'chrome.exe').write_bytes(b'MZfixture')
    link = tmp_path / 'linked'
    subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(real)], check=True, capture_output=True)
    try:
        with pytest.raises(ValueError):
            producer._installed_browser(link / 'chrome.exe')
    finally:
        link.rmdir()
