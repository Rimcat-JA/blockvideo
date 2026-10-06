"""Synthetic D39 production-seam probes; no real evaluation or user browser."""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from tests.test_d39_verifier_driver import tools_repo as tools_repo
from tests.test_d39_release_verification import publication as publication

import pytest
from pydantic import ValidationError

from evaluation.smoke_contracts import BrowserSummary

DOC_KEYS = ('setup_paths', 'locked_versions', 'mode_commands', 'recovery_codes', 'migration_restore', 'limitation_boundary')


def _browser(**changes: Any) -> dict[str, Any]:
    return dict(stage='browser', narrow_width=390, wide_width=1440, waiting_ok=True,
                safe_retry_ok=True, unknown_remote_blocked=True, migration_failed_ok=True,
                keyboard_ok=True, duplicate_post_count=1, horizontal_overflow=False,
                playback_ok=True, documentation_checks=dict.fromkeys(DOC_KEYS, True)) | changes


def _api(module: str) -> Any:
    assert importlib.util.find_spec(module) is not None, f'{module} D39 boundary missing'
    return importlib.import_module(module)


def test_browser_documentation_schema_requires_six_individual_raw_checks() -> None:
    with pytest.raises(ValidationError):
        BrowserSummary.model_validate(_browser(documentation_checks=True))
    assert BrowserSummary.model_validate(_browser()).documentation_checks.model_dump() == dict.fromkeys(DOC_KEYS, True)


@pytest.mark.parametrize('drift', ['missing', 'extra', 'int', 'false'])
def test_browser_documentation_checks_preserve_failure_but_reject_schema_drift(drift: str) -> None:
    checks: dict[str, Any] = dict.fromkeys(DOC_KEYS, True)
    if drift == 'missing':
        checks.pop('setup_paths')
    elif drift == 'extra':
        checks['human_approved'] = True
    elif drift == 'int':
        checks['setup_paths'] = 1
    else:
        checks['locked_versions'] = False
    if drift == 'false':
        assert not BrowserSummary.model_validate(_browser(documentation_checks=checks)).documentation_checks['locked_versions']
    else:
        with pytest.raises(ValidationError):
            BrowserSummary.model_validate(_browser(documentation_checks=checks))


def _protocol() -> dict[str, Any]:
    methods = {
        'Browser': ['getVersion'], 'Page': ['navigate', 'captureScreenshot'],
        'Runtime': ['evaluate'], 'Input': ['dispatchKeyEvent'],
        'Emulation': ['setDeviceMetricsOverride'],
    }
    return {'domains': [{'domain': d, 'commands': [{'name': m} for m in ms]} for d, ms in methods.items()]}


def test_installed_cdp_protocol_requires_all_fixed_methods() -> None:
    browser = _api('evaluation.browser_smoke')
    browser.validate_protocol(_protocol())
    protocol = _protocol()
    protocol['domains'][3]['commands'] = []
    with pytest.raises(ValueError):
        browser.validate_protocol(protocol)


@pytest.mark.parametrize('url', ['ws://remote.invalid/devtools/page/x', 'wss://127.0.0.1:9222/x',
                                 'ws://name:secret@127.0.0.1:9222/x', 'ws://127.0.0.1:9222/x?token=secret',
                                 'ws://localhost:9222/x', 'ws://127.0.0.1:0/x'])
def test_cdp_endpoint_must_be_owned_literal_loopback(url: str) -> None:
    browser = _api('evaluation.browser_smoke')
    with pytest.raises(ValueError):
        browser.validate_cdp_url(url, port=9222)


@pytest.mark.parametrize('response', [
    {'id': 1, 'error': {'message': 'private sentinel'}},
    {'id': 2, 'result': {}},
    {'id': 1, 'result': {'exceptionDetails': {'text': 'private sentinel'}}},
])
def test_cdp_roundtrip_rejects_errors_exception_details_and_wrong_ids(response: dict[str, Any]) -> None:
    from websockets.sync.server import serve
    browser = _api('evaluation.browser_smoke')
    def handle(connection: Any) -> None:
        connection.recv(timeout=5)
        connection.send(json.dumps(response))
    with serve(handle, '127.0.0.1', 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        try:
            with browser.CDPSession(f'ws://127.0.0.1:{port}/devtools/page/test', port=port) as session:
                with pytest.raises(ValueError) as refusal:
                    session.call('Browser.getVersion', {})
                assert 'private sentinel' not in str(refusal.value)
        finally:
            server.shutdown()
            thread.join(5)
            assert not thread.is_alive()


def test_cdp_roundtrip_uses_real_transport_and_binds_result() -> None:
    from websockets.sync.server import serve
    browser = _api('evaluation.browser_smoke')
    def handle(connection: Any) -> None:
        request = json.loads(connection.recv(timeout=5))
        assert request == {'id': 1, 'method': 'Browser.getVersion', 'params': {}}
        connection.send(json.dumps({'id': 1, 'result': {'product': 'Synthetic/1.0'}}))
    with serve(handle, '127.0.0.1', 0, compression=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        try:
            with browser.CDPSession(f'ws://127.0.0.1:{port}/devtools/page/test', port=port) as session:
                assert session.call('Browser.getVersion', {}) == {'product': 'Synthetic/1.0'}
                with pytest.raises(ValueError):
                    session.call('Network.getAllCookies', {})
        finally:
            server.shutdown()
            thread.join(5)
            assert not thread.is_alive()


def test_candidate_bootstrap_and_smoke_producer_boundaries_exist() -> None:
    _api('evaluation.scripts.d39_candidate_smoke')
    producer = _api('evaluation.scripts.d39_smoke')
    assert callable(producer.run_candidate_smokes)


@pytest.fixture(scope='module')
def pinned_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.d37_pinned_support import verified_archive
    directory = tmp_path_factory.mktemp('d39-pinned-smoke')
    root = directory / 'source'
    verified_archive(Path(__file__).parents[2], root)
    return root


def _child(pinned_source: Path, group: Path, action: str, *, provider_url: str = 'http://127.0.0.1:1234/v1', storage: Path | None = None) -> dict[str, Any]:
    producer = _api('evaluation.scripts.d39_smoke')
    from evaluation.tool_attestation import canonical_json_bytes
    # Verified source is read-only test input; only the owned test group's state changes.
    frontend = group / 'frontend'
    frontend.mkdir(exist_ok=True)
    (frontend / 'index.html').write_text('<html>synthetic</html>', encoding='ascii')
    profile = group / 'profile.json'
    profile.write_bytes(canonical_json_bytes(producer._profile()) + b'\n')
    summary = group / (action + '.summary.json')
    config = dict(group_root=str(group), source_root=str(pinned_source), storage=str(storage or group / 'storage'),
                  frontend=str(frontend), profile=str(profile), index=str(group / 'index'),
                  provider_url=provider_url, model='synthetic-d39-chat', port=producer._port(), summary_path=str(summary),
                  scenario='stateful' if action == 'stateful_startup' else 'normal', owner_token='a' * 64, **{role: str(Path(found).resolve(strict=True)) if (found := shutil.which(role)) else None for role in ('ffmpeg', 'ffprobe')})
    # Source is a verified archive alongside the state root; the real production
    # sandbox contains both. Use its common parent as the owned root in this test.
    config['group_root'] = str(group.parent)
    path = group / (action + '.config.json')
    path.write_bytes(canonical_json_bytes(config) + b'\n')
    script = Path(__file__).parents[1] / 'evaluation/scripts/d39_candidate_smoke.py'
    env = {name: os.environ[name] for name in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP') if name in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output = group / (action + '.out')
    error = group / (action + '.err')
    with output.open('wb') as stdout, error.open('wb') as stderr:
        server = None
        try:
            if action.endswith('_startup') or action == 'seed_browser':
                import httpx
                import time
                serve_path = group / (action + '.serve.json')
                serve_path.write_bytes(canonical_json_bytes(config | {'summary_path': str(group / (action + '.counter.json'))}) + b'\n')
                server = subprocess.Popen([sys.executable, '-B', '-c', producer._BOOTSTRAP, str(script), '--action', 'serve', '--configuration', str(serve_path)],
                                          cwd=pinned_source / 'backend', env=env, stdout=stdout, stderr=stderr)
                with httpx.Client(trust_env=False, timeout=2) as client:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        assert server.poll() is None, error.read_bytes().decode(errors='replace')
                        try:
                            response = client.get(f"http://127.0.0.1:{config['port']}/__d39-owned")
                            if response.status_code == 200 and response.json() == {'owner': config['owner_token']}:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.05)
                    else:
                        pytest.fail('pinned owned server readiness deadline exceeded')
            bootstrap = producer._BOOTSTRAP
            if action == 'seed_browser':
                bootstrap = "import runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); c=m['read_configuration'](Path(sys.argv[-1])); Path(c['summary_path']).write_bytes(m['_canonical'](m['_seed_browser'](c)))"
            result = subprocess.run([sys.executable, '-B', '-c', bootstrap, str(script), '--action', action, '--configuration', str(path)],
                                    cwd=pinned_source / 'backend', env=env, stdout=stdout, stderr=stderr, timeout=120)
        finally:
            if server is not None:
                server.terminate()
                server.wait(timeout=10)
    assert result.returncode == 0, error.read_bytes()[:8192].decode('utf-8', errors='replace')
    return json.loads(summary.read_bytes())


def test_pinned_candidate_migration_and_restore_execute_real_database_contracts(pinned_source: Path) -> None:
    group = pinned_source.parent / 'migration-group'
    group.mkdir()
    migration = _child(pinned_source, group, 'legacy_migration')
    assert migration['from_version'] == 0 and migration['to_version'] == 1
    assert all(migration[name] is True for name in ('integrity_ok', 'foreign_keys_enabled', 'rows_preserved', 'identities_preserved'))
    assert migration['backup_size'] > 0 and len(migration['backup_sha256']) == 64
    restored = _child(pinned_source, group, 'restore')
    assert restored['restored_version'] == 1
    assert all(restored[name] is True for name in ('integrity_ok', 'foreign_keys_enabled', 'rows_equal', 'identities_equal', 'lease_exclusion_passed'))


def test_pinned_candidate_index_and_both_startup_modes_use_fake_loopback_provider(pinned_source: Path) -> None:
    producer = _api('evaluation.scripts.d39_smoke')
    group = pinned_source.parent / 'startup-group'
    group.mkdir()
    index = _child(pinned_source, group, 'build_index')
    assert index['document_count'] > 1
    with producer.fake_providers() as (url, counts):
        for mode in ('all_tools', 'stateful'):
            result = _child(pinned_source, group, mode + '_startup', provider_url=url, storage=group / mode)
            assert all(result[name] is True for name in ('startup_ready', 'health_ok', 'request_completed'))
            assert result['model_calls'] == 1
            if mode == 'stateful':
                assert result['embedding_calls'] == 1 and result['retrieval_verified'] is True
                assert result['index_sha256'] == index['bundle_sha256']
        assert counts == {'chat': 2, 'embedding': 1}


def test_pinned_candidate_owned_serve_and_seed_register_all_models(pinned_source: Path) -> None:
    group = pinned_source.parent / 'seed-group'
    group.mkdir()
    producer = _api('evaluation.scripts.d39_smoke')
    with producer.fake_providers() as (url, _):
        result = _child(pinned_source, group, 'seed_browser', provider_url=url)
    assert set(result) == {'waiting', 'safe_retry', 'safe_retry_wide', 'unknown_remote'}
    assert len(set(result.values())) == 4


def test_pinned_d35_tracked_credential_literals_still_fail_secret_scan(pinned_source: Path) -> None:
    from evaluation import release_verification as verification
    from tests.d37_pinned_support import PINNED_D35
    raw = subprocess.check_output(['git', '-C', str(Path(__file__).parents[2]), 'ls-tree', '-r', '-z', '--name-only', PINNED_D35])
    paths = tuple(path.decode('utf-8') for path in raw.split(b'\0') if path)
    counts = verification.scan_public_files(pinned_source, paths, ())
    assert counts['credential_token'] > 0


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='native pinned-candidate media requires installed FFmpeg/ffprobe; no downloads')
def test_pinned_candidate_fake_provider_generation_has_real_bound_media_and_probe(pinned_source: Path) -> None:
    group = pinned_source.parent / 'media-group'
    group.mkdir()
    producer = _api('evaluation.scripts.d39_smoke')
    with producer.fake_providers() as (url, _):
        summary = _child(pinned_source, group, 'ffmpeg', provider_url=url)
    assert summary['ffmpeg_exit_code'] == summary['ffprobe_exit_code'] == 0
    assert 0 < summary['duration_ms'] <= 60000
    assert all(summary[name] is True for name in ('providers_fake', 'video_present', 'subtitle_present', 'publication_bound'))
    media = json.loads((group / 'ffmpeg.summary.media.json').read_bytes())
    for name in ('video', 'subtitle'):
        assert 0 < (group / 'storage' / media[name]).stat().st_size <= 32 * 1024 * 1024


@pytest.mark.parametrize('prior_exit', [None, 7])
def test_administrative_server_stop_does_not_poison_reusable_scope_but_exit_failure_does(prior_exit: int | None) -> None:
    from evaluation import blinded_runtime as runtime
    # Lifecycle unit seam, not operational memory/Job acceptance.
    async def exercise() -> None:
        scope = runtime.OwnedProcessScope()
        child = runtime.OwnedProcess(scope, deadline_seconds=30)
        class Process:
            returncode = prior_exit
        terminal = ('exited', prior_exit, False) if prior_exit is not None else ('running', None, False)
        class Stdin:
            def is_closing(self) -> bool:
                return False
            def write(self, raw: bytes) -> None:
                nonlocal terminal
                assert raw == b'2'
                if prior_exit is None:
                    terminal = ('exited', -9, True)
            async def drain(self) -> None:
                pass
        child._process = Process()
        child._process.stdin = Stdin()
        child._observation = lambda: terminal
        child._readers = tuple(asyncio.create_task(asyncio.sleep(0)) for _ in range(2))
        async def terminate() -> None:
            if child._process.returncode is None:
                child._process.returncode = -9
            child._tree_confirmed = True
        child._terminate = terminate
        child._task = asyncio.create_task(child._watch())
        outcome = await child.stop()
        assert outcome.outcome == 'completed'
        assert outcome.exit_code == (7 if prior_exit is not None else -9)
        assert scope._failed is (prior_exit is not None)
    asyncio.run(exercise())


@pytest.mark.parametrize('failure', ['memory_limit', 'output_limit', 'timeout', 'teardown_failed'])
def test_administrative_stop_preserves_late_guard_failures(failure: str) -> None:
    from evaluation import blinded_runtime as runtime
    # A bounded lifecycle unit seam, including failures discovered during teardown.
    async def exercise() -> None:
        scope = runtime.OwnedProcessScope()
        child = runtime.OwnedProcess(scope, deadline_seconds=30)
        class Process:
            returncode = None
        terminal = ('running', None, False)
        class Stdin:
            def is_closing(self) -> bool:
                return False
            def write(self, raw: bytes) -> None:
                nonlocal terminal
                terminal = ('exited', -9, True)
            async def drain(self) -> None:
                pass
        child._process = Process()
        child._process.stdin = Stdin()
        child._observation = lambda: terminal
        child._readers = tuple(asyncio.create_task(asyncio.sleep(0)) for _ in range(2))
        calls = 0
        async def terminate() -> None:
            nonlocal calls
            calls += 1
            child._process.returncode = -9
            if failure == 'memory_limit':
                scope._memory_lost.set()
            elif failure == 'output_limit':
                child._overflow.set()
            elif failure == 'timeout':
                child._stop_reason = 'timeout'
            elif calls > 1:
                raise OSError('synthetic teardown accounting failure')
            child._tree_confirmed = True
        child._terminate = terminate
        child._task = asyncio.create_task(child._watch())
        result = await child.stop()
        assert result.outcome == failure and scope._failed is True
    asyncio.run(exercise())


def test_observed_nonzero_probe_emits_failed_receipt_instead_of_fabricated_pass() -> None:
    producer = _api('evaluation.scripts.d39_smoke')
    from evaluation.smoke_contracts import FFmpegSummary
    from evaluation.smoke_contracts import SmokeStageReceipt
    from tests.test_d39_release_verification import _bytes, _receipt
    from evaluation.evidence_json import parse_canonical_model
    assert hasattr(producer, '_make_receipt'), 'missing actual-observation receipt boundary'
    original = parse_canonical_model(_bytes(_receipt('ffmpeg')), SmokeStageReceipt, maximum=65536)
    data = {name: getattr(original, name) for name in SmokeStageReceipt.model_fields if name not in {'outcome', 'stage', 'summary'}}
    summary = original.summary.model_dump()
    summary['ffprobe_exit_code'] = 7
    receipt = producer._make_receipt(summary=FFmpegSummary.model_validate(summary), **data)
    assert receipt.outcome == 'failed' and receipt.summary.ffprobe_exit_code == 7


def test_frozen_readme_documentation_gate_is_failed_not_waived(pinned_source: Path) -> None:
    producer = _api('evaluation.scripts.d39_smoke')
    inventory = frozenset(path.relative_to(pinned_source).as_posix() for path in pinned_source.rglob('*') if path.is_file())
    checks = producer.documentation_checks(pinned_source, contracts_passed=True, inventory=inventory)
    # Full frozen tree: only the README prerequisite mismatch fails; the stricter
    # fail-closed link, coverage and limitation parsing accept D35's real text.
    assert checks == dict.fromkeys(DOC_KEYS, True) | {'locked_versions': False}


@pytest.mark.parametrize('mutated', [False, True])
def test_candidate_bounded_read_permits_distinct_timestamp_views_but_refuses_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated: bool) -> None:
    from types import SimpleNamespace
    child = _api('evaluation.scripts.d39_candidate_smoke')
    path = tmp_path / 'public.json'
    path.write_bytes(b'{"public":true}\n')
    original = os.fstat
    calls = 0
    def handle_view(fd: int) -> Any:
        nonlocal calls
        calls += 1
        metadata = original(fd)
        if mutated and calls == 2:
            named = path.stat()
            os.utime(path, ns=(named.st_atime_ns, named.st_mtime_ns + 2_000_000))
        return SimpleNamespace(st_dev=metadata.st_dev, st_ino=metadata.st_ino, st_size=metadata.st_size,
                               st_mtime_ns=metadata.st_mtime_ns, st_ctime_ns=metadata.st_ctime_ns + 12345)
    monkeypatch.setattr(child.os, 'fstat', handle_view)
    if mutated:
        with pytest.raises(ValueError, match='changed'):
            child._read(path, maximum=64)
    else:
        assert child._read(path, maximum=64) == b'{"public":true}\n'


@pytest.mark.parametrize('matching', [False, True])
def test_owned_health_requires_listener_nonce_before_trusting_health(matching: bool) -> None:
    import time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from types import SimpleNamespace
    producer = _api('evaluation.scripts.d39_smoke')
    visits: list[str] = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass
        def do_GET(self) -> None:
            visits.append(self.path)
            result = {'owner': ('a' if matching else 'b') * 64} if self.path == '/__d39-owned' else {'status': 'ok'}
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe = producer._http_health(server.server_port, owner_token='a' * 64, child=SimpleNamespace(is_running=lambda: True), deadline=time.monotonic() + 2)
        if matching:
            asyncio.run(probe)
            assert visits == ['/__d39-owned', '/api/health']
        else:
            with pytest.raises(ValueError, match='deadline'):
                asyncio.run(probe)
            assert visits and '/api/health' not in visits
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize('source_drift', [False, True])
def test_synthetic_smoke_driver_failure_keeps_actual_prefix_and_cleans_only_group(
    publication: tuple[Path, Path, Path, Path], tools_repo: Path,
    monkeypatch: pytest.MonkeyPatch, source_drift: bool,
) -> None:
    # Driver failure/identity seam. Installation/native tool/stage observation doubles
    # deliberately do NOT constitute operational acceptance.
    from contextlib import contextmanager
    from types import SimpleNamespace
    from evaluation import release_verification as verification
    from evaluation.smoke_contracts import SmokeStageReceipt
    from evaluation.evidence_json import parse_canonical_model
    from tests.test_d39_release_verification import _bytes, _cleanup, _materialize, _summary
    from tests.test_d39_verifier_driver import _synthetic_smoke
    producer = _api('evaluation.scripts.d39_smoke')
    record, runtime, digest = _materialize(publication)
    fixture = _synthetic_smoke(record, digest)
    bindings = parse_canonical_model(_bytes(fixture['stage_receipts'][0]), SmokeStageReceipt, maximum=65536).tools
    counts = {'chat': 0, 'embedding': 0}
    stages: list[str] = []
    groups: list[Path] = []
    closed: list[bool] = []
    class UnitScope:
        async def __aenter__(self) -> Any:
            return self
        async def close(self) -> None:
            closed.append(True)
    def tool(roles: set[str]) -> Any:
        return SimpleNamespace(executable=Path(sys._base_executable), bindings=tuple(item for item in bindings if item.role in roles), verify=lambda: None)
    async def bootstrap(*args: Any) -> Any:
        return tool({'python_bootstrap', 'uv'})
    async def python(*args: Any) -> Any:
        return tool({'python'})
    async def frontend(*args: Any) -> Any:
        return tool({'node', 'npx', 'pnpm'})
    async def command(**kwargs: Any) -> Any:
        return SimpleNamespace(outcome='completed', exit_code=0)
    async def preflight(*args: Any) -> None:
        pass
    async def probe(*args: Any) -> bytes:
        return b'fixed native probe fixture\n'
    async def action(scope: Any, group: Any, python: Any, env: Any, config: Any, name: str, **kwargs: Any) -> dict[str, Any]:
        if name == 'build_index':
            groups.append(group.root)
            return {}
        stages.append(name)
        summary = _summary(name)
        if name == 'restore':
            summary['rows_equal'] = False
        if source_drift:
            changed = group.source / record.files[0].path
            changed.unlink()
            changed.write_bytes(b'observed synthetic source drift\n')
        return summary
    @contextmanager
    def provider() -> Any:
        yield 'http://127.0.0.1:1234/v1', counts
    monkeypatch.setattr(verification, '_trusted_tool_root', lambda: tools_repo)
    monkeypatch.setattr(producer.blinded_runtime, 'OwnedProcessScope', UnitScope)
    monkeypatch.setattr(verification, '_bootstrap_python', bootstrap)
    monkeypatch.setattr(verification, '_sandbox_python', python)
    monkeypatch.setattr(verification, '_frontend_tools', frontend)
    monkeypatch.setattr(verification, '_execute_command', command)
    monkeypatch.setattr(verification, '_esbuild_preflight', preflight)
    monkeypatch.setattr(verification, '_probe', probe)
    monkeypatch.setattr(producer, '_candidate_action', action)
    monkeypatch.setattr(producer, 'fake_providers', provider)
    # Host symlink privilege is a separate documented prerequisite (own test).
    monkeypatch.setattr(producer, '_symlink_prerequisite', lambda root: None)
    shared_preflight: list[Path] = []
    async def frontend_preflight(scope: Any, group: Any, tools: Any, env: Any) -> None:
        shared_preflight.append(group.root)
    monkeypatch.setattr(verification, '_frontend_preflight', frontend_preflight)
    output = tools_repo / 'evidence/smoke'
    try:
        with pytest.raises(ValueError, match='no complete smoke evidence'):
            producer.run_candidate_smokes(candidate_root=publication[0], freeze_manifest_path=publication[1], runtime_root=runtime,
                                         materialization_path=publication[3], expected_materialization_sha256=digest,
                                         work_root=publication[2], output_dir=output)
        assert closed == [True]
        # The smoke's own frontend group runs the shared local-pnpm guard (M9).
        assert shared_preflight == groups
        assert groups and not any(path.exists() for path in groups)
        assert runtime.exists() and list(publication[2].iterdir()) == [runtime]
        assert not (output / 'smoke-manifest.json').exists()
        if source_drift:
            assert stages == ['legacy_migration'] and list(output.iterdir()) == []
        else:
            assert stages == ['legacy_migration', 'restore']
            receipt = parse_canonical_model((output / 'restore.receipt.json').read_bytes(), SmokeStageReceipt, maximum=65536)
            assert receipt.outcome == 'failed' and receipt.summary.rows_equal is False
            assert not (output / 'all_tools_startup.receipt.json').exists()
    finally:
        _cleanup(publication, runtime, digest)
