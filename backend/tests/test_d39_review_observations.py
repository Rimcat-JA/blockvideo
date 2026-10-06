"""D39 observed browser/startup/media regressions; no operational publication."""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import os
import time
from pathlib import Path
from typing import Any

import pytest

from evaluation import browser_smoke as browser
from evaluation.scripts import d39_candidate_smoke as candidate
from evaluation.tool_attestation import canonical_json_bytes
from tests.test_d39_review_runtime import synthetic_agent as synthetic_agent


def test_every_browser_journey_uses_measured_width_keyboard_and_client_width(tmp_path: Path) -> None:
    class Session:
        width = 0
        calls: list[tuple[int, str, dict[str, Any]]] = []
        visits: list[tuple[int, str]] = []
        expressions: list[str] = []

        def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((self.width, method, params))
            if method == 'Emulation.setDeviceMetricsOverride':
                self.width = params['width']
            return {'data': base64.b64encode(b'\x89PNG\r\n\x1a\n').decode()}

        def navigate(self, url: str) -> None:
            self.visits.append((self.width, url))

        def evaluate(self, expression: str) -> Any:
            self.expressions.append(expression)
            if expression == 'innerWidth':
                return self.width
            if 'scrollWidth' in expression:
                return 'clientWidth' in expression  # Classic scrollbar overflow.
            return True

    session = Session()
    counter = tmp_path / 'counter.json'
    counter.write_bytes(canonical_json_bytes({'operation_posts': 0}) + b'\n')
    original = session.call

    def dispatch(method: str, params: dict[str, Any]) -> dict[str, Any]:
        result = original(method, params)
        if method == 'Input.dispatchKeyEvent' and params.get('type') == 'keyDown' and params.get('key') == 'Enter':
            counter.write_bytes(canonical_json_bytes({'operation_posts': 1 if session.width == 390 else 2}) + b'\n')
        return result

    session.call = dispatch  # CDP transport seam; journey itself is exercised.
    result = browser.recovery_journey(session, base_url='http://127.0.0.1:1234', migration_url='http://127.0.0.1:1235/',
                                     projects={'waiting': 1, 'safe_retry': 2, 'safe_retry_wide': 5, 'unknown_remote': 3, 'media': 4},
                                     screenshots=tmp_path, counter_path=counter)
    assert result['narrow_width'] == 390 and result['wide_width'] == 1440
    assert result['horizontal_overflow'] is True
    assert result['duplicate_post_count'] == 1
    for width in (390, 1440):
        assert (width, 'http://127.0.0.1:1234/projects/4') in session.visits
        assert (width, 'http://127.0.0.1:1235/') in session.visits
        enters = [p for w, m, p in session.calls if w == width and m == 'Input.dispatchKeyEvent' and p.get('key') == 'Enter' and p['type'] == 'keyDown']
        assert enters == [{'type': 'keyDown', 'key': 'Enter', 'code': 'Enter', 'text': '\r', 'windowsVirtualKeyCode': 13}] * 2
    assert any('#generation-history' in item for item in session.expressions)
    assert len(list(tmp_path.glob('*.png'))) == 10


def test_ffmpeg_observer_captures_actual_nonzero_and_restores_subprocess(tmp_path: Path) -> None:
    original = asyncio.create_subprocess_exec
    async def exercise() -> None:
        with candidate._observe_ffmpeg(Path(sys.executable)) as exits:
            process = await asyncio.create_subprocess_exec(sys.executable, '-c', 'raise SystemExit(7)')
            assert await process.wait() == 7
        assert exits == [7]
    asyncio.run(exercise())
    assert asyncio.create_subprocess_exec is original


def test_failed_ffprobe_keeps_real_code_without_parsing_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    def run(argv: tuple[str, ...], **options: Any) -> subprocess.CompletedProcess[str]:
        options['stdout'].write(b'{}')
        return subprocess.CompletedProcess(argv, 9)
    monkeypatch.setattr(candidate.subprocess, 'run', run)
    assert candidate._probe_duration('ffprobe', tmp_path / 'video', tmp_path / 'probe.json') == (9, None)


def test_startup_requires_http_listener_instead_of_inprocess_fallback(tmp_path: Path) -> None:
    # No listener and deliberately no candidate source: it must first connect to
    # the owned HTTP listener, rather than importing/starting an in-process app.
    import httpx
    with pytest.raises(httpx.ConnectError):
        candidate._startup({'port': 1, 'owner_token': 'a' * 64}, 'all_tools')


def test_browser_main_binds_version_from_owned_cdp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('profile', 'screenshots'):
        (tmp_path / name).mkdir()
    config = dict(profile=str(tmp_path / 'profile'), screenshots=str(tmp_path / 'screenshots'), summary_path=str(tmp_path / 'result.json'),
                  counter_path=str(tmp_path / 'counter.json'), base_url='http://127.0.0.1:1234', migration_url='http://127.0.0.1:1235/',
                  projects=dict(waiting=1, safe_retry=2, safe_retry_wide=5, unknown_remote=3, media=4))
    path = tmp_path / 'config.json'
    path.write_bytes(canonical_json_bytes(config) + b'\n')
    class Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __enter__(self) -> Session:
            return self
        def __exit__(self, *args: Any) -> None:
            pass
        def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == 'Browser.getVersion'
            return {'product': 'Chrome/154.0.1.2'}
    monkeypatch.setattr(browser, 'discover_owned_browser', lambda *a, **kw: ('ws://127.0.0.1:1234/devtools/page/synthetic', 1234))
    monkeypatch.setattr(browser, 'CDPSession', Session)
    monkeypatch.setattr(browser, 'recovery_journey', lambda *a, **kw: {'observed': True})
    assert browser.main(['--configuration', str(path)]) == 0
    assert json.loads((tmp_path / 'result.json').read_bytes()) == {'observed': True, 'browser_version': 'Chrome/154.0.1.2'}


def test_browser_stage_never_probes_chrome_without_owned_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluation.scripts import d39_smoke as producer
    from evaluation import release_verification as verification
    from types import SimpleNamespace
    executable = Path(sys._base_executable)
    module = tmp_path / 'env/client.py'
    module.parent.mkdir()
    module.write_bytes(b'fixture')
    monkeypatch.setattr(producer, '_installed_browser', lambda value: executable)
    async def probe(scope: Any, group: Any, argv: tuple[str, ...], env: Any) -> bytes:
        assert '--version' not in argv
        return json.dumps({'version': '16.1.1', 'origin': str(module)}).encode()
    async def action(*args: Any, **kwargs: Any) -> None:
        raise ValueError('synthetic stop before media')
    monkeypatch.setattr(verification, '_probe', probe)
    monkeypatch.setattr(producer, '_candidate_action', action)
    with pytest.raises(ValueError, match='synthetic stop before media'):
        asyncio.run(producer._browser_stage(None, SimpleNamespace(root=tmp_path), SimpleNamespace(executable=executable), {}, {}, None, (), None))


@pytest.mark.skipif(os.name != 'nt', reason='installed Windows Chrome native seam')
def test_real_owned_chrome_tab_enter_and_version(tmp_path: Path, synthetic_agent: None) -> None:
    from evaluation import blinded_runtime as runtime
    from evaluation.scripts import d39_smoke as producer
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    try:
        chrome = producer._installed_browser(None)
    except ValueError:
        pytest.skip('no installed Chrome; native browser seam not run')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass
        def do_GET(self) -> None:
            raw = b"<html><body><input id='before'><button id='retry' onclick='window.posts++;this.disabled=true'>Retry</button><script>window.posts=0</script></body></html>"
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tmp_path / 'profile'
    profile.mkdir()
    async def exercise() -> None:
        async with runtime.OwnedProcessScope() as scope:
            child = await runtime.start_owned_process(scope=scope, argv=(str(chrome), '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions', '--disable-default-apps', '--no-proxy-server', '--renderer-process-limit=1', '--disk-cache-size=1048576', '--media-cache-size=1048576', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', '--user-data-dir=' + str(profile), 'about:blank'), cwd=tmp_path, env=dict(os.environ), stdout_path=tmp_path / 'out', stderr_path=tmp_path / 'err', deadline_seconds=240)
            def observe() -> None:
                # A cold Chrome start can take far longer than a warm one (disk cache,
                # antivirus scanning), so allow 3 minutes for its DevTools endpoint.
                url, port = browser.discover_owned_browser(profile, deadline=time.monotonic() + 180)
                with browser.CDPSession(url, port=port) as session:
                    assert session.call('Browser.getVersion', {})['product'].startswith(('Chrome/', 'HeadlessChrome/'))
                    session.navigate(f'http://127.0.0.1:{server.server_port}/')
                    browser._wait(session, "!!document.querySelector('#retry')")
                    assert session.evaluate("(()=>{document.querySelector('#before').focus();return true})()")
                    for kind in ('keyDown', 'keyUp'):
                        session.call('Input.dispatchKeyEvent', {'type': kind, 'key': 'Tab', 'code': 'Tab'})
                    assert session.evaluate("document.activeElement.id==='retry'")
                    for _ in range(2):
                        session.call('Input.dispatchKeyEvent', {'type': 'keyDown', 'key': 'Enter', 'code': 'Enter', 'text': '\r', 'windowsVirtualKeyCode': 13})
                        session.call('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': 'Enter', 'code': 'Enter', 'windowsVirtualKeyCode': 13})
                    assert session.evaluate('window.posts') == 1
            await asyncio.to_thread(observe)
            await child.stop()
            assert child._tree_confirmed and not scope._failed
    try:
        asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.mark.skipif(os.name != 'nt', reason='installed Windows Chrome native seam')
def test_installed_chrome_opens_devtools_under_the_group_environment(tmp_path: Path) -> None:
    # Chrome resolves its default data folder from %USERPROFILE%\AppData\Local; without
    # that layout it treats every profile as default and refuses remote debugging.
    import subprocess
    from evaluation import release_verification as release
    from evaluation.scripts import d39_smoke as producer
    try:
        chrome = producer._installed_browser(None)
    except ValueError:
        pytest.skip('no installed Chrome; native browser seam not run')
    env = release.build_group_environment(tmp_path, python_executable=Path(sys._base_executable), node_executable=None)
    profile = tmp_path / 'profile'
    profile.mkdir()
    process = subprocess.Popen([str(chrome), '--headless=new', '--no-first-run', '--remote-debugging-address=127.0.0.1',
                                '--remote-debugging-port=0', '--user-data-dir=' + str(profile), 'about:blank'],
                               env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and not (profile / 'DevToolsActivePort').exists():
            time.sleep(0.2)
        assert (profile / 'DevToolsActivePort').exists()
    finally:
        process.kill()
        process.wait(timeout=30)
