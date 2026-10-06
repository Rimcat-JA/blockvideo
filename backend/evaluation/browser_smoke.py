"""Bounded CDP for a newly owned synthetic D39 browser, never a user's session."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import stat
import time
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

METHODS: frozenset[str] = frozenset({'Browser.getVersion', 'Page.navigate', 'Runtime.evaluate', 'Input.dispatchKeyEvent',
                     'Emulation.setDeviceMetricsOverride', 'Page.captureScreenshot'})
_LIMIT = 2 * 1024 * 1024


def validate_cdp_url(url: str, *, port: int) -> str:
    try:
        parsed = urlsplit(url)
        if (type(port) is not int or not 1 <= port <= 65535 or parsed.scheme != 'ws'
                or parsed.hostname != '127.0.0.1' or parsed.port != port
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or not parsed.path.startswith('/devtools/')):
            raise ValueError('CDP endpoint refused')
    except (ValueError, TypeError, AttributeError):
        raise ValueError('CDP endpoint refused') from None
    return url


def validate_protocol(protocol: dict[str, Any]) -> None:
    try:
        domains = protocol['domains']
        if type(domains) is not list or len(domains) > 256:
            raise ValueError
        methods: set[str] = set()
        for domain in domains:
            commands = domain.get('commands', [])
            if type(commands) is not list or len(commands) > 512:
                raise ValueError
            methods.update(domain['domain'] + '.' + item['name'] for item in commands)
        if not METHODS <= methods:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError('installed CDP protocol lacks required methods') from None


class CDPSession:
    def __init__(self, url: str, *, port: int) -> None:
        self.url = validate_cdp_url(url, port=port)
        self._connection: Any = None
        self._id = 0

    def __enter__(self) -> Self:
        from websockets.sync.client import connect
        self._connection = connect(self.url, proxy=None, compression=None, open_timeout=5,
                                   close_timeout=5, max_size=_LIMIT, max_queue=4)
        return self

    def __exit__(self, *exception: object) -> None:
        if self._connection is not None:
            self._connection.close()

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method not in METHODS or self._connection is None:
            raise ValueError('CDP method refused')
        self._id += 1
        try:
            raw = json.dumps({'id': self._id, 'method': method, 'params': params}, allow_nan=False)
            if len(raw.encode()) > 65536:
                raise ValueError
            self._connection.send(raw)
            # Unsolicited events are bounded and ignored; a wrong response ID is refused.
            for _ in range(64):
                frame = self._connection.recv(timeout=5)
                if type(frame) is not str or len(frame.encode()) > _LIMIT:
                    raise ValueError
                result = _json(frame.encode(), maximum=_LIMIT)
                if type(result) is not dict:
                    raise ValueError
                if 'id' not in result:
                    continue
                body = result.get('result')
                if type(result['id']) is not int or result['id'] != self._id or 'error' in result or type(body) is not dict or 'exceptionDetails' in body:
                    raise ValueError
                return body
        except Exception:
            raise ValueError('CDP roundtrip refused') from None
        raise ValueError('CDP event limit exceeded')

    def evaluate(self, expression: str) -> Any:
        result = self.call('Runtime.evaluate', {'expression': expression, 'returnByValue': True,
                                               'awaitPromise': True, 'timeout': 5000})
        value = result.get('result')
        if type(value) is not dict or value.get('type') in {'undefined', 'error'} or 'value' not in value:
            raise ValueError('CDP observation unavailable')
        return value['value']

    def navigate(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port or parsed.username or parsed.password:
            raise ValueError('browser navigation refused')
        if 'errorText' in self.call('Page.navigate', {'url': url}):
            raise ValueError('browser navigation failed')


def _http_json(port: int, path: str, *, maximum: int) -> dict[str, Any] | list[Any]:
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client:
        with client.stream('GET', f'http://127.0.0.1:{port}{path}') as response:
            if response.status_code != 200:
                raise ValueError('owned browser endpoint unavailable')
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                if len(raw) > maximum:
                    raise ValueError('browser discovery size exceeded')
    # /json/protocol can be larger than a tool probe; no canonical spelling required.
    try:
        result = _json(bytes(raw), maximum=maximum)
        if type(result) not in (dict, list):
            raise ValueError
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError('browser discovery malformed') from None


def discover_owned_browser(profile: Path, *, deadline: float) -> tuple[str, int]:
    while time.monotonic() < deadline:
        try:
            raw = _read_regular(profile / 'DevToolsActivePort', maximum=4096).decode('ascii')
            port_text, browser_path = raw.splitlines()
            port = int(port_text)
            validate_cdp_url(f'ws://127.0.0.1:{port}{browser_path}', port=port)
            version = _http_json(port, '/json/version', maximum=65536)
            if type(version) is not dict or version.get('webSocketDebuggerUrl') != f'ws://127.0.0.1:{port}{browser_path}':
                raise ValueError
            protocol = _http_json(port, '/json/protocol', maximum=_LIMIT)
            if type(protocol) is not dict:
                raise ValueError
            validate_protocol(protocol)
            pages = _http_json(port, '/json/list', maximum=65536)
            if type(pages) is not list:
                raise ValueError
            candidates = [item for item in pages if type(item) is dict and item.get('type') == 'page' and item.get('url') == 'about:blank']
            if len(candidates) == 1:
                return validate_cdp_url(candidates[0]['webSocketDebuggerUrl'], port=port), port
        except FileNotFoundError:
            pass
        except (OSError, ValueError, KeyError, TypeError):
            # Once ownership data exists, malformed or mismatched discovery is fatal.
            if (profile / 'DevToolsActivePort').exists():
                raise ValueError('owned browser discovery refused') from None
        time.sleep(0.05)
    raise ValueError('owned browser startup deadline exceeded')


def _wait(session: CDPSession, predicate: str, *, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if session.evaluate(predicate) is True:
            return
        time.sleep(0.05)
    raise ValueError('browser observation deadline exceeded')


def recovery_journey(session: CDPSession, *, base_url: str, migration_url: str,
                     projects: dict[str, int], screenshots: Path, counter_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = dict(waiting_ok=True, safe_retry_ok=True, unknown_remote_blocked=True,
                                  migration_failed_ok=True, keyboard_ok=True, duplicate_post_count=0,
                                  horizontal_overflow=False, playback_ok=True)
    def read_count() -> int:
        for attempt in range(40):
            try:
                value = _json(_read_regular(counter_path, maximum=4096), maximum=4096)
                break
            except (OSError, ValueError):
                # The server replaces the counter atomically; a read racing that
                # replacement is retried, never interpreted.
                if attempt == 39:
                    raise
                time.sleep(0.05)
        if type(value) is not dict or set(value) != {'operation_posts'} or type(value['operation_posts']) is not int or not 0 <= value['operation_posts'] <= 16:
            raise ValueError('operation counter invalid')
        return value['operation_posts']

    def count() -> int:
        """Counter value after a quiet second, so a late duplicate POST is observed."""
        value = read_count()
        quiet_since = time.monotonic()
        deadline = quiet_since + 6
        while time.monotonic() - quiet_since < 1.0:
            if time.monotonic() >= deadline:
                raise ValueError('operation counter never settled')
            time.sleep(0.1)
            current = read_count()
            if current != value:
                value, quiet_since = current, time.monotonic()
        return value

    initial = count()
    retry_label = json.dumps('現在の設定で再実行')
    no_enabled_retry = ("(async()=>{for(let i=0;i<25;i++){if([...document.querySelectorAll('button')].some(b=>b.textContent.includes("
                        + retry_label + ")&&!b.disabled))return false;await new Promise(r=>setTimeout(r,100));}return true})()")

    def capture(width: int, state: str) -> None:
        if session.evaluate('document.documentElement.scrollWidth > document.documentElement.clientWidth') is True:
            summary['horizontal_overflow'] = True
        data = session.call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False}).get('data')
        if type(data) is not str or len(data) > 6 * 1024 * 1024:
            raise ValueError('screenshot unavailable or oversized')
        screenshot = base64.b64decode(data, validate=True)
        if len(screenshot) > 4 * 1024 * 1024 or not screenshot.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('screenshot format/size refused')
        _write_exclusive(screenshots / f'{width}-{state}.png', screenshot)

    for width in (390, 1440):
        session.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 900, 'deviceScaleFactor': 1, 'mobile': False})
        summary['narrow_width' if width == 390 else 'wide_width'] = session.evaluate('innerWidth')
        for state, label in (('waiting', 'お待ち'), ('safe_retry', '現在の設定で再実行'), ('unknown_remote', '外部')):
            project = projects['safe_retry_wide' if width == 1440 and state == 'safe_retry' else state]
            session.navigate(base_url + '/projects/' + str(project))
            _wait(session, "document.readyState==='complete' && !!document.querySelector('#generation-history li') && document.querySelector('#generation-history').innerText.includes(" + json.dumps(label) + ')')
            # Require rendered controls that stay unchanged for 600 ms after the
            # durable history is present. A pre-history negation is not evidence.
            _wait(session, "(async()=>{const h=document.querySelector('#generation-history');const state=()=>h.innerText+'|'+[...document.querySelectorAll('button')].map(b=>b.disabled).join();const a=state();for(let i=0;i<4;i++){await new Promise(r=>setTimeout(r,150));if(state()!==a)return false;}return true})()")
            if state == 'waiting':
                # Negations are sampled across 2.5 s (longer than the 2 s polling
                # refetch), so a fetch-disabled moment cannot satisfy them.
                summary['waiting_ok'] &= session.evaluate(no_enabled_retry) is True
            elif state == 'safe_retry':
                _wait(session, "[...document.querySelectorAll('#generation-history button')].some(b=>b.textContent.includes('現在の設定で再実行')&&!b.disabled)")
                summary['safe_retry_ok'] &= session.evaluate("[...document.querySelectorAll('button')].some(b=>b.textContent.includes('現在の設定で再実行')&&!b.disabled)") is True
                previous = session.evaluate("(()=>{const e=[...document.querySelectorAll('button,a,input,textarea,select,[tabindex]')].filter(e=>e.tabIndex>=0&&!e.disabled&&e.getClientRects().length);const i=e.findIndex(e=>e.textContent.includes('現在の設定で再実行'));if(i<1)return false;e[i-1].focus();return true})()")
                for event in ({'type': 'keyDown', 'key': 'Tab', 'code': 'Tab'}, {'type': 'keyUp', 'key': 'Tab', 'code': 'Tab'}):
                    session.call('Input.dispatchKeyEvent', event)
                summary['keyboard_ok'] &= previous is True and session.evaluate("document.activeElement.textContent.includes('現在の設定で再実行')") is True
                before = count()
                for _ in range(2):
                    session.call('Input.dispatchKeyEvent', {'type': 'keyDown', 'key': 'Enter', 'code': 'Enter', 'text': '\r', 'windowsVirtualKeyCode': 13})
                    session.call('Input.dispatchKeyEvent', {'type': 'keyUp', 'key': 'Enter', 'code': 'Enter', 'windowsVirtualKeyCode': 13})
                _wait(session, "document.body.innerText.includes('現在の設定での再実行を受け付けました。')")
                delta = count() - before
                summary['duplicate_post_count'] = max(summary['duplicate_post_count'], delta)
                summary['keyboard_ok'] &= delta == 1
            elif state == 'unknown_remote':
                summary['unknown_remote_blocked'] &= session.evaluate(no_enabled_retry) is True
            capture(width, state)
        session.navigate(base_url + '/projects/' + str(projects['media']))
        _wait(session, "document.readyState==='complete' && !!document.querySelector('video')")
        summary['playback_ok'] &= session.evaluate("(async()=>{const v=document.querySelector('video');v.muted=true;await v.play();return v.readyState>=2&&!v.paused})()") is True
        capture(width, 'playback')
        session.navigate(migration_url)
        _wait(session, "document.readyState==='complete' && document.body.innerText.includes('移行')")
        # The migration-failed alert itself must carry the stop-the-app guidance
        # (both D35 variants); generic startup text containing '起動' is not enough.
        summary['migration_failed_ok'] &= session.evaluate("[...document.querySelectorAll('[role=alert]')].some(a=>a.innerText.includes('移行')&&a.innerText.includes('アプリを停止'))") is True
        capture(width, 'migration')
    total = count() - initial
    if total != 2:
        # Exactly one accepted POST per viewport across the whole journey.
        summary['keyboard_ok'] = False
        summary['duplicate_post_count'] = 0 if total < 2 else min(4, max(2, total - 1))
    return summary


def _json(raw: bytes, *, maximum: int) -> Any:
    # Ephemeral CDP/HTTP JSON, not canonical evidence. Bound frames before parsing.
    if len(raw) > maximum:
        raise ValueError('browser JSON size exceeded')
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate browser JSON key')
            result[key] = value
        return result
    def constant(value: str) -> None:
        raise ValueError('nonfinite browser JSON value')
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    def check(value: Any, depth: int) -> None:
        if depth > 32 or isinstance(value, float) and not math.isfinite(value):
            raise ValueError('invalid browser JSON value')
        if isinstance(value, str):
            value.encode('utf-8', errors='strict')
        elif isinstance(value, dict):
            for key, child in value.items():
                check(key, depth + 1)
                check(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                check(child, depth + 1)
    check(result, 0)
    return result


def _read_regular(path: Path, *, maximum: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, 'st_file_attributes', 0) & 0x400:
        raise ValueError('browser file type refused')
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError('browser file identity changed')
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino) or len(raw) > maximum:
            raise ValueError('browser file changed or oversized')
        return raw


def _write_exclusive(path: Path, raw: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--configuration', type=Path, required=True)
    values = parser.parse_args(argv)
    try:
        config = _json(_read_regular(values.configuration, maximum=65536), maximum=65536)
        if type(config) is not dict or set(config) != {'profile', 'base_url', 'migration_url', 'projects', 'screenshots', 'summary_path', 'counter_path'}:
            raise ValueError('browser configuration refused')
        root = values.configuration.absolute().parent
        for name in ('profile', 'screenshots', 'summary_path', 'counter_path'):
            path = Path(config[name])
            if not path.is_absolute() or not path.is_relative_to(root) or '..' in path.parts:
                raise ValueError('browser configuration path refused')
            current = path.parent if name in ('summary_path', 'counter_path') else path
            while current != root.parent:
                metadata = current.lstat()
                if not stat.S_ISDIR(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('browser directory type refused')
                current = current.parent
        if type(config['projects']) is not dict or set(config['projects']) != {'waiting', 'safe_retry', 'safe_retry_wide', 'unknown_remote', 'media'} or any(type(value) is not int or value < 1 for value in config['projects'].values()):
            raise ValueError('browser project identities refused')
        url, port = discover_owned_browser(Path(config['profile']), deadline=time.monotonic() + 30)
        with CDPSession(url, port=port) as session:
            product = session.call('Browser.getVersion', {}).get('product')
            if type(product) is not str or not product:
                raise ValueError('browser identity missing')
            summary = recovery_journey(session, base_url=config['base_url'], migration_url=config['migration_url'], projects=config['projects'], screenshots=Path(config['screenshots']), counter_path=Path(config['counter_path']))
            summary['browser_version'] = product
        raw = json.dumps(summary, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii') + b'\n'
        _write_exclusive(Path(config['summary_path']), raw)
        return 0
    except Exception:
        import sys
        print('D39 browser smoke refused', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
