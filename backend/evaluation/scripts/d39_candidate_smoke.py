"""Fixed D39 synthetic actions, executed only in a verified candidate-rooted child.

This file imports no evaluation or current application modules. The external owner
binds its bytes and copies no tooling into candidate source. No arbitrary expression,
import, endpoint or action can be supplied through configuration.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import runpy
import sqlite3
import stat
import subprocess
import sys
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

ACTIONS: tuple[str, ...] = ('build_index', 'legacy_migration', 'restore', 'all_tools_startup', 'stateful_startup',
           'seed_browser', 'serve', 'ffmpeg', 'contract_tests')
_KEYS = {'group_root', 'source_root', 'storage', 'frontend', 'profile', 'index', 'provider_url',
         'model', 'port', 'summary_path', 'scenario', 'ffmpeg', 'ffprobe', 'owner_token'}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False).encode('ascii') + b'\n'


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate configuration key')
        result[key] = value
    return result


def _regular_path(path: Path, *, root: Path | None = None) -> Path:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute bounded smoke path required')
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if not component.exists():
            continue
        metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & 0x400:
            raise ValueError('smoke path link refused')
    if root is not None and not absolute.is_relative_to(root):
        raise ValueError('smoke path escapes owned group')
    return absolute


def _read(path: Path, *, maximum: int) -> bytes:
    _regular_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('regular smoke file required')
    def identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
    with path.open('rb') as stream:
        opened = os.fstat(stream.fileno())
        if identity(opened) != identity(before):
            raise ValueError('smoke file changed before read')
        raw = stream.read(maximum + 1)
        final = os.fstat(stream.fileno())
        current = path.lstat()
        # Windows named/handle ctime views can differ without mutation. Compare
        # each view to itself, as the shared streamed fingerprint reader does.
        if len(raw) > maximum or len(raw) != opened.st_size or identity(final) != identity(before) or identity(current) != identity(before) or opened.st_ctime_ns != final.st_ctime_ns or before.st_ctime_ns != current.st_ctime_ns:
            raise ValueError('smoke file changed or oversized')
        return raw


def read_configuration(path: Path) -> dict[str, Any]:
    raw = _read(path, maximum=8192)
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite configuration')))
    if type(value) is not dict or set(value) != _KEYS or _canonical(value) != raw:
        raise ValueError('invalid smoke configuration')
    root = _regular_path(Path(value['group_root']))
    for name in ('source_root', 'storage', 'frontend', 'profile', 'index', 'summary_path'):
        if type(value[name]) is not str:
            raise ValueError('invalid smoke path')
        _regular_path(Path(value[name]), root=root)
    source = Path(value['source_root'])
    for name in ('storage', 'index', 'summary_path'):
        owned = Path(value[name])
        if owned.is_relative_to(source) or source.is_relative_to(owned):
            raise ValueError('writable state overlaps candidate source')
    if type(value['owner_token']) is not str or len(value['owner_token']) != 64 or any(character not in '0123456789abcdef' for character in value['owner_token']):
        raise ValueError('server owner identity invalid')
    if Path.cwd().absolute() != source / 'backend' or not (source / 'backend/scripts/plan_c_demo.py').is_file():
        raise ValueError('candidate import root mismatch')
    provider = urlsplit(value['provider_url'])
    if provider.scheme != 'http' or provider.hostname != '127.0.0.1' or not provider.port or provider.path != '/v1' or provider.username or provider.password or provider.query or provider.fragment:
        raise ValueError('smoke provider must be literal loopback')
    if type(value['port']) is not int or not 1024 <= value['port'] <= 65535 or type(value['model']) is not str or not 1 <= len(value['model']) <= 128:
        raise ValueError('smoke mode/port invalid')
    if value['scenario'] not in ('normal', 'stateful', 'migration_failed'):
        raise ValueError('unknown smoke scenario')
    for name in ('ffmpeg', 'ffprobe'):
        if value[name] is not None:
            native = _regular_path(Path(value[name]))
            if not native.is_file() or native.suffix.lower() in ('.cmd', '.bat', '.py', '.sh', '.ps1'):
                raise ValueError('native media executable required')
    return value


def _connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA foreign_keys=ON')
    if connection.execute('PRAGMA foreign_keys').fetchone() != (1,):
        connection.close()
        raise ValueError('foreign keys unavailable')
    return connection


def _state(path: Path) -> tuple[tuple[str, tuple[tuple[Any, ...], ...]], ...]:
    with closing(_connection(path)) as db:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        result = []
        for table in tables:
            if not table.replace('_', '').isalnum():
                raise ValueError('unsafe table identifier')
            # Preserve all old columns/rows, ignoring additive new tables/columns.
            rows = tuple(sorted(db.execute('SELECT * FROM "' + table + '"').fetchall(), key=repr))
            result.append((table, rows))
        return tuple(result)


def _old_state(path: Path, before: dict[str, tuple[str, ...]]) -> dict[str, tuple[tuple[Any, ...], ...]]:
    with closing(_connection(path)) as db:
        return {table: tuple(sorted(db.execute('SELECT ' + ','.join('"' + c + '"' for c in columns) + ' FROM "' + table + '"').fetchall(), key=repr)) for table, columns in before.items()}


def _migration(config: dict[str, Any]) -> dict[str, Any]:
    from app.db import Base, register_models
    from app.migrations.lease import acquire_database_lease
    from app.migrations.runner import migrate_database
    storage = Path(config['storage'])
    storage.mkdir(parents=True, exist_ok=True)
    database = storage / 'migration.db'
    sql = Path(config['source_root']) / 'backend/tests/fixtures/migrations/upstream_v0.sql'
    with sql.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('legacy SQL limit exceeded')
    with closing(_connection(database)) as db:
        db.executescript(raw.decode('utf-8'))
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        columns = {table: tuple(row[1] for row in db.execute('PRAGMA table_info("' + table + '")')) for table in tables}
    before = _old_state(database, columns)
    register_models()
    url = 'sqlite:///' + database.as_posix()
    lease = acquire_database_lease(url)
    try:
        observed = migrate_database(url, Base.metadata, lease=lease)
    finally:
        lease.release()
    with closing(_connection(database)) as db:
        integrity = db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        version = db.execute('PRAGMA user_version').fetchone()[0]
    candidates = [p for p in (storage / '.backups').iterdir() if not p.name.endswith('.metadata.json')]
    if len(candidates) != 1 or observed.backup_sha256 is None:
        raise ValueError('migration backup missing')
    backup = candidates[0]
    digest = _hash(backup, maximum=32 * 1024 * 1024)
    if digest[1] != observed.backup_sha256:
        raise ValueError('backup digest mismatch')
    preserved = _old_state(database, columns) == before
    return dict(stage='legacy_migration', from_version=observed.from_version, to_version=version,
                integrity_ok=integrity, foreign_keys_enabled=True, rows_preserved=preserved,
                identities_preserved=preserved, backup_size=digest[0], backup_sha256=digest[1])


def _restore(config: dict[str, Any]) -> dict[str, Any]:
    from app.db import Base, register_models
    from app.migrations.lease import acquire_database_lease
    from app.migrations.runner import migrate_database, restore_database_backup
    from app.migrations.contracts import MigrationError
    storage = Path(config['storage'])
    database = storage / 'migration.db'
    before = _state(database)
    register_models()
    url = 'sqlite:///' + database.as_posix()
    # A verified current-schema backup permits exact logical-state restore at v1.
    from app.migrations.backup import create_verified_backup
    from app.migrations.schema import critical_identity_snapshot
    lease = acquire_database_lease(url)
    try:
        with closing(_connection(database)) as db:
            backup = create_verified_backup(database, db, Base.metadata, critical_identity_snapshot(db, Base.metadata), 1, before_publish=lambda: lease.assert_held_for(url))
        excluded = False
        try:
            restore_database_backup(url, backup.path, backup.sha256, Base.metadata)
        except MigrationError as error:
            excluded = error.reason_code == 'database_lease_unavailable'
    finally:
        lease.release()
    with closing(_connection(database)) as db:
        db.execute("UPDATE projects SET title='synthetic changed title'")
        db.commit()
    restore_database_backup(url, backup.path, backup.sha256, Base.metadata)
    with closing(_connection(database)) as db:
        integrity = db.execute('PRAGMA integrity_check').fetchone() == ('ok',)
        version = db.execute('PRAGMA user_version').fetchone()[0]
    lease = acquire_database_lease(url)
    try:
        migrate_database(url, Base.metadata, lease=lease)
    finally:
        lease.release()
    equal = _state(database) == before
    return dict(stage='restore', restored_version=version, integrity_ok=integrity,
                foreign_keys_enabled=True, rows_equal=equal, identities_equal=equal,
                lease_exclusion_passed=excluded)


def _demo(config: dict[str, Any], mode: str) -> tuple[dict[str, Any], Any, Path]:
    demo = runpy.run_path(str(Path(config['source_root']) / 'backend/scripts/plan_c_demo.py'))
    storage = demo['prepare_storage'](Path(config['storage']))
    settings = demo['demo_settings'](storage, mode, config['model'], Path(config['index']) if mode == 'stateful' else None)
    settings.language_base_url = config['provider_url']
    settings.voicevox_url = config['provider_url']
    settings.language_embedding_base_url = config['provider_url']
    settings.language_retrieval_profile = Path(config['profile'])
    settings.language_retrieval_index = Path(config['index']) if mode == 'stateful' else None
    settings.ffmpeg_path, settings.ffprobe_path = config['ffmpeg'], config['ffprobe']
    settings.output_width, settings.output_height, settings.output_fps = 320, 240, 12
    settings.subtitle_band_height = 48
    # The candidate's db module captures get_settings at import time. Install
    # the public demo settings before model registration imports that module.
    from app.core import config as candidate_config
    candidate_config.get_settings = lambda: settings
    from app.db import register_models
    register_models()
    if mode == 'stateful':
        from app.retrieval import contracts, reader, sources
        profile = contracts.EmbeddingProfile.model_validate_json(_read(Path(config['profile']), maximum=65536))
        reader.load_index(Path(config['index']), sources.load_sources(), profile)
    return demo, settings, storage


def _build_index(config: dict[str, Any]) -> dict[str, Any]:
    from app.retrieval import builder, reader, sources, contracts
    profile = contracts.EmbeddingProfile.model_validate_json(_read(Path(config['profile']), maximum=65536))
    source = sources.load_sources()
    vectors = tuple((1.0, 0.0) if document.key == ('project.subtitle-font-size.set', 1) else (0.0, 1.0) for document in source.documents)
    manifest = builder.publish_index(Path(config['index']), source, profile, vectors)
    verified = reader.load_index(Path(config['index']), source, profile)
    if verified.bundle.documents != source.documents:
        raise ValueError('synthetic candidate index load failed')
    return {'document_count': len(verified.bundle.documents), 'bundle_sha256': manifest.bundle_sha256,
            'profile_sha256': _hash(Path(config['profile']), maximum=65536)[1]}


def _startup(config: dict[str, Any], mode: str) -> dict[str, Any]:
    import httpx
    with httpx.Client(base_url='http://127.0.0.1:' + str(config['port']), trust_env=False, follow_redirects=False, timeout=30) as client:
        owner = client.get('/__d39-owned')
        if owner.status_code != 200 or len(owner.content) > 4096 or owner.json() != {'owner': config['owner_token']}:
            raise ValueError('startup server ownership mismatch')
        health = client.get('/api/health')
        startup = client.get('/api/startup')
        # Route uses startup-status in the frozen candidate.
        if startup.status_code == 404:
            startup = client.get('/api/startup/status')
        created = client.post('/api/projects', json={'title': 'D39 synthetic startup', 'source_script': '合成テストです。', 'use_fake_providers': True, 'voicevox_url': config['provider_url']})
        created.raise_for_status()
        project = created.json()
        response = client.post('/api/language/requests', json={'request_id': 'd39-' + mode, 'text': '字幕を64pxにして', 'target': {'project_id': project['id']}, 'base_revision': project['revision']})
        response.raise_for_status()
        result = response.json()
        selected = (result.get('diagnostics') or {}).get('retrieval') or {}
        interpretation = result.get('interpretation') or {}
        saved = client.get('/api/projects/' + str(project['id']))
        saved.raise_for_status()
        effect = saved.json()
        operation = result.get('result') or {}
        completed = result.get('status') == 'completed' and result.get('executed') is True and operation.get('operation_id') == 'project.subtitle-font-size.set' and effect.get('subtitle_font_size') == 64 and effect.get('revision') == project['revision'] + 1
        summary = dict(stage=mode + '_startup', mode=mode, startup_ready=startup.status_code == 200 and startup.json().get('status') == 'ready',
                       health_ok=health.status_code == 200 and health.json().get('status') == 'ok', request_completed=completed,
                       model_calls=interpretation.get('attempts', 0), index_sha256=None)
        if mode == 'stateful':
            manifest = json.loads(_read(Path(config['index']) / 'manifest.json', maximum=65536))
            summary.update(index_sha256=manifest['bundle_sha256'], profile_sha256=_hash(Path(config['profile']), maximum=65536)[1],
                           embedding_calls=selected.get('embedding_calls', 0), retrieval_verified=result.get('mode') == 'semantic' and selected.get('reason') == 'ranked' and selected.get('all_tools_count') == 0 and selected.get('index_sha256') == manifest['bundle_sha256'])
        return summary


def _hash(path: Path, *, maximum: int) -> tuple[int, str]:
    _regular_path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(min(65536, maximum + 1 - size)):
            size += len(chunk)
            if size > maximum:
                raise ValueError('smoke artifact byte limit exceeded')
            digest.update(chunk)
    return size, digest.hexdigest()


def _seed_browser(config: dict[str, Any]) -> dict[str, int]:
    from app.db import register_models
    register_models()
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import Session
    from app.models.project import Project
    from app.models.job import GenerationJob, JobStatus
    from app.models.external_call import ExternalCall
    engine = create_engine('sqlite:///' + (Path(config['storage']) / 'demo.db').as_posix())
    @event.listens_for(engine, 'connect')
    def foreign_keys(db: Any, _: Any) -> None:
        db.execute('PRAGMA foreign_keys=ON')
        if db.execute('PRAGMA foreign_keys').fetchone() != (1,):
            raise ValueError('foreign keys not enabled')
    try:
        import httpx
        base_url = 'http://127.0.0.1:' + str(config['port'])
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=5) as client, Session(engine) as db:
            owner = client.get(base_url + '/__d39-owned')
            if owner.status_code != 200 or len(owner.content) > 4096 or owner.json() != {'owner': config['owner_token']}:
                raise ValueError('seed server ownership mismatch')
            result = {}
            for name, status in (('waiting', JobStatus.running), ('safe_retry', JobStatus.cancelled), ('safe_retry_wide', JobStatus.cancelled), ('unknown_remote', JobStatus.unknown)):
                created = client.post(base_url + '/api/projects', json={'title': 'D39 ' + name, 'source_script': '合成テストです。', 'use_fake_providers': True, 'voicevox_url': config['provider_url']})
                created.raise_for_status()
                if len(created.content) > 65536:
                    raise ValueError('seed response oversized')
                project = db.get(Project, created.json()['id'])
                if project is None:
                    raise ValueError('seed project publication missing')
                # Public creation initializes the durable settings/history ledger;
                # only the synthetic terminal/running job states are seeded directly.
                job = GenerationJob(project_id=project.id, status=status, current_stage='synthetic')
                db.add(job)
                db.flush()
                if name == 'unknown_remote':
                    db.add(ExternalCall(job_id=job.id, fingerprint='9' * 64, provider='synthetic', endpoint='https://provider.invalid/jobs', remote_side_effect=True, status='unknown'))
                result[name] = project.id
                db.commit()  # Release the writer before the next public creation.
            return result
    finally:
        engine.dispose()


def _serve(config: dict[str, Any]) -> None:
    import uvicorn
    demo, settings, storage = _demo(config, 'stateful' if config['scenario'] == 'stateful' else 'all_tools')
    if config['scenario'] == 'migration_failed':
        with closing(_connection(storage / 'demo.db')) as db:
            db.execute('PRAGMA user_version=2')
            db.commit()
    with demo['exclusive_demo'](storage):
        app = demo['create_demo_app'](settings, Path(config['frontend']))
        # Observation only: bind owned listener and count fixed operation POSTs.
        from starlette.responses import JSONResponse
        counts = {'operation_posts': 0}
        published = {'sequence': 0}
        publish = asyncio.Lock()
        counter = Path(config['summary_path'])
        with counter.open('xb') as stream:
            stream.write(_canonical(counts))
        @app.middleware('http')
        async def count_posts(request: Any, next_call: Any) -> Any:
            if request.method == 'GET' and request.url.path == '/__d39-owned':
                return JSONResponse({'owner': config['owner_token']})
            if request.method == 'POST' and request.url.path == '/api/operations/execute':
                counts['operation_posts'] += 1
                # Publications are serialized and each writes the latest count, so a
                # retried replacement can never move the counter backwards.
                async with publish:
                    published['sequence'] += 1
                    temporary = counter.with_suffix('.count-tmp-' + str(published['sequence']))
                    with temporary.open('xb') as stream:
                        stream.write(_canonical(counts))
                        stream.flush()
                        os.fsync(stream.fileno())
                    # The browser helper may hold the counter open for a moment (no
                    # share-delete on Windows); retry the atomic replacement briefly.
                    for attempt in range(100):
                        try:
                            os.replace(temporary, counter)
                            break
                        except PermissionError:
                            if attempt == 99:
                                raise
                            await asyncio.sleep(0.01)
            return await next_call(request)
        uvicorn.run(app, host='127.0.0.1', port=config['port'], workers=1, reload=False, log_level='warning')


@contextmanager
def _observe_ffmpeg(executable: Path) -> Iterator[list[int]]:
    """Observe the native waits used by the candidate without changing its files."""
    original = asyncio.create_subprocess_exec
    exits: list[int] = []
    async def spawn(*argv: Any, **kwargs: Any) -> Any:
        process = await original(*argv, **kwargs)
        if argv and Path(str(argv[0])).resolve() == executable.resolve():
            wait = process.wait
            recorded = False
            async def observed_wait() -> int:
                nonlocal recorded
                code = await wait()
                if not recorded:
                    exits.append(code)
                    recorded = True
                return code
            process.wait = observed_wait
        return process
    asyncio.create_subprocess_exec = spawn
    try:
        yield exits
    finally:
        asyncio.create_subprocess_exec = original


def _probe_duration(executable: str, video: Path, output_path: Path) -> tuple[int, int | None]:
    with output_path.open('xb') as output:
        probe = subprocess.run((executable, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video)), stdout=output, stderr=subprocess.DEVNULL, timeout=15, check=False)
    if probe.returncode != 0:
        return probe.returncode, None
    raw = _read(output_path, maximum=65536)
    try:
        duration = int(float(json.loads(raw.decode('utf-8'))['format']['duration']) * 1000)
        return probe.returncode, duration if 0 <= duration <= 60000 else None
    except (ValueError, KeyError, TypeError, OverflowError):
        return probe.returncode, None


def _ffmpeg(config: dict[str, Any]) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from sqlalchemy import select
    demo, settings, storage = _demo(config, 'all_tools')
    with demo['exclusive_demo'](storage), _observe_ffmpeg(Path(config['ffmpeg'])) as exits:
        app = demo['create_demo_app'](settings, Path(config['frontend']))
        from app.models.artifact import GenerationArtifact
        from app.models.project import Project
        with TestClient(app) as client:
            response = client.post('/api/projects/quick', json={'source_script': '合成テストです。', 'use_fake_providers': True, 'voicevox_url': config['provider_url']})
            response.raise_for_status()
            project_id = response.json()['project']['id']
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                detail = client.get(f'/api/projects/{project_id}').json()
                if detail['status'] in ('completed', 'failed'):
                    break
                time.sleep(0.1)
            from app.db import get_session_factory
            with get_session_factory()() as db:
                artifact = db.scalar(select(GenerationArtifact).where(GenerationArtifact.project_id == project_id))
                current = db.get(Project, project_id)
                observed = dict(stage='ffmpeg', providers_fake=current is not None and current.use_fake_providers is True,
                                ffmpeg_exit_code=next((code for code in exits if code != 0), 0 if exits else None),
                                ffprobe_exit_code=None, video_present=False, subtitle_present=False, publication_bound=False, duration_ms=None)
                if artifact is None or current is None:
                    return observed
                manifest = artifact.manifest_json
                video = _regular_path(storage / artifact.video_path, root=storage)
                subtitle = _regular_path(storage / artifact.subtitle_path, root=storage)
                identities = {}
                for name, path in (('video', video), ('subtitle', subtitle)):
                    size, digest = _hash(path, maximum=32 * 1024 * 1024)
                    identities[name] = {'size': size, 'sha256': digest}
                    if not isinstance(manifest.get(name), dict) or manifest[name].get('sha256') != digest or manifest[name].get('size') != size:
                        raise ValueError('artifact manifest file binding mismatch')
                probe_path = Path(config['summary_path']).with_suffix('.ffprobe.json')
                code, duration = _probe_duration(config['ffprobe'], video, probe_path)
                Path(config['summary_path']).with_suffix('.media.json').write_bytes(_canonical({'project_id': project_id, 'video': video.relative_to(storage).as_posix(), 'subtitle': subtitle.relative_to(storage).as_posix(), 'manifest': manifest}))
                return observed | dict(ffprobe_exit_code=code, video_present=video.is_file(), subtitle_present=subtitle.is_file(),
                            # The observed exits do not replace the job's own completion:
                            # a publication is bound only for a job that ended 'completed'.
                            publication_bound=detail.get('status') == 'completed' and current.current_artifact_id == artifact.id and detail.get('output_video_path') == artifact.video_path and artifact.revision == detail['revision'] and manifest.get('job_id') == artifact.job_id and manifest.get('revision') == artifact.revision and manifest.get('input_fingerprint') == artifact.input_fingerprint, duration_ms=duration)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--action', choices=ACTIONS, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = read_configuration(args.configuration)
        if args.action == 'serve':
            _serve(config)
            return 0
        if args.action == 'contract_tests':
            import pytest
            class Results:
                def __init__(self) -> None:
                    self.results: dict[str, list[bool]] = {'migration_restore': [], 'recovery_codes': []}
                    self.skipped: dict[str, int] = {'migration_restore': 0, 'recovery_codes': 0}
                def pytest_runtest_logreport(self, report: Any) -> None:
                    if report.when == 'call' or report.failed or report.skipped:
                        key = 'migration_restore' if 'test_d34_migrations.py' in report.nodeid else 'recovery_codes'
                        # A skip is recorded separately and is never a passed contract test.
                        self.results[key].append(report.passed)
                        self.skipped[key] += bool(report.skipped)
            results = Results()
            status = pytest.main(['tests/test_d34_migrations.py', 'tests/test_d35_startup_recovery_api.py', '-q', '-p', 'no:cacheprovider'], plugins=[results])
            summary: dict[str, bool | int] = {key: bool(values) and all(values) and int(status) in (0, 1) for key, values in results.results.items()}
            summary.update({key + '_skipped': count for key, count in results.skipped.items()})
            Path(config['summary_path']).write_bytes(_canonical(summary))
            return 0
        actions = {'build_index': _build_index, 'legacy_migration': _migration, 'restore': _restore,
                   'all_tools_startup': lambda c: _startup(c, 'all_tools'), 'stateful_startup': lambda c: _startup(c, 'stateful'),
                   'seed_browser': _seed_browser, 'ffmpeg': _ffmpeg}
        summary = actions[args.action](config)
        raw = _canonical(summary)
        if len(raw) > 65536:
            raise ValueError('summary exceeds byte limit')
        with Path(config['summary_path']).open('xb') as output:
            output.write(raw)
        return 0
    except Exception:
        print('D39 candidate smoke refused', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
