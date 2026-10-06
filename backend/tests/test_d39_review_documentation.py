"""Per-key documentation gates with consistent and contradictory metadata."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts.d39_smoke import DOC_KEYS, documentation_checks


@pytest.fixture
def documented(tmp_path: Path) -> Path:
    files = {
        'README.md': 'Audited lane: Python 3.12.12, Node.js 24.11.1, pnpm 10.18.3.\n[setup](backend/.env.example)\n',
        'Makefile': 'demo:\n\tpython -m scripts.plan_c_demo --mode all_tools\n',
        'backend/.env.example': '# synthetic\n',
        'backend/pyproject.toml': '[project]\nname="synthetic"\nversion="1"\nrequires-python=">=3.12"\ndependencies=[]\n',
        'backend/uv.lock': 'requires-python=">=3.12"\npackage=[]\n',
        'frontend/package.json': json.dumps({'packageManager': 'pnpm@10.18.3', 'engines': {'node': '>=24.11.1'}, 'dependencies': {'react': '^18.3.1'}}),
        'frontend/pnpm-lock.yaml': "lockfileVersion: '9.0'\nimporters:\n  .:\n    dependencies:\n      react:\n        specifier: ^18.3.1\n        version: 18.3.1\npackages:\n  react@18.3.1:\n    engines: {node: '>=18'}\n",
        'backend/scripts/plan_c_demo.py': 'def prepare_storage(): pass\ndef exclusive_demo(): pass\ndef demo_settings(): pass\ndef create_demo_app(): pass\ndef main():\n    parser.add_argument("--mode", choices=("all_tools", "stateful"))\n',
        'backend/tests/test_d34_migrations.py': 'def test_restore():\n    assert restore_database_backup is not None\n    assert reason == "database_lease_unavailable"\n',
        'backend/tests/test_d35_startup_recovery_api.py': 'def test_codes():\n    assert codes == ("safe_retry", "external_outcome_unknown", "migration_failed")\n',
        'frontend/src/test/recovery-status.test.tsx': "it.each(['safe_retry', 'external_outcome_unknown', 'migration_failed'])('%s', () => {});\n",
        'specification.md': 'Automated evidence is not human acceptance.\nReal held-out execution is external.\n',
    }
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')
    return tmp_path


def checks(source: Path, **results: bool) -> dict[str, bool]:
    inventory = frozenset(path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file())
    return documentation_checks(source, contracts_passed={'recovery_codes': True, 'migration_restore': True, 'ui_recovery': True} | results,
                                inventory=inventory)


def test_consistent_successor_documentation_passes_all_six(documented: Path) -> None:
    assert checks(documented) == dict.fromkeys(DOC_KEYS, True)


@pytest.mark.parametrize(('name', 'old', 'new', 'key'), [
    ('README.md', 'backend/.env.example', 'missing.md', 'setup_paths'),
    ('backend/pyproject.toml', '>=3.12', '>=3.14', 'locked_versions'),
    ('frontend/package.json', '^18.3.1', '^19', 'locked_versions'),
    ('frontend/pnpm-lock.yaml', "node: '>=18'", "node: '>=26'", 'locked_versions'),
    ('backend/scripts/plan_c_demo.py', 'def create_demo_app', 'def private_app', 'mode_commands'),
    ('backend/scripts/plan_c_demo.py', '"all_tools", "stateful"', '"all_tools", "other"', 'mode_commands'),
    ('specification.md', 'is not human', 'IS human', 'limitation_boundary'),
    ('specification.md', 'is external', 'is internal', 'limitation_boundary'),
])
def test_documentation_contradictions_fail_one_key(documented: Path, name: str, old: str, new: str, key: str) -> None:
    path = documented / name
    path.write_text(path.read_text(encoding='utf-8').replace(old, new), encoding='utf-8')
    assert checks(documented)[key] is False


@pytest.mark.parametrize(('name', 'body', 'key'), [
    ('backend/tests/test_d34_migrations.py', 'def test_trivial(): pass\n', 'migration_restore'),
    ('backend/tests/test_d34_migrations.py', '# restore_database_backup database_lease_unavailable\ndef test_trivial(): pass\n', 'migration_restore'),
    ('backend/tests/test_d34_migrations.py', 'def test_restore_database_backup_database_lease_unavailable():\n    """restore_database_backup database_lease_unavailable"""\n', 'migration_restore'),
    ('backend/tests/test_d34_migrations.py', 'TERMS = ("restore_database_backup", "database_lease_unavailable")\ndef test_trivial(): pass\n', 'migration_restore'),
    ('backend/tests/test_d35_startup_recovery_api.py', 'def test_x():\n    assert "safe_retry_external_outcome_unknown_migration_failed"\n', 'recovery_codes'),
    ('backend/tests/test_d35_startup_recovery_api.py', 'def test_trivial(): pass\n', 'recovery_codes'),
    ('frontend/src/test/recovery-status.test.tsx', 'it("trivial", () => {});\n', 'recovery_codes'),
])
def test_passing_but_uncovering_contract_tests_fail_the_key(documented: Path, name: str, body: str, key: str) -> None:
    (documented / name).write_text(body, encoding='utf-8')
    assert checks(documented)[key] is False


@pytest.mark.parametrize('spec', [
    'Checks are automated technical evidence, not human acceptance. The held-out corpus remains mounted outside.',
    "Automated checks aren't human acceptance; held-out execution happens externally.",
    'No automated evidence is human acceptance. Held-out data stays outside the process.',
    'Human acceptance is not automated evidence. Held-out execution is external.',
])
def test_conforming_limitation_phrasings_pass(documented: Path, spec: str) -> None:
    (documented / 'specification.md').write_text(spec, encoding='utf-8')
    assert checks(documented)['limitation_boundary'] is True


@pytest.mark.parametrize('extra', [
    'Automated evidence also counts as human acceptance.',
    'Automated technical evidence is also human acceptance.',
    'Held-out execution must remain internal.',
    'Held-out evaluation runs inside the candidate process.',
    'It is false that real held-out execution is external.',
    'Human acceptance is provided by automated evidence.',
    'Held-out evaluation runs inside the candidate process, not outside.',
    "Automated evidence isn't merely technical evidence, it is human acceptance.",
    'Automated evidence is not, in practice, distinct from human acceptance.',
    'Automated evidence is human acceptance, not merely a technical check.',
    'Automated evidence is human acceptance and not a technical check.',
    'Automated is human acceptance is not human acceptance.',
    'No doubt automated evidence is human acceptance.',
    'No question automated evidence is human acceptance.',
    'Not only automated evidence is human acceptance.',
    'Automated evidence isn\u2019t merely technical evidence, it is human acceptance.',
])
def test_appended_contradictions_fail_limitation_boundary(documented: Path, extra: str) -> None:
    path = documented / 'specification.md'
    path.write_text(path.read_text(encoding='utf-8') + extra + '\n', encoding='utf-8')
    assert checks(documented)['limitation_boundary'] is False


def test_npm_engine_spellings_in_real_locks_are_understood(documented: Path) -> None:
    lock = documented / 'frontend/pnpm-lock.yaml'
    spellings = ["'>= 0.4'", "'>= 8'", "'>=v12.22.7'", "'^12 || ^14 || >= 16'", "'>= 14.16'", "'>=18.x'", "'*'"]
    lock.write_text(lock.read_text(encoding='utf-8') + ''.join(
        f'  synthetic-{index}@1.0.0:\n    engines: {{node: {value}}}\n' for index, value in enumerate(spellings)), encoding='utf-8')
    assert checks(documented)['locked_versions'] is True
    lock.write_text(lock.read_text(encoding='utf-8') + "  synthetic-old@1.0.0:\n    engines: {node: '<= 20'}\n", encoding='utf-8')
    assert checks(documented)['locked_versions'] is False


@pytest.mark.parametrize(('engine', 'accepted'), [
    ('0 - 24.11.1', True), ('0 - 24.11.1-rc.1', False), ('>=24 || nonsense', False), ('nonsense || >=24', False),
    ('>=024', False), ('>=24.11.1-rc.01', False), ('>=24.11.1x', False), ('>=24.0.0-0 <25', True),
    ('>=24.11.1+a..b', False), ('>=24.11.1+...', False), ('>=24.11.1+', False), ('>=24.11.1+build.1', True),
    ('>=2\u0664.11.1', False), ('>=24.1\u0661.1', False), ('>=24.11.\u0661', False), ('\u0662\u0664 - 25', False),
])
def test_node_ranges_fail_closed_like_npm_semver(documented: Path, engine: str, accepted: bool) -> None:
    # Each expectation matches npm's own semver 7.7.3 for Node 24.11.1.
    package = documented / 'frontend/package.json'
    data = json.loads(package.read_text(encoding='utf-8'))
    data['engines']['node'] = engine
    package.write_text(json.dumps(data), encoding='utf-8')
    assert checks(documented)['locked_versions'] is accepted


def test_locked_versions_with_non_ascii_digits_are_refused(documented: Path) -> None:
    lock = documented / 'frontend/pnpm-lock.yaml'
    lock.write_text(lock.read_text(encoding='utf-8').replace('version: 18.3.1', 'version: 1\u0668.3.1'), encoding='utf-8')
    assert checks(documented)['locked_versions'] is False


@pytest.mark.parametrize(('old', 'new', 'accepted'), [
    ('Node.js 24.11.1', 'Node.js 24.11.\u0669', False), ('Node.js 24.11.1', 'Node.js 24.\u0661', False),
    ('Python 3.12.12', 'Python 3.12.1\u0662', False), ('pnpm 10.18.3', 'pnpm 10.18.\u0663', False),
    ('Python 3.12.12', 'Python 3.12.12rc1', False), ('Node.js 24.11.1', 'Node.js 24.11.1\u4ee5\u4e0a', True),
    ('Node.js 24.11.1', 'Node.js 24', True), ('Python 3.12.12', 'Python 3.12+', True),
])
def test_readme_version_tokens_are_never_cut_short(documented: Path, old: str, new: str, accepted: bool) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8').replace(old, new), encoding='utf-8')
    assert checks(documented)['locked_versions'] is accepted


def test_optional_manifest_fields_may_be_absent(documented: Path) -> None:
    package = documented / 'frontend/package.json'
    data = json.loads(package.read_text(encoding='utf-8'))
    del data['packageManager'], data['engines']
    package.write_text(json.dumps(data), encoding='utf-8')
    assert checks(documented)['locked_versions'] is True


@pytest.mark.parametrize('link', ['/README.md', 'readme.MD', 'BACKEND/.env.example', 'C:/Windows/win.ini', r'backend\.env.example'])
def test_setup_links_are_case_exact_and_repository_relative(documented: Path, link: str) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + f'[x]({link})\n', encoding='utf-8')
    assert checks(documented)['setup_paths'] is (link == '/README.md')


@pytest.mark.parametrize('link', [
    '[x]( docs/missing.md )', "[x](docs/missing.md 'T')", '[x](docs/missing.md (T))', '[x](<docs/missing.md>)',
    '<a href=docs/missing.md>x</a>', "<img src='docs/missing.png'>", '[x](C:docs/missing.md)', '[x](docs/a(1).md)',
])
def test_every_commonmark_link_form_is_checked_or_fails_closed(documented: Path, link: str) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + link + '\n', encoding='utf-8')
    assert checks(documented)['setup_paths'] is False


def test_links_to_files_written_after_materialization_fail(documented: Path) -> None:
    inventory = frozenset(path.relative_to(documented).as_posix() for path in documented.rglob('*') if path.is_file())
    (documented / 'frontend/.npmrc').write_text('verifier-written\n', encoding='utf-8')
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '[rc](frontend/.npmrc) [ok]( backend/.env.example "T")\n', encoding='utf-8')
    result = documentation_checks(documented, contracts_passed=True, inventory=inventory)
    assert result['setup_paths'] is False
    readme.write_text(readme.read_text(encoding='utf-8').replace('[rc](frontend/.npmrc) ', ''), encoding='utf-8')
    assert documentation_checks(documented, contracts_passed=True, inventory=inventory)['setup_paths'] is True


def test_reference_style_links_are_checked(documented: Path) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '[guide]: docs/missing-guide.md\n', encoding='utf-8')
    assert checks(documented)['setup_paths'] is False


@pytest.mark.parametrize(('definition', 'accepted'), [
    ('[setup\n label]: docs/missing.md\n', False), ('[setup\n label]: backend/.env.example\n', True),
    ('[a\\]b]: docs/missing.md\n', False), ('[a\\]b]: backend/.env.example\n', True),
    ('[setup\n\n label]: backend/.env.example\n', False),
])
def test_multiline_and_escaped_reference_labels_are_checked(documented: Path, definition: str, accepted: bool) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '[guide][setup label]\n\n' + definition, encoding='utf-8')
    assert checks(documented)['setup_paths'] is accepted


@pytest.mark.parametrize('container', ['> ', '- ', '* ', '1. ', '> - ', '- > ', '    '])
@pytest.mark.parametrize(('destination', 'accepted'), [('docs/missing.md', False), ('backend/.env.example', True)])
def test_reference_definitions_inside_containers_are_checked(documented: Path, container: str, destination: str,
                                                            accepted: bool) -> None:
    readme = documented / 'README.md'
    body = '[guide][setup]\n\n- item\n\n' if container == '    ' else '[guide][setup]\n\n'
    readme.write_text(readme.read_text(encoding='utf-8') + body + f'{container}[setup]: {destination}\n', encoding='utf-8')
    assert checks(documented)['setup_paths'] is accepted


@pytest.mark.parametrize(('body', 'accepted'), [
    # Code never hides a link or definition: a missing target shown as code is still
    # refused (documented fail-closed over-rejection); an existing one passes.
    ('Intro text.\n\n    [setup]: docs/missing.md\n', False),
    ('Intro text.\n\n    [x](docs/missing.md)\n', False),
    ('```\n[setup]: docs/missing.md\n[x](docs/missing.md)\n```\n', False),
    ('Use `[x](docs/missing.md)` as an example.\n', False),
    ('- a\n\n        [setup]: docs/missing.md\n', False),
    ('```\n[x](backend/.env.example)\n```\n', True),
    ('Use `[x](backend/.env.example)` as an example.\n', True),
    # Real definitions in containers are checked.
    ('[guide][setup]\n\n[setup]: docs/missing.md\n', False),
    ('> - a\n>\n>     [setup]: docs/missing.md\n', False),
    ('- a\n  - b\n\n      [setup]: docs/missing.md\n', False),
    # R-11: a fence inside a block quote must not hide a later real link.
    ('> ```\n> example\n> ```\n\n[bad](docs/missing.md)\n', False),
    ('> ```\n> example\n\n[bad](docs/missing.md)\n', False),
    # R-12: unequal backtick runs are not a code span and must not hide the link.
    ('``[bad](docs/missing.md)`\n', False),
    ('`[bad](docs/missing.md)``\n', False),
    # R-13: a multi-line code span is still scanned (over-rejection only).
    ('`example\n[bad](docs/missing.md)`\n', False),
])
def test_code_never_hides_a_link_or_definition(documented: Path, body: str, accepted: bool) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + body, encoding='utf-8')
    assert checks(documented)['setup_paths'] is accepted


def test_reference_definition_split_across_quoted_lines_is_checked(documented: Path) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '[guide][setup]\n\n> [setup]:\n>   docs/missing.md\n', encoding='utf-8')
    assert checks(documented)['setup_paths'] is False


@pytest.mark.parametrize(('definition', 'accepted'), [
    ('[setup]:\n  docs/missing.md\n', False), ('[setup]:\n  backend/.env.example\n', True),
    ('[setup]: backend/.env.example "Title"\n', True), ('[setup]:\n  backend/.env.example "Title"\n', True),
    ('[setup]:\n\n  backend/.env.example\n', False), ('[setup]:\n', False),
])
def test_reference_destinations_on_the_next_line_are_checked(documented: Path, definition: str, accepted: bool) -> None:
    readme = documented / 'README.md'
    readme.write_text(readme.read_text(encoding='utf-8') + '[guide][setup]\n\n' + definition, encoding='utf-8')
    assert checks(documented)['setup_paths'] is accepted


@pytest.mark.parametrize(('failed', 'key'), [('recovery_codes', 'recovery_codes'), ('ui_recovery', 'recovery_codes'), ('migration_restore', 'migration_restore')])
def test_actual_contract_failures_are_individual_false_values(documented: Path, failed: str, key: str) -> None:
    result = checks(documented, **{failed: False})
    assert result[key] is False
    assert result['migration_restore' if key == 'recovery_codes' else 'recovery_codes'] is True
