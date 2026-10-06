"""Separate owned D39 six-stage synthetic smoke producer; never release approval."""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, NoReturn

import httpx
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from pydantic import TypeAdapter, ValidationError

from evaluation import blinded_io, blinded_runtime, browser_smoke, release_verification as verification
from evaluation import runtime_materialization as materialization
from evaluation.evidence_json import parse_canonical_model
from evaluation.smoke_contracts import (
    AllToolsStartupSummary, BrowserSummary, FFmpegSummary, MigrationSummary, RestoreSummary,
    SMOKE_STAGES, SmokeManifest, SmokeStageReceipt, StatefulStartupSummary, ToolExecutionBinding,
    TOOL_RULES,
)
from evaluation.tool_attestation import FileFingerprint, canonical_json_bytes

_DEADLINES = (120, 120, 180, 180, 300, 300)
_SUMMARIES = (MigrationSummary, RestoreSummary, AllToolsStartupSummary, StatefulStartupSummary, BrowserSummary, FFmpegSummary)
DOC_KEYS: tuple[str, ...] = ('setup_paths', 'locked_versions', 'mode_commands', 'recovery_codes', 'migration_restore', 'limitation_boundary')
_BOOTSTRAP = "import runpy,sys; p=sys.argv.pop(1); runpy.run_path(p,run_name='__main__')"


@contextmanager
def fake_providers() -> Iterator[tuple[str, dict[str, int]]]:
    counts = {'chat': 0, 'embedding': 0}
    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)
        def log_message(self, *arguments: Any) -> None:
            pass
        def do_GET(self) -> None:
            if self.path != '/v1/speakers':
                self.send_error(404)
                return
            raw = json.dumps([{'name': 'D39 synthetic', 'speaker_uuid': 'd39-synthetic', 'styles': [{'name': 'synthetic', 'id': 1}], 'version': 'synthetic'}]).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_POST(self) -> None:
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                self.send_error(400)
                return
            if not 0 < length <= 131072:
                self.send_error(413)
                return
            try:
                body = browser_smoke._json(self.rfile.read(length), maximum=131072)
                if type(body) is not dict:
                    raise ValueError
            except (ValueError, TypeError, RecursionError):
                self.send_error(400)
                return
            if self.path == '/v1/embeddings' and body.get('model') == 'synthetic-d39-embedding':
                counts['embedding'] += 1
                response = {'object': 'list', 'model': body['model'], 'data': [{'object': 'embedding', 'index': 0, 'embedding': [1.0, 0.0]}]}
            elif self.path == '/v1/chat/completions' and body.get('model') == 'synthetic-d39-chat':
                counts['chat'] += 1
                proposal = {'result': {'kind': 'operation', 'operation_id': 'project.subtitle-font-size.set', 'operation_version': 1, 'arguments': {'value': 64}}}
                response = {'model': body['model'], 'choices': [{'message': {'role': 'assistant', 'content': json.dumps(proposal)}, 'finish_reason': 'stop'}]}
            else:
                self.send_error(400)
                return
            if max(counts.values()) > 8:
                self.send_error(429)
                return
            raw = json.dumps(response, allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    server.timeout = 1
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='d39-fake-provider')
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', counts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        if thread.is_alive():
            raise ValueError('fake provider teardown unconfirmed')


# Numeric identifiers follow SemVer/node-semver: no leading zeros (`024` is invalid).
_SEMVER_COMPARATOR = re.compile(
    r'(?P<op>>=|<=|>|<|=|\^|~>|~)?\s*v?(?P<major>0|[1-9]\d*|[xX*])(?:\.(?P<minor>0|[1-9]\d*|[xX*]))?'
    r'(?:\.(?P<patch>0|[1-9]\d*|[xX*]))?(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?', re.ASCII)
_SEMVER_HYPHEN = re.compile(r'\s*(\S+)\s+-\s+(\S+)\s*', re.ASCII)
_RELEASE_VERSION = re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', re.ASCII)
Bound = tuple[str, tuple[int, int, int]]


def _semver_prefix(match: re.Match[str]) -> list[int]:
    """Concrete numeric prefix of a (possibly partial or x-range) version."""
    values: list[int] = []
    for part in (match['major'], match['minor'], match['patch']):
        if part is None or part in 'xX*':
            break
        values.append(int(part))
    return values


def _semver_bump(values: list[int], position: int) -> tuple[int, int, int]:
    padded = values[:position] + [0] * (3 - position)
    padded[position - 1] += 1
    return padded[0], padded[1], padded[2]


def _semver_bounds(operator: str, values: list[int], prerelease: bool = False) -> list[Bound] | None:
    """npm node-semver comparator semantics for release versions."""
    count = len(values)
    low = values + [0] * (3 - count)
    lower: Bound = ('>=', (low[0], low[1], low[2]))
    if prerelease and count == 3:
        # Release versions only: X.Y.Z-pre < X.Y.Z, so an exact prerelease never
        # matches, '>' behaves like '>=' and '<=' like '<'; ~ and ^ are unchanged.
        if operator in ('', '='):
            return [('<', (0, 0, 0))]
        operator = {'>': '>=', '<=': '<'}.get(operator, operator)
    if operator in ('', '='):
        return [] if count == 0 else [('==', lower[1])] if count == 3 else [lower, ('<', _semver_bump(values, count))]
    if operator == '>=':
        return [] if count == 0 else [lower]
    if operator == '>':
        return [('<', (0, 0, 0))] if count == 0 else [('>', lower[1])] if count == 3 else [('>=', _semver_bump(values, count))]
    if operator == '<':
        return [('<', (0, 0, 0) if count == 0 else lower[1])]
    if operator == '<=':
        return [] if count == 0 else [('<=', lower[1])] if count == 3 else [('<', _semver_bump(values, count))]
    if operator in ('~', '~>'):
        return [] if count == 0 else [lower, ('<', _semver_bump(values, 1 if count == 1 else 2))]
    if operator == '^':
        if count == 0:
            return []
        position = next((index + 1 for index, value in enumerate(values) if value), count)
        return [lower, ('<', _semver_bump(values, min(position, count) if count < 3 or any(values) else 3))]
    return None


def _valid_prerelease(match: re.Match[str], values: list[int]) -> bool:
    """A prerelease needs a full version and non-empty identifiers without leading zeros."""
    if match['pre'] is None:
        return True
    identifiers = match['pre'].split('.')
    return len(values) == 3 and all(item and not (item.isdecimal() and len(item) > 1 and item[0] == '0') for item in identifiers)


def _range_bounds(alternative: str) -> list[Bound] | None:
    """Bounds of one `||` alternative for release versions, or None when invalid."""
    hyphen = _SEMVER_HYPHEN.fullmatch(alternative)
    if hyphen is not None:
        first, last = (_SEMVER_COMPARATOR.fullmatch(side) for side in hyphen.groups())
        if first is None or last is None or first['op'] or last['op']:
            return None
        start, end = _semver_prefix(first), _semver_prefix(last)
        if not _valid_prerelease(first, start) or not _valid_prerelease(last, end):
            return None
        # For release versions `>= X.Y.Z-pre` equals `>= X.Y.Z`, while `<= X.Y.Z-pre`
        # admits no X.Y.Z release (X.Y.Z-pre < X.Y.Z), so it becomes `< X.Y.Z`.
        bounds = _semver_bounds('>=', start) or []
        if len(end) == 3:
            bounds.append(('<' if last['pre'] is not None else '<=', (end[0], end[1], end[2])))
        elif end:
            bounds.append(('<', _semver_bump(end, len(end))))
        return bounds
    bounds = []
    text = alternative.strip()
    position = 0
    while position < len(text):
        match = _SEMVER_COMPARATOR.match(text, position)
        if match is None or match.end() == position:
            return None
        if match.end() < len(text) and not text[match.end()].isspace():
            return None  # comparators are whitespace-separated (`>=024`, `1.2.3x` are invalid)
        values = _semver_prefix(match)
        if not _valid_prerelease(match, values):
            return None
        found = _semver_bounds(match['op'] or '', values, match['pre'] is not None)
        if found is None:
            return None
        bounds += found
        position = match.end()
        while position < len(text) and text[position].isspace():
            position += 1
    return bounds


def _node_range(version: str, expression: str) -> bool:
    """npm-compatible range satisfaction for a release version.

    Every `||` alternative is parsed before any is evaluated, so one valid
    branch can never hide invalid syntax elsewhere (fails closed).
    """
    release = _RELEASE_VERSION.fullmatch(version) if type(version) is str else None
    if release is None or type(expression) is not str or len(expression) > 512:
        return False
    actual = tuple(int(item) for item in release.groups())
    alternatives = [_range_bounds(item) for item in expression.split('||')]
    if any(bounds is None for bounds in alternatives):
        return False
    checks = {'==': lambda a, b: a == b, '>=': lambda a, b: a >= b, '>': lambda a, b: a > b,
              '<': lambda a, b: a < b, '<=': lambda a, b: a <= b}
    return any(all(checks[operator](actual, target) for operator, target in bounds or ()) for bounds in alternatives)


def _locked_documentation(values: dict[str, str]) -> bool:
    try:
        project = tomllib.loads(values['backend/pyproject.toml'])['project']
        uv = tomllib.loads(values['backend/uv.lock'])
        package = json.loads(values['frontend/package.json'])
        lock = yaml.safe_load(values['frontend/pnpm-lock.yaml'])
        python = TOOL_RULES['python'][0]
        node = TOOL_RULES['node'][0]
        pnpm = TOOL_RULES['pnpm'][0]
        if project['requires-python'] != uv['requires-python'] or python not in SpecifierSet(project['requires-python']):
            return False
        # Optional manifest declarations must agree with the lane when present;
        # their absence is not a disagreement (the README statement is checked below).
        manager = package.get('packageManager')
        if manager is not None and (type(manager) is not str or manager.split('+', 1)[0] != 'pnpm@' + pnpm):
            return False
        engines = package.get('engines', {})
        if type(engines) is not dict or ('node' in engines and not _node_range(node, engines['node'])):
            return False
        readme = values['README.md']
        for word, pattern, expected in (('Python', r'(\d+\.\d+(?:\.\d+)?)\+?', python), (r'Node(?:\.js)?', r'(\d+(?:\.\d+){0,2})\+?', node), ('pnpm', r'(\d+\.\d+\.\d+)', pnpm)):
            stated = []
            # The whole token after the tool name (Unicode digits included, so a
            # non-ASCII digit can never cut it short) must be one ASCII version.
            for token in re.findall(word + r'\s+(\d(?:\d|[A-Za-z._+-])*)', readme, re.IGNORECASE):
                token = token.rstrip('.')
                if not token.isascii():
                    return False
                version = re.match(pattern, token, re.ASCII)
                if version is None:
                    continue  # not a version statement (for example "Python 3")
                if version.end() != len(token):
                    return False
                stated.append(version.group(1))
            if not stated or any(not (expected == value or expected.startswith(value + '.')) for value in stated):
                return False
        locked = {item['name'].lower().replace('_', '-'): item['version'] for item in uv.get('package', [])}
        for dependency in project.get('dependencies', []):
            item = Requirement(dependency)
            if item.marker is None or item.marker.evaluate({'python_version': '3.12', 'python_full_version': python}):
                if locked.get(item.name.lower().replace('_', '-')) not in item.specifier:
                    return False
        importer = lock['importers']['.']
        for category in ('dependencies', 'devDependencies', 'optionalDependencies'):
            specified = package.get(category, {})
            resolved = importer.get(category, {})
            if set(specified) != set(resolved):
                return False
            for name, expression in specified.items():
                entry = resolved[name]
                if entry['specifier'] != expression or not _node_range(entry['version'].split('(')[0], expression):
                    return False
        return all(_node_range(node, entry['engines']['node']) for entry in lock.get('packages', {}).values() if 'node' in entry.get('engines', {}))
    except (ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
        return False


_NEGATIONS = re.compile(r"\b(?:not|never|no|cannot|can't|isn't|aren't|doesn't|don't|neither|without|false|distinct|separate|different|rather than|instead of)\b")
_EQUATES = re.compile(r"\b(?:is|are|isn't|aren't|counts? as|constitutes?|equals?|serves? as|replaces?|substitutes? for|amounts? to|means?)\b")


# The only accepted ways to deny that automated evidence is human acceptance. The
# negation must govern the human-acceptance predicate itself; any other equating
# sentence about the two is treated as an affirmation (fail closed).
# Subject words exclude verbs, negations and the predicate, so no second clause or
# affirmation can hide inside a subject phrase.
_WORD = (r"(?!(?:is|are|was|were|be|been|not|never|no|nor|human|acceptance|counts?|means?|equals?|constitutes?"
         r"|replaces?|but|except|unless|if|only|than|without|doubt|question|also|too)\b)[a-z][a-z/-]*")
# Only determiners may precede "automated"; an adverbial or quantifying prefix
# ("no doubt", "not only") could otherwise reverse the sentence.
_DETERMINERS = r"(?:(?:the|all|any|these|those|such|our|its|their|this|that) ){0,2}"
_SUBJECT = rf"{_DETERMINERS}automated(?: {_WORD}){{0,4}}"
_AUTOMATED_DENIALS: tuple[re.Pattern[str], ...] = (
    re.compile(_SUBJECT + r" (?:is|are) (?:not|never) human acceptance"),
    re.compile(_SUBJECT + r" (?:isn't|aren't) human acceptance"),
    re.compile(rf"no automated(?: {_WORD}){{0,4}} (?:is|are) human acceptance"),
    re.compile(rf"(?:{_WORD} ){{1,4}}(?:is|are) automated(?: {_WORD}){{0,4}}, not human acceptance"),
    re.compile(rf"human acceptance (?:is|are) (?:not|never) automated(?: {_WORD}){{0,4}}"),
)


def _negated(clause: str) -> bool:
    """Odd negation count means the clause denies its predicate."""
    return len(_NEGATIONS.findall(clause)) % 2 == 1


def _limitation_boundary(specification: str) -> bool:
    """Sentence-level check of the two DTD limitation statements; fails closed.

    A sentence naming automated evidence and human acceptance with any equating
    verb must be one of the exact denial forms in `_AUTOMATED_DENIALS`; every
    other equating sentence (affirmations, mixed or unclassifiable clauses) is a
    contradiction. Negation words are never counted across clauses. A held-out
    sentence must place execution/data outside; inside, both placements or a
    negated outside placement fail the key regardless of other sentences.
    """
    text = specification.lower().translate({0x2018: "'", 0x2019: "'"})
    text = re.sub(r'\s+', ' ', re.sub(r'[*_`>#]', ' ', text))
    sentences = [item.strip() for item in re.split(r'(?<=[.!?;:])\s+', text) if item.strip()]
    automated = held_out = contradiction = False
    for sentence in sentences:
        if 'automated' in sentence and 'human acceptance' in sentence:
            # A list that merely names both categories ("distinguish automated
            # checks, ..., human acceptance") neither affirms nor denies.
            if _EQUATES.search(sentence):
                clause = sentence.rstrip('.!?;: ')
                if any(form.fullmatch(clause) for form in _AUTOMATED_DENIALS):
                    automated = True
                else:
                    contradiction = True
        if 'held-out' in sentence or 'held out' in sentence:
            outside = re.search(r'\b(?:external(?:ly)?|outside)\b', sentence)
            inside = re.search(r'\b(?:internal(?:ly)?|inside|in-house)\b', sentence)
            if outside and inside:
                contradiction = True
            elif outside:
                if _negated(sentence[:outside.end()]):
                    contradiction = True
                else:
                    held_out = True
            elif inside and not _negated(sentence[:inside.end()]):
                contradiction = True
    return automated and held_out and not contradiction


# CommonMark inline destinations (optionally <bracketed>) with an optional title,
# HTML src/href in any quoting, and reference definitions. Every "](" must be a
# parsed inline link; an unparsed one fails setup_paths (fail closed).
_INLINE_LINK = re.compile(r'\]\(\s*(?:<([^<>\n]*)>|([^\s()<>]+))(?:\s+(?:"[^"\n]*"|\'[^\'\n]*\'|\([^()\n]*\)))?\s*\)')
_HTML_LINK = re.compile(r'\b(?:src|href)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'<>`=]+))', re.IGNORECASE)
# A reference definition's destination may follow on the next line (CommonMark
# allows one line ending); every definition label must yield a parsed destination.
_REFERENCE_LABEL = r'\[(?:[^\[\]\\]|\\.){1,999}\]'
_REFERENCE_START = re.compile(r'^[ \t]*' + _REFERENCE_LABEL + ':', re.MULTILINE | re.DOTALL)
_REFERENCE_LINK = re.compile(r'^[ \t]*' + _REFERENCE_LABEL + r':[ \t]*(?:\r?\n[ \t]*)?(?:<([^<>\n]*)>|([^\s<>]\S*))',
                             re.MULTILINE | re.DOTALL)
# The scan never hides README text. Block-quote and list-item markers are removed
# (they are container syntax, so a definition inside `> `, `- `, `1. ` or nested
# containers is still found) and definitions are recognized at any indentation.
# Code blocks and code spans are deliberately NOT excluded: a mis-parsed code
# region could otherwise hide a real link, so a link or definition shown as code is
# still checked. That can only over-reject (fail closed), never pass a missing file.
_CONTAINER_PREFIX = re.compile(r'^(?:[ \t]*(?:>[ \t]?|[-+*][ \t]+|[0-9]{1,9}[.)][ \t]+))+', re.MULTILINE | re.ASCII)


def _readme_links(readme: str) -> list[str] | None:
    inline = _INLINE_LINK.findall(readme)
    flattened = _CONTAINER_PREFIX.sub('', readme)
    references = _REFERENCE_LINK.findall(flattened)
    starts = _REFERENCE_START.findall(flattened)
    # A blank line ends a label, so a "label" spanning one is not a definition here;
    # refusing it keeps unparsed definitions from being silently skipped.
    if (len(inline) != readme.count('](') or len(references) != len(starts)
            or any(re.search(r'\n[ \t\r]*\n', start) for start in starts)):
        return None
    groups = [*inline, *_HTML_LINK.findall(readme), *references]
    return [next((item for item in group if item), '') for group in groups]


_COVERAGE_TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    'migration_restore': ('backend/tests/test_d34_migrations.py', ('restore_database_backup', 'database_lease_unavailable')),
    'recovery_codes': ('backend/tests/test_d35_startup_recovery_api.py', ('safe_retry', 'external_outcome_unknown', 'migration_failed')),
}


def _python_terms(source: str) -> set[str]:
    """Exact code tokens used inside committed `test*` functions.

    Names, attributes and string constants in test bodies and their decorators
    count; comments, docstrings, function names and module-level values cannot
    satisfy coverage.
    """
    tree = ast.parse(source)
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                  and isinstance(node.body[0].value.value, str)}
    terms: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or not function.name.startswith('test'):
            continue
        for statement in (*function.decorator_list, *function.body):
            for node in ast.walk(statement):
                if isinstance(node, ast.Name):
                    terms.add(node.id)
                elif isinstance(node, ast.Attribute):
                    terms.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    terms.add(node.value)
    return terms


def _covers(source: str, required: tuple[str, ...]) -> bool:
    terms = _python_terms(source)
    return all(term in terms for term in required)


def documentation_checks(source: Path, *, contracts_passed: bool | dict[str, bool], inventory: frozenset[str]) -> dict[str, bool]:
    """Six documentation keys. README links must name files of `inventory` (the
    materialized record), never files the verifier later wrote into the group."""
    checks = dict.fromkeys(DOC_KEYS, False)
    if type(inventory) is not frozenset:
        return checks
    required = ('README.md', 'Makefile', 'backend/.env.example', 'backend/pyproject.toml',
                'backend/uv.lock', 'frontend/package.json', 'frontend/pnpm-lock.yaml',
                'backend/scripts/plan_c_demo.py', 'backend/tests/test_d34_migrations.py',
                'backend/tests/test_d35_startup_recovery_api.py', 'frontend/src/test/recovery-status.test.tsx', 'specification.md')
    values = {}
    for name in required:
        try:
            materialization._directory((source / name).parent)
            values[name] = blinded_io.read_regular(source / name, maximum=8 * 1024 * 1024).decode('utf-8')
        except (OSError, ValueError, UnicodeError):
            return checks
    readme = values['README.md']
    links = _readme_links(readme)
    checks['setup_paths'] = links is not None
    for target in links or ():
        if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]+:', target) or target.startswith('#'):
            continue  # external URL schemes (two or more characters) and in-page anchors
        reference = target.split('#', 1)[0].split('?', 1)[0]
        try:
            # GitHub resolves '/x' against the repository root; anything that is
            # not a plain forward-slash relative path to an inventoried file is refused.
            relative = PurePosixPath(reference[1:] if reference.startswith('/') else reference)
            if (not reference or '\\' in reference or re.match(r'[a-zA-Z]:', reference) or relative.is_absolute()
                    or any(part in {'', '.', '..'} or part.endswith(('.', ' ')) for part in relative.parts)
                    or any(part in {'node_modules', '.venv', 'dist', '.git'} for part in relative.parts)):
                raise ValueError('setup reference not inventoried')
            if relative.as_posix() not in inventory:
                raise ValueError('setup reference not inventoried')
            materialization._directory((source / relative).parent)
            blinded_io.fingerprint_regular(source / relative, maximum=8 * 1024 * 1024)
        except (OSError, ValueError):
            checks['setup_paths'] = False
    checks['locked_versions'] = _locked_documentation(values)
    demo = values['backend/scripts/plan_c_demo.py']
    try:
        tree = ast.parse(demo)
        functions = {item.name for item in tree.body if isinstance(item, ast.FunctionDef)}
        modes = [ast.literal_eval(keyword.value) for item in ast.walk(tree) if isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == 'add_argument' and item.args and isinstance(item.args[0], ast.Constant) and item.args[0].value == '--mode' for keyword in item.keywords if keyword.arg == 'choices']
        checks['mode_commands'] = {'prepare_storage', 'exclusive_demo', 'demo_settings', 'create_demo_app', 'main'} <= functions and len(modes) == 1 and set(modes[0]) == {'all_tools', 'stateful'}
    except (SyntaxError, ValueError, TypeError):
        pass
    results = dict.fromkeys(('recovery_codes', 'migration_restore', 'ui_recovery'), contracts_passed) if type(contracts_passed) is bool else contracts_passed
    # The committed tests must still cover the DTD cases (live-lease rejection,
    # backup restore and the three recovery codes); passing an unrelated test
    # file is not evidence. Coverage is structural (AST), results are executed.
    covered: dict[str, bool] = {}
    for key, (name, terms) in _COVERAGE_TERMS.items():
        try:
            covered[key] = _covers(values[name], terms)
        except (SyntaxError, ValueError):
            covered[key] = False
    ui_codes = all(code in values['frontend/src/test/recovery-status.test.tsx'] for code in _COVERAGE_TERMS['recovery_codes'][1])
    checks['recovery_codes'] = results.get('recovery_codes') is True and results.get('ui_recovery') is True and covered['recovery_codes'] and ui_codes
    checks['migration_restore'] = results.get('migration_restore') is True and covered['migration_restore']
    checks['limitation_boundary'] = _limitation_boundary(values['specification.md'])
    return checks


def _installed_browser(value: Path | None) -> Path:
    if value is not None:
        materialization._directory(value.absolute().parent)
        before = value.absolute().lstat()
        if value.is_symlink() or blinded_io.is_reparse(before):
            raise ValueError('browser executable link refused')
        return verification._native_path(value)
    choices = [Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Google/Chrome/Application/chrome.exe',
               Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / 'Google/Chrome/Application/chrome.exe'] if os.name == 'nt' else [Path(p) for name in ('google-chrome', 'chromium', 'chromium-browser') if (p := shutil.which(name))]
    for path in choices:
        if path.is_file():
            materialization._directory(path.absolute().parent)
            if blinded_io.is_reparse(path.lstat()) or path.is_symlink():
                raise ValueError('browser executable link refused')
            return verification._native_path(path)
    raise ValueError('installed native browser unavailable')


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _profile() -> dict[str, Any]:
    return dict(model='synthetic-d39-embedding', weights_sha256='0' * 64, dimensions=2,
                document_prefix='passage: ', query_prefix='query: ', normalization='l2-full-v1',
                transport='local-openai-embeddings-v1', tokenizer_sha256=None, source_revision=None)


async def _candidate_action(scope: blinded_runtime.OwnedProcessScope, group: verification._ExecutionGroup,
                            python: verification._ToolSet, environment: dict[str, str],
                            configuration: dict[str, Any], action: str, *, deadline: int,
                            serving: bool = False) -> Any:
    token = secrets.token_hex(16)
    config_path = group.root / (token + '.config.json')
    await asyncio.to_thread(blinded_io.write_exclusive, config_path, canonical_json_bytes(configuration) + b'\n')
    path = Path(__file__).with_name('d39_candidate_smoke.py')
    # Origin fixed to the current attested bootstrap; caller cannot replace its path.
    await asyncio.to_thread(group.assert_source)
    await asyncio.to_thread(python.verify)
    argv = (str(python.executable), '-B', '-c', _BOOTSTRAP, str(path), '--action', action, '--configuration', str(config_path))
    arguments = dict(scope=scope, argv=argv, cwd=group.source / 'backend', env=environment,
                     stdout_path=group.root / (token + '.out'), stderr_path=group.root / (token + '.err'), deadline_seconds=deadline)
    if serving:
        return await blinded_runtime.start_owned_process(**arguments)
    outcome = await blinded_runtime.run_owned_command(**arguments)
    if outcome.outcome != 'completed' or outcome.exit_code != 0:
        raise ValueError('candidate smoke action failed')
    await asyncio.to_thread(group.assert_source)
    await asyncio.to_thread(python.verify)
    return verification._probe_json(await asyncio.to_thread(blinded_io.read_regular, Path(configuration['summary_path']), maximum=65536))


async def _http_health(port: int, *, owner_token: str, child: blinded_runtime.OwnedProcess,
                       deadline: float, expected: str = 'ok') -> None:
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=2) as client:
        async def read(path: str) -> dict[str, Any] | None:
            async with client.stream('GET', f'http://127.0.0.1:{port}' + path) as response:
                if response.status_code != 200:
                    return None
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    raw.extend(chunk)
                    if len(raw) > 4096:
                        raise ValueError('owned health size exceeded')
                return verification._probe_json(bytes(raw))
        while time.monotonic() < deadline:
            if not await asyncio.to_thread(child.is_running):
                if child.outcome is not None or child._observation()[0] in {'exited', 'launch_failed'}:
                    raise ValueError('owned server terminated before readiness')
                await asyncio.sleep(0.05)
                continue  # Gate assigned, target not yet launched; no health trusted.
            try:
                owner = await read('/__d39-owned')
                if owner == {'owner': owner_token}:
                    health = await read('/api/health')
                    if health is not None and health.get('status') == expected and await asyncio.to_thread(child.is_running):
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)
    raise ValueError('owned server startup deadline exceeded')


async def _startup_stage(scope: blinded_runtime.OwnedProcessScope, group: verification._ExecutionGroup,
                         python: verification._ToolSet, env: dict[str, str], config: dict[str, Any], stage: str) -> dict[str, Any]:
    configuration = config | {'port': _port(), 'owner_token': secrets.token_hex(32),
                              'scenario': 'stateful' if stage == 'stateful_startup' else 'normal'}
    server = await _candidate_action(scope, group, python, env, configuration | {'summary_path': str(group.root / (stage + '.server.json'))}, 'serve', deadline=180, serving=True)
    try:
        await _http_health(configuration['port'], owner_token=configuration['owner_token'], child=server, deadline=time.monotonic() + 30)
        result = await _candidate_action(scope, group, python, env, configuration, stage, deadline=120)
        if not await asyncio.to_thread(server.is_running):
            raise ValueError('startup server terminated during observation')
        return result
    finally:
        await server.stop()
        await server.wait()
        if scope._failed or not server._tree_confirmed:
            raise ValueError('startup server teardown failed')


def _make_receipt(*, summary: Any, **fields: Any) -> SmokeStageReceipt:
    try:
        return SmokeStageReceipt(stage=summary.stage, outcome='passed', summary=summary, **fields)
    except ValidationError:
        # Passed-only constraints fail without changing any observed field.
        return SmokeStageReceipt(stage=summary.stage, outcome='failed', summary=summary, **fields)


async def _browser_stage(scope: blinded_runtime.OwnedProcessScope, group: verification._ExecutionGroup,
                         python: verification._ToolSet, env: dict[str, str], config: dict[str, Any],
                         browser_executable: Path | None, base_tools: tuple[ToolExecutionBinding, ...], frontend: verification._ToolSet) -> tuple[dict[str, Any], tuple[ToolExecutionBinding, ...], tuple[FileFingerprint, ...]]:
    chrome = await asyncio.to_thread(_installed_browser, browser_executable)
    chrome_fp = await asyncio.to_thread(verification._native_file, chrome, 'tools/chrome')
    # The sandbox owns browser transport packages too; prove its installed origin/version.
    probe = verification._probe_json(await verification._probe(scope, group, (str(python.executable), '-I', '-B', '-c', "import importlib.metadata,websockets.sync.client,json; print(json.dumps({'version':importlib.metadata.version('websockets'),'origin':websockets.sync.client.__file__}))"), env))
    if probe.get('version') != TOOL_RULES['websockets'][0] or not Path(str(probe.get('origin'))).is_relative_to(group.root / 'env'):
        raise ValueError('websockets sandbox origin/version mismatch')
    sockets_fp = await asyncio.to_thread(verification._tool_file, Path(str(probe['origin'])), 'tools/websockets_module')
    normal = config | {'storage': str(group.root / 'media-storage'), 'port': _port(), 'summary_path': str(group.root / 'post-count.json'), 'owner_token': secrets.token_hex(32)}
    failed = config | {'storage': str(group.root / 'failed-storage'), 'port': _port(), 'scenario': 'migration_failed', 'summary_path': str(group.root / 'failed-post-count.json'), 'owner_token': secrets.token_hex(32)}
    server = None
    broken = None
    owned_browser = None
    try:
        # Real fake-provider media gives a usable playback target.
        media_config = config | {'storage': str(group.root / 'media-storage'), 'summary_path': str(group.root / 'browser-media-summary.json')}
        media = await _candidate_action(scope, group, python, env, media_config, 'ffmpeg', deadline=240)
        if not all(media.get(name) is True for name in ('providers_fake', 'video_present', 'subtitle_present', 'publication_bound')):
            raise ValueError('browser media generation failed')
        media_record = verification._probe_json(blinded_io.read_regular(Path(media_config['summary_path']).with_suffix('.media.json'), maximum=65536))
        for kind in ('video', 'subtitle'):
            await asyncio.to_thread(_media_fingerprint, Path(media_config['storage']), media_record, kind)
        # Playback server is the same real candidate app started on the media storage.
        normal['storage'] = media_config['storage']
        server = await _candidate_action(scope, group, python, env, normal, 'serve', deadline=300, serving=True)
        await _http_health(normal['port'], owner_token=normal['owner_token'], child=server, deadline=time.monotonic() + 30)
        projects = await _candidate_action(scope, group, python, env, normal | {'summary_path': str(group.root / 'media-browser-projects.json')}, 'seed_browser', deadline=30)
        projects['media'] = media_record['project_id']
        broken = await _candidate_action(scope, group, python, env, failed, 'serve', deadline=300, serving=True)
        await _http_health(failed['port'], owner_token=failed['owner_token'], child=broken, deadline=time.monotonic() + 30, expected='degraded')
        profile = group.root / 'browser-profile'
        profile.mkdir()
        owned_browser = await blinded_runtime.start_owned_process(scope=scope, argv=(str(chrome), '--headless=new', '--no-first-run', '--no-default-browser-check', '--disable-background-networking', '--disable-component-update', '--disable-sync', '--disable-extensions', '--disable-default-apps', '--no-proxy-server', '--renderer-process-limit=1', '--disk-cache-size=1048576', '--media-cache-size=1048576', '--remote-debugging-address=127.0.0.1', '--remote-debugging-port=0', '--user-data-dir=' + str(profile), 'about:blank'), cwd=group.root, env=env, stdout_path=group.root / 'chrome.out', stderr_path=group.root / 'chrome.err', deadline_seconds=300)
        screenshots = group.root / 'screenshots'
        screenshots.mkdir()
        browser_config = group.root / 'browser.config.json'
        browser_summary = group.root / 'browser-cdp.summary.json'
        await asyncio.to_thread(blinded_io.write_exclusive, browser_config, canonical_json_bytes(dict(profile=str(profile), base_url=f"http://127.0.0.1:{normal['port']}", migration_url=f"http://127.0.0.1:{failed['port']}/", projects=projects, screenshots=str(screenshots), summary_path=str(browser_summary), counter_path=normal['summary_path'])) + b'\n')
        # Actual CDP transport uses the attested sandbox module, not the controller's.
        browser_result = await blinded_runtime.run_owned_command(scope=scope, argv=(str(python.executable), '-I', '-B', '-c', _BOOTSTRAP, str(Path(browser_smoke.__file__)), '--configuration', str(browser_config)), cwd=group.root, env=env, stdout_path=group.root / 'browser-cdp.out', stderr_path=group.root / 'browser-cdp.err', deadline_seconds=180)
        if browser_result.outcome != 'completed' or browser_result.exit_code != 0:
            raise ValueError('owned browser observations failed')
        observations = verification._probe_json(await asyncio.to_thread(blinded_io.read_regular, browser_summary, maximum=65536))
        version = observations.pop('browser_version')
        tools = tuple(sorted((*base_tools, ToolExecutionBinding(role='chrome', version=version, executable=chrome_fp, launcher=None),
                              ToolExecutionBinding(role='websockets', version=TOOL_RULES['websockets'][0], executable=python.bindings[0].executable, launcher=sockets_fp)), key=lambda item: item.role))
        # Release browser/server memory before running independent contract tests.
        for child in (owned_browser, broken, server):
            await child.stop()
            await child.wait()
        if scope._failed:
            raise ValueError('browser/server teardown failed')
        contracts = await _candidate_action(scope, group, python, env, config | {'summary_path': str(group.root / 'contract-test-summary.json')}, 'contract_tests', deadline=180)
        await asyncio.to_thread(frontend.verify)
        ui = await blinded_runtime.run_owned_command(scope=scope, argv=(str(frontend.executable), str(frontend.launcher), '-y', 'pnpm@' + TOOL_RULES['pnpm'][0], 'test', 'src/test/recovery-status.test.tsx', '--maxWorkers=1', '--minWorkers=1', '--no-file-parallelism'), cwd=group.source / 'frontend', env=env,
                                                    stdout_path=group.root / 'ui-contract.out', stderr_path=group.root / 'ui-contract.err', deadline_seconds=120)
        await asyncio.to_thread(frontend.verify)
        contracts['ui_recovery'] = ui.outcome == 'completed' and ui.exit_code == 0
        observations.update(stage='browser', documentation_checks=await asyncio.to_thread(documentation_checks, group.source, contracts_passed=contracts,
                                                                                           inventory=frozenset(item.path for item in group.record.files)))
        artifacts = []
        for path in sorted(screenshots.iterdir()):
            artifacts.append(await asyncio.to_thread(_screenshot_fingerprint, path))
        if await asyncio.to_thread(verification._native_file, chrome, 'tools/chrome') != chrome_fp or await asyncio.to_thread(verification._tool_file, Path(probe['origin']), 'tools/websockets_module') != sockets_fp:
            raise ValueError('browser transport/tool source changed')
        return observations, tools, tuple(artifacts)
    finally:
        for child in (owned_browser, broken, server):
            if child is not None:
                await child.stop()
                await child.wait()


def _symlink_prerequisite(root: Path) -> None:
    """The committed D34 contract tests need symlink creation; refuse early otherwise.

    Without it (Windows lacking SeCreateSymbolicLinkPrivilege/Developer Mode)
    three tests skip, and a skip never counts as a passed contract test.
    """
    probe = root / ('.symlink-probe-' + secrets.token_hex(8))
    try:
        os.symlink('symlink-probe-target', probe)
    except OSError:
        raise ValueError('symlink creation prerequisite unavailable') from None
    os.unlink(probe)


def _screenshot_fingerprint(path: Path) -> FileFingerprint:
    materialization._directory(path.parent)
    size, digest = blinded_io.fingerprint_regular(path, maximum=4 * 1024 * 1024)
    return FileFingerprint(path='screenshots/' + path.name, size=size, sha256=digest)


def _media_fingerprint(storage: Path, media: dict[str, Any], name: str) -> FileFingerprint:
    # Internal transient output still cannot select a path outside owned storage.
    relative = FileFingerprint(path=media[name], size=0, sha256='0' * 64).path
    file = storage / relative
    materialization._directory(file.parent)
    size, digest = blinded_io.fingerprint_regular(file, maximum=32 * 1024 * 1024)
    expected = media['manifest'][name]
    if type(expected) is not dict or expected.get('path') != relative or type(expected.get('size')) is not int or expected['size'] != size or expected.get('sha256') != digest:
        raise ValueError('media publication fingerprint mismatch')
    return FileFingerprint(path='media/' + name, size=size, sha256=digest)


async def _run(*, candidate: Path, frozen_path: Path, runtime: Path, path: Path, digest: str,
               work: Path, output: Path, browser_executable: Path | None) -> SmokeManifest:
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1, thread_name_prefix='d39-smoke'))
    record = materialization._bound_materialization(path, digest)
    root = verification._trusted_tool_root()
    attestation = await asyncio.to_thread(verification._attest_verifier, root)
    verification._validate_output(output, root, candidate, runtime, work, path, work / 'not-used-input' / 'smoke.json', frozen_path)
    # Constructed before any owned directory exists: an invalid D39_AGENT_PID
    # refuses without leaving an output directory or anchor behind.
    scope = blinded_runtime.OwnedProcessScope()
    output.mkdir()
    output_anchor = materialization._open_anchor(output)
    output_identity = output_anchor.identity
    group = None
    entered = False
    receipts = []
    async def boundary() -> None:
        await asyncio.to_thread(verification._inventory_boundary, candidate, runtime, work, path, digest, record, root, attestation, frozen_path)
        materialization._assert_directory(output, output_anchor)
    try:
        await boundary()
        group = await asyncio.to_thread(verification._ExecutionGroup, work, runtime, record)
        await asyncio.to_thread(_symlink_prerequisite, group.root)
        native = await asyncio.to_thread(verification._native_path, Path(getattr(sys, '_base_executable', sys.executable)))
        node = await asyncio.to_thread(verification._installed_node)
        env = await asyncio.to_thread(verification.build_group_environment, group.root, python_executable=native, node_executable=node)
        await scope.__aenter__()
        entered = True
        bootstrap = await verification._bootstrap_python(scope, group, env)
        install = await verification._execute_command(index=0, group=group, scope=scope, tools=bootstrap, env=env, commands=[])
        if install.outcome != 'completed' or install.exit_code != 0:
            raise ValueError('smoke backend extras installation failed')
        python = await verification._sandbox_python(scope, group, env, bootstrap)
        frontend = await verification._frontend_tools(scope, group, env, node)
        for index in (5, 7):
            command = await verification._execute_command(index=index, group=group, scope=scope, tools=frontend, env=env, commands=[])
            if command.outcome != 'completed' or command.exit_code != 0:
                raise ValueError('smoke frontend bootstrap failed')
            if index == 5:
                await verification._frontend_preflight(scope, group, frontend, env)
        base_tools = tuple(sorted((*bootstrap.bindings, *python.bindings, *frontend.bindings), key=lambda item: item.role))
        profile = group.root / 'embedding-profile.json'
        await asyncio.to_thread(blinded_io.write_exclusive, profile, canonical_json_bytes(_profile()) + b'\n')
        # Actual installed media executables are mandatory; never package-downloaded.
        media_tools = []
        media_paths = {}
        for role in ('ffmpeg', 'ffprobe'):
            found = shutil.which(role)
            if found is None:
                raise ValueError('installed media executable missing')
            executable = await asyncio.to_thread(verification._native_path, Path(found))
            fingerprint = await asyncio.to_thread(verification._native_file, executable, 'tools/' + role)
            lines = (await verification._probe(scope, group, (str(executable), '-version'), env)).decode('utf-8').splitlines()
            if not lines or not lines[0].strip():
                raise ValueError('installed media version unavailable')
            version = lines[0]
            media_tools.append(ToolExecutionBinding(role=role, version=version[:512], executable=fingerprint, launcher=None))
            media_paths[role] = str(executable)
        with fake_providers() as (provider_url, provider_counts):
            config: dict[str, Any] = dict(group_root=str(group.root), source_root=str(group.source), storage=str(group.root / 'startup-storage'), frontend=str(group.source / 'frontend/dist'),
                          profile=str(profile), index=str(group.root / 'index'), provider_url=provider_url,
                          model='synthetic-d39-chat', port=_port(), summary_path=str(group.root / 'index-summary.json'), scenario='normal', owner_token=secrets.token_hex(32), **media_paths)
            await _candidate_action(scope, group, python, env, config, 'build_index', deadline=60)
            for position, stage in enumerate(SMOKE_STAGES):
                await asyncio.to_thread(group.assert_source)
                summary_path = group.root / (stage + '.summary.json')
                stage_config = config | {'summary_path': str(summary_path)}
                if stage in ('legacy_migration', 'restore'):
                    stage_config['storage'] = str(group.root / 'migration-storage')
                elif stage.endswith('_startup'):
                    stage_config['storage'] = str(group.root / (stage + '-storage'))
                elif stage == 'ffmpeg':
                    stage_config['storage'] = str(group.root / 'ffmpeg-storage')
                tools = base_tools
                artifacts: tuple[FileFingerprint, ...] = ()
                started = time.monotonic()
                if stage in ('browser', 'ffmpeg'):
                    for media_tool in media_tools:
                        if await asyncio.to_thread(verification._native_file, Path(media_paths[media_tool.role]), media_tool.executable.path) != media_tool.executable:
                            raise ValueError('media tool changed before use')
                if stage == 'browser':
                    observations, tools, artifacts = await asyncio.wait_for(_browser_stage(scope, group, python, env, stage_config, browser_executable, base_tools, frontend), timeout=_DEADLINES[position])
                else:
                    calls = dict(provider_counts)
                    if stage.endswith('_startup'):
                        observations = await _startup_stage(scope, group, python, env, stage_config, stage)
                    else:
                        observations = await _candidate_action(scope, group, python, env, stage_config, stage, deadline=_DEADLINES[position])
                    if stage.endswith('_startup'):
                        if observations['model_calls'] != provider_counts['chat'] - calls['chat']:
                            raise ValueError('startup model counter mismatch')
                        if stage == 'stateful_startup' and observations['embedding_calls'] != provider_counts['embedding'] - calls['embedding']:
                            raise ValueError('startup embedding counter mismatch')
                    if stage == 'ffmpeg':
                        tools = tuple(sorted((*base_tools, *media_tools), key=lambda item: item.role))
                        items = []
                        if observations['video_present'] and observations['subtitle_present']:
                            media = verification._probe_json(blinded_io.read_regular(summary_path.with_suffix('.media.json'), maximum=65536))
                            for name in ('video', 'subtitle'):
                                items.append(await asyncio.to_thread(_media_fingerprint, Path(stage_config['storage']), media, name))
                        artifacts = tuple(items)
                if time.monotonic() - started > _DEADLINES[position]:
                    raise ValueError('smoke stage deadline exceeded')
                summary = TypeAdapter(_SUMMARIES[position]).validate_python(observations, strict=True)
                raw_summary = canonical_json_bytes(summary) + b'\n'
                # Bind only observations bracketed by all source/tool boundaries.
                await asyncio.to_thread(group.assert_source)
                for tool in (bootstrap, python, frontend):
                    await asyncio.to_thread(tool.verify)
                if stage in ('browser', 'ffmpeg'):
                    for media_tool in media_tools:
                        if await asyncio.to_thread(verification._native_file, Path(media_paths[media_tool.role]), media_tool.executable.path) != media_tool.executable:
                            raise ValueError('media tool source changed')
                # Child summary output is disposable; persist only the validated model.
                materialization._assert_directory(output, output_anchor)
                blinded_io.publish_immutable(output / (stage + '.summary.json'), raw_summary, 'smoke_summary', maximum=65536)
                summary_artifact = FileFingerprint(path=stage + '.summary.json', size=len(raw_summary), sha256=hashlib.sha256(raw_summary).hexdigest())
                artifacts = tuple(sorted((*artifacts, summary_artifact), key=lambda item: item.path))
                binding = dict(candidate_id=record.candidate_id, git_commit=record.git_commit, freeze_sha256=record.freeze_sha256,
                               materialization_sha256=digest, runtime_instance_id=record.runtime_instance_id, runtime_source_sha256=record.runtime_source_sha256)
                receipt = _make_receipt(schema_version=1, tools=tools, artifacts=artifacts, summary=summary, **binding)
                raw = canonical_json_bytes(receipt) + b'\n'
                blinded_io.publish_immutable(output / (stage + '.receipt.json'), raw, 'smoke_receipt', maximum=65536)
                if parse_canonical_model(blinded_io.read_regular(output / (stage + '.receipt.json'), maximum=65536), SmokeStageReceipt, maximum=65536) != receipt:
                    raise ValueError('smoke receipt readback drift')
                receipts.append(receipt)
                if receipt.outcome != 'passed':
                    raise ValueError('required smoke observation failed')
        smoke = SmokeManifest(schema_version=1, producer_tool_sha256=attestation.aggregate_sha256, stage_receipts=tuple(receipts), **binding,
                              **{r.stage + '_sha256': hashlib.sha256(canonical_json_bytes(r) + b'\n').hexdigest() for r in receipts})
    finally:
        try:
            if group is not None:
                await verification._close_group(group, scope, entered)
        finally:
            verification.freeze._close_directory_anchor(output_anchor)
            await asyncio.to_thread(verification._inventory_boundary, candidate, runtime, work, path, digest, record, root, attestation, frozen_path)
    # No completed publication until owned group teardown/source checks succeed.
    final_anchor = materialization._open_anchor(output)
    try:
        if final_anchor.identity != output_identity:
            raise ValueError('smoke publication output identity changed')
        materialization._assert_directory(output, final_anchor)
        blinded_io.publish_immutable(output / 'smoke-manifest.json', canonical_json_bytes(smoke) + b'\n', 'smoke_manifest', maximum=1024 * 1024)
    finally:
        verification.freeze._close_directory_anchor(final_anchor)
    return smoke


def run_candidate_smokes(*, candidate_root: Path, freeze_manifest_path: Path, runtime_root: Path,
                         materialization_path: Path, expected_materialization_sha256: str,
                         work_root: Path, output_dir: Path, browser_executable: Path | None = None) -> SmokeManifest:
    try:
        return asyncio.run(_run(candidate=candidate_root.absolute(), frozen_path=freeze_manifest_path.absolute(), runtime=runtime_root.absolute(),
                                path=materialization_path.absolute(), digest=expected_materialization_sha256, work=work_root.absolute(),
                                output=output_dir.absolute(), browser_executable=browser_executable))
    except (OSError, ValueError, TimeoutError, RuntimeError, subprocess.SubprocessError):
        raise ValueError('D39 smoke refused; no complete smoke evidence') from None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError('D39 smoke arguments refused')


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    for name in ('candidate-root', 'freeze-manifest', 'runtime-root', 'materialization', 'work-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-materialization-sha256', required=True)
    parser.add_argument('--browser-executable', type=Path)
    try:
        values = parser.parse_args(argv)
        run_candidate_smokes(candidate_root=values.candidate_root, freeze_manifest_path=values.freeze_manifest,
                             runtime_root=values.runtime_root, materialization_path=values.materialization,
                             expected_materialization_sha256=values.expected_materialization_sha256,
                             work_root=values.work_root, output_dir=values.output, browser_executable=values.browser_executable)
        print('{"status":"completed"}')
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        print('D39 smoke refused', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
