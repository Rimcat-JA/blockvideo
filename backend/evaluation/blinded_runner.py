"""Crash-safe external D37 blinded evaluator orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from evaluation.blinded_io import (
    DEFAULT_JSON_BYTES,
    ensure_writable_directory as _ensure_writable_directory,
    fingerprint_regular as _fingerprint_regular,
    is_reparse as _is_reparse,
    publish_immutable as _publish_immutable,
    read_regular as _read_regular,
    require_directory as _require_directory,
    validate_directory as _validate_directory,
    write_atomic as _write_atomic,
    write_exclusive as _write_exclusive,
    write_or_validate_immutable as _write_or_validate_immutable,
)
from evaluation.blinded_runtime import (
    CandidateAnchor as _CandidateAnchor,
    assert_candidate_anchor as _assert_candidate_anchor,
    canonical_candidate_root as _canonical_candidate_root,
    close_candidate_anchor as _close_candidate_anchor,
    invoke_trial_host as _invoke_trial_host,
    open_candidate_anchor as _open_candidate_anchor,
)

from evaluation.blinded_contracts import (
    MAX_PROTOCOL_BYTES,
    EvaluationProtocol,
    case_category_bindings,
    opaque_case_token,
    token_key,
)
from evaluation.blinded_scoring import TrialScore, score_trial
from evaluation.evidence_json import parse_canonical_model
from evaluation.contracts import Case
from evaluation.corpus import load_cases, load_review
from evaluation.release_candidate.freeze import read_frozen_candidate
from evaluation.release_candidate.fingerprints import aggregate_fingerprints
from evaluation.release_candidate.contracts import FreezeManifest
from evaluation.result_contracts import (
    MAX_RESULT_BUNDLE_BYTES,
    CategoryResult,
    EvaluationResultBundle,
    ExcludedCaseToken,
    ModeResult,
    approval_partition,
)
from evaluation.sealed_evidence import seal_evidence
from evaluation.scripts.evaluation_trial_host import (
    TrialObservation, _embedding_configuration, _check_embedding_index,
)
from evaluation.tool_attestation import (
    FileFingerprint,
    ToolAttestation,
    aggregate_fingerprints as aggregate_tool_fingerprints,
    attest_tool,
    canonical_json_bytes,
    fingerprint_committed_file,
    fingerprint_file,
    validate_git_repository,
)
from evaluation.case_normalization import complete_settings, normalize_case
from evaluation.unlabeled_contracts import (
    UnlabeledTrialCase,
    canonical_case_sha256,
)

_D36_HOST_PATH = "backend/evaluation/scripts/evaluation_trial_host.py"
_D36_SOURCE_PATHS = (
    "backend/evaluation/blinded_io.py",
    "backend/evaluation/evidence_json.py",
    "backend/evaluation/final_protocol.json",
    "backend/evaluation/release_candidate/__init__.py",
    "backend/evaluation/release_candidate/contracts.py",
    "backend/evaluation/release_candidate/fingerprints.py",
    "backend/evaluation/release_candidate/freeze.py",
    _D36_HOST_PATH,
    "backend/evaluation/scripts/freeze_candidate.py",
    "backend/evaluation/tool_attestation.py",
    "backend/evaluation/unlabeled_contracts.py",
)
_D37_REQUIRED_PATHS = (
    "backend/app/operations/definitions.json",
    "backend/pyproject.toml",
    "backend/scripts/run_blinded_evaluation.py",
    "backend/uv.lock",
)
_PUBLIC_RESULT_NAME = "result-bundle.json"
_PARTIAL_NAME = "partial-result.json"
_PROTOCOL_NAME = "protocol.json"
_TEMP_PATTERN = re.compile(r"^\.(?P<final>[^/\\]+)\.tmp-[0-9a-f]{64}$")
_TOKEN_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODES = ("all_tools", "stateful")


class _TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    protocol_sha256: str
    case_token: str
    category_token: str
    mode: Literal["all_tools", "stateful"]
    outcome: Literal["completed", "transport_failure", "deadline_failure"]
    score: TrialScore | None
    candidate_snapshot_sha256: str | None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_primitive(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("trial schema version must be an integer")
        return value


class _RunContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    manifest: FreezeManifest
    d36_attestation: ToolAttestation
    d37_attestation: ToolAttestation
    freeze_sha256: str
    tool_root: Path
    candidate_anchor: _CandidateAnchor


def _tool_repo_root() -> Path:
    return Path(__file__).parents[2]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_file_bytes(value: BaseModel | dict[str, object]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def write_run_protocol_exclusive(
    output_root: Path, protocol: EvaluationProtocol
) -> Path:
    """Create the canonical protocol once, or validate exact bytes on resume."""
    root = _require_directory(output_root, "evaluation output", create=True)
    path = root / _PROTOCOL_NAME
    expected = _canonical_file_bytes(protocol)
    if len(expected) > MAX_PROTOCOL_BYTES:
        raise ValueError("protocol exceeds its maximum canonical size")
    parse_canonical_model(expected, EvaluationProtocol, maximum=MAX_PROTOCOL_BYTES)
    return _write_or_validate_immutable(
        path, expected, "protocol", maximum=MAX_PROTOCOL_BYTES
    )


def _unlabeled_bytes(case: UnlabeledTrialCase) -> bytes:
    return canonical_json_bytes(case.model_dump(mode="json", exclude_unset=True))


def _assert_no_forbidden_wire_keys(value: object) -> None:
    forbidden = {
        "expected",
        "known_limitation",
        "rationale",
        "rule_ids",
        "review",
        "decision",
        "score",
        "labels",
    }
    if isinstance(value, dict):
        if set(value) & forbidden:
            raise ValueError("unlabeled case projection contains private evaluation fields")
        for item in value.values():
            _assert_no_forbidden_wire_keys(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_wire_keys(item)


def _wire_settings(value: dict[str, object]) -> dict[str, object]:
    return {
        **value,
        "pronunciation_overrides": tuple(value["pronunciation_overrides"]),
    }


def _wire_projects(case: Case) -> tuple[dict[str, object], ...]:
    """Seed other projects referenced by jobs exactly as the D24 development seed does."""
    seen = {case.initial.project_id}
    additional = []
    for job in case.initial.jobs:
        if job["project_id"] in seen:
            continue
        seen.add(job["project_id"])
        additional.append({
            "project_id": job["project_id"],
            "revision": job["input_revision"],
            "settings": _wire_settings(job["input_settings"]),
            "project_status": "generating"
            if job["status"] in {"pending", "running"} else "failed",
        })
    # Prior turns and a changed request may name a project that only exists in prose;
    # seed it with product defaults so the reference is representable.
    referenced = [
        (turn["project_id"], max(turn["base_revision"], turn.get("result_revision") or 1))
        for turn in case.initial.prior_turns
    ]
    replacement = case.event.details.get("replacement_target_project_id")
    if case.event.kind == "same_id_different_body" and type(replacement) is int:
        referenced.append((replacement, 1))
    for project_id, revision in referenced:
        if project_id in seen:
            continue
        seen.add(project_id)
        additional.append({
            "project_id": project_id,
            "revision": max(
                [revision] + [r for p, r in referenced if p == project_id]
            ),
            "settings": _wire_settings(complete_settings({})),
            "project_status": "completed",
        })
    return tuple(additional)


def _wire_prior_turn(value: dict[str, object]) -> dict[str, object]:
    allowed = {
        "request_id",
        "project_id",
        "base_revision",
        "status",
        "question",
        "result_revision",
        "settings_saved",
        "proposal",
        "text",
        "relation",
        "parent_request_id",
        "successor_request_id",
    }
    turn = {key: item for key, item in value.items() if key in allowed}
    proposal = turn.get("proposal")
    if isinstance(proposal, dict):
        proposal = dict(proposal)
        if proposal.get("kind") == "clarification" and isinstance(
            proposal.get("missing_fields"), list
        ):
            proposal["missing_fields"] = tuple(proposal["missing_fields"])
        arguments = proposal.get("arguments")
        if isinstance(arguments, dict):
            arguments = dict(arguments)
            if isinstance(arguments.get("pronunciation_overrides"), list):
                arguments["pronunciation_overrides"] = tuple(
                    arguments["pronunciation_overrides"]
                )
            settings = arguments.get("settings")
            if isinstance(settings, dict):
                arguments["settings"] = {
                    **settings,
                    **(
                        {
                            "pronunciation_overrides": tuple(
                                settings["pronunciation_overrides"]
                            )
                        }
                        if "pronunciation_overrides" in settings
                        else {}
                    ),
                }
            proposal["arguments"] = arguments
        turn["proposal"] = proposal
    return turn


def case_to_unlabeled(case: Case) -> UnlabeledTrialCase:
    """Project one label-bearing D24 case through an explicit wire allowlist."""
    case = normalize_case(case)
    if case.split != "held_out" or not case.tags:
        raise ValueError("only categorized held-out cases can be projected")
    request = case.request.model_dump(mode="json")
    event: dict[str, object] = {"kind": case.event.kind, "request": request}
    details = case.event.details
    if case.event.kind == "same_id_different_body":
        event.update(
            {
                "replacement_text": details["replacement_text"],
                "replacement_target_project_id": details.get(
                    "replacement_target_project_id"
                ),
            }
        )
    elif case.event.kind == "revision_race":
        event.update(
            {
                "external_revision": details["external_revision"],
                "timing": details["timing"],
                "external_settings": {
                    **details["external_settings"],
                    **(
                        {
                            "pronunciation_overrides": tuple(
                                details["external_settings"]["pronunciation_overrides"]
                            )
                        }
                        if "pronunciation_overrides" in details["external_settings"]
                        else {}
                    ),
                },
            }
        )
    elif case.event.kind == "switch_target":
        event.update(
            {
                "selected_project_id_after": details["selected_project_id_after"],
                "action": details["action"],
            }
        )
    additional_projects = _wire_projects(case)
    initial: dict[str, object] = {
        "project_id": case.initial.project_id,
        "revision": case.initial.revision,
        "settings": _wire_settings(case.initial.settings),
        "project_status": case.initial.project_status,
        "jobs": tuple(
            {**item, "input_settings": _wire_settings(item["input_settings"])}
            for item in case.initial.jobs
        ),
        "history": tuple(
            {**item, "settings": _wire_settings(item["settings"])}
            for item in case.initial.history
        ),
        "artifact_revisions": tuple(case.initial.artifact_revisions),
        "prior_turns": tuple(_wire_prior_turn(item) for item in case.initial.prior_turns),
    }
    if additional_projects:
        initial["additional_projects"] = additional_projects
    payload: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "group_id": case.group_id,
        "category": case.tags[0],
        "split": "held_out",
        "event": event,
        "initial": initial,
    }
    payload["case_sha256"] = canonical_case_sha256(payload)
    projected = UnlabeledTrialCase.model_validate(payload, strict=True)
    serialized = _unlabeled_bytes(projected)
    decoded = json.loads(serialized)
    _assert_no_forbidden_wire_keys(decoded)
    if UnlabeledTrialCase.model_validate_json(serialized, strict=True) != projected:
        raise ValueError("unlabeled case did not survive strict serialization")
    return projected


def _git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    if completed.returncode != 0:
        raise ValueError("Git identity validation failed")
    return completed.stdout


def _d37_source_paths(root: Path) -> tuple[str, ...]:
    tracked = _git(root, "ls-files", "-z").split(b"\0")
    paths: set[str] = set(_D37_REQUIRED_PATHS)
    for raw in tracked:
        if not raw:
            continue
        try:
            relative = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("tracked source path is not UTF-8") from None
        if (
            relative.startswith("backend/evaluation/")
            and relative.endswith(".py")
        ) or (
            relative.startswith("backend/app/") and relative.endswith(".py")
        ):
            paths.add(relative)
    missing = paths - {
        raw.decode("utf-8") for raw in tracked if raw
    }
    if missing:
        raise ValueError("D37 runtime attestation source is not tracked")
    return tuple(sorted(paths))


_D37_SOURCE_PATHS = tuple(
    sorted(
        {
            *_D37_REQUIRED_PATHS,
            "backend/evaluation/__init__.py",
            "backend/evaluation/blinded_contracts.py",
            "backend/evaluation/blinded_runner.py",
            "backend/evaluation/blinded_scoring.py",
            "backend/evaluation/result_contracts.py",
            "backend/evaluation/sealed_evidence.py",
        }
    )
)


def _validate_candidate(candidate_root: Path, manifest: Any) -> None:
    root = _canonical_candidate_root(candidate_root)
    if _git(root, "branch", "--show-current").strip():
        raise ValueError("candidate must be a detached checkout")
    commit = validate_git_repository(root, expected_commit=manifest.git_commit)
    actual = [fingerprint_committed_file(root, commit, item.path) for item in manifest.files]
    if actual != manifest.files or aggregate_fingerprints(actual) != manifest.aggregate_sha256:
        raise ValueError("candidate bytes do not match the frozen manifest")
    validate_git_repository(root, expected_commit=manifest.git_commit)


def _verify_tool_sources(root: Path, attestation: ToolAttestation) -> None:
    if not attestation.files:
        raise ValueError("tool attestation is empty")
    _git(root, "cat-file", "-e", f"{attestation.git_commit}^{{commit}}")
    actual = [fingerprint_file(root, item.path) for item in attestation.files]
    if actual != attestation.files or aggregate_tool_fingerprints(actual) != (
        attestation.aggregate_sha256
    ):
        raise ValueError("tool source bytes do not match their attestation")
    for item in attestation.files:
        committed = _git(root, "show", f"{attestation.git_commit}:{item.path}")
        if len(committed) != item.size or _sha256_bytes(committed) != item.sha256:
            raise ValueError("tool attestation does not match its declared commit")


def _load_run_context(
    candidate_root: Path,
    freeze_manifest: Path,
    candidate_anchor: _CandidateAnchor,
) -> _RunContext:
    if freeze_manifest.name != "freeze-manifest.json":
        raise ValueError("freeze manifest must belong to a completed D36 publication")
    publication = freeze_manifest.parent
    parse_canonical_model(
        _read_regular(freeze_manifest), FreezeManifest, maximum=DEFAULT_JSON_BYTES,
    )
    parse_canonical_model(
        _read_regular(publication / "d36-tool-attestation.json"), ToolAttestation,
        maximum=DEFAULT_JSON_BYTES,
    )
    manifest, d36_attestation = read_frozen_candidate(publication)
    if freeze_manifest.resolve(strict=True) != (
        publication.resolve(strict=True) / "freeze-manifest.json"
    ):
        raise ValueError("freeze manifest path is invalid")
    raw = _read_regular(freeze_manifest)
    expected_raw = _canonical_file_bytes(manifest)
    if raw != expected_raw:
        raise ValueError("freeze manifest bytes are not canonical")
    if d36_attestation.tool_name != "d36_candidate_freezer_and_trial_host":
        raise ValueError("D36 trial tool attestation has the wrong identity")
    if tuple(item.path for item in d36_attestation.files) != _D36_SOURCE_PATHS:
        raise ValueError("D36 trial tool attestation has the wrong source allowlist")
    tool_root = _tool_repo_root().resolve(strict=True)
    _verify_tool_sources(tool_root, d36_attestation)
    tool_commit = validate_git_repository(tool_root)
    d37_attestation = attest_tool(
        repo_root=tool_root,
        tool_name="d37_blinded_evaluator",
        git_commit=tool_commit,
        source_paths=_d37_source_paths(tool_root),
    )
    _assert_candidate_anchor(candidate_anchor)
    _validate_candidate(candidate_root, manifest)
    _assert_candidate_anchor(candidate_anchor)
    return _RunContext(
        manifest=manifest,
        d36_attestation=d36_attestation,
        d37_attestation=d37_attestation,
        freeze_sha256=_sha256_bytes(raw),
        tool_root=tool_root,
        candidate_anchor=candidate_anchor,
    )


def _validate_run_context(
    candidate_root: Path, freeze_manifest: Path, expected: _RunContext
) -> None:
    try:
        _assert_candidate_anchor(expected.candidate_anchor)
        current = _load_run_context(
            candidate_root, freeze_manifest, expected.candidate_anchor
        )
        _assert_candidate_anchor(expected.candidate_anchor)
    except (OSError, ValueError) as error:
        raise ValueError(
            "candidate, freeze, or tool identity changed during evaluation"
        ) from error
    if (
        current.manifest != expected.manifest
        or current.d36_attestation != expected.d36_attestation
        or current.d37_attestation != expected.d37_attestation
        or current.freeze_sha256 != expected.freeze_sha256
    ):
        raise ValueError("candidate, freeze, or tool identity changed during evaluation")


def _fingerprint_directory(root: Path) -> str:
    directory = _require_directory(root, "stateful index")
    files: list[FileFingerprint] = []
    total = 0

    def walk_error(error: OSError) -> None:
        raise ValueError("stateful index cannot be read") from None

    for current, directories, names in os.walk(
        directory, topdown=True, followlinks=False, onerror=walk_error,
    ):
        current_path = Path(current)
        _require_directory(current_path, "stateful index directory")
        for name in sorted(directories):
            _require_directory(current_path / name, "stateful index directory")
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            relative = path.relative_to(directory).as_posix()
            PurePosixPath(relative)
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISREG(
                metadata.st_mode
            ):
                raise ValueError("stateful index contains a non-regular entry")
            if len(files) >= 4096:
                raise ValueError("stateful index exceeds its file count limit")
            maximum = min(512 * 1024 * 1024 - total, 64_000_000)
            if name == "manifest.json":
                maximum = min(maximum, 64_000)
            size, digest = _fingerprint_regular(path, maximum=maximum)
            total += size
            files.append(FileFingerprint(path=relative, sha256=digest, size=size))
    if not files:
        raise ValueError("stateful index must contain at least one regular file")
    files.sort(key=lambda item: item.path)
    return aggregate_tool_fingerprints(files)


def _model_configuration_sha256(model: str) -> str:
    if not model or model != model.strip() or len(model) > 256:
        raise ValueError("model identifier is invalid")
    return _sha256_bytes(canonical_json_bytes({"model": model}))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return _path_is_within(left, right) or _path_is_within(right, left)


def _validate_output_location(output_root: Path, protected_roots: tuple[Path, ...]) -> Path:
    output = output_root.absolute().resolve(strict=False)
    for protected in protected_roots:
        if _paths_overlap(output, protected):
            raise ValueError("evaluation output overlaps a protected root")
    return output


def _validate_run_artifacts(
    root: Path, protocol_raw: bytes, tool_attestation_raw: bytes
) -> None:
    if _read_regular(
        root / _PROTOCOL_NAME, maximum=MAX_PROTOCOL_BYTES
    ) != protocol_raw:
        raise ValueError("immutable protocol changed during evaluation")
    if _read_regular(
        root / "tool-attestation.json", maximum=len(tool_attestation_raw)
    ) != tool_attestation_raw:
        raise ValueError("immutable D37 tool attestation changed during evaluation")


def _partial_payload(
    case_tokens: tuple[str, ...], records: list[_TrialRecord], active: str | None
) -> dict[str, object]:
    by_case: dict[str, list[_TrialRecord]] = defaultdict(list)
    for record in records:
        by_case[record.case_token].append(record)
    completed = sorted(
        token
        for token, items in by_case.items()
        if len(items) == 2 and all(item.outcome == "completed" for item in items)
    )
    failed = sorted(
        token
        for token, items in by_case.items()
        if len(items) == 2 and any(item.outcome != "completed" for item in items)
    )
    started = sorted(
        token
        for token, items in by_case.items()
        if token not in completed and token not in failed and items
    )
    if active is not None and active not in completed and active not in failed:
        started = sorted(set(started) | {active})
    remaining = sorted(set(case_tokens) - set(completed) - set(failed) - set(started))
    return {
        "schema_version": 1,
        "started_case_tokens": started,
        "completed_case_tokens": completed,
        "failed_case_tokens": failed,
        "remaining_case_tokens": remaining,
    }


def _write_partial(
    root: Path,
    case_tokens: tuple[str, ...],
    records: list[_TrialRecord],
    active: str | None = None,
) -> None:
    _write_atomic(
        root / _PARTIAL_NAME,
        _canonical_file_bytes(_partial_payload(case_tokens, records, active)),
    )


def _record_path(root: Path, group_ordinal: int, case_token: str, mode: str) -> Path:
    return (
        root
        / "groups"
        / f"{group_ordinal:06d}"
        / "cases"
        / case_token
        / mode
        / "trial-result.json"
    )


def _load_trial_record(path: Path, protocol_sha256: str) -> _TrialRecord:
    raw = _read_regular(path)
    record = parse_canonical_model(raw, _TrialRecord, maximum=DEFAULT_JSON_BYTES)
    if raw != _canonical_file_bytes(record) or record.protocol_sha256 != protocol_sha256:
        raise ValueError("completed trial record is invalid")
    return record


def _safe_directory_entries(path: Path, description: str) -> list[os.DirEntry[str]]:
    _validate_directory(path.absolute(), description)
    entries = sorted(os.scandir(path), key=lambda entry: entry.name)
    for entry in entries:
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            raise ValueError(f"{description} contains an unsafe entry")
    return entries


def _is_valid_interrupted_temp(entry: os.DirEntry[str]) -> bool:
    match = _TEMP_PATTERN.fullmatch(entry.name)
    if match is None:
        return False
    metadata = entry.stat(follow_symlinks=False)
    return stat.S_ISREG(metadata.st_mode) and not _is_reparse(metadata)


def _load_records(root: Path, protocol_sha256: str) -> list[_TrialRecord]:
    root = _validate_directory(root.absolute(), "evaluation output")
    groups = root / "groups"
    try:
        group_entries = _safe_directory_entries(groups, "trial groups")
    except FileNotFoundError:
        return []
    paths: list[Path] = []
    for group in group_entries:
        metadata = group.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or not re.fullmatch(r"[0-9]{6}", group.name):
            raise ValueError("trial groups contain an invalid entry")
        cases = Path(group.path) / "cases"
        for case in _safe_directory_entries(cases, "trial cases"):
            metadata = case.stat(follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode) or _TOKEN_PATTERN.fullmatch(case.name) is None:
                raise ValueError("trial cases contain an invalid entry")
            for mode in _safe_directory_entries(Path(case.path), "trial modes"):
                metadata = mode.stat(follow_symlinks=False)
                if not stat.S_ISDIR(metadata.st_mode) or mode.name not in _MODES:
                    raise ValueError("trial modes contain an invalid entry")
                for child in _safe_directory_entries(Path(mode.path), "trial mode"):
                    child_metadata = child.stat(follow_symlinks=False)
                    if child.name == "trial-result.json":
                        if not stat.S_ISREG(child_metadata.st_mode):
                            raise ValueError("completed trial record is not regular")
                        paths.append(Path(child.path))
                    elif child.name in {"input.json", "attempts"}:
                        expected_directory = child.name == "attempts"
                        if expected_directory != stat.S_ISDIR(child_metadata.st_mode):
                            raise ValueError("trial mode entry has the wrong type")
                    elif not _is_valid_interrupted_temp(child):
                        raise ValueError("trial mode contains an invalid entry")
    records = [_load_trial_record(path, protocol_sha256) for path in paths]
    identities = {(record.case_token, record.mode) for record in records}
    if len(identities) != len(records):
        raise ValueError("duplicate completed trial records")
    return records


def _next_attempt(mode_root: Path) -> Path:
    root = _validate_directory(mode_root.absolute(), "trial mode")
    attempts = _ensure_writable_directory(root, "attempts")
    existing = []
    for entry in _safe_directory_entries(attempts, "trial attempts"):
        metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or not re.fullmatch(r"[0-9]{6}", entry.name):
            raise ValueError("trial attempts contain an invalid entry")
        existing.append(int(entry.name))
    name = f"{max(existing, default=0) + 1:06d}"
    return _ensure_writable_directory(attempts, name)


def _observation(path: Path) -> tuple[TrialObservation, bytes]:
    raw = _read_regular(path)
    observation = parse_canonical_model(raw, TrialObservation, maximum=DEFAULT_JSON_BYTES)
    if raw != _canonical_file_bytes(observation):
        raise ValueError("trial observation is not canonical")
    return observation, raw


async def _run_trial(
    *,
    root: Path,
    group_ordinal: int,
    case: Case,
    case_token: str,
    category_token: str,
    mode: str,
    protocol_sha256: str,
    context: _RunContext,
    candidate_root: Path,
    freeze_manifest: Path,
    model: str,
    index: Path,
    index_sha256: str,
    embedding_profile: Path | None = None,
    embedding_base_url: str | None = None,
) -> _TrialRecord:
    mode_root = _ensure_writable_directory(
        root,
        "groups",
        f"{group_ordinal:06d}",
        "cases",
        case_token,
        mode,
    )
    record_path = mode_root / "trial-result.json"
    try:
        record_path.lstat()
        record_exists = True
    except FileNotFoundError:
        record_exists = False
    if record_exists:
        record = _load_trial_record(record_path, protocol_sha256)
        if (
            record.case_token != case_token
            or record.category_token != category_token
            or record.mode != mode
        ):
            raise ValueError("completed trial record does not match its path")
        return record

    projected = case_to_unlabeled(case)
    input_raw = _unlabeled_bytes(projected) + b"\n"
    input_path = mode_root / "input.json"
    try:
        _write_exclusive(input_path, input_raw)
    except FileExistsError:
        if _read_regular(input_path, maximum=len(input_raw)) != input_raw:
            raise ValueError("existing unlabeled trial input changed") from None

    attempt = _next_attempt(mode_root)
    output_path = attempt / "observation.json"
    _validate_run_context(candidate_root, freeze_manifest, context)
    if _fingerprint_directory(index) != index_sha256:
        raise ValueError("stateful index changed before trial execution")
    outcome: Literal["completed", "transport_failure", "deadline_failure"]
    try:
        outcome = await _invoke_trial_host(
            tool_root=context.tool_root,
            candidate_root=context.candidate_anchor.execution_path,
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            storage=attempt / "storage",
            model=model,
            index=index,
            stdout_path=attempt / "stdout.log",
            stderr_path=attempt / "stderr.log",
            candidate_identity=context.candidate_anchor.identity,
            embedding_profile=embedding_profile if mode == "stateful" else None,
            embedding_base_url=embedding_base_url if mode == "stateful" else None,
        )
    finally:
        _validate_run_context(candidate_root, freeze_manifest, context)
        if _fingerprint_directory(index) != index_sha256:
            raise ValueError("stateful index changed during trial execution")

    storage_path = attempt / "storage"
    try:
        storage_path.lstat()
    except FileNotFoundError:
        pass
    else:
        _validate_directory(storage_path, "trial storage")
    score: TrialScore | None = None
    candidate_snapshot: str | None = None
    if outcome == "completed":
        try:
            observation, _ = _observation(output_path)
            if observation.mode != mode or observation.case_sha256 != projected.case_sha256:
                raise ValueError("trial observation does not match its invocation")
            score = score_trial(normalize_case(case), observation.model_dump(mode="json"))
            candidate_snapshot = observation.candidate_snapshot_sha256
        except (OSError, ValidationError, ValueError):
            outcome = "transport_failure"
    record = _TrialRecord(
        schema_version=1,
        protocol_sha256=protocol_sha256,
        case_token=case_token,
        category_token=category_token,
        mode=mode,
        outcome=outcome,
        score=score,
        candidate_snapshot_sha256=candidate_snapshot,
    )
    _publish_immutable(record_path, _canonical_file_bytes(record), "trial result")
    return record


def _mode_result(
    mode: str,
    records: list[_TrialRecord],
    category_tokens: tuple[str, ...],
    denominators: dict[str, int],
) -> ModeResult:
    selected = [record for record in records if record.mode == mode]

    def counts(items: list[_TrialRecord]) -> dict[str, int]:
        completed = [item for item in items if item.outcome == "completed"]
        return {
            "included": len(items),
            "completed": len(completed),
            "task_complete": sum(item.score is not None and item.score.task_complete for item in completed),
            "unauthorized_effects": sum(
                item.score is not None and item.score.unauthorized_effect for item in completed
            ),
            "unauthorized_replays": sum(
                item.score is not None and item.score.unauthorized_replay for item in completed
            ),
            "secret_disclosures": sum(
                item.score is not None and item.score.secret_disclosure for item in completed
            ),
        }

    categories = []
    for token in category_tokens:
        values = counts([item for item in selected if item.category_token == token])
        if values["included"] != denominators[token]:
            raise ValueError("trial records do not cover the exact category topology")
        categories.append(CategoryResult(category_token=token, **values))
    values = counts(selected)
    return ModeResult(
        mode=mode,
        **values,
        transport_failures=sum(item.outcome == "transport_failure" for item in selected),
        deadline_failures=sum(item.outcome == "deadline_failure" for item in selected),
        categories=tuple(categories),
    )


def _validate_existing_bundle(
    root: Path,
    protocol: EvaluationProtocol,
    protocol_sha256: str,
    context: _RunContext,
    included: tuple[str, ...],
    excluded: tuple[ExcludedCaseToken, ...],
    evaluator_name: str,
) -> EvaluationResultBundle | None:
    path = root / _PUBLIC_RESULT_NAME
    if not path.exists():
        return None
    raw = _read_regular(path, maximum=MAX_RESULT_BUNDLE_BYTES)
    bundle = parse_canonical_model(raw, EvaluationResultBundle, maximum=MAX_RESULT_BUNDLE_BYTES)
    if raw != _canonical_file_bytes(bundle):
        raise ValueError("existing result bundle is not canonical")
    if (
        bundle.protocol_sha256 != protocol_sha256
        or bundle.candidate_id != protocol.candidate_id
        or bundle.freeze_sha256 != protocol.freeze_sha256
        or bundle.corpus_sha256 != protocol.corpus_sha256
        or bundle.human_approval_sha256 != protocol.human_approval_sha256
        or bundle.independent_approval_sha256 != protocol.independent_approval_sha256
        or bundle.d36_trial_tool_sha256 != context.d36_attestation.aggregate_sha256
        or bundle.d37_evaluator_tool_sha256 != context.d37_attestation.aggregate_sha256
        or bundle.protocol_case_count != protocol.case_count
        or bundle.protocol_case_tokens != protocol.case_tokens
        or bundle.protocol_category_count != protocol.category_count
        or bundle.protocol_category_tokens != protocol.category_tokens
        or bundle.case_categories != protocol.case_categories
        or bundle.included_case_tokens != included
        or bundle.excluded_cases != excluded
        or bundle.evaluator_name != evaluator_name
    ):
        raise ValueError("existing result bundle does not match this run")
    _, sealed = seal_evidence(root)
    if sealed != bundle.sealed_evidence_sha256:
        raise ValueError("existing result bundle detailed evidence changed")
    return bundle


async def _run_blinded_evaluation_anchored(
    *,
    candidate_root: Path,
    freeze_manifest: Path,
    corpus: Path,
    human_review: Path,
    independent_review: Path,
    output_root: Path,
    model: str,
    index: Path,
    evaluator_name: str,
    token_key_file: Path,
    context: _RunContext,
    embedding_profile: Path | None = None,
    embedding_base_url: str | None = None,
) -> EvaluationResultBundle:
    embedding = _embedding_configuration(
        mode="stateful", embedding_profile=embedding_profile, embedding_base_url=embedding_base_url,
    )
    if embedding is not None:
        embedding_profile, embedding_base_url, _, profile_raw = embedding
        _check_embedding_index(profile_raw, index)
    protected_roots = (
        candidate_root,
        context.tool_root,
        freeze_manifest.parent.resolve(strict=True),
        corpus.parent.resolve(strict=True),
        human_review.parent.resolve(strict=True),
        independent_review.parent.resolve(strict=True),
        index.resolve(strict=True),
        token_key_file.parent.resolve(strict=True),
    ) + ((embedding_profile.parent,) if embedding_profile is not None else ())
    expected_output = _validate_output_location(output_root, protected_roots)
    output = _require_directory(output_root, "evaluation output", create=True)
    if output != expected_output:
        raise ValueError("evaluation output changed while creating its root")

    corpus_raw = _read_regular(corpus)
    human_raw = _read_regular(human_review)
    independent_raw = _read_regular(independent_review)
    cases = load_cases(corpus)
    if _read_regular(corpus) != corpus_raw:
        raise ValueError("evaluation corpus changed while loading")
    if any(case.split != "held_out" for case in cases):
        raise ValueError("evaluation corpus must contain held-out cases only")
    human = load_review(human_review, cases, "human")
    if _read_regular(human_review) != human_raw:
        raise ValueError("human review changed while loading")
    independent = load_review(independent_review, cases, "independent_ai")
    if _read_regular(independent_review) != independent_raw:
        raise ValueError("independent review changed while loading")
    index_sha256 = _fingerprint_directory(index)
    model_sha256 = _model_configuration_sha256(model)

    with token_key(token_key_file) as key:
        bindings = case_category_bindings(cases, key)
        included, excluded = approval_partition(cases, human, independent, key)
        token_by_id = {
            case.case_id: opaque_case_token(key, case.case_id) for case in cases
        }
    # Reject unprojectable included cases before any protocol or trial is written.
    included_tokens = set(included)
    for case in cases:
        if token_by_id[case.case_id] in included_tokens:
            case_to_unlabeled(case)
    case_tokens = tuple(binding.case_token for binding in bindings)
    category_tokens = tuple(sorted({binding.category_token for binding in bindings}))
    protocol = EvaluationProtocol(
        schema_version=1,
        candidate_id=context.manifest.candidate_id,
        modes=_MODES,
        per_call_deadline_seconds=180,
        maximum_model_calls=4,
        isolation="fresh_case_state_under_source_group",
        corpus_sha256=_sha256_bytes(corpus_raw),
        human_approval_sha256=_sha256_bytes(human_raw),
        independent_approval_sha256=_sha256_bytes(independent_raw),
        freeze_sha256=context.freeze_sha256,
        d36_trial_tool_sha256=context.d36_attestation.aggregate_sha256,
        d37_evaluator_tool_sha256=context.d37_attestation.aggregate_sha256,
        model_configuration_sha256=model_sha256,
        stateful_index_sha256=index_sha256,
        category_count=len(category_tokens),
        category_tokens=category_tokens,
        case_count=len(case_tokens),
        case_tokens=case_tokens,
        case_categories=bindings,
    )
    d37_attestation_raw = _canonical_file_bytes(context.d37_attestation)
    parse_canonical_model(d37_attestation_raw, ToolAttestation, maximum=DEFAULT_JSON_BYTES)
    _write_or_validate_immutable(
        output / "tool-attestation.json",
        d37_attestation_raw,
        "D37 tool attestation",
    )
    protocol_path = write_run_protocol_exclusive(output, protocol)
    protocol_raw = _read_regular(protocol_path, maximum=MAX_PROTOCOL_BYTES)
    parse_canonical_model(protocol_raw, EvaluationProtocol, maximum=MAX_PROTOCOL_BYTES)
    protocol_sha256 = _sha256_bytes(protocol_raw)
    _validate_run_artifacts(output, protocol_raw, d37_attestation_raw)
    existing_bundle = _validate_existing_bundle(
        output,
        protocol,
        protocol_sha256,
        context,
        included,
        excluded,
        evaluator_name,
    )
    if existing_bundle is not None:
        _validate_run_context(candidate_root, freeze_manifest, context)
        return existing_bundle

    included_set = set(included)
    binding_by_token = {item.case_token: item.category_token for item in bindings}
    tally = Counter(
        binding.category_token for binding in bindings if binding.case_token in included_set
    )
    denominators = {category: tally[category] for category in category_tokens}
    if not included or any(value < 1 for value in denominators.values()):
        raise ValueError("approved evaluation coverage must be non-empty in every category")

    case_by_token = {token_by_id[case.case_id]: case for case in cases}
    if set(case_by_token) != set(case_tokens):
        raise ValueError("case/token topology could not be reconstructed")

    group_ordinals = {
        group: index + 1
        for index, group in enumerate(sorted({case.group_id for case in cases}))
    }
    ordered_cases = sorted(
        (case_by_token[token] for token in included),
        key=lambda case: (case.group_id, case.case_id),
    )
    records = _load_records(output, protocol_sha256)
    expected_trials = {(token, mode) for token in included for mode in _MODES}
    if {(record.case_token, record.mode) for record in records} - expected_trials:
        raise ValueError("partial records do not belong to the approved protocol")
    _write_partial(output, included, records)

    for case_index, case in enumerate(ordered_cases):
        case_token = token_by_id[case.case_id]
        category_token = binding_by_token[case_token]
        mode_order = _MODES if case_index % 2 == 0 else tuple(reversed(_MODES))
        for mode in mode_order:
            active = case_token
            _validate_run_artifacts(output, protocol_raw, d37_attestation_raw)
            _write_partial(output, included, records, active)
            try:
                record = await _run_trial(
                    root=output,
                    group_ordinal=group_ordinals[case.group_id],
                    case=case,
                    case_token=case_token,
                    category_token=category_token,
                    mode=mode,
                    protocol_sha256=protocol_sha256,
                    context=context,
                    candidate_root=candidate_root,
                    freeze_manifest=freeze_manifest,
                    model=model,
                    index=index,
                    index_sha256=index_sha256,
                    embedding_profile=embedding_profile,
                    embedding_base_url=embedding_base_url,
                )
                records = [
                    item
                    for item in records
                    if (item.case_token, item.mode) != (case_token, mode)
                ] + [record]
                records.sort(key=lambda item: (item.case_token, item.mode))
                _validate_run_artifacts(output, protocol_raw, d37_attestation_raw)
                _write_partial(output, included, records)
            except BaseException:
                _validate_run_artifacts(output, protocol_raw, d37_attestation_raw)
                _write_partial(output, included, records, active)
                raise

    if {(record.case_token, record.mode) for record in records} != expected_trials:
        raise ValueError("trial records do not exactly cover the approved protocol")
    snapshots = {
        record.candidate_snapshot_sha256
        for record in records
        if record.candidate_snapshot_sha256 is not None
    }
    if len(snapshots) > 1:
        raise ValueError("candidate trial snapshots differ across runs")
    _validate_run_context(candidate_root, freeze_manifest, context)
    _validate_run_artifacts(output, protocol_raw, d37_attestation_raw)
    if _embedding_configuration(
        mode="stateful", embedding_profile=embedding_profile, embedding_base_url=embedding_base_url,
    ) != embedding:
        raise ValueError("embedding configuration changed during evaluation")
    _, sealed_sha256 = seal_evidence(output)
    modes = tuple(
        _mode_result(mode, records, category_tokens, denominators) for mode in _MODES
    )
    bundle = EvaluationResultBundle(
        schema_version=1,
        candidate_id=protocol.candidate_id,
        freeze_sha256=protocol.freeze_sha256,
        corpus_sha256=protocol.corpus_sha256,
        human_approval_sha256=protocol.human_approval_sha256,
        independent_approval_sha256=protocol.independent_approval_sha256,
        protocol_sha256=protocol_sha256,
        d36_trial_tool_sha256=protocol.d36_trial_tool_sha256,
        d37_evaluator_tool_sha256=protocol.d37_evaluator_tool_sha256,
        protocol_case_count=protocol.case_count,
        protocol_case_tokens=protocol.case_tokens,
        protocol_category_count=protocol.category_count,
        protocol_category_tokens=protocol.category_tokens,
        case_categories=protocol.case_categories,
        included_count=len(included),
        excluded_count=len(excluded),
        included_case_tokens=included,
        excluded_cases=excluded,
        evaluator_role="independent_evaluator",
        evaluator_name=evaluator_name,
        executed_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        sealed_evidence_sha256=sealed_sha256,
        modes=modes,
    )
    result_path = output / _PUBLIC_RESULT_NAME
    _publish_immutable(
        result_path,
        _canonical_file_bytes(bundle),
        "result bundle",
        maximum=MAX_RESULT_BUNDLE_BYTES,
    )
    published_raw = _read_regular(result_path, maximum=MAX_RESULT_BUNDLE_BYTES)
    if parse_canonical_model(
        published_raw, EvaluationResultBundle, maximum=MAX_RESULT_BUNDLE_BYTES,
    ) != bundle:
        raise ValueError("result bundle changed after publication")
    _validate_run_context(candidate_root, freeze_manifest, context)
    return bundle


async def run_blinded_evaluation(
    *,
    candidate_root: Path,
    freeze_manifest: Path,
    corpus: Path,
    human_review: Path,
    independent_review: Path,
    output_root: Path,
    model: str,
    index: Path,
    evaluator_name: str,
    token_key_file: Path,
    embedding_profile: Path | None = None,
    embedding_base_url: str | None = None,
) -> EvaluationResultBundle:
    """Run or resume one externally isolated synthetic-or-held-out evaluation."""
    canonical_candidate = _canonical_candidate_root(candidate_root)
    anchor = _open_candidate_anchor(canonical_candidate)
    try:
        context = _load_run_context(canonical_candidate, freeze_manifest, anchor)
        return await _run_blinded_evaluation_anchored(
            candidate_root=canonical_candidate,
            freeze_manifest=freeze_manifest,
            corpus=corpus,
            human_review=human_review,
            independent_review=independent_review,
            output_root=output_root,
            model=model,
            index=index,
            evaluator_name=evaluator_name,
            token_key_file=token_key_file,
            embedding_profile=embedding_profile,
            embedding_base_url=embedding_base_url,
            context=context,
        )
    finally:
        _close_candidate_anchor(anchor)


__all__ = [
    "run_blinded_evaluation",
    "write_run_protocol_exclusive",
]
