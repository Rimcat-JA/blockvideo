"""Run one label-free trial against a detached D35 candidate subprocess."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_WORKER_OUTPUT_BYTES = 512 * 1024
FILE_HASH_CHUNK_BYTES = 1024 * 1024
PROTOCOL_DEADLINE_SECONDS = 180
MODEL_CALL_LIMIT = 4
MAX_OBSERVED_PROJECTS = 1 + 16
MAX_OBSERVED_HISTORY = 32 + MODEL_CALL_LIMIT
MAX_OBSERVED_JOBS = 32 + MODEL_CALL_LIMIT
MAX_OBSERVED_ARTIFACTS = 32 + 32 + MODEL_CALL_LIMIT
MAX_OBSERVED_RECEIPTS = 32 + MODEL_CALL_LIMIT
MAX_OBSERVED_EXTERNAL_CALLS = 32 + MODEL_CALL_LIMIT
MAX_OBSERVED_LANGUAGE_REQUESTS = 8 + MODEL_CALL_LIMIT
MAX_OBSERVED_LANGUAGE_TURNS = 8 + MODEL_CALL_LIMIT
SUBPROCESS_TIMEOUT_SECONDS = PROTOCOL_DEADLINE_SECONDS * MODEL_CALL_LIMIT + 30
_REPARSE_POINT = 0x400
_POSIX_FD_PATH = re.compile(r"^/proc/(?P<pid>[1-9][0-9]*)/fd/(?P<fd>0|[1-9][0-9]*)$")

ResponseStatus = Literal["interpreting", "ready", "needs_input", "unsupported", "blocked", "error", "completed", "dismissed", "http_error"]
ResponseMode = Literal["all_tools", "semantic", "stateful"]
OperationId = Literal[
    "project.subtitle-font-size.set", "project.subtitle-font-size.adjust", "project.settings.update",
    "project.status.get", "project.generation.start", "project.generation.cancel",
    "project.generation.retry", "project.settings.restore",
]
ReasonCode = Literal[
    "invalid_request", "stale_state", "project_busy", "request_conflict", "dialogue_superseded",
    "dialogue_unavailable", "dialogue_limit", "parent_not_found", "dialogue_target_mismatch",
    "dialogue_not_pending", "dialogue_stale", "target_required", "target_conflict", "target_not_found",
    "request_not_ready", "confirmation_mismatch", "generation_confirmation_required",
    "core_request_conflict", "request_not_found", "request_id_conflict", "invalid_arguments",
    "operation_not_found", "invalid_generation_request", "job_not_found", "job_not_retryable",
    "settings_revision_not_found", "artifact_not_found", "artifact_unavailable", "database_busy", "external_outcome_unknown",
    "interpretation_failed", "interpretation_interrupted", "model_not_configured", "not_ready",
    "invalid_input", "invalid_json", "invalid_output", "candidate_not_offered", "invalid_response",
    "refused", "incomplete_response", "response_too_large", "http_error", "timeout",
    "configuration_error", "connection_failed", "model_mismatch", "retrieval_deadline",
    "retrieval_integrity_failed", "retrieval_unavailable", "deadline_exceeded", "internal_error",
]
FailureClass = Literal[
    "http_4xx", "http_5xx", "candidate_error", "model_call_limit", "timeout", "output_limit",
    "invalid_observation",
]
ProjectStatusValue = Literal[
    "pending", "splitting", "planning", "generating", "rendering", "completed", "failed", "cancelled",
]
OpaqueIdentityHash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


async def _quiescent_candidate_dispatcher() -> None:
    await asyncio.Event().wait()


def _stable_path_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size


def _canonical_stored_relative_path(relative: str) -> tuple[str, ...]:
    if not relative or "\0" in relative or "\\" in relative or ":" in relative:
        raise ValueError("stored media path must be canonical relative POSIX")
    if relative.startswith("/") or re.match(r"^[A-Za-z]:", relative):
        raise ValueError("stored media path must be relative")
    components = tuple(relative.split("/"))
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("stored media path contains an unsafe component")
    if PurePosixPath(relative).as_posix() != relative:
        raise ValueError("stored media path is not canonical POSIX")
    return components


def _checked_path_component(path: Path, *, directory: bool) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise ValueError("stored media path contains a link or reparse point")
    if directory and not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("stored media path parent is not a directory")
    return metadata


def _file_identity(storage_root: Path, relative: str | None) -> dict[str, Any] | None:
    if relative is None:
        return None
    components = _canonical_stored_relative_path(relative)
    path_sha256 = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    root = storage_root.absolute()
    root_metadata = _checked_path_component(root, directory=True)
    if root.resolve(strict=True) != root:
        raise ValueError("storage root is not canonical")

    checked: list[tuple[Path, tuple[int, int, int, int], bool]] = [
        (root, _stable_path_identity(root_metadata), True)
    ]
    path = root
    missing = False
    for index, component in enumerate(components):
        path /= component
        is_directory = index < len(components) - 1
        try:
            metadata = _checked_path_component(path, directory=is_directory)
        except FileNotFoundError:
            missing = True
            break
        checked.append((path, _stable_path_identity(metadata), is_directory))

    if missing:
        for checked_path, expected, is_directory in checked:
            current = _checked_path_component(checked_path, directory=is_directory)
            if _stable_path_identity(current) != expected:
                raise ValueError("stored media path identity changed")
        return {
            "exists": False,
            "path_sha256": path_sha256,
            "size": None,
            "sha256": None,
        }

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or _is_reparse(before):
        raise ValueError("stored media path final is a link or reparse point")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("stored media path final is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _is_reparse(opened)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError("stored media file identity changed while opening")
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, FILE_HASH_CHUNK_BYTES):
            size += len(chunk)
            if size > 2**63 - 1:
                raise ValueError("stored media file is too large")
            digest.update(chunk)
        final = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _checked_path_component(path, directory=False)
    identities = {
        _stable_path_identity(before),
        _stable_path_identity(opened),
        _stable_path_identity(final),
        _stable_path_identity(after),
    }
    if len(identities) != 1 or size != final.st_size:
        raise ValueError("stored media file identity changed while hashing")
    for checked_path, expected, is_directory in checked[:-1]:
        current = _checked_path_component(checked_path, directory=is_directory)
        if _stable_path_identity(current) != expected:
            raise ValueError("stored media path component identity changed")
    return {
        "exists": True,
        "path_sha256": path_sha256,
        "size": size,
        "sha256": digest.hexdigest(),
    }


class _ModelCallBudget:
    def __init__(self, remaining: int) -> None:
        if type(remaining) is not int or not 0 <= remaining <= MODEL_CALL_LIMIT:
            raise ValueError("model-call budget must be between 0 and 4")
        self._remaining = remaining
        self.calls = 0

    async def complete(self, invoke: Callable[[], Awaitable[str]]) -> str:
        if self.calls >= self._remaining:
            raise ValueError("budget_exhausted")
        self.calls += 1
        return await invoke()


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RedactedResponse(_StrictRecord):
    http_status: int = Field(ge=100, le=599)
    status: ResponseStatus
    mode: ResponseMode
    executed: bool
    requires_confirmation: bool
    operation_id: OperationId | None = None
    operation_version: int | None = Field(default=None, ge=1, le=2)
    arguments_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generate_after_save: bool | None = None
    generation_requested: bool | None = None
    clarification_missing_fields: tuple[Literal["target", "arguments", "intent"], ...] | None = Field(
        default=None, min_length=1, max_length=3, strict=False
    )
    reason_code: ReasonCode | None = None
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def details_match_status(self) -> RedactedResponse:
        operation_values = (
            self.operation_id,
            self.operation_version,
            self.arguments_sha256,
            self.generate_after_save,
            self.generation_requested,
        )
        has_operation = self.operation_id is not None
        if has_operation != all(value is not None for value in operation_values):
            raise ValueError("operation projection fields must be all present or all null")
        if self.status == "needs_input":
            if has_operation or self.clarification_missing_fields is None:
                raise ValueError("clarification status requires only missing fields")
        elif self.clarification_missing_fields is not None:
            raise ValueError("clarification missing fields require needs_input status")
        if self.status in {"ready", "completed"} and not has_operation:
            raise ValueError("operation response status requires operation fields")
        if self.status in {
            "interpreting", "unsupported", "error", "dismissed", "http_error"
        } and has_operation:
            raise ValueError("non-operation response status requires null operation fields")
        if self.clarification_missing_fields is not None:
            if tuple(sorted(set(self.clarification_missing_fields))) != self.clarification_missing_fields:
                raise ValueError("clarification missing fields must be unique and sorted")
        return self


class RedactedFileIdentity(_StrictRecord):
    exists: bool
    path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int | None = Field(default=None, ge=0, le=2**63 - 1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def content_matches_existence(self) -> RedactedFileIdentity:
        has_size = self.size is not None
        has_content_hash = self.sha256 is not None
        if has_size != has_content_hash or self.exists != has_size:
            raise ValueError("file content identity must match existence")
        return self


class RedactedProjectEntry(_StrictRecord):
    id: int = Field(ge=1, le=2**63 - 1)
    revision: int = Field(ge=1, le=10**12)
    status: ProjectStatusValue
    settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    title_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_script_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    global_visual_style_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    progress: float = Field(ge=0, le=1)
    current_stage: str | None = Field(default=None, max_length=64)
    current_artifact_id: int | None = Field(default=None, ge=1, le=2**63 - 1)
    output_video: RedactedFileIdentity | None = None
    output_subtitle: RedactedFileIdentity | None = None
    error_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RedactedHistoryEntry(_StrictRecord):
    project_id: int = Field(ge=1, le=2**63 - 1)
    revision: int = Field(ge=1, le=10**12)
    settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changed_fields: tuple[Literal[
        "subtitle_font_size", "voicevox_speed_scale", "voicevox_pitch_scale",
        "voicevox_speaker_id", "pronunciation_overrides", "narration_pacing_mode",
        "narration_sentence_pause_seconds",
    ], ...] = Field(max_length=7, strict=False)
    restored_from_revision: int | None = Field(default=None, ge=1, le=10**12)

    @model_validator(mode="after")
    def changed_fields_are_canonical(self) -> RedactedHistoryEntry:
        if tuple(sorted(set(self.changed_fields))) != self.changed_fields:
            raise ValueError("changed fields must be unique and sorted")
        return self


class RedactedJobEntry(_StrictRecord):
    id: int = Field(ge=1, le=2**63 - 1)
    project_id: int = Field(ge=1, le=2**63 - 1)
    status: Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]
    current_stage: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")
    progress: float = Field(ge=0, le=1)
    stage_progress: float = Field(ge=0, le=1)
    input_revision: int = Field(ge=1, le=10**12)
    cancel_requested: bool
    kind: Literal["full", "rerender"]
    block_index: int | None = Field(default=None, ge=0, le=100000)
    parent_job_id: int | None = Field(default=None, ge=1, le=2**63 - 1)
    input_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recovery_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    error_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RedactedArtifactEntry(_StrictRecord):
    id: int = Field(ge=1, le=2**63 - 1)
    project_id: int = Field(ge=1, le=2**63 - 1)
    job_id: int | None = Field(default=None, ge=1, le=2**63 - 1)
    revision: int | None = Field(default=None, ge=1, le=10**12)
    input_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    video_path_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    video_size: int | None = Field(default=None, ge=0, le=2**63 - 1)
    video_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    subtitle_path_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    subtitle_size: int | None = Field(default=None, ge=0, le=2**63 - 1)
    subtitle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def content_identities_are_complete(self) -> RedactedArtifactEntry:
        if (self.video_size is None) != (self.video_sha256 is None):
            raise ValueError("video size and content hash must both be present or absent")
        if self.subtitle_path_sha256 is None:
            if self.subtitle_size is not None or self.subtitle_sha256 is not None:
                raise ValueError("subtitle content identity requires subtitle path hash")
        elif (self.subtitle_size is None) != (self.subtitle_sha256 is None):
            raise ValueError("subtitle size and content hash must both be present or absent")
        return self


class RedactedState(_StrictRecord):
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_status: ProjectStatusValue
    settings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projects_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    history_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    jobs_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_calls_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_requests_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language_turns_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    project_count: int = Field(ge=1, le=MAX_OBSERVED_PROJECTS)
    history_count: int = Field(ge=0, le=MAX_OBSERVED_HISTORY)
    job_count: int = Field(ge=0, le=MAX_OBSERVED_JOBS)
    artifact_count: int = Field(ge=0, le=MAX_OBSERVED_ARTIFACTS)
    receipt_count: int = Field(ge=0, le=MAX_OBSERVED_RECEIPTS)
    external_call_count: int = Field(ge=0, le=MAX_OBSERVED_EXTERNAL_CALLS)
    language_request_count: int = Field(ge=0, le=MAX_OBSERVED_LANGUAGE_REQUESTS)
    language_turn_count: int = Field(ge=0, le=MAX_OBSERVED_LANGUAGE_TURNS)
    project_entries: tuple[RedactedProjectEntry, ...] = Field(
        max_length=MAX_OBSERVED_PROJECTS, strict=False
    )
    history_entries: tuple[RedactedHistoryEntry, ...] = Field(
        max_length=MAX_OBSERVED_HISTORY, strict=False
    )
    job_entries: tuple[RedactedJobEntry, ...] = Field(
        max_length=MAX_OBSERVED_JOBS, strict=False
    )
    artifact_entries: tuple[RedactedArtifactEntry, ...] = Field(
        max_length=MAX_OBSERVED_ARTIFACTS, strict=False
    )
    receipt_identity_sha256s: tuple[OpaqueIdentityHash, ...] = Field(
        max_length=MAX_OBSERVED_RECEIPTS, strict=False
    )
    external_call_identity_sha256s: tuple[OpaqueIdentityHash, ...] = Field(
        max_length=MAX_OBSERVED_EXTERNAL_CALLS, strict=False
    )
    language_request_identity_sha256s: tuple[OpaqueIdentityHash, ...] = Field(
        max_length=MAX_OBSERVED_LANGUAGE_REQUESTS, strict=False
    )
    language_turn_identity_sha256s: tuple[OpaqueIdentityHash, ...] = Field(
        max_length=MAX_OBSERVED_LANGUAGE_TURNS, strict=False
    )

    @model_validator(mode="after")
    def projections_match_counts_and_order(self) -> RedactedState:
        if len(self.project_entries) != self.project_count:
            raise ValueError("project entries must match project count")
        if len(self.history_entries) != self.history_count:
            raise ValueError("history entries must match history count")
        if len(self.job_entries) != self.job_count:
            raise ValueError("job entries must match job count")
        if len(self.artifact_entries) != self.artifact_count:
            raise ValueError("artifact entries must match artifact count")
        opaque_identities = (
            (self.receipt_identity_sha256s, self.receipt_count),
            (self.external_call_identity_sha256s, self.external_call_count),
            (self.language_request_identity_sha256s, self.language_request_count),
            (self.language_turn_identity_sha256s, self.language_turn_count),
        )
        if any(len(identities) != count for identities, count in opaque_identities):
            raise ValueError("opaque record identities must match collection counts")
        if any(tuple(sorted(set(identities))) != identities for identities, _ in opaque_identities):
            raise ValueError("opaque record identities must be unique and sorted")
        project_ids = [item.id for item in self.project_entries]
        history_keys = [(item.project_id, item.revision) for item in self.history_entries]
        job_ids = [item.id for item in self.job_entries]
        artifact_ids = [item.id for item in self.artifact_entries]
        if project_ids != sorted(set(project_ids)):
            raise ValueError("project entries must be unique and sorted")
        if history_keys != sorted(set(history_keys)):
            raise ValueError("history entries must be unique and sorted")
        if job_ids != sorted(set(job_ids)):
            raise ValueError("job entries must be unique and sorted")
        if artifact_ids != sorted(set(artifact_ids)):
            raise ValueError("artifact entries must be unique and sorted")
        return self


class ObservedEffects(_StrictRecord):
    settings: int = Field(ge=0, le=1)
    revision: int = Field(ge=0, le=10**12)
    jobs: int = Field(ge=0, le=1)
    cancellations: int = Field(ge=0, le=1)
    receipts: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    artifacts: int = Field(ge=0, le=1)
    external_calls: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    history: int = Field(ge=0, le=1)
    language_records: int = Field(ge=0, le=1)
    language_requests: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    language_turns: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    prior_receipts_preserved: bool
    prior_external_calls_preserved: bool
    prior_language_requests_preserved: bool
    prior_language_turns_preserved: bool


class ReplayObservation(_StrictRecord):
    attempted: bool
    model_calls: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    state_unchanged: bool
    same_response: bool
    response: RedactedResponse | None = None
    failure_class: FailureClass | None = None

    @model_validator(mode="after")
    def response_matches_attempt(self) -> ReplayObservation:
        if self.attempted != (self.response is not None):
            raise ValueError("replay response must match attempted state")
        return self


class ConfirmationObservation(_StrictRecord):
    attempted: bool
    duplicate_attempted: bool
    state_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duplicate_same_response: bool | None = None
    response: RedactedResponse | None = None
    duplicate_response: RedactedResponse | None = None
    failure_class: FailureClass | None = None

    @model_validator(mode="after")
    def responses_match_attempts(self) -> ConfirmationObservation:
        if self.attempted != (self.response is not None) or self.attempted != (self.state_sha256 is not None):
            raise ValueError("confirmation evidence must match attempted state")
        if self.duplicate_attempted and not self.attempted:
            raise ValueError("duplicate confirmation requires an initial attempt")
        if self.duplicate_attempted != (self.duplicate_response is not None):
            raise ValueError("duplicate confirmation response must match attempted state")
        if self.duplicate_attempted != (self.duplicate_same_response is not None):
            raise ValueError("duplicate comparison must match attempted state")
        return self


class _WorkerObservation(_StrictRecord):
    schema_version: Literal[1]
    response: RedactedResponse
    before: RedactedState
    after: RedactedState
    effects: ObservedEffects
    model_calls: int = Field(ge=0, le=MODEL_CALL_LIMIT)
    failure_class: FailureClass | None = None
    replay: ReplayObservation
    confirmation: ConfirmationObservation

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_primitive(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("observation schema version must be an integer")
        return value


class TrialObservation(_WorkerObservation):
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["all_tools", "stateful"]

    @model_validator(mode="after")
    def responses_match_mode(self) -> TrialObservation:
        responses = (
            self.response,
            self.replay.response,
            self.confirmation.response,
            self.confirmation.duplicate_response,
        )
        if any(response is not None and response.mode != self.mode for response in responses):
            raise ValueError("every response mode must equal the trial mode")
        return self


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value).rstrip(b"\n")).hexdigest()


_VOLATILE_LOGICAL_FIELDS = frozenset({
    "created_at", "updated_at", "started_at", "finished_at", "lease_until", "owner_token",
})
_OPAQUE_LOGICAL_FIELDS = frozenset({"confirmation_token", "core_request_id"})


def _identity_view(collection: str, item: Any) -> Any:
    """Answering a question links the parent turn to its successor; that link is the
    product's normal chaining, not a rewrite of the earlier record."""
    if collection == "language_turns" and isinstance(item, dict):
        return {key: value for key, value in item.items() if key != "successor_request_id"}
    return item


def _record_identity(value: object) -> str:
    def stable_opaque(item: object) -> object:
        if isinstance(item, dict):
            return {key: stable_opaque(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [stable_opaque(child) for child in item]
        if isinstance(item, str):
            return re.sub(r"opaque-\d+", "opaque", item)
        return item

    return _hash(stable_opaque(value))


def _logical_state(value: object) -> object:
    opaque_bindings: dict[str, tuple[str, str]] = {}
    unbound_values: list[str] = []

    def collect(item: object, owner: str | None = None) -> None:
        if isinstance(item, dict):
            request_id = item.get("request_id")
            current_owner = request_id if isinstance(request_id, str) and "core_request_id" in item else owner
            for key, child in item.items():
                if key in _OPAQUE_LOGICAL_FIELDS and isinstance(child, str):
                    if current_owner is None:
                        if child not in unbound_values:
                            unbound_values.append(child)
                    else:
                        opaque_bindings[child] = (key, current_owner)
                collect(child, current_owner)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child, owner)

    collect(value)
    ordered = sorted(opaque_bindings, key=lambda item: opaque_bindings[item])
    ordered.extend(item for item in unbound_values if item not in opaque_bindings)
    normalized_ids = {item: f"opaque-{index}" for index, item in enumerate(ordered, start=1)}

    def normalize(item: object) -> object:
        if isinstance(item, dict):
            return {
                key: normalize(child)
                for key, child in item.items()
                if key not in _VOLATILE_LOGICAL_FIELDS and not key.endswith("_ms")
            }
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, str):
            for opaque, replacement in normalized_ids.items():
                item = item.replace(opaque, replacement)
            return item
        return item

    return normalize(value)


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _lstat_regular(path: Path, error: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(error)
    return metadata


def _lstat_directory(path: Path, error: str) -> os.stat_result:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(error)
    return metadata


def _candidate_identity(
    expected_candidate_dev: int | None, expected_candidate_ino: int | None
) -> tuple[int, int] | None:
    if (expected_candidate_dev is None) != (expected_candidate_ino is None):
        raise ValueError("candidate root is invalid")
    if expected_candidate_dev is None or expected_candidate_ino is None:
        return None
    if (
        isinstance(expected_candidate_dev, bool)
        or isinstance(expected_candidate_ino, bool)
        or expected_candidate_dev < 0
        or expected_candidate_ino < 0
    ):
        raise ValueError("candidate root is invalid")
    return expected_candidate_dev, expected_candidate_ino


def _assert_candidate_root_identity(path: Path, expected: tuple[int, int]) -> None:
    link_metadata = path.lstat()
    match = _POSIX_FD_PATH.fullmatch(path.as_posix())
    if (
        sys.platform != "linux"
        or match is None
        or int(match.group("pid")) != os.getppid()
        or not stat.S_ISLNK(link_metadata.st_mode)
        or _is_reparse(link_metadata)
    ):
        raise ValueError("candidate root is invalid")
    target_metadata = path.stat()
    if not stat.S_ISDIR(target_metadata.st_mode) or _is_reparse(target_metadata):
        raise ValueError("candidate root is invalid")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        retained_metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(retained_metadata.st_mode)
        or _is_reparse(retained_metadata)
        or (target_metadata.st_dev, target_metadata.st_ino) != expected
        or (retained_metadata.st_dev, retained_metadata.st_ino) != expected
    ):
        raise ValueError("candidate root is invalid")


def _resolve_candidate_root(
    path: Path,
    *,
    expected_candidate_dev: int | None = None,
    expected_candidate_ino: int | None = None,
) -> Path:
    absolute = path.absolute()
    metadata = absolute.lstat()
    expected = _candidate_identity(expected_candidate_dev, expected_candidate_ino)
    if stat.S_ISLNK(metadata.st_mode):
        if expected is None:
            raise ValueError("candidate root is invalid")
        _assert_candidate_root_identity(absolute, expected)
        return absolute
    if expected is not None:
        raise ValueError("candidate root is invalid")
    _lstat_directory(absolute, "candidate root is invalid")
    return absolute.resolve(strict=True)


def _bounded_bytes(path: Path, limit: int, error: str) -> bytes:
    before = _lstat_regular(path, error)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(error)
        value = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(value) > limit:
        raise ValueError(error)
    return value


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _outside_candidate(path: Path, candidate_root: Path) -> bool:
    return not _contained(path, candidate_root)


_SNAPSHOT_IGNORED_DIRECTORIES = frozenset({
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__",
    "dist", "node_modules", "release-evidence",
})


def _candidate_snapshot(candidate_root: Path) -> str:
    entries: list[dict[str, object]] = []
    pending = [candidate_root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(candidate_root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                raise ValueError("candidate snapshot contains an invalid path")
            if stat.S_ISDIR(metadata.st_mode):
                if path.name not in _SNAPSHOT_IGNORED_DIRECTORIES:
                    pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("candidate snapshot contains an invalid path")
            digest = hashlib.sha256()
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_size) != (
                    metadata.st_dev, metadata.st_ino, metadata.st_size
                ):
                    raise ValueError("candidate changed during snapshot")
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
            finally:
                os.close(descriptor)
            entries.append({"path": relative, "sha256": digest.hexdigest(), "size": metadata.st_size})
    return _hash(sorted(entries, key=lambda item: str(item["path"])))


def _secure_directory(path: Path, *, create: bool, empty: bool, error: str) -> Path:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        _lstat_directory(path.parent, error)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
    _lstat_directory(path, error)
    resolved = path.resolve(strict=True)
    if empty and any(path.iterdir()):
        raise ValueError(error)
    return resolved


def _exclusive_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags, 0o600)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = _exclusive_file(path)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clean_environment(candidate_backend: Path, storage: Path, case_path: Path, worker_output: Path,
                       mode: str, model: str, index: Path | None, base_url: str, *, phase: str,
                       previous_output: Path | None = None,
                       embedding_profile: Path | None = None,
                       embedding_base_url: str | None = None) -> dict[str, str]:
    retained = ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
    env = {name: os.environ[name] for name in retained if name in os.environ}
    env.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(candidate_backend), "D36_WORKER_INPUT": str(case_path),
        "D36_WORKER_OUTPUT": str(worker_output), "D36_STORAGE_ROOT": str(storage),
        "D36_WORKER_PHASE": phase, "D36_MODE": mode, "D36_MODEL": model,
        "D36_BASE_URL": base_url, "DATABASE_URL": f"sqlite:///{(storage / 'trial.db').as_posix()}",
        "STORAGE_ROOT": str(storage), "LANGUAGE_MODEL": model, "LANGUAGE_BASE_URL": base_url,
        "LANGUAGE_REVIEW_ALL": "false",
    })
    if previous_output is not None:
        env["D36_PREVIOUS_OUTPUT"] = str(previous_output)
    if index is not None:
        env.update({"D36_INDEX": str(index), "LANGUAGE_RETRIEVAL_INDEX": str(index),
                    "LANGUAGE_RETRIEVAL_READINESS": "true"})
    if mode == "stateful" and embedding_profile is not None and embedding_base_url is not None:
        env.update({"LANGUAGE_RETRIEVAL_PROFILE": str(embedding_profile),
                    "LANGUAGE_EMBEDDING_BASE_URL": embedding_base_url})
    return env


def _embedding_configuration(
    *, mode: str, embedding_profile: Path | None, embedding_base_url: str | None,
) -> tuple[Path, str, tuple[int, int, int, int, int], bytes] | None:
    if embedding_profile is None and embedding_base_url is None:
        return None
    if mode != "stateful" or embedding_profile is None or embedding_base_url is None:
        raise ValueError("invalid embedding configuration")
    try:
        from app.retrieval.contracts import EmbeddingProfile
        from app.retrieval.embeddings import local_url
        from app.retrieval.serialization import decode
        from evaluation.blinded_io import read_regular, validate_directory

        if type(embedding_base_url) is not str or len(embedding_base_url) > 512:
            raise ValueError()
        endpoint = local_url(embedding_base_url)
        path = embedding_profile.absolute()
        validate_directory(path.parent, "embedding profile parent")
        metadata = _lstat_regular(path, "invalid embedding configuration")
        if path.resolve(strict=True) != path:
            raise ValueError()
        raw = read_regular(path, maximum=16_000)
        profile = EmbeddingProfile.model_validate(decode(raw), strict=True)
        if profile.transport != "local-openai-embeddings-v1":
            raise ValueError()
        identity = (metadata.st_dev, metadata.st_ino, metadata.st_size,
                    metadata.st_mtime_ns, metadata.st_ctime_ns)
        return path, endpoint, identity, raw
    except (OSError, ValueError, TypeError):
        raise ValueError("invalid embedding configuration") from None


def _check_embedding_index(profile_raw: bytes, index: Path) -> None:
    try:
        from app.retrieval.contracts import EmbeddingProfile
        from app.retrieval.serialization import decode
        from evaluation.blinded_io import read_regular

        manifest = decode(read_regular(index / "manifest.json", maximum=64_000))
        if EmbeddingProfile.model_validate(manifest["profile"], strict=True) != (
            EmbeddingProfile.model_validate(decode(profile_raw), strict=True)
        ):
            raise ValueError()
    except (OSError, ValueError, TypeError, KeyError):
        raise ValueError("invalid embedding configuration") from None


def _run_candidate(command: list[str], *, cwd: Path, env: dict[str, str], storage: Path) -> None:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except OSError:
        raise ValueError("candidate subprocess failed") from None
    deadline = time.monotonic() + SUBPROCESS_TIMEOUT_SECONDS
    failure: str | None = None
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                failure = "candidate subprocess timed out"
                process.kill()
                break
            time.sleep(0.02)
        process.wait(timeout=5)
        if failure is not None or process.returncode != 0:
            raise ValueError(failure or "candidate subprocess failed")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def _atomic_publish(output_path: Path, payload: bytes, *, candidate_root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent = _secure_directory(output_path.parent, create=False, empty=False, error="output directory is invalid")
    if not _outside_candidate(parent, candidate_root):
        raise ValueError("trial storage and output must be external to candidate")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("output must not exist")
    temporary = parent / f".{output_path.name}.{secrets.token_hex(16)}.tmp"
    try:
        _write_exclusive(temporary, payload)
        temporary_metadata = _lstat_regular(temporary, "output temporary file is invalid")
        if not _contained(temporary, parent):
            raise ValueError("output temporary file escaped output directory")
        try:
            os.link(temporary, output_path, follow_symlinks=False)
        except FileExistsError:
            raise ValueError("output must not exist") from None
        published_metadata = _lstat_regular(output_path, "published output is invalid")
        if (temporary_metadata.st_dev, temporary_metadata.st_ino) != (
            published_metadata.st_dev, published_metadata.st_ino
        ):
            raise ValueError("published output identity mismatch")
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


def run_trial_host(*, candidate_root: Path, mode: str, input_path: Path, output_path: Path,
                   storage: Path, model: str, index: Path | None = None,
                   base_url: str = "http://127.0.0.1:1234/v1",
                   embedding_profile: Path | None = None,
                   embedding_base_url: str | None = None,
                   expected_candidate_dev: int | None = None,
                   expected_candidate_ino: int | None = None) -> TrialObservation:
    embedding = _embedding_configuration(
        mode=mode, embedding_profile=embedding_profile, embedding_base_url=embedding_base_url,
    )
    if embedding is not None:
        embedding_profile, embedding_base_url, _, profile_raw = embedding
    if type(mode) is not str or mode not in {"all_tools", "stateful"}:
        raise ValueError("unsupported evaluation mode")
    if mode == "stateful" and index is None:
        raise ValueError("stateful mode requires an index")
    if mode == "all_tools" and index is not None:
        raise ValueError("all_tools mode rejects an index")
    if (type(model) is not str or not model or model != model.strip()
            or len(model) > 128 or type(base_url) is not str or len(base_url) > 512):
        raise ValueError("model configuration is invalid")

    raw_input = _bounded_bytes(input_path, MAX_INPUT_BYTES, "invalid unlabeled trial input")
    try:
        from evaluation.unlabeled_contracts import UnlabeledTrialCase
        trial = UnlabeledTrialCase.model_validate_json(raw_input)
    except (ValidationError, ValueError):
        raise ValueError("invalid unlabeled trial case") from None

    expected_candidate_identity = _candidate_identity(
        expected_candidate_dev, expected_candidate_ino
    )
    candidate_root = _resolve_candidate_root(
        candidate_root,
        expected_candidate_dev=expected_candidate_dev,
        expected_candidate_ino=expected_candidate_ino,
    )
    candidate_backend = candidate_root / "backend"
    _lstat_directory(candidate_backend, "candidate root is invalid")
    if not _contained(candidate_backend, candidate_root):
        raise ValueError("candidate root is invalid")
    candidate_snapshot_sha256 = _candidate_snapshot(candidate_root)
    if not _outside_candidate(storage, candidate_root) or not _outside_candidate(output_path, candidate_root):
        raise ValueError("trial storage and output must be external to candidate")
    if embedding is not None:
        assert embedding_profile is not None and index is not None
        parent = embedding_profile.parent
        if (not _outside_candidate(embedding_profile, candidate_root)
                or any(_contained(path, parent) or _contained(parent, path)
                       for path in (storage, output_path))):
            raise ValueError("invalid embedding configuration")
        _check_embedding_index(profile_raw, index)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("output must not exist")
    storage = _secure_directory(storage, create=True, empty=True, error="storage must be empty")
    if index is not None:
        _lstat_directory(index, "stateful index is invalid")
        index = index.resolve(strict=True)
    worker_input = storage / f"validated-case-{secrets.token_hex(16)}.json"
    _write_exclusive(worker_input, _canonical_json_bytes(trial.model_dump(mode="json", exclude_unset=True)))
    _fsync_directory(storage)

    worker_script = str(Path(__file__).resolve())
    bootstrap = (
        "import runpy,sys;"
        f"sys.argv=[{worker_script!r},'--candidate-worker',*sys.argv[1:]];"
        f"runpy.run_path({worker_script!r},run_name='__main__')"
    )

    def worker_command(model_call_budget: int) -> list[str]:
        return [
            sys.executable,
            "-B",
            "-c",
            bootstrap,
            "--model-call-budget",
            str(model_call_budget),
        ]

    def run_candidate(command: list[str], *, env: dict[str, str]) -> None:
        if expected_candidate_identity is not None:
            _assert_candidate_root_identity(candidate_root, expected_candidate_identity)
        try:
            _run_candidate(command, cwd=candidate_backend, env=env, storage=storage)
        finally:
            if expected_candidate_identity is not None:
                _assert_candidate_root_identity(candidate_root, expected_candidate_identity)

    event_kind = trial.event.kind
    first_worker: _WorkerObservation | None = None
    if event_kind == "restart_resend":
        first_output = storage / f"worker-phase1-{secrets.token_hex(16)}.json"
        env = _clean_environment(candidate_backend, storage, worker_input, first_output, mode, model, index, base_url,
                                 phase="restart_prepare", embedding_profile=embedding_profile,
                                 embedding_base_url=embedding_base_url)
        run_candidate(worker_command(MODEL_CALL_LIMIT), env=env)
        first_raw = _bounded_bytes(first_output, MAX_WORKER_OUTPUT_BYTES, "invalid candidate observation")
        try:
            from evaluation.evidence_json import parse_canonical_model
            first_worker = parse_canonical_model(
                first_raw, _WorkerObservation, maximum=MAX_WORKER_OUTPUT_BYTES,
            )
        except (ValidationError, ValueError):
            raise ValueError("invalid candidate observation") from None
        worker_output = storage / f"worker-observation-{secrets.token_hex(16)}.json"
        env = _clean_environment(candidate_backend, storage, worker_input, worker_output, mode, model, index, base_url,
                                 phase="restart_replay", previous_output=first_output,
                                 embedding_profile=embedding_profile, embedding_base_url=embedding_base_url)
        remaining = MODEL_CALL_LIMIT - first_worker.model_calls
        run_candidate(worker_command(remaining), env=env)
    else:
        worker_output = storage / f"worker-observation-{secrets.token_hex(16)}.json"
        env = _clean_environment(candidate_backend, storage, worker_input, worker_output, mode, model, index, base_url,
                                 phase="single", embedding_profile=embedding_profile,
                                 embedding_base_url=embedding_base_url)
        run_candidate(worker_command(MODEL_CALL_LIMIT), env=env)

    if embedding is not None:
        if _embedding_configuration(
            mode=mode, embedding_profile=embedding_profile, embedding_base_url=embedding_base_url,
        ) != embedding:
            raise ValueError("embedding configuration changed during trial")
        assert index is not None
        _check_embedding_index(profile_raw, index)
    if _candidate_snapshot(candidate_root) != candidate_snapshot_sha256:
        raise ValueError("candidate changed during trial")
    raw_observation = _bounded_bytes(worker_output, MAX_WORKER_OUTPUT_BYTES, "invalid candidate observation")
    try:
        from evaluation.evidence_json import parse_canonical_model
        worker = parse_canonical_model(
            raw_observation, _WorkerObservation, maximum=MAX_WORKER_OUTPUT_BYTES,
        )
    except (ValidationError, ValueError):
        raise ValueError("invalid candidate observation") from None
    if first_worker is not None and worker.model_calls != first_worker.model_calls + worker.replay.model_calls:
        raise ValueError("invalid candidate observation")
    observation = TrialObservation(**worker.model_dump(mode="python"), case_sha256=trial.case_sha256,
                                   candidate_snapshot_sha256=candidate_snapshot_sha256,
                                   input_sha256=hashlib.sha256(raw_input).hexdigest(), mode=mode)
    _atomic_publish(output_path, _canonical_json_bytes(observation.model_dump(mode="json")), candidate_root=candidate_root)
    return observation


def _candidate_worker(model_call_budget: int) -> int:
    try:
        from fastapi.testclient import TestClient
        from sqlalchemy import select

        from app.api import routes_language
        from app.core.config import get_settings
        from app.db import get_session_factory
        from app.interpretation.contracts import ClarificationProposal, InterpretationOutcome
        from app.interpretation.local_chat import LocalChatAdapter
        from app.language_operations.contracts import LanguageResponse
        from app.models.artifact import GenerationArtifact
        from app.models.external_call import ExternalCall
        from app.models.job import GenerationJob, JobStatus
        from app.models.language_request import LanguageRequestRecord
        from app.models.language_turn import LanguageTurn
        from app.models.operation_request import OperationReceipt
        from app.models.project import Project, ProjectStatus
        from app.models.settings_revision import SettingsRevision
        from app.operations.contracts import OperationRequest, OperationResult, OperationTarget
        from app.schemas import ProjectCreate
        from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
        from app.services.settings_history import configuration
        from app.workers import operation_dispatcher as candidate_operation_dispatcher

        candidate_operation_dispatcher.run_operation_dispatcher = _quiescent_candidate_dispatcher
        from app.main import create_app

        case = json.loads(Path(os.environ["D36_WORKER_INPUT"]).read_text(encoding="ascii"))
        initial, event = case["initial"], case["event"]
        settings = get_settings()
        settings.language_model, settings.language_base_url = os.environ["D36_MODEL"], os.environ["D36_BASE_URL"]
        settings.language_retrieval_index = Path(os.environ["D36_INDEX"]) if os.environ["D36_MODE"] == "stateful" else None
        if "LANGUAGE_RETRIEVAL_PROFILE" in os.environ:
            settings.language_retrieval_profile = Path(os.environ["LANGUAGE_RETRIEVAL_PROFILE"])
            settings.language_embedding_base_url = os.environ["LANGUAGE_EMBEDDING_BASE_URL"]
        model_budget = _ModelCallBudget(model_call_budget)
        race_applied = False
        after_submit_race = False

        def apply_external_race() -> None:
            nonlocal race_applied
            if race_applied or event["kind"] != "revision_race":
                return
            if event.get("timing", "before_execution") != "before_execution" and not after_submit_race:
                return
            race_applied = True
            with get_session_factory()() as race_db:
                project = race_db.get(Project, initial["project_id"])
                for key, value in event["external_settings"].items():
                    setattr(project, key, value)
                # After submit, never reuse a revision the request itself just recorded.
                project.revision = (max(event["external_revision"], project.revision + 1)
                                    if after_submit_race else event["external_revision"])
                race_db.add(SettingsRevision(project_id=project.id, revision=project.revision,
                                              settings_json=configuration(project), changed_fields=sorted(event["external_settings"])))
                race_db.commit()

        async def counted_complete(self: Any, messages: Any, schema: Any) -> str:
            async def invoke() -> str:
                async with LocalChatAdapter(settings.language_base_url, settings.language_model,
                                            timeout_seconds=PROTOCOL_DEADLINE_SECONDS,
                                            reasoning_effort=settings.language_reasoning_effort) as adapter:
                    return await adapter.complete(messages, schema)

            result = await model_budget.complete(invoke)
            apply_external_race()
            return result

        routes_language._LocalAdapter.complete = counted_complete

        def make_project(item: dict[str, Any], *, primary: bool = False) -> Project:
            values = ProjectCreate(title="D36 synthetic", source_script="D36 synthetic source.",
                                   use_fake_providers=True, **item["settings"]).model_dump(exclude={"providers"})
            return Project(id=item["project_id"], revision=item["revision"],
                           status=ProjectStatus(item["project_status"]), **values)

        def seed() -> None:
            primary = {"project_id": initial["project_id"], "revision": initial["revision"],
                       "settings": initial["settings"], "project_status": initial["project_status"],
                       "current_artifact_id": initial.get("current_artifact_id")}
            with get_session_factory()() as db:
                projects = [make_project(primary, primary=True), *(make_project(item) for item in initial.get("additional_projects", []))]
                db.add_all(projects)
                db.flush()
                project_map = {project.id: project for project in projects}
                defaults = {project.id: configuration(project) for project in projects}
                for history in initial["history"]:
                    owner_id = history.get("project_id") or initial["project_id"]
                    db.add(SettingsRevision(project_id=owner_id, revision=history["revision"],
                        settings_json={**defaults[owner_id], **history["settings"]},
                        changed_fields=history.get("changed_fields", []),
                        restored_from_revision=history.get("restored_from_revision")))
                for item in initial["jobs"]:
                    owner = project_map[item["project_id"]]
                    snapshot = capture_inputs(owner)
                    snapshot["project"].update(item["input_settings"])
                    db.add(GenerationJob(id=item["id"], project_id=owner.id, status=JobStatus(item["status"]),
                        kind=item["kind"], block_index=item.get("block_index"), parent_job_id=item.get("parent_job_id"),
                        cancel_requested=item["cancel_requested"], input_revision=item["input_revision"],
                        input_snapshot=snapshot, input_fingerprint=fingerprint_inputs(snapshot)))
                artifacts = list(initial.get("artifacts", []))
                next_id = max([item["id"] for item in artifacts], default=0) + 1
                for revision in initial.get("artifact_revisions", []):
                    artifacts.append({"id": next_id, "project_id": initial["project_id"], "job_id": None,
                                      "revision": revision, "file_size": 1,
                                      "file_sha256": hashlib.sha256(b"x").hexdigest()})
                    next_id += 1
                for item in artifacts:
                    relative = f"projects/{item['project_id']}/history/d36-{item['id']}/video.mp4"
                    absolute = Path(os.environ["D36_STORAGE_ROOT"]) / relative
                    absolute.parent.mkdir(parents=True, exist_ok=True)
                    content = bytes.fromhex(item.get("file_content_hex", "78"))
                    if (len(content) != item["file_size"]
                            or hashlib.sha256(content).hexdigest() != item["file_sha256"]):
                        raise ValueError("artifact identity mismatch")
                    absolute.write_bytes(content)
                    artifact = GenerationArtifact(id=item["id"], project_id=item["project_id"], job_id=item.get("job_id"),
                        revision=item.get("revision"), video_path=relative,
                        input_fingerprint=None, manifest_json={"schema_version": 1, "synthetic_placeholder": True,
                                                               "file_sha256": item["file_sha256"]})
                    db.add(artifact)
                db.flush()
                artifact_map = {item["id"]: item for item in artifacts}
                project_inputs = [primary, *initial.get("additional_projects", [])]
                for project_input in project_inputs:
                    project = project_map[project_input["project_id"]]
                    current_id = project_input.get("current_artifact_id")
                    if current_id is not None:
                        current = artifact_map[current_id]
                        project.current_artifact_id = current_id
                        project.output_video_path = (
                            f"projects/{current['project_id']}/history/d36-{current_id}/video.mp4"
                        )
                for item in initial.get("receipts", []):
                    canonical_request = json.dumps({"operation_id": item["operation_id"], "operation_version": item["operation_version"],
                        "project_id": item["project_id"], "base_revision": item["base_revision"]},
                        ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                    result = {"operation_id": item["operation_id"], "project_id": item["project_id"],
                              "revision": item["result_revision"], "changed": item["result_revision"] != item["base_revision"]}
                    canonical_sha256 = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
                    if canonical_sha256 != item["canonical_request_sha256"] or _hash(result) != item["result_sha256"]:
                        raise ValueError("receipt identity mismatch")
                    db.add(OperationReceipt(request_id=item["request_id"], canonical_request=canonical_request,
                        operation_id=item["operation_id"], operation_version=item["operation_version"],
                        project_id=item["project_id"], base_revision=item["base_revision"],
                        result_revision=item["result_revision"], resolved_arguments={},
                        generation_requested=item["generation_requested"], job_id=item.get("job_id"),
                        result_ref=f"d36:{item['request_id']}", result_json=result))
                for item in initial.get("external_calls", []):
                    body = (bytes.fromhex(item["response_body_hex"])
                            if item.get("response_body_hex") is not None else None)
                    if body is not None and hashlib.sha256(body).hexdigest() != item["response_body_sha256"]:
                        raise ValueError("external response identity mismatch")
                    db.add(ExternalCall(id=item["id"], job_id=item["job_id"], fingerprint=item["fingerprint"],
                        provider=item["provider"], endpoint=item["endpoint"], remote_side_effect=item["remote_side_effect"],
                        status=item["status"], attempts=item["attempts"], response_status=item.get("response_status"),
                        response_body=body, response_content_type=item.get("response_content_type"),
                        provider_response_id=item.get("provider_response_id"), error_code=item.get("error_code")))
                explicit_turn_links = any(
                    "parent_request_id" in turn or "successor_request_id" in turn
                    for turn in initial["prior_turns"]
                )
                for index, turn in enumerate(initial["prior_turns"]):
                    proposal = turn.get("proposal")
                    outcome = InterpretationOutcome.model_validate({
                        "status": {"operation": "proposed", "unsupported": "unsupported",
                                   "no_operation": "dismissed"}.get(proposal["kind"] if proposal else "", "needs_input"),
                        "proposal": proposal,
                    })
                    response = LanguageResponse(request_id=turn["request_id"], core_request_id="seed-" + turn["request_id"],
                        project_id=turn["project_id"], base_revision=turn["base_revision"], status=turn["status"],
                        interpretation=outcome, clarification=ClarificationProposal(kind="clarification",
                            question=turn["question"], missing_fields=["arguments"]) if turn.get("question") else None,
                        dialogue_available=True)
                    if turn.get("result_revision") and proposal and proposal["kind"] == "operation":
                        response = response.model_copy(update={"result": OperationResult(operation_id=proposal["operation_id"],
                            project_id=turn["project_id"], changed=turn["settings_saved"],
                            state_revision=str(turn["result_revision"]), revision=turn["result_revision"], data={}),
                            "executed": True})
                    if proposal and proposal["kind"] == "operation":
                        response = response.model_copy(update={"prepared_request": OperationRequest(
                            operation_id=proposal["operation_id"], operation_version=proposal["operation_version"],
                            target=OperationTarget(project_id=turn["project_id"]), arguments={k: v for k, v in proposal["arguments"].items() if v is not None},
                            base_revision=turn["base_revision"], request_id=response.core_request_id,
                            generation_requested=proposal["generate_after_save"])})
                    db.add(LanguageRequestRecord(request_id=response.request_id, core_request_id=response.core_request_id,
                        input_fingerprint="0" * 64, project_id=response.project_id, base_revision=response.base_revision,
                        status=response.status, owner_token="seed", lease_until=0, created_at=0,
                        request_json=None, response_json=response.model_dump(mode="json")))
                    parent_request_id = (
                        turn.get("parent_request_id") if explicit_turn_links
                        else initial["prior_turns"][index - 1]["request_id"] if index else None
                    )
                    successor_request_id = (
                        turn.get("successor_request_id") if explicit_turn_links
                        else initial["prior_turns"][index + 1]["request_id"]
                        if index + 1 < len(initial["prior_turns"]) else None
                    )
                    db.add(LanguageTurn(request_id=response.request_id, text=turn["text"],
                        parent_request_id=parent_request_id, relation=turn.get("relation"),
                        successor_request_id=successor_request_id))
                db.commit()

        storage_root = Path(os.environ["D36_STORAGE_ROOT"])

        def file_identity(relative: str | None) -> dict[str, Any] | None:
            return _file_identity(storage_root, relative)

        settings_fields = tuple(initial["settings"])

        def projected_settings(value: Any) -> dict[str, Any]:
            if isinstance(value, dict):
                return {name: value[name] for name in settings_fields}
            return {name: getattr(value, name) for name in settings_fields}

        def canonical_state() -> dict[str, Any]:
            with get_session_factory()() as db:
                projects = list(db.scalars(select(Project).order_by(Project.id)))
                history = list(db.scalars(select(SettingsRevision).order_by(SettingsRevision.project_id, SettingsRevision.revision)))
                jobs = list(db.scalars(select(GenerationJob).order_by(GenerationJob.id)))
                receipts = list(db.scalars(select(OperationReceipt).order_by(OperationReceipt.request_id)))
                artifacts = list(db.scalars(select(GenerationArtifact).order_by(GenerationArtifact.id)))
                calls_rows = list(db.scalars(select(ExternalCall).order_by(ExternalCall.id)))
                requests = list(db.scalars(select(LanguageRequestRecord).order_by(LanguageRequestRecord.request_id)))
                turns = list(db.scalars(select(LanguageTurn).order_by(LanguageTurn.request_id)))
                primary = db.get(Project, initial["project_id"])
                state = {
                    "primary": {"revision": primary.revision, "status": primary.status.value,
                                "settings": projected_settings(primary)},
                    "projects": [{"id": row.id, "revision": row.revision, "status": row.status.value,
                        "settings": projected_settings(row), "title_sha256": _hash(row.title),
                        "source_script_sha256": _hash(row.source_script),
                        "global_visual_style_sha256": _hash(row.global_visual_style),
                        "progress": row.progress, "current_stage": row.current_stage,
                        "current_artifact_id": row.current_artifact_id,
                        "output_video": file_identity(row.output_video_path),
                        "output_subtitle": file_identity(row.output_subtitle_path),
                        "error_sha256": _hash(row.error_message),
                        "created_at": row.created_at.isoformat(), "updated_at": row.updated_at.isoformat()} for row in projects],
                    "history": [{"project_id": row.project_id, "revision": row.revision,
                        "settings": projected_settings(row.settings_json), "changed_fields": sorted(row.changed_fields or []),
                        "restored_from_revision": row.restored_from_revision,
                        "created_at": row.created_at.isoformat()} for row in history],
                    "jobs": [{"id": row.id, "project_id": row.project_id, "status": row.status.value,
                        "current_stage": row.current_stage, "progress": row.progress,
                        "stage_progress": row.stage_progress, "input_revision": row.input_revision,
                        "cancel_requested": row.cancel_requested, "kind": row.kind,
                        "block_index": row.block_index, "parent_job_id": row.parent_job_id,
                        "input_snapshot_sha256": _hash(row.input_snapshot), "input_fingerprint": row.input_fingerprint,
                        "plan_sha256": _hash(row.plan_json), "recovery_sha256": _hash(row.recovery_message),
                        "error_sha256": _hash(row.error_message),
                        "started_at": row.started_at.isoformat() if row.started_at else None,
                        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                        "created_at": row.created_at.isoformat()} for row in jobs],
                    "receipts": [{"request_id": row.request_id,
                        "canonical_request": json.loads(row.canonical_request),
                        "operation_id": row.operation_id, "operation_version": row.operation_version,
                        "project_id": row.project_id, "base_revision": row.base_revision,
                        "result_revision": row.result_revision, "resolved_arguments": row.resolved_arguments,
                        "generation_requested": row.generation_requested, "job_id": row.job_id,
                        "result_ref": row.result_ref, "result": row.result_json,
                        "created_at": row.created_at.isoformat()} for row in receipts],
                    "artifacts": [{"id": row.id, "project_id": row.project_id, "job_id": row.job_id,
                        "revision": row.revision, "input_fingerprint": row.input_fingerprint,
                        "video": file_identity(row.video_path), "subtitle": file_identity(row.subtitle_path),
                        "manifest_sha256": _hash(row.manifest_json),
                        "created_at": row.created_at.isoformat()} for row in artifacts],
                    "external_calls": [{"id": row.id, "job_id": row.job_id, "fingerprint": row.fingerprint,
                        "provider": row.provider, "endpoint_sha256": _hash(row.endpoint),
                        "remote_side_effect": row.remote_side_effect, "status": row.status, "attempts": row.attempts,
                        "response_status": row.response_status,
                        "response_body_sha256": hashlib.sha256(row.response_body).hexdigest() if row.response_body is not None else None,
                        "response_content_type": row.response_content_type,
                        "provider_response_id_sha256": _hash(row.provider_response_id), "error_code": row.error_code,
                        "started_at": row.started_at.isoformat(),
                        "finished_at": row.finished_at.isoformat() if row.finished_at else None} for row in calls_rows],
                    "language_requests": [{"request_id": row.request_id, "input_fingerprint": row.input_fingerprint,
                        "core_request_id": row.core_request_id, "project_id": row.project_id,
                        "base_revision": row.base_revision, "status": row.status,
                        "created_at": row.created_at, "request": row.request_json,
                        "response": row.response_json} for row in requests],
                    "language_turns": [{"request_id": row.request_id, "parent_request_id": row.parent_request_id,
                        "relation": row.relation, "text_sha256": _hash(row.text),
                        "successor_request_id": row.successor_request_id} for row in turns],
                }
                normalized = _logical_state(state)
                if not isinstance(normalized, dict):
                    raise ValueError("logical state normalization failed")
                return normalized

        def redact_state(value: dict[str, Any]) -> dict[str, Any]:
            primary = value["primary"]
            result = {"state_sha256": _hash(value), "project_status": primary["status"],
                      "settings_sha256": _hash(primary["settings"]),
                      "project_entries": [{"id": item["id"], "revision": item["revision"],
                          "status": item["status"], "settings_sha256": _hash(item["settings"]),
                          "title_sha256": item["title_sha256"],
                          "source_script_sha256": item["source_script_sha256"],
                          "global_visual_style_sha256": item["global_visual_style_sha256"],
                          "progress": item["progress"], "current_stage": item["current_stage"],
                          "current_artifact_id": item["current_artifact_id"],
                          "output_video": item["output_video"],
                          "output_subtitle": item["output_subtitle"],
                          "error_sha256": item["error_sha256"]} for item in value["projects"]],
                      "history_entries": [{"project_id": item["project_id"],
                          "revision": item["revision"], "settings_sha256": _hash(item["settings"]),
                          "changed_fields": item["changed_fields"],
                          "restored_from_revision": item["restored_from_revision"]}
                          for item in value["history"]],
                      "job_entries": [{key: item[key] for key in ("id", "project_id", "status",
                          "current_stage", "progress", "stage_progress", "input_revision",
                          "cancel_requested", "kind", "block_index", "parent_job_id",
                          "input_fingerprint", "input_snapshot_sha256", "plan_sha256",
                          "recovery_sha256", "error_sha256")} for item in value["jobs"]],
                      "artifact_entries": [{"id": item["id"], "project_id": item["project_id"],
                          "job_id": item["job_id"], "revision": item["revision"],
                          "input_fingerprint": item["input_fingerprint"],
                          "video_path_sha256": item["video"]["path_sha256"],
                          "video_size": item["video"]["size"],
                          "video_sha256": item["video"]["sha256"],
                          "subtitle_path_sha256": (
                              item["subtitle"]["path_sha256"] if item["subtitle"] is not None else None
                          ),
                          "subtitle_size": (
                              item["subtitle"]["size"] if item["subtitle"] is not None else None
                          ),
                          "subtitle_sha256": (
                              item["subtitle"]["sha256"] if item["subtitle"] is not None else None
                          ), "manifest_sha256": item["manifest_sha256"]}
                          for item in value["artifacts"]],
                      "receipt_identity_sha256s": sorted(
                          _record_identity(item) for item in value["receipts"]
                      ),
                      "external_call_identity_sha256s": sorted(
                          _record_identity(item) for item in value["external_calls"]
                      ),
                      "language_request_identity_sha256s": sorted(
                          _record_identity(item) for item in value["language_requests"]
                      ),
                      "language_turn_identity_sha256s": sorted(
                          _record_identity(_identity_view("language_turns", item)) for item in value["language_turns"]
                      )}
            for name in ("projects", "history", "jobs", "receipts", "artifacts", "external_calls", "language_requests", "language_turns"):
                result[f"{name}_sha256"] = _hash(value[name])
                singular = {"projects": "project", "history": "history", "jobs": "job", "receipts": "receipt",
                            "artifacts": "artifact", "external_calls": "external_call",
                            "language_requests": "language_request", "language_turns": "language_turn"}[name]
                result[f"{singular}_count"] = len(value[name])
            return result

        def response_projection(response: dict[str, Any], status_code: int) -> dict[str, Any]:
            failure = response.get("failure") or response.get("detail") or {}
            interpretation = response.get("interpretation") or {}
            proposal = interpretation.get("proposal") or {}
            clarification = response.get("clarification") or (
                proposal if proposal.get("kind") == "clarification" else {}
            )
            operation = proposal if proposal.get("kind") == "operation" and not clarification else {}
            prepared = response.get("prepared_request") or {}
            mode = response.get("mode", "all_tools")
            if mode == "semantic":
                mode = "stateful" if os.environ["D36_MODE"] == "stateful" else "semantic"
            status = response.get("status", "http_error" if status_code >= 400 else "error")
            reason = failure.get("reason_code") if isinstance(failure, dict) else None
            missing_fields = clarification.get("missing_fields")
            canonical_missing_fields = (
                sorted(set(missing_fields)) if isinstance(missing_fields, list) else None
            )
            public = {"http_status": status_code, "status": status, "mode": mode,
                      "executed": bool(response.get("executed")),
                      "requires_confirmation": bool(response.get("requires_confirmation")),
                      "operation_id": operation.get("operation_id"),
                      "operation_version": operation.get("operation_version"),
                      "arguments_sha256": (
                          _hash(operation.get("arguments")) if operation else None
                      ),
                      "generate_after_save": operation.get("generate_after_save") if operation else None,
                      "generation_requested": (
                          prepared.get("generation_requested") if operation else None
                      ),
                      "clarification_missing_fields": canonical_missing_fields,
                      "reason_code": reason,
                      "project_id": response.get("project_id"), "base_revision": response.get("base_revision"),
                      "result_revision": (response.get("result") or {}).get("revision"),
                      "job_id": (response.get("generation_result") or response.get("result") or {}).get("job_id")}
            return {**{key: public[key] for key in (
                        "http_status", "status", "mode", "executed", "requires_confirmation",
                        "operation_id", "operation_version", "arguments_sha256",
                        "generate_after_save", "generation_requested",
                        "clarification_missing_fields", "reason_code")},
                    "response_sha256": _hash(public)}

        def request_payload(request: dict[str, Any], *, text: str | None = None,
                            target: int | None | object = ...) -> dict[str, Any]:
            target_id = request["target_project_id"] if target is ... else target
            payload: dict[str, Any] = {"request_id": request["request_id"], "text": text or request["text"],
                "base_revision": request["base_revision"], "target": {"project_id": target_id}}
            if request.get("continuation") is not None:
                payload["continuation"] = request["continuation"]
            return payload

        phase = os.environ["D36_WORKER_PHASE"]
        client = TestClient(create_app())
        client.__enter__()
        try:
            if phase != "restart_replay":
                seed()
            request = event["request"]
            payload = request_payload(request)
            before = canonical_state()
            if event["kind"] == "concurrent_identical":
                with ThreadPoolExecutor(max_workers=2) as executor:
                    submitted = list(executor.map(
                        lambda _: client.post("/api/language/requests", json=payload), range(2)
                    ))
                # One submission usually observes the other in flight; read its final
                # stored response (bounded) so both sides are compared after completion.
                settled = []
                for item in submitted:
                    for _ in range(600):
                        if item.json().get("status") != "interpreting":
                            break
                        time.sleep(0.05)
                        item = client.get(f"/api/language/requests/{request['request_id']}")
                    settled.append(item)
                response_http, concurrent_http = settled
                response = response_http.json()
                concurrent_response = concurrent_http.json()
            else:
                response_http = client.post("/api/language/requests", json=payload)
                response = response_http.json()
                concurrent_http = None
                concurrent_response = None
            after_submit = canonical_state()
            confirmation_http: Any | None = None
            confirmation_response: dict[str, Any] | None = None
            confirmed_state: dict[str, Any] | None = None
            duplicate_confirmation_http: Any | None = None
            duplicate_confirmation: dict[str, Any] | None = None
            replay_http: Any | None = None
            replay_response: dict[str, Any] | None = None
            replay_before, replay_after = after_submit, after_submit
            calls_before_event = model_budget.calls

            if phase == "restart_replay":
                previous = _WorkerObservation.model_validate_json(Path(os.environ["D36_PREVIOUS_OUTPUT"]).read_bytes())
                replay_after = canonical_state()
                replay_projection = RedactedResponse.model_validate(
                    response_projection(response, response_http.status_code)
                )
                output = previous.model_copy(update={
                    "after": RedactedState.model_validate(redact_state(replay_after)),
                    "model_calls": previous.model_calls + model_budget.calls,
                    "replay": ReplayObservation(attempted=True, model_calls=model_budget.calls,
                        state_unchanged=previous.after.state_sha256 == _hash(replay_after),
                        same_response=previous.response.response_sha256 == replay_projection.response_sha256,
                        response=replay_projection, failure_class=None),
                })
                Path(os.environ["D36_WORKER_OUTPUT"]).write_bytes(_canonical_json_bytes(output.model_dump(mode="json")))
                return 0

            kind = event["kind"]
            if kind == "revision_race" and event.get("timing") == "after_submit_before_confirmation":
                after_submit_race = True
                apply_external_race()
                if response.get("confirmation_token"):
                    permission = {"confirmation_token": response["confirmation_token"], "confirm_generation": True}
                    confirmation_http = client.post(f"/api/language/requests/{request['request_id']}/execute", json=permission)
                    confirmation_response = confirmation_http.json()
                    confirmed_state = canonical_state()
            elif kind in {"confirm_generation", "confirm_twice"} and response.get("confirmation_token"):
                permission = {"confirmation_token": response["confirmation_token"], "confirm_generation": True}
                confirmation_http = client.post(f"/api/language/requests/{request['request_id']}/execute", json=permission)
                confirmation_response = confirmation_http.json()
                confirmed_state = canonical_state()
                if kind == "confirm_twice":
                    duplicate_confirmation_http = client.post(
                        f"/api/language/requests/{request['request_id']}/execute", json=permission
                    )
                    duplicate_confirmation = duplicate_confirmation_http.json()
            elif kind == "resend_identical":
                replay_before = canonical_state()
                replay_http = client.post("/api/language/requests", json=payload)
                replay_response = replay_http.json()
                replay_after = canonical_state()
            elif kind == "same_id_different_body":
                replacement = request_payload(request, text=event["replacement_text"],
                    target=event.get("replacement_target_project_id", request["target_project_id"]))
                replay_before = canonical_state()
                replay_http = client.post("/api/language/requests", json=replacement)
                replay_response = replay_http.json()
                replay_after = canonical_state()
            elif kind == "concurrent_identical":
                replay_before = before
                replay_http = concurrent_http
                replay_response = concurrent_response
                replay_after = after_submit
            elif kind == "switch_target":
                if event["action"] != "read_original_request":
                    raise ValueError("unsupported switch-target action")
                selected_project_id = event["selected_project_id_after"]
                if selected_project_id == request["target_project_id"]:
                    raise ValueError("switch-target event did not change selection")
                replay_before = canonical_state()
                replay_http = client.post("/api/language/requests", json=payload)
                replay_response = replay_http.json()
                replay_after = canonical_state()

            after = canonical_state()
            collection_changes = {name: int(_hash(before[name]) != _hash(after[name])) for name in
                ("history", "jobs", "receipts", "artifacts", "external_calls", "language_requests", "language_turns")}
            opaque_collections = (
                "receipts", "external_calls", "language_requests", "language_turns"
            )
            before_identities = {
                name: {_record_identity(_identity_view(name, item)) for item in before[name]}
                for name in opaque_collections
            }
            after_identities = {
                name: {_record_identity(_identity_view(name, item)) for item in after[name]}
                for name in opaque_collections
            }
            additions = {
                name: len(after_identities[name] - before_identities[name])
                for name in opaque_collections
            }
            prior_preserved = {
                name: before_identities[name] <= after_identities[name]
                for name in opaque_collections
            }
            cancel_before = [(item["id"], item["cancel_requested"]) for item in before["jobs"]]
            cancel_after = [(item["id"], item["cancel_requested"]) for item in after["jobs"]]
            redacted_response = response_projection(response, response_http.status_code)
            replay_projection = (
                response_projection(replay_response, replay_http.status_code)
                if replay_response is not None and replay_http is not None else None
            )
            confirmation_projection = (
                response_projection(confirmation_response, confirmation_http.status_code)
                if confirmation_response is not None and confirmation_http is not None else None
            )
            duplicate_confirmation_projection = (
                response_projection(duplicate_confirmation, duplicate_confirmation_http.status_code)
                if duplicate_confirmation is not None and duplicate_confirmation_http is not None else None
            )
            output = {
                "schema_version": 1, "response": redacted_response,
                "before": redact_state(before), "after": redact_state(after),
                "effects": {"settings": int(before["primary"]["settings"] != after["primary"]["settings"]),
                    "revision": abs(after["primary"]["revision"] - before["primary"]["revision"]),
                    "jobs": collection_changes["jobs"], "cancellations": int(cancel_before != cancel_after),
                    "receipts": additions["receipts"], "artifacts": collection_changes["artifacts"],
                    "external_calls": additions["external_calls"], "history": collection_changes["history"],
                    "language_records": int(collection_changes["language_requests"] or collection_changes["language_turns"]),
                    "language_requests": additions["language_requests"],
                    "language_turns": additions["language_turns"],
                    "prior_receipts_preserved": prior_preserved["receipts"],
                    "prior_external_calls_preserved": prior_preserved["external_calls"],
                    "prior_language_requests_preserved": prior_preserved["language_requests"],
                    "prior_language_turns_preserved": prior_preserved["language_turns"]},
                "model_calls": model_budget.calls, "failure_class": None,
                "replay": {"attempted": replay_projection is not None, "model_calls": model_budget.calls - calls_before_event,
                    "state_unchanged": replay_before == replay_after,
                    "same_response": replay_projection is not None and redacted_response["response_sha256"] == replay_projection["response_sha256"],
                    "response": replay_projection, "failure_class": None},
                "confirmation": {"attempted": confirmation_projection is not None,
                    "duplicate_attempted": duplicate_confirmation_projection is not None,
                    "state_sha256": _hash(confirmed_state if confirmed_state is not None else after)
                    if confirmation_projection is not None else None,
                    "duplicate_same_response": (
                        confirmation_projection["response_sha256"] == duplicate_confirmation_projection["response_sha256"]
                    ) if duplicate_confirmation_projection is not None and confirmation_projection is not None else None,
                    "response": confirmation_projection,
                    "duplicate_response": duplicate_confirmation_projection,
                    "failure_class": None},
            }
            Path(os.environ["D36_WORKER_OUTPUT"]).write_bytes(_canonical_json_bytes(output))
        finally:
            client.__exit__(None, None, None)
        return 0
    except BaseException:
        return 2


def main() -> int:
    if sys.argv[1:2] == ["--candidate-worker"]:
        worker_parser = argparse.ArgumentParser(add_help=False)
        worker_parser.add_argument("--model-call-budget", type=int, choices=range(MODEL_CALL_LIMIT + 1), required=True)
        worker_args = worker_parser.parse_args(sys.argv[2:])
        return _candidate_worker(worker_args.model_call_budget)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("all_tools", "stateful"), required=True)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--embedding-profile", type=Path)
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--expected-candidate-dev", type=int)
    parser.add_argument("--expected-candidate-ino", type=int)
    args = parser.parse_args()
    try:
        run_trial_host(**vars(args))
    except (OSError, ValueError):
        print("Candidate trial failed without exposing trial or model content.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
