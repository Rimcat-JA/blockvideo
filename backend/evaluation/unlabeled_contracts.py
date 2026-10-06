"""Strict label-free wire contracts for one external D36 candidate trial."""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_IDENTIFIER = 128
MAX_REVISION = 10**12
MAX_DATABASE_ID = 2**63 - 1


class StrictUnlabeledRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class UnlabeledPronunciation(StrictUnlabeledRecord):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)

    surface: str = Field(min_length=1, max_length=80)
    reading: str = Field(min_length=1, max_length=160, pattern=r"^[ァ-ヴー]+$")
    accent: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def candidate_compatible(self) -> Self:
        if any(character in self.surface for character in "。！？\n\r"):
            raise ValueError("pronunciation surface contains a delimiter")
        if self.reading[0] in "ァィゥェォャュョヮー":
            raise ValueError("pronunciation reading has an invalid start")
        mora_count = sum(character not in "ァィゥェォャュョヮ" for character in self.reading)
        if self.accent is not None and self.accent > mora_count:
            raise ValueError("pronunciation accent exceeds reading mora count")
        return self


def _unique_pronunciation_surfaces(
    overrides: tuple[UnlabeledPronunciation, ...],
) -> tuple[UnlabeledPronunciation, ...]:
    surfaces = [item.surface for item in overrides]
    if len(surfaces) != len(set(surfaces)):
        raise ValueError("pronunciation surfaces must be unique")
    return overrides


UnlabeledPronunciations = Annotated[
    tuple[UnlabeledPronunciation, ...],
    AfterValidator(_unique_pronunciation_surfaces),
]


class UnlabeledSettings(StrictUnlabeledRecord):
    subtitle_font_size: int = Field(ge=16, le=120)
    voicevox_speed_scale: float = Field(ge=0.5, le=2.0)
    voicevox_pitch_scale: float = Field(default=0.0, ge=-1.0, le=1.0)
    voicevox_speaker_id: int = Field(ge=0, le=100000)
    pronunciation_overrides: UnlabeledPronunciations = Field(max_length=100, strict=False)
    narration_pacing_mode: Literal["adaptive", "fixed"]
    narration_sentence_pause_seconds: float = Field(ge=0, le=5)


class UnlabeledSettingsPatch(StrictUnlabeledRecord):
    subtitle_font_size: int | None = Field(default=None, ge=16, le=120)
    voicevox_speed_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    voicevox_pitch_scale: float | None = Field(default=None, ge=-1.0, le=1.0)
    voicevox_speaker_id: int | None = Field(default=None, ge=0, le=100000)
    pronunciation_overrides: UnlabeledPronunciations | None = Field(default=None, max_length=100, strict=False)
    narration_pacing_mode: Literal["adaptive", "fixed"] | None = None
    narration_sentence_pause_seconds: float | None = Field(default=None, ge=0, le=5)

    @model_validator(mode="after")
    def not_empty(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("settings patch must contain a consumed field")
        if any(getattr(self, field) is None for field in self.model_fields_set):
            raise ValueError("settings patch fields must not be null")
        return self


class UnlabeledProject(StrictUnlabeledRecord):
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    revision: int = Field(ge=1, le=MAX_REVISION)
    settings: UnlabeledSettings
    project_status: Literal[
        "pending", "splitting", "planning", "generating", "rendering",
        "completed", "failed", "cancelled",
    ]
    current_artifact_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)


class UnlabeledInitialJob(StrictUnlabeledRecord):
    id: int = Field(ge=1, le=MAX_DATABASE_ID)
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    status: Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]
    input_revision: int = Field(ge=1, le=MAX_REVISION)
    cancel_requested: bool
    input_settings: UnlabeledSettings
    kind: Literal["full", "rerender"]
    block_index: int | None = Field(default=None, ge=0, le=100000)
    parent_job_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)


class UnlabeledInitialHistory(StrictUnlabeledRecord):
    project_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    revision: int = Field(ge=1, le=MAX_REVISION)
    settings: UnlabeledSettings
    changed_fields: tuple[Literal[
        "subtitle_font_size", "voicevox_speed_scale", "voicevox_pitch_scale",
        "voicevox_speaker_id", "pronunciation_overrides", "narration_pacing_mode",
        "narration_sentence_pause_seconds",
    ], ...] = Field(default=(), max_length=7, strict=False)
    restored_from_revision: int | None = Field(default=None, ge=1, le=MAX_REVISION)


class UnlabeledInitialArtifact(StrictUnlabeledRecord):
    id: int = Field(ge=1, le=MAX_DATABASE_ID)
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    job_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    revision: int | None = Field(default=None, ge=1, le=MAX_REVISION)
    file_content_hex: str = Field(default="78", max_length=8192, pattern=r"^(?:[0-9a-f]{2})*$")
    file_size: int = Field(default=1, ge=0, le=4096)
    file_sha256: str = Field(default=hashlib.sha256(b"x").hexdigest(), pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def content_identity_matches(self) -> Self:
        content = bytes.fromhex(self.file_content_hex)
        if len(content) != self.file_size or hashlib.sha256(content).hexdigest() != self.file_sha256:
            raise ValueError("artifact content identity mismatch")
        return self


class UnlabeledInitialReceipt(StrictUnlabeledRecord):
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    operation_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9.-]*$")
    operation_version: int = Field(ge=1, le=100)
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    base_revision: int = Field(ge=1, le=MAX_REVISION)
    result_revision: int = Field(ge=1, le=MAX_REVISION)
    generation_requested: bool
    job_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    canonical_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_matches_worker_seed(self) -> Self:
        canonical_request = {
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "project_id": self.project_id,
            "base_revision": self.base_revision,
        }
        result = {
            "operation_id": self.operation_id,
            "project_id": self.project_id,
            "revision": self.result_revision,
            "changed": self.result_revision != self.base_revision,
        }
        if (hashlib.sha256(_canonical_json_bytes(canonical_request)).hexdigest()
                != self.canonical_request_sha256
                or hashlib.sha256(_canonical_json_bytes(result)).hexdigest() != self.result_sha256):
            raise ValueError("receipt identity mismatch")
        return self


class UnlabeledInitialExternalCall(StrictUnlabeledRecord):
    id: int = Field(ge=1, le=MAX_DATABASE_ID)
    job_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    fingerprint: str = Field(min_length=1, max_length=64)
    provider: Literal["synthetic"] = "synthetic"
    endpoint: Literal["https://synthetic.invalid/d36"] = "https://synthetic.invalid/d36"
    remote_side_effect: bool
    status: Literal["in_flight", "succeeded", "failed", "unknown"]
    attempts: int = Field(default=1, ge=1, le=100)
    response_status: int | None = Field(default=None, ge=100, le=599)
    response_body_hex: str | None = Field(default=None, max_length=8192, pattern=r"^(?:[0-9a-f]{2})*$")
    response_body_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response_content_type: Literal["application/json"] | None = None
    provider_response_id: str | None = Field(default=None, min_length=1, max_length=128)
    error_code: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")

    @model_validator(mode="after")
    def response_identity_matches(self) -> Self:
        if (self.response_body_hex is None) != (self.response_body_sha256 is None):
            raise ValueError("external response body and hash must be supplied together")
        if self.response_body_hex is not None:
            content = bytes.fromhex(self.response_body_hex)
            if hashlib.sha256(content).hexdigest() != self.response_body_sha256:
                raise ValueError("external response identity mismatch")
        return self


class UnlabeledOperationArguments(StrictUnlabeledRecord):
    settings: UnlabeledSettingsPatch | None = None
    value: int | None = Field(default=None, ge=16, le=120)
    delta: int | None = Field(default=None, ge=-104, le=104)
    subtitle_font_size: int | None = Field(default=None, ge=16, le=120)
    subtitle_font_size_delta: int | None = Field(default=None, ge=-104, le=104)
    voicevox_speed_scale: float | None = Field(default=None, ge=0.5, le=2.0)
    voicevox_pitch_scale: float | None = Field(default=None, ge=-1.0, le=1.0)
    voicevox_speaker_id: int | None = Field(default=None, ge=0, le=100000)
    pronunciation_overrides: UnlabeledPronunciations | None = Field(default=None, max_length=100, strict=False)
    narration_pacing_mode: Literal["adaptive", "fixed"] | None = None
    narration_sentence_pause_seconds: float | None = Field(default=None, ge=0, le=5)
    revision: int | None = Field(default=None, ge=1, le=MAX_REVISION)
    job_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    block_index: int | None = Field(default=None, ge=0, le=100000)
    kind: Literal["full", "rerender"] | None = None


class UnlabeledOperationProposal(StrictUnlabeledRecord):
    kind: Literal["operation"]
    operation_id: Literal[
        "project.subtitle-font-size.set", "project.subtitle-font-size.adjust",
        "project.settings.update", "project.status.get", "project.generation.start",
        "project.generation.cancel", "project.generation.retry", "project.settings.restore",
    ]
    operation_version: int = Field(ge=1, le=2)
    arguments: UnlabeledOperationArguments
    generate_after_save: bool

    @model_validator(mode="after")
    def arguments_match_operation(self) -> Self:
        supplied = set(self.arguments.model_dump(exclude_none=True))
        allowed = {
            "project.subtitle-font-size.set": {"value"},
            "project.subtitle-font-size.adjust": {"delta"},
            "project.settings.update": ({
                "subtitle_font_size", "voicevox_speed_scale", "voicevox_pitch_scale", "voicevox_speaker_id",
                "pronunciation_overrides", "narration_pacing_mode", "narration_sentence_pause_seconds",
            } if self.operation_version == 1 else {"settings", "subtitle_font_size_delta"}),
            "project.status.get": set(),
            "project.generation.start": {"kind", "block_index"},
            "project.generation.cancel": {"job_id"},
            "project.generation.retry": {"job_id"},
            "project.settings.restore": {"revision"},
        }[self.operation_id]
        if supplied - allowed:
            raise ValueError("proposal arguments do not belong to operation")
        return self


class UnlabeledClarificationProposal(StrictUnlabeledRecord):
    kind: Literal["clarification"]
    question: str = Field(min_length=1, max_length=240)
    missing_fields: tuple[Literal["target", "arguments", "intent"], ...] = Field(
        min_length=1, max_length=3, strict=False,
    )


class UnlabeledUnsupportedProposal(StrictUnlabeledRecord):
    kind: Literal["unsupported"]
    reason: str = Field(min_length=1, max_length=240, pattern=r"\S")


class UnlabeledNoOperationProposal(StrictUnlabeledRecord):
    kind: Literal["no_operation"]
    reason: str = Field(min_length=1, max_length=240, pattern=r"\S")


UnlabeledPriorProposal = Annotated[
    UnlabeledOperationProposal | UnlabeledClarificationProposal
    | UnlabeledUnsupportedProposal | UnlabeledNoOperationProposal,
    Field(discriminator="kind"),
]


class UnlabeledInitialPriorTurn(StrictUnlabeledRecord):
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    base_revision: int = Field(ge=1, le=MAX_REVISION)
    status: Literal["ready", "needs_input", "unsupported", "blocked", "error", "completed", "dismissed"]
    question: str | None = Field(default=None, min_length=1, max_length=240)
    result_revision: int | None = Field(default=None, ge=1, le=MAX_REVISION)
    settings_saved: bool
    proposal: UnlabeledPriorProposal | None = None
    text: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    relation: Literal["answer", "correction", "dismiss"] | None = None
    parent_request_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    )
    successor_request_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
    )


class UnlabeledInitialState(StrictUnlabeledRecord):
    project_id: int = Field(ge=1, le=MAX_DATABASE_ID)
    revision: int = Field(ge=1, le=MAX_REVISION)
    settings: UnlabeledSettings
    project_status: Literal[
        "pending", "splitting", "planning", "generating", "rendering",
        "completed", "failed", "cancelled",
    ]
    current_artifact_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    additional_projects: tuple[UnlabeledProject, ...] = Field(default=(), max_length=16, strict=False)
    jobs: tuple[UnlabeledInitialJob, ...] = Field(max_length=32, strict=False)
    history: tuple[UnlabeledInitialHistory, ...] = Field(max_length=32, strict=False)
    artifact_revisions: tuple[Annotated[int, Field(ge=1, le=MAX_REVISION)], ...] = Field(
        default=(), max_length=32, strict=False
    )
    artifacts: tuple[UnlabeledInitialArtifact, ...] = Field(default=(), max_length=32, strict=False)
    receipts: tuple[UnlabeledInitialReceipt, ...] = Field(default=(), max_length=32, strict=False)
    external_calls: tuple[UnlabeledInitialExternalCall, ...] = Field(default=(), max_length=32, strict=False)
    prior_turns: tuple[UnlabeledInitialPriorTurn, ...] = Field(max_length=8, strict=False)

    @model_validator(mode="after")
    def seed_graph_is_consistent(self) -> Self:
        projects = {
            self.project_id: (self.revision, self.current_artifact_id),
            **{
                item.project_id: (item.revision, item.current_artifact_id)
                for item in self.additional_projects
            },
        }
        if len(projects) != 1 + len(self.additional_projects):
            raise ValueError("project IDs must be unique")

        collections = (
            ("job", [item.id for item in self.jobs]),
            ("artifact", [item.id for item in self.artifacts]),
            ("receipt", [item.request_id for item in self.receipts]),
            ("external call", [item.id for item in self.external_calls]),
            ("prior turn", [item.request_id for item in self.prior_turns]),
            ("history", [(item.project_id or self.project_id, item.revision) for item in self.history]),
        )
        for name, identities in collections:
            if len(identities) != len(set(identities)):
                raise ValueError(f"{name} identities must be unique")

        history_ids = {(item.project_id or self.project_id, item.revision) for item in self.history}
        for item in self.history:
            owner_id = item.project_id or self.project_id
            if owner_id not in projects:
                raise ValueError("history references unknown project")
            if item.revision > projects[owner_id][0]:
                raise ValueError("history revision exceeds owner revision")
            if (item.restored_from_revision is not None
                    and (owner_id, item.restored_from_revision) not in history_ids):
                raise ValueError("history restore references unknown revision")

        jobs = {item.id: item for item in self.jobs}
        for item in self.jobs:
            if item.project_id not in projects:
                raise ValueError("job references unknown project")
            owner_revision = projects[item.project_id][0]
            if item.input_revision > owner_revision:
                raise ValueError("job revision exceeds owner revision")
            if item.status in {"pending", "running"} and item.input_revision != owner_revision:
                raise ValueError("active job input revision must equal owner revision")
            if item.parent_job_id is not None:
                parent = jobs.get(item.parent_job_id)
                if parent is None:
                    raise ValueError("parent job does not exist")
                if parent.project_id != item.project_id:
                    raise ValueError("parent job ownership mismatch")
        for item in self.jobs:
            seen: set[int] = set()
            current: UnlabeledInitialJob | None = item
            while current is not None:
                if current.id in seen:
                    raise ValueError("parent job cycle")
                seen.add(current.id)
                current = jobs.get(current.parent_job_id) if current.parent_job_id is not None else None

        next_artifact_id = max((item.id for item in self.artifacts), default=0) + 1
        derived_ids = list(range(next_artifact_id, next_artifact_id + len(self.artifact_revisions)))
        if derived_ids and derived_ids[-1] > MAX_DATABASE_ID:
            raise ValueError("derived artifact IDs exceed database range")
        artifact_ids = [item.id for item in self.artifacts] + derived_ids
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact identities must be unique across representations")
        if any(revision > self.revision for revision in self.artifact_revisions):
            raise ValueError("artifact revision exceeds owner revision")
        artifacts = {item.id: item for item in self.artifacts}
        artifacts.update({
            artifact_id: UnlabeledInitialArtifact(
                id=artifact_id,
                project_id=self.project_id,
                revision=revision,
            )
            for artifact_id, revision in zip(derived_ids, self.artifact_revisions, strict=True)
        })
        artifact_job_ids = [item.job_id for item in self.artifacts if item.job_id is not None]
        if len(artifact_job_ids) != len(set(artifact_job_ids)):
            raise ValueError("artifact job identities must be unique")
        for item in self.artifacts:
            if item.project_id not in projects:
                raise ValueError("artifact references unknown project")
            if item.revision is not None and item.revision > projects[item.project_id][0]:
                raise ValueError("artifact revision exceeds owner revision")
            if item.job_id is not None:
                job = jobs.get(item.job_id)
                if job is None:
                    raise ValueError("artifact references unknown job")
                if job.project_id != item.project_id:
                    raise ValueError("artifact job ownership mismatch")
        for project_id, (_, current_artifact_id) in projects.items():
            if current_artifact_id is None:
                continue
            artifact = artifacts.get(current_artifact_id)
            if artifact is None:
                raise ValueError("current artifact does not exist")
            if artifact.project_id != project_id:
                raise ValueError("current artifact ownership mismatch")

        receipt_job_ids = [item.job_id for item in self.receipts if item.job_id is not None]
        if len(receipt_job_ids) != len(set(receipt_job_ids)):
            raise ValueError("receipt job identities must be unique")
        for item in self.receipts:
            if item.project_id not in projects:
                raise ValueError("receipt references unknown project")
            if max(item.base_revision, item.result_revision) > projects[item.project_id][0]:
                raise ValueError("receipt revision exceeds owner revision")
            if item.job_id is not None:
                job = jobs.get(item.job_id)
                if job is None:
                    raise ValueError("receipt references unknown job")
                if job.project_id != item.project_id:
                    raise ValueError("receipt job ownership mismatch")

        call_ids = [(item.job_id, item.fingerprint) for item in self.external_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("external call job fingerprints must be unique")
        if any(item.job_id not in jobs for item in self.external_calls):
            raise ValueError("external call references unknown job")

        turns = {item.request_id: item for item in self.prior_turns}
        explicit_links = any(
            {"parent_request_id", "successor_request_id"}.intersection(item.model_fields_set)
            for item in self.prior_turns
        )
        if explicit_links:
            parent_ids = {item.request_id: item.parent_request_id for item in self.prior_turns}
            successor_ids = {item.request_id: item.successor_request_id for item in self.prior_turns}
        else:
            parent_ids = {
                item.request_id: self.prior_turns[index - 1].request_id if index else None
                for index, item in enumerate(self.prior_turns)
            }
            successor_ids = {
                item.request_id: (
                    self.prior_turns[index + 1].request_id
                    if index + 1 < len(self.prior_turns) else None
                )
                for index, item in enumerate(self.prior_turns)
            }
        for item in self.prior_turns:
            if item.project_id not in projects:
                raise ValueError("prior turn references unknown project")
            if max(item.base_revision, item.result_revision or 1) > projects[item.project_id][0]:
                raise ValueError("prior turn revision exceeds owner revision")
            if isinstance(item.proposal, UnlabeledOperationProposal):
                proposal_job_id = item.proposal.arguments.job_id
                if proposal_job_id is not None:
                    proposal_job = jobs.get(proposal_job_id)
                    if proposal_job is None:
                        raise ValueError("prior turn proposal references unknown job")
                    if proposal_job.project_id != item.project_id:
                        raise ValueError("prior turn proposal job ownership mismatch")
                proposal_revision = item.proposal.arguments.revision
                if (proposal_revision is not None
                        and (item.project_id, proposal_revision) not in history_ids):
                    raise ValueError("prior turn proposal references unknown history")
            parent_id = parent_ids[item.request_id]
            successor_id = successor_ids[item.request_id]
            if parent_id is not None and parent_id not in turns:
                raise ValueError("prior turn parent does not exist")
            if successor_id is not None and successor_id not in turns:
                raise ValueError("prior turn successor does not exist")
        for item in self.prior_turns:
            parent_id = parent_ids[item.request_id]
            successor_id = successor_ids[item.request_id]
            if parent_id is not None and turns[parent_id].project_id != item.project_id:
                raise ValueError("prior turn link ownership mismatch")
            if successor_id is not None and turns[successor_id].project_id != item.project_id:
                raise ValueError("prior turn link ownership mismatch")
            if parent_id is not None and successor_ids[parent_id] != item.request_id:
                raise ValueError("prior turn links must be reciprocal")
            if successor_id is not None and parent_ids[successor_id] != item.request_id:
                raise ValueError("prior turn links must be reciprocal")
        for item in self.prior_turns:
            seen: set[str] = set()
            current_id: str | None = item.request_id
            while current_id is not None:
                if current_id in seen:
                    raise ValueError("prior turn cycle")
                seen.add(current_id)
                current_id = parent_ids[current_id]
        return self


class UnlabeledContinuation(StrictUnlabeledRecord):
    parent_request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    relation: Literal["answer", "correction", "dismiss"]


class UnlabeledRequest(StrictUnlabeledRecord):
    request_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    text: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    target_project_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)
    base_revision: int | None = Field(default=None, ge=1, le=MAX_REVISION)
    continuation: UnlabeledContinuation | None = None


class _RequestEvent(StrictUnlabeledRecord):
    request: UnlabeledRequest


class UnlabeledNormalEvent(_RequestEvent):
    kind: Literal["none", "resend_identical", "restart_resend", "concurrent_identical", "confirm_generation", "confirm_twice"]


class UnlabeledDifferentBodyEvent(_RequestEvent):
    kind: Literal["same_id_different_body"]
    replacement_text: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    replacement_target_project_id: int | None = Field(default=None, ge=1, le=MAX_DATABASE_ID)


class UnlabeledRevisionRaceEvent(_RequestEvent):
    kind: Literal["revision_race"]
    external_revision: int = Field(ge=1, le=MAX_REVISION)
    external_settings: UnlabeledSettingsPatch
    # before_execution: the external save lands while the request is interpreted.
    # after_submit_before_confirmation: it lands after the request's own save, and
    # the original confirmation is then attempted.
    timing: Literal["before_execution", "after_submit_before_confirmation"] = "before_execution"


class UnlabeledSwitchTargetEvent(_RequestEvent):
    kind: Literal["switch_target"]
    selected_project_id_after: int = Field(ge=1, le=MAX_DATABASE_ID)
    action: Literal["read_original_request"]


UnlabeledEvent = Annotated[
    UnlabeledNormalEvent | UnlabeledDifferentBodyEvent | UnlabeledRevisionRaceEvent | UnlabeledSwitchTargetEvent,
    Field(discriminator="kind"),
]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def canonical_case_sha256(value: object) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude={"case_sha256"}, exclude_unset=True)
    elif isinstance(value, dict):
        payload = {key: item for key, item in value.items() if key != "case_sha256"}
    else:
        raise TypeError("case must be a model or mapping")
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


class UnlabeledTrialCase(StrictUnlabeledRecord):
    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_primitive(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("unlabeled schema version must be an integer")
        return value

    schema_version: Literal[1]
    case_id: str = Field(pattern=r"^D24-H\d{3}$")
    group_id: str = Field(pattern=r"^D24-HG\d{2}$")
    category: Literal[
        "paraphrase", "negation", "omission", "correction", "compound", "blocked",
        "unsupported", "resend", "race", "boundary", "target", "confirmation",
        "history", "failure", "status",
    ]
    split: Literal["held_out"]
    event: UnlabeledEvent
    initial: UnlabeledInitialState
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def case_references_seeded_graph(self) -> Self:
        projects = {
            self.initial.project_id: self.initial.revision,
            **{item.project_id: item.revision for item in self.initial.additional_projects},
        }
        request = self.event.request
        if request.target_project_id is not None:
            if request.target_project_id not in projects:
                raise ValueError("request target references unknown project")
            if request.base_revision is not None and request.base_revision > projects[request.target_project_id]:
                raise ValueError("request revision exceeds target revision")
        turns = {item.request_id: item for item in self.initial.prior_turns}
        if request.request_id in turns:
            raise ValueError("request would create prior turn cycle")
        if request.continuation is not None:
            parent = turns.get(request.continuation.parent_request_id)
            if parent is None:
                raise ValueError("continuation parent does not exist")
            explicit_links = any(
                {"parent_request_id", "successor_request_id"}.intersection(item.model_fields_set)
                for item in self.initial.prior_turns
            )
            if explicit_links:
                successor_id = parent.successor_request_id
            else:
                parent_index = self.initial.prior_turns.index(parent)
                successor_id = (
                    self.initial.prior_turns[parent_index + 1].request_id
                    if parent_index + 1 < len(self.initial.prior_turns) else None
                )
            if successor_id is not None:
                raise ValueError("continuation parent already has successor")
            # A continuation naming another project is a product-rule case
            # (dialogue_target_mismatch); the product, not the seed, decides it.
        if isinstance(self.event, UnlabeledRevisionRaceEvent):
            primary_history_revisions = {
                item.revision
                for item in self.initial.history
                if (item.project_id or self.initial.project_id) == self.initial.project_id
            }
            if self.event.external_revision in primary_history_revisions:
                raise ValueError("external revision collides with seeded history")
            if self.event.external_revision <= self.initial.revision:
                raise ValueError("external revision must exceed primary initial revision")
            # After submit, the external save follows the request's own save when it
            # saved (initial+2), or the unchanged initial revision when it only prepared.
            offsets = {1, 2} if self.event.timing == "after_submit_before_confirmation" else {1}
            if (self.initial.revision > MAX_REVISION - max(offsets)
                    or self.event.external_revision - self.initial.revision not in offsets):
                raise ValueError("external revision must follow the primary revision for its timing")
        if isinstance(self.event, UnlabeledDifferentBodyEvent):
            replacement_target = self.event.replacement_target_project_id
            if replacement_target is not None and replacement_target not in projects:
                raise ValueError("replacement target references unknown project")
        return self

    @model_validator(mode="after")
    def verify_content_hash(self) -> Self:
        if self.case_sha256 != canonical_case_sha256(self):
            raise ValueError("case_sha256 does not match canonical case content")
        return self
