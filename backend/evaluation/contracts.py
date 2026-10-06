"""D24 synthetic cases and content-bound review records; no model execution."""
from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Provenance(StrictRecord):
    kind: Literal["new_synthetic", "prior_development"]
    reference: str = Field(min_length=1)


class InitialState(StrictRecord):
    project_id: int = Field(ge=1)
    revision: int = Field(ge=1)
    settings: dict[str, Any]
    project_status: str
    jobs: list[dict[str, Any]]
    history: list[dict[str, Any]]
    artifact_revisions: list[int]
    prior_turns: list[dict[str, Any]]


class Continuation(StrictRecord):
    parent_request_id: str = Field(min_length=1)
    relation: Literal["answer", "correction", "dismiss"]


class Request(StrictRecord):
    request_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)
    target_project_id: int | None = Field(ge=1)
    base_revision: int | None = Field(ge=1)
    continuation: Continuation | None


class Event(StrictRecord):
    kind: Literal["none", "resend_identical", "restart_resend", "same_id_different_body",
                  "concurrent_identical", "revision_race", "confirm_generation",
                  "confirm_twice", "switch_target"]
    details: dict[str, Any]


class Proposal(StrictRecord):
    operation_id: str
    operation_version: int = Field(ge=1)
    arguments: dict[str, Any]
    generate_after_save: bool


class Effects(StrictRecord):
    outcome: Literal["saved", "saved_awaiting_confirmation", "awaiting_confirmation", "queried",
                     "needs_input", "unsupported", "dismissed", "blocked", "replayed",
                     "cancel_requested", "cancelled", "unchanged", "generation_queued"]
    reason: str = Field(min_length=1)
    question_for: list[str]
    settings_delta: dict[str, Any]
    revision_delta: int = Field(ge=0, le=1)
    new_jobs: int = Field(ge=0, le=1)
    confirmation_required: bool
    job_assertions: dict[str, Any]
    artifact_policy: Literal["preserve_all_no_new_publication", "job_may_publish_on_success"]
    receipt_rule: Literal["new_request", "first_result", "same_id_conflict", "none"]

    @model_validator(mode="after")
    def consistent_effects(self) -> Self:
        if self.settings_delta and self.revision_delta != 1:
            raise ValueError("setting changes must create one revision")
        if self.outcome in {"needs_input", "unsupported", "dismissed"} and (
            self.settings_delta or self.new_jobs or self.job_assertions
        ):
            raise ValueError("non-execution labels cannot mutate settings or jobs")
        if self.outcome == "needs_input" and not self.question_for:
            raise ValueError("a clarification must identify missing information")
        if self.new_jobs and self.artifact_policy != "job_may_publish_on_success":
            raise ValueError("new job requires an explicit publication policy")
        return self


class Expected(StrictRecord):
    interpretation: Literal["operation", "clarification", "unsupported", "no_operation", "not_called"]
    operations: list[Proposal]
    target_project_id: int | None = Field(ge=1)
    submit: Effects
    after_event: Effects | None
    rationale: str = Field(min_length=1)
    rule_ids: list[str] = Field(min_length=1)


class Case(StrictRecord):
    schema_version: Literal[1]
    case_id: str = Field(pattern=r"^D24-[DH]\d{3}$")
    group_id: str = Field(pattern=r"^D24-[DH]G\d{2}$")
    split: Literal["development", "held_out"]
    source_request: str = Field(min_length=1)
    provenance: Provenance
    tags: list[Literal["paraphrase", "negation", "omission", "correction", "compound", "blocked",
                       "unsupported", "resend", "race", "boundary", "target", "confirmation",
                       "history", "failure", "status"]] = Field(min_length=1)
    situation: str = Field(min_length=1)
    initial: InitialState
    request: Request
    event: Event
    expected: Expected
    known_limitation: str | None = None

    @model_validator(mode="after")
    def consistent_case(self) -> Self:
        prefix = "D" if self.split == "development" else "H"
        if not self.case_id.startswith(f"D24-{prefix}") or not self.group_id.startswith(f"D24-{prefix}G"):
            raise ValueError("case/group ID does not match split")
        if self.split == "held_out" and self.provenance.kind != "new_synthetic":
            raise ValueError("prior development cannot become held-out")
        if (self.event.kind == "none") != (self.expected.after_event is None):
            raise ValueError("event and after_event must be present together")
        if (self.expected.interpretation == "operation") != bool(self.expected.operations):
            raise ValueError("only operation labels have accepted proposals")
        if self.request.target_project_id != self.expected.target_project_id:
            raise ValueError("expected target must stay bound to selected target")
        if not set(self.expected.rule_ids) <= {f"R{i:02d}" for i in range(1, 16)}:
            raise ValueError("unknown rubric rule")
        if self.expected.submit.new_jobs:
            raise ValueError("language submit cannot create a generation job without confirmation")
        restore_only = bool(self.expected.operations) and all(
            operation.operation_id == "project.settings.restore" for operation in self.expected.operations
        )
        for effect in (self.expected.submit, self.expected.after_event):
            if effect is None:
                continue
            if effect.revision_delta and not effect.settings_delta and not restore_only:
                raise ValueError("only a recorded restore can add a revision without changing values")
            if restore_only and effect.outcome in {"saved", "unchanged", "replayed"} and effect.revision_delta != 1:
                raise ValueError("a completed restore records one new revision even for identical values")
        return self


class ReviewEntry(StrictRecord):
    case_id: str
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["pending", "approved", "rejected"]
    note: str


class ReviewLedger(StrictRecord):
    schema_version: Literal[1]
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: Literal["human", "independent_ai"]
    reviewer: str
    reviewed_at: str
    entries: list[ReviewEntry]

    @model_validator(mode="after")
    def explicit_reviewer(self) -> Self:
        if any(entry.decision != "pending" for entry in self.entries):
            if not self.reviewer.strip() or not self.reviewed_at.strip():
                raise ValueError("decisions need reviewer attribution and timestamp")
        return self
