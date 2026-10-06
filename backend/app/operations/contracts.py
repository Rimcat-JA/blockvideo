"""Transport-independent operation requests, readiness, and results."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.operations.limits import MAX_PROMPT_CANDIDATES


class Readiness(str, Enum):
    """Final operation availability after target, value, and state checks."""

    ready = "ready"
    needs_input = "needs_input"
    blocked = "blocked"
    unsupported = "unsupported"


class OperationTarget(BaseModel):
    """Project target supplied explicitly or by a selected UI context."""

    model_config = ConfigDict(extra="forbid")

    project_id: int | None = Field(default=None, ge=1, strict=True)
    selected_project_id: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def ids_must_agree(self) -> "OperationTarget":
        if (
            self.project_id is not None
            and self.selected_project_id is not None
            and self.project_id != self.selected_project_id
        ):
            raise ValueError("project_id and selected_project_id must agree")
        return self

    @property
    def resolved_id(self) -> int | None:
        """Return the single unambiguous project ID, if supplied."""
        return self.project_id if self.project_id is not None else self.selected_project_id


class OperationRequest(BaseModel):
    """A fully structured request accepted by readiness and execution."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1)
    operation_version: int = Field(default=1, ge=1, strict=True)
    target: OperationTarget = Field(default_factory=OperationTarget)
    arguments: dict[str, Any] = Field(default_factory=dict)
    observed_state_revision: str | None = None
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    base_revision: int | None = Field(default=None, ge=1, strict=True)
    generation_requested: bool = Field(default=False, strict=True)


class ReadinessResult(BaseModel):
    """Machine-readable reason why an operation can or cannot execute."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    readiness: Readiness
    reason_code: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    project_id: int | None = None
    state_revision: str | None = None
    revision: int | None = None


class CandidateReadiness(BaseModel):
    """Target/state hint before argument interpretation, never an execution permit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    operation_id: str = Field(min_length=1, max_length=128)
    operation_version: int = Field(ge=1)
    phase: Literal["candidate_preview"] = "candidate_preview"
    arguments_checked: Literal[False] = False
    readiness: Literal["ready", "needs_input", "blocked", "unsupported"]
    reason_code: Literal["target_required", "target_not_found", "project_busy",
                         "arguments_unchecked", "operation_not_found"] | None = None
    missing_fields: tuple[str, ...] = Field(default=(), max_length=32, strict=False)
    project_id: int | None = Field(default=None, ge=1)
    revision: int | None = Field(default=None, ge=1)


class CandidateReadinessSnapshot(BaseModel):
    """Time-bound observation; generation state can change without a new revision."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    observed_at: float = Field(ge=0)
    candidates: tuple[CandidateReadiness, ...] = Field(max_length=MAX_PROMPT_CANDIDATES, strict=False)


class OperationResult(BaseModel):
    """Non-secret output from one registered operation handler."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    project_id: int
    changed: bool
    state_revision: str
    data: dict[str, Any]
    revision: int = 1
    request_id: str | None = None
    base_revision: int | None = None
    resolved_arguments: dict[str, Any] = Field(default_factory=dict)
    generation_requested: bool = False
    job_id: int | None = None
    result_ref: str | None = None


class OperationDefinition(BaseModel):
    """One versioned operation definition loaded from the Git catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1, strict=True)
    operation_id: str = Field(pattern=r"^[a-z][a-z0-9.-]*$")
    operation_version: int = Field(ge=1, strict=True)
    description: str = Field(min_length=1)
    examples: list[str] = Field(min_length=1)
    input_schema: dict[str, Any]
    handler_key: str = Field(pattern=r"^[a-z][a-z0-9._-]*$")
    affected_artifacts: list[str]
    precondition_key: str = Field(min_length=1)
    postcondition_key: str = Field(min_length=1)
