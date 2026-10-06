"""Bounded, strict values crossing the model interpretation boundary."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.operations.contracts import CandidateReadinessSnapshot
from app.operations.limits import MAX_CATALOG_OPERATIONS, MAX_PLAN_STEPS

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=240, pattern=r"\S")]


class StrictValue(BaseModel):
    """Reject implicit coercion and properties outside the declared contract."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CandidateRef(StrictValue):
    """An exact offered operation version; never a handler name."""

    operation_id: str = Field(min_length=1, max_length=128)
    operation_version: int = Field(default=1, ge=1)


class MinimalState(StrictValue):
    """Explicit allowlist, not an ORM serialization or authoritative readiness."""

    selected_project_id: int | None = Field(default=None, ge=1)
    revision: int | None = Field(default=None, ge=1)
    subtitle_font_size: int | None = Field(default=None, ge=16, le=120)
    status: Literal[
        "pending", "splitting", "planning", "generating", "rendering",
        "completed", "failed", "cancelled",
    ] | None = None


class InterpretationInput(StrictValue):
    """One interpretation request with explicitly selected candidates."""

    text: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    candidates: tuple[CandidateRef, ...] = Field(min_length=1, max_length=MAX_CATALOG_OPERATIONS)
    state: MinimalState = Field(default_factory=MinimalState)
    dialogue: tuple[DialogueContextTurn, ...] = Field(default=(), max_length=8)
    # Unattended mode: the model must not ask; it proposes its best guess instead.
    guess_missing: bool = False
    candidate_state: CandidateReadinessSnapshot | None = None

    @model_validator(mode="after")
    def annotation_scope(self) -> "InterpretationInput":
        if self.candidate_state is not None:
            refs = {(r.operation_id, r.operation_version) for r in self.candidates}
            rows = self.candidate_state.candidates
            keys = {(r.operation_id, r.operation_version) for r in rows}
            if keys != refs or len(keys) != len(rows):
                raise ValueError("candidate state must cover exactly the offered versions")
            if any(r.project_id != self.state.selected_project_id for r in rows):
                raise ValueError("candidate state target differs from selected target")
        return self


class OperationProposal(StrictValue):
    """Schema-valid suggestion, without target/receipt/revision/execution power."""

    kind: Literal["operation"]
    operation_id: str = Field(min_length=1, max_length=128)
    operation_version: int = Field(ge=1)
    arguments: dict[str, Any]
    generate_after_save: bool = False


class ClarificationProposal(StrictValue):
    """A question only; it cannot carry a partial executable operation."""

    kind: Literal["clarification"]
    question: ShortText
    missing_fields: list[Literal["target", "arguments", "intent"]] = Field(
        min_length=1, max_length=3,
    )


class UnsupportedProposal(StrictValue):
    """An explanation of an unsupported request, never executable text."""

    kind: Literal["unsupported"]
    reason: ShortText


class NoOperationProposal(StrictValue):
    """A negative or withdrawn intent, never a substitute executable operation."""

    kind: Literal["no_operation"]
    reason: ShortText


class PlanProposal(StrictValue):
    """Ordered steps the application runs one by one; each step is an ordinary
    operation proposal. The model never wires results between steps."""

    kind: Literal["plan"]
    steps: list[OperationProposal] = Field(min_length=2, max_length=MAX_PLAN_STEPS)


Proposal = Annotated[
    OperationProposal | ClarificationProposal | UnsupportedProposal | NoOperationProposal | PlanProposal,
    Field(discriminator="kind"),
]


class ProposalEnvelope(StrictValue):
    """The entire JSON response must conform to this envelope."""

    result: Proposal


class DialogueContextTurn(StrictValue):
    text: str = Field(max_length=2000)
    status: str = Field(max_length=32)
    proposal: Proposal | None = None
    question: ShortText | None = None
    relation: Literal["answer", "correction", "dismiss"] | None = None
    settings_saved: bool = False


InterpretationInput.model_rebuild()


class FailureView(StrictValue):
    """Safe fixed copy for CLI and future product rendering."""

    reason_code: str
    message: str
    http_status: int | None = Field(default=None, ge=100, le=599)


class InterpretationOutcome(StrictValue):
    """Interpretation state is distinct from readiness or execution success."""

    status: Literal["proposed", "needs_input", "unsupported", "error", "dismissed"]
    executed: Literal[False] = False
    proposal: Proposal | None = None
    failure: FailureView | None = None
    attempts: int = Field(default=0, ge=0, le=2)
    repair_codes: list[str] = Field(default_factory=list, max_length=1)
