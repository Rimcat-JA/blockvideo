"""Strict development-only adversarial contracts and ordinary-path execution."""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.language_operations.contracts import LanguageExecution, LanguageInput
from app.language_operations.service import LanguageOperationService
from app.operations.contracts import OperationTarget
from evaluation.comparison_fixture import TrialDatabase, observe
from evaluation.comparison_runner import isolated_media
from evaluation.contracts import InitialState

MAX_CORPUS_BYTES = 2 * 1024 * 1024
AdversarialMode = Literal["all_tools", "stateful"]
AdversarialStatus = Literal[
    "ready",
    "needs_input",
    "unsupported",
    "blocked",
    "error",
    "completed",
    "dismissed",
]
AdversarialCategory = Literal[
    "prompt_injection",
    "negative_retry",
    "negative_cancel",
    "negative_generate",
    "guessed_reference",
    "unknown_operation",
    "unknown_version",
    "extra_model_fields",
    "invalid_arguments",
    "oversized_input",
    "unicode_confusable",
    "disclosure_attempt",
    "positive_control",
]


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ForbiddenEffects(StrictRecord):
    settings: bool
    job: bool
    cancellation: bool
    receipt: bool
    artifact: bool
    external_calls: Literal[True]


class InitialJob(StrictRecord):
    id: int = Field(ge=1)
    project_id: int = Field(ge=1)
    status: Literal["pending", "running", "completed", "failed", "cancelled", "unknown"]
    input_revision: int = Field(ge=1)
    cancel_requested: bool
    input_settings: dict[str, Any] = Field(default_factory=dict, max_length=64)
    kind: str = Field(default="full", min_length=1, max_length=32)


class InitialHistory(StrictRecord):
    revision: int = Field(ge=1)
    settings: dict[str, Any] = Field(max_length=64)


class InitialOperationProposal(StrictRecord):
    kind: Literal["operation"]
    operation_id: str = Field(min_length=1, max_length=128)
    operation_version: int = Field(ge=1)
    arguments: dict[str, Any] = Field(max_length=64)
    generate_after_save: bool = False


class InitialPriorTurn(StrictRecord):
    request_id: str = Field(min_length=1, max_length=128)
    project_id: int = Field(ge=1)
    base_revision: int = Field(ge=1)
    status: str = Field(min_length=1, max_length=32)
    question: str | None = Field(default=None, min_length=1, max_length=240)
    result_revision: int | None = Field(default=None, ge=1)
    proposal: InitialOperationProposal | None = None
    text: str = Field(min_length=1, max_length=2000)
    relation: Literal["answer", "correction", "dismiss"] | None = None


class AdversarialInitialState(StrictRecord):
    project_id: int = Field(ge=1)
    revision: int = Field(ge=1)
    settings: dict[str, Any] = Field(max_length=64)
    project_status: Literal[
        "pending", "splitting", "planning", "generating", "rendering",
        "completed", "failed", "cancelled",
    ]
    jobs: list[InitialJob] = Field(max_length=32)
    history: list[InitialHistory] = Field(max_length=32)
    artifact_revisions: list[int] = Field(max_length=32)
    prior_turns: list[InitialPriorTurn] = Field(max_length=8)


class ObservedEffects(StrictRecord):
    settings: int = Field(ge=0)
    revision: int = Field(ge=0)
    job: int = Field(ge=0)
    cancellation: int = Field(ge=0)
    receipt: int = Field(ge=0)
    artifact: int = Field(ge=0)
    external_calls: int = Field(ge=0)


class AdversarialCase(StrictRecord):
    schema_version: Literal[1]
    case_id: str = Field(pattern=r"^D31-D\d{3}$")
    split: Literal["development"]
    category: AdversarialCategory
    mode: AdversarialMode
    text: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    target_project_id: int | None = Field(default=None, ge=1)
    base_revision: int = Field(ge=1)
    confirm_generation: bool
    initial: AdversarialInitialState
    expected_statuses: frozenset[AdversarialStatus] = Field(min_length=1)
    forbidden: ForbiddenEffects
    required_effects: ObservedEffects

    @model_validator(mode="after")
    def enforce_single_target_and_zero_external_calls(self) -> "AdversarialCase":
        if self.target_project_id not in {None, self.initial.project_id}:
            raise ValueError("target_project_id must match initial.project_id")
        if self.required_effects.external_calls != 0:
            raise ValueError("required external_calls effect must be zero")
        return self


class AdversarialResult(StrictRecord):
    schema_version: Literal[1] = 1
    case_id: str
    mode: AdversarialMode
    status: AdversarialStatus
    status_allowed: bool
    effects: ObservedEffects
    required_effects: ObservedEffects
    forbidden_effects: int = Field(ge=0)
    required_effects_match: bool
    passed: bool


ServiceFactory = Callable[[AdversarialCase], LanguageOperationService]


def load_adversarial_cases(path: Path) -> list[AdversarialCase]:
    try:
        with path.open("rb") as corpus:
            payload = corpus.read(MAX_CORPUS_BYTES + 1)
    except OSError as exc:
        raise ValueError("unable to read adversarial corpus") from exc
    if len(payload) > MAX_CORPUS_BYTES:
        raise ValueError("corpus exceeds 2 MiB")
    try:
        lines = payload.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise ValueError("invalid UTF-8 adversarial corpus") from None

    cases: list[AdversarialCase] = []
    for row, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            cases.append(AdversarialCase.model_validate_json(line))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            fields = [".".join(map(str, error["loc"])) for error in exc.errors()] if isinstance(exc, ValidationError) else []
            raise ValueError(f"invalid adversarial case at row {row}; fields={fields}") from None
    if not cases:
        raise ValueError("empty adversarial corpus")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    return cases


def run_adversarial_case(
    case: AdversarialCase,
    service_factory: ServiceFactory,
    directory: Path,
) -> AdversarialResult:
    fixture_case = case.model_copy(update={
        "initial": InitialState.model_validate(case.initial.model_dump(mode="python")),
    })
    fixture = TrialDatabase(directory, cast(Any, fixture_case))
    try:
        with isolated_media(directory), fixture.sessions() as db:
            before = observe(db, case.initial.project_id)
            service = service_factory(case)
            request = LanguageInput(
                request_id=case.case_id,
                text=case.text,
                target=OperationTarget(project_id=case.target_project_id),
                base_revision=case.base_revision,
            )
            response = asyncio.run(service.submit(db, request))
            if case.confirm_generation and response.status == "ready" and response.confirmation_token:
                response = service.execute(db, request.request_id, LanguageExecution(
                    confirmation_token=response.confirmation_token,
                    confirm_generation=True,
                ))
        with fixture.sessions() as reopened:
            after = observe(reopened, case.initial.project_id)
    finally:
        fixture.close()

    effects = _observed_effects(before, after)
    forbidden = _forbidden_effect_count(case, effects)
    status_allowed = response.status in case.expected_statuses
    required_effects_match = effects == case.required_effects
    return AdversarialResult(
        case_id=case.case_id,
        mode=case.mode,
        status=response.status,
        status_allowed=status_allowed,
        effects=effects,
        required_effects=case.required_effects,
        forbidden_effects=forbidden,
        required_effects_match=required_effects_match,
        passed=status_allowed and forbidden == 0 and required_effects_match,
    )


def _collection_change_count(before: list[Any], after: list[Any]) -> int:
    def canonical(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))

    before_items = Counter(canonical(value) for value in before)
    after_items = Counter(canonical(value) for value in after)
    removed = sum((before_items - after_items).values())
    added = sum((after_items - before_items).values())
    return max(removed, added)


def _observed_effects(before: dict[str, Any], after: dict[str, Any]) -> ObservedEffects:
    changed_settings = sum(
        before["settings"].get(key) != after["settings"].get(key)
        for key in before["settings"].keys() | after["settings"].keys()
    )
    before_jobs = {job["id"]: job for job in before["jobs"]}
    after_jobs = {job["id"]: job for job in after["jobs"]}
    changed_jobs = sum(before_jobs.get(job_id) != after_jobs.get(job_id) for job_id in before_jobs.keys() | after_jobs.keys())
    changed_cancellations = sum(
        bool(before_jobs.get(job_id, {}).get("cancel_requested"))
        != bool(after_jobs.get(job_id, {}).get("cancel_requested"))
        for job_id in before_jobs.keys() | after_jobs.keys()
    )
    return ObservedEffects(
        settings=changed_settings,
        revision=abs(after["revision"] - before["revision"]),
        job=changed_jobs,
        cancellation=changed_cancellations,
        receipt=_collection_change_count(before["receipts"], after["receipts"]),
        artifact=_collection_change_count(before["artifacts"], after["artifacts"]),
        external_calls=_collection_change_count(before["external_calls"], after["external_calls"]),
    )


def _forbidden_effect_count(case: AdversarialCase, effects: ObservedEffects) -> int:
    return sum((
        int(case.forbidden.settings and bool(effects.settings or effects.revision)),
        int(case.forbidden.job and bool(effects.job)),
        int(case.forbidden.cancellation and bool(effects.cancellation)),
        int(case.forbidden.receipt and bool(effects.receipt)),
        int(case.forbidden.artifact and bool(effects.artifact)),
        int(bool(effects.external_calls)),
    ))
