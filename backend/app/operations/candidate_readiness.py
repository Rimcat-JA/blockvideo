"""Read-only provisional state; do not fabricate arguments to call the executor."""
from __future__ import annotations

from time import time

from sqlalchemy.orm import Session

from app.operations.contracts import (
    CandidateReadiness, CandidateReadinessSnapshot, OperationDefinition, OperationTarget,
)
from app.operations.readiness import evaluate_readiness


def candidate_snapshot(db: Session, definitions: tuple[OperationDefinition, ...],
                       target: OperationTarget) -> CandidateReadinessSnapshot:
    """Known state restrictions precede unknown argument/reference constraints."""
    rows: list[CandidateReadiness] = []
    observed_at = time()
    for definition in definitions:
        result = evaluate_readiness(db, definition, target)
        readiness = result.readiness.value
        reason = result.reason_code
        missing = tuple(result.missing_fields)
        # Even optional/default arguments may determine block/job/history validity.
        # Do not invent values or claim a full ready check before interpretation.
        if readiness == "ready" and definition.input_schema.get("properties"):
            readiness, reason = "needs_input", "arguments_unchecked"
            missing = tuple(definition.input_schema.get("required") or ["arguments"])
        rows.append(CandidateReadiness(operation_id=definition.operation_id,
            operation_version=definition.operation_version, readiness=readiness,
            reason_code=reason, missing_fields=missing, project_id=result.project_id,
            revision=result.revision))
    return CandidateReadinessSnapshot(observed_at=observed_at, candidates=tuple(rows))
