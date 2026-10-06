"""DB-owned observations for the read-only interpreter and explicit UI refresh."""
from __future__ import annotations

from time import time

from sqlalchemy.orm import Session

from app.interpretation.contracts import CandidateRef
from app.language_operations.contracts import LanguageResponse
from app.operations.candidate_readiness import candidate_snapshot
from app.operations.contracts import CandidateReadiness, CandidateReadinessSnapshot, OperationTarget
from app.operations.service import OperationService


def current_candidates(db: Session, core: OperationService, refs: tuple[CandidateRef, ...],
                       project_id: int | None) -> CandidateReadinessSnapshot:
    """End stale read transactions; never retain a DB transaction during inference."""
    db.rollback()
    db.expire_all()
    try:
        by_key = {(d.operation_id, d.operation_version): d for d in core.list_definitions()}
        valid = tuple(by_key[(r.operation_id, r.operation_version)] for r in refs
                      if (r.operation_id, r.operation_version) in by_key)
        snapshot = candidate_snapshot(db, valid, OperationTarget(project_id=project_id))
        known = {(r.operation_id, r.operation_version): r for r in snapshot.candidates}
        rows = tuple(known.get((r.operation_id, r.operation_version)) or CandidateReadiness(
            operation_id=r.operation_id, operation_version=r.operation_version,
            readiness="unsupported", reason_code="operation_not_found", project_id=project_id) for r in refs)
        return snapshot.model_copy(update={"candidates": rows})
    finally:
        db.rollback()
        db.expire_all()


def refresh_recorded_candidates(db: Session, core: OperationService,
                                response: LanguageResponse) -> CandidateReadinessSnapshot:
    trace = response.diagnostics.retrieval
    if trace is None or not trace.stages or trace.stages[-1].candidate_state is None:
        return CandidateReadinessSnapshot(observed_at=time(), candidates=())
    return current_candidates(db, core, trace.stages[-1].candidates, response.project_id)
