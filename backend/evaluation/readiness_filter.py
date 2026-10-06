"""D28 experiment-only Hard Filter; never imported by product entry points."""
from __future__ import annotations

from app.interpretation.contracts import CandidateRef
from app.operations.contracts import CandidateReadinessSnapshot


def hard_filter_candidates(refs: tuple[CandidateRef, ...],
                           snapshot: CandidateReadinessSnapshot) -> tuple[CandidateRef, ...]:
    """Drop known unavailable states; unknown arguments remain eligible to ask.

    Callers must bind the returned exact refs as the entire experimental scope,
    including fallback. Empty output is valid and must never force an operation.
    No execution, ranking, authority checks, or prompt weakening happens here.
    """
    rows = {(r.operation_id, r.operation_version): r for r in snapshot.candidates}
    keys = {(r.operation_id, r.operation_version) for r in refs}
    if keys != set(rows) or len(rows) != len(snapshot.candidates) or len(keys) != len(refs):
        raise ValueError("Hard Filter requires an exact candidate-state snapshot")
    return tuple(r for r in refs if rows[(r.operation_id, r.operation_version)].readiness
                 not in {"blocked", "unsupported"})
