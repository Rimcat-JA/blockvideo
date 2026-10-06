"""Canonical request identity and transaction-local receipt persistence."""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models.operation_request import OperationReceipt
from app.operations.contracts import OperationRequest, OperationResult
from app.operations.errors import OperationError
from app.services.job_records import create_pending_job


def canonical_request(request: OperationRequest) -> str:
    """Compare original intent, never its newly calculated absolute value."""
    payload = request.model_dump(mode="json", exclude={"request_id", "target"})
    payload["project_id"] = request.target.resolved_id
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise OperationError("invalid_arguments", "arguments must be finite JSON values") from exc


def find_replay(db: Session, request: OperationRequest, identity: str) -> OperationResult | None:
    """Look up durable identity before validating current state or resolving delta."""
    if request.request_id is None:
        return None
    receipt = db.get(OperationReceipt, request.request_id)
    if receipt is None:
        return None
    if receipt.canonical_request != identity:
        raise OperationError("request_id_conflict", "request ID already belongs to different content")
    return OperationResult.model_validate(receipt.result_json)


def save_receipt(
    db: Session, request: OperationRequest, identity: str, result: OperationResult,
    *, base_revision: int, resolved_arguments: dict[str, Any],
) -> OperationResult:
    """Flush the pending job and receipt without committing either separately."""
    if request.request_id is None:
        return result
    job_id: int | None = result.job_id
    if request.generation_requested:
        job = create_pending_job(db, result.project_id)
        job_id = job.id
    result_ref = f"/api/operations/requests/{request.request_id}"
    result = result.model_copy(update={
        "request_id": request.request_id, "base_revision": base_revision,
        "resolved_arguments": resolved_arguments,
        "generation_requested": request.generation_requested or result.generation_requested,
        "job_id": job_id, "result_ref": result_ref,
    })
    db.add(OperationReceipt(
        request_id=request.request_id, canonical_request=identity,
        operation_id=request.operation_id, operation_version=request.operation_version,
        project_id=result.project_id, base_revision=base_revision,
        result_revision=result.revision, resolved_arguments=resolved_arguments,
        generation_requested=result.generation_requested, job_id=job_id,
        result_ref=result_ref, result_json=result.model_dump(mode="json"),
    ))
    db.flush()
    return result
