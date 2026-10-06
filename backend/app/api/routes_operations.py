"""Thin structured HTTP entry for the validated operation core."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.logging import log
from app.db import get_db
from app.operations.bootstrap import operation_service
from app.operations.contracts import (
    OperationDefinition,
    OperationRequest,
    OperationResult,
    Readiness,
    ReadinessResult,
)
from app.operations.service import OperationError

router = APIRouter(prefix="/operations")


def _raise_http(exc: OperationError) -> None:
    """Map a domain rejection to one stable HTTP error body."""
    status = 404 if exc.reason_code in {"operation_not_found", "request_not_found", "job_not_found", "settings_revision_not_found",
                                          "artifact_not_found"} else 422
    if exc.reason_code in {"request_id_conflict", "external_outcome_unknown", "job_not_retryable"}:
        status = 409
    elif exc.reason_code == "database_busy":
        status = 503
    detail: dict[str, object] = {
        "reason_code": exc.reason_code,
        "message": str(exc),
    }
    if exc.result is not None:
        status = 409
        detail.update(
            {
                "readiness": exc.result.readiness.value,
                "missing_fields": exc.result.missing_fields,
                "project_id": exc.result.project_id,
                "state_revision": exc.result.state_revision,
                "revision": exc.result.revision,
            }
        )
    raise HTTPException(status_code=status, detail=detail,
                        headers={"Retry-After": "1"} if status == 503 else None) from exc


@router.get("", response_model=list[OperationDefinition])
def list_operations() -> list[OperationDefinition]:
    """Return versioned definitions generated from the package catalog."""
    return operation_service.list_definitions()


@router.get("/requests/{request_id}", response_model=OperationResult)
def get_operation_result(request_id: str, db: Session = Depends(get_db)) -> OperationResult:
    """Recover the original acknowledgement after a lost response or restart."""
    try:
        return operation_service.get_result(db, request_id)
    except OperationError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")


@router.post("/readiness", response_model=ReadinessResult)
def check_readiness(
    request: OperationRequest, db: Session = Depends(get_db)
) -> ReadinessResult:
    """Validate arguments and report current readiness without executing."""
    try:
        result = operation_service.readiness(db, request)
    except OperationError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    log.info(
        "operation readiness operation_id={operation_id} project_id={project_id} readiness={readiness} reason={reason}",
        operation_id=request.operation_id,
        project_id=result.project_id,
        readiness=result.readiness.value,
        reason=result.reason_code,
    )
    if result.readiness != Readiness.ready:
        _raise_http(
            OperationError(
                result.reason_code or "not_ready",
                f"operation is not ready: {result.readiness.value}",
                readiness=result.readiness,
                result=result,
            )
        )
    return result


@router.post("/execute", response_model=OperationResult)
def execute_operation(
    request: OperationRequest, db: Session = Depends(get_db)
) -> OperationResult:
    """Perform final validation and call one registered operation handler."""
    try:
        result = operation_service.execute(db, request)
    except OperationError as exc:
        _raise_http(exc)
        raise AssertionError("unreachable")
    log.info(
        "operation executed operation_id={operation_id} project_id={project_id} changed={changed}",
        operation_id=result.operation_id,
        project_id=result.project_id,
        changed=result.changed,
    )
    return result
