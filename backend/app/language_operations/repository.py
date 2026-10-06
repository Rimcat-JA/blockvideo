"""Short transactions for language identity and immutable prepared requests."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.interpretation.contracts import FailureView, MinimalState
from app.language_operations.context import resolve_context
from app.language_operations import dialogue
from app.language_operations.generation_followup import generation_request
from app.language_operations.contracts import LanguageDiagnostics, LanguageError, LanguageInput, LanguageResponse
from app.models.language_request import LanguageRequestRecord
from app.models.operation_request import OperationReceipt
from app.models.language_turn import LanguageTurn
from app.operations.contracts import OperationResult
from app.operations.receipts import canonical_request
from app.services.transactions import atomic_write


def digest(value: Any) -> str:
    # JSON may contain an escaped lone surrogate. Fingerprint it safely so the
    # interpreter can reject invalid Unicode without a 500 or echoed raw text.
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def input_fingerprint(request: LanguageInput) -> str:
    value = request.model_dump(mode="json", exclude={"request_id", "target"})
    if request.continuation is None:
        value.pop("continuation", None)  # Preserve D17/D18 input fingerprints.
    value["project_id"] = request.target.resolved_id
    return digest(value)


def response_for(db: Session, record: LanguageRequestRecord) -> LanguageResponse:
    """Receipt wins even after a crash between core commit and acknowledgement."""
    response = LanguageResponse.model_validate(record.response_json)
    turn = db.get(LanguageTurn, record.request_id)
    if turn:
        response = response.model_copy(update={"dialogue_available": True,
            "superseded_by": turn.successor_request_id})
    receipt = db.get(OperationReceipt, record.core_request_id)
    if receipt is not None:
        if response.prepared_request is None or receipt.canonical_request != canonical_request(response.prepared_request):
            raise LanguageError("core_request_conflict", "保存済みの実行要求と結果が一致しません。新しい要求で確認してください。")
        result = OperationResult.model_validate(receipt.result_json)
        response = response.model_copy(update={"status": "completed", "result": result,
                                           "executed": True, "failure": None,
                                           "requires_confirmation": False})
        if response.generate_after_save:
            prepared = generation_request(response)
            generated = db.get(OperationReceipt, prepared.request_id)
            token = digest({"request_id": record.request_id, "prepared_request": prepared.model_dump(mode="json")})
            if generated is not None:
                if generated.canonical_request != canonical_request(prepared):
                    raise LanguageError("core_request_conflict", "保存済みの生成要求と結果が一致しません。")
                response = response.model_copy(update={"generation_request": prepared,
                    "generation_result": OperationResult.model_validate(generated.result_json),
                    "confirmation_token": token})
            else:
                stored = LanguageResponse.model_validate(record.response_json)
                blocked = stored.status == "blocked" or bool(response.superseded_by)
                response = response.model_copy(update={"status": "blocked" if blocked else "ready",
                    "generation_request": prepared, "confirmation_token": token,
                    "requires_confirmation": not blocked,
                    "failure": FailureView(reason_code="dialogue_superseded", message="この生成依頼は訂正・取り下げによって置き換えられています。")
                        if response.superseded_by else stored.failure if blocked else None})
    if response.plan:
        # Step receipts are authoritative, also after a crash between core commit and
        # acknowledgement: results are rebuilt from them in plan order.
        results = []
        for step in response.plan:
            stored_step = db.get(OperationReceipt, step.request_id)
            if stored_step is None:
                break
            results.append(OperationResult.model_validate(stored_step.result_json))
        response = response.model_copy(update={"plan_results": results})
        if len(results) == len(response.plan):
            response = response.model_copy(update={"status": "completed", "executed": True, "failure": None,
                                                   "requires_confirmation": False})
    return response


def _expire(record: LanguageRequestRecord) -> None:
    if record.status == "interpreting" and record.lease_until <= time.time():
        response = LanguageResponse.model_validate(record.response_json).model_copy(update={
            "status": "error", "failure": FailureView(reason_code="interpretation_interrupted",
            message="解釈処理が中断されました。自動再送はしていません。新しい要求として送ってください。"),
        })
        record.status = response.status
        record.response_json = response.model_dump(mode="json")


def lookup(db: Session, request_id: str) -> LanguageResponse:
    with atomic_write(db):
        record = db.get(LanguageRequestRecord, request_id)
        if record is None:
            raise LanguageError("request_not_found", "要求が見つかりません。", 404)
        _expire(record)
        response = response_for(db, record)
    return response


def claim(
    db: Session, request: LanguageInput,
) -> tuple[LanguageResponse, str | None, MinimalState | None]:
    """Return a single durable inference owner or an existing response."""
    fingerprint = input_fingerprint(request)
    with atomic_write(db):
        record = db.get(LanguageRequestRecord, request.request_id)
        if record is not None:
            if record.input_fingerprint != fingerprint:
                raise LanguageError("request_id_conflict", "この要求IDは別の内容に使われています。")
            _expire(record)
            return response_for(db, record), None, None
        owner = uuid.uuid4().hex
        response = LanguageResponse(request_id=request.request_id,
                                    core_request_id=f"nl-{uuid.uuid4().hex}", status="interpreting",
                                    project_id=request.target.resolved_id, base_revision=request.base_revision,
                                    diagnostics=LanguageDiagnostics(started_at=time.time()),
                                    parent_request_id=request.continuation.parent_request_id if request.continuation else None,
                                    relation=request.continuation.relation if request.continuation else None)
        state: MinimalState | None = None
        try:
            if request.continuation is not None:
                dialogue.require_not_superseded(
                    db, request.continuation.parent_request_id
                )
            state = resolve_context(db, request)
            parent_record = db.get(LanguageRequestRecord, request.continuation.parent_request_id) if request.continuation else None
            parent = response_for(db, parent_record) if parent_record else None
            dialogue.attach(db, request, state, parent)
            response = response.model_copy(update={"project_id": state.selected_project_id,
                                                   "base_revision": state.revision, "dialogue_available": True})
        except LanguageError as exc:
            state = None
            if exc.code == "target_required" and request.continuation is None:
                dialogue.attach(db, request, None, None)
            response = response.model_copy(update={
                "status": "needs_input" if exc.code == "target_required" else "blocked",
                "failure": FailureView(reason_code=exc.code, message=str(exc)),
                "dialogue_available": exc.code == "target_required" and request.continuation is None,
            })
        record = LanguageRequestRecord(
            request_id=request.request_id, input_fingerprint=fingerprint,
            core_request_id=response.core_request_id, project_id=response.project_id,
            base_revision=response.base_revision, status=response.status, owner_token=owner,
            created_at=time.time(), lease_until=time.time() + 210,
            response_json=response.model_dump(mode="json"),
        )
        db.add(record)
    return response, owner if state is not None else None, state


def finish_interpretation(
    db: Session, request_id: str, owner: str, response: LanguageResponse,
) -> LanguageResponse:
    with atomic_write(db):
        record = db.get(LanguageRequestRecord, request_id)
        if record is None:
            raise LanguageError("request_not_found", "要求が見つかりません。", 404)
        _expire(record)
        if record.owner_token != owner or record.status != "interpreting":
            return response_for(db, record)
        record.status = response.status
        record.response_json = response.model_dump(mode="json")
        record.request_json = response.prepared_request.model_dump(mode="json") if response.prepared_request else None
    return response


def acknowledge(db: Session, request_id: str, response: LanguageResponse) -> LanguageResponse:
    with atomic_write(db):
        record = db.get(LanguageRequestRecord, request_id)
        if record is None:
            raise LanguageError("request_not_found", "要求が見つかりません。", 404)
        observed = LanguageResponse.model_validate(record.response_json).diagnostics
        # Concurrent execution callers can recover the same receipt. Keep the
        # first durably observed duration for each phase instead of replacing it
        # with a later replay's duration. Missing crash-gap measurements stay null.
        response = response.model_copy(update={"diagnostics": response.diagnostics.model_copy(update={
            name: getattr(observed, name) for name in (
                "started_at", "interpretation_ms", "execution_ms", "generation_execution_ms",
            ) if getattr(observed, name) is not None
        })})
        record.status = response.status
        record.response_json = response.model_dump(mode="json")
        response = response_for(db, record)
        record.status = response.status
        record.response_json = response.model_dump(mode="json")
    return response
