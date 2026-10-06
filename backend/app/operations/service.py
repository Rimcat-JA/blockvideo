"""Validate, resolve and commit one operation with durable replay protection."""
from __future__ import annotations

from typing import Any
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.models.operation_request import OperationReceipt
from app.models.project import Project
from app.operations.catalog import CatalogError, OperationCatalog, validate_arguments
from app.operations.contracts import (
    OperationDefinition,
    OperationRequest,
    OperationResult,
    Readiness,
    ReadinessResult,
)
from app.operations.readiness import evaluate_readiness, resolve_project
from app.operations.errors import OperationError
from app.operations.receipts import canonical_request, find_replay, save_receipt
from app.operations.registry import HandlerRegistry, RegistryError
from app.services.transactions import WriteBusyError, atomic_write
from app.services.job_records import UnresolvedExternalWorkError

_ADJUST = "project.subtitle-font-size.adjust"
_SET = "project.subtitle-font-size.set"
_LEGACY = {_SET, "project.status.get"}
_GENERATING_SETTINGS = {_SET, _ADJUST, "project.settings.update"}


class OperationService:
    """Own operation lookup, validation, final readiness, and dispatch order."""

    def __init__(self, catalog: OperationCatalog, registry: HandlerRegistry) -> None:
        self._catalog = catalog.model_copy(deep=True)
        self._registry = registry
        definition_keys = {item.handler_key for item in self._catalog.definitions}
        for key in definition_keys:
            registry.require(key)
        unreferenced = registry.keys() - definition_keys
        if unreferenced:
            raise RegistryError(f"registered handlers are not defined: {sorted(unreferenced)}")

    def list_definitions(self) -> list[OperationDefinition]:
        """Return detached catalog definitions in stable operation-ID order."""
        return [item.model_copy(deep=True) for item in self._catalog.definitions]

    def get_result(self, db: Session, request_id: str) -> OperationResult:
        """Return the original acknowledgement even if the project changed."""
        receipt = db.get(OperationReceipt, request_id)
        if receipt is None:
            raise OperationError("request_not_found", "committed request not found")
        return OperationResult.model_validate(receipt.result_json)

    def _definition_and_arguments(
        self, request: OperationRequest
    ) -> tuple[OperationDefinition, dict[str, Any]]:
        try:
            definition = self._catalog.require(
                request.operation_id, request.operation_version
            )
        except CatalogError as exc:
            raise OperationError("operation_not_found", str(exc)) from exc
        try:
            arguments = validate_arguments(definition, request.arguments)
        except CatalogError as exc:
            raise OperationError("invalid_arguments", str(exc)) from exc
        if request.generation_requested and request.operation_id not in _GENERATING_SETTINGS:
            raise OperationError("invalid_generation_request", "この操作では生成フラグを指定できません")
        return definition, arguments

    def _current_readiness(
        self, db: Session, definition: OperationDefinition, request: OperationRequest,
    ) -> ReadinessResult:
        result = evaluate_readiness(db, definition, request.target)
        if result.readiness != Readiness.ready:
            return result
        durable_write = request.operation_id not in _LEGACY or (definition.precondition_key == "project_editable" and (
            request.request_id is not None or request.base_revision is not None
            or request.operation_id == _ADJUST or request.generation_requested
        ))
        missing = []
        if durable_write:
            if request.request_id is None:
                missing.append("request_id")
            if request.base_revision is None:
                missing.append("base_revision")
        if missing:
            return result.model_copy(update={
                "readiness": Readiness.needs_input, "reason_code": "request_metadata_required",
                "missing_fields": missing,
            })
        stale = (
            request.base_revision is not None and request.base_revision != result.revision
        ) or (
            request.observed_state_revision is not None
            and request.observed_state_revision != result.state_revision
        )
        if stale:
            return result.model_copy(update={"readiness": Readiness.blocked, "reason_code": "stale_state"})
        return result

    def _resolve_arguments(
        self, project: Project, request: OperationRequest, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if request.operation_id == "project.settings.update" and request.operation_version == 2:
            resolved = dict(arguments["settings"])
            delta = arguments["subtitle_font_size_delta"]
            if delta is not None:
                if "subtitle_font_size" in resolved:
                    raise OperationError("invalid_arguments", "字幕サイズの絶対値と増減は同時に指定できません")
                resolved["subtitle_font_size"] = project.subtitle_font_size + delta
            if not resolved:
                raise OperationError("invalid_arguments", "変更する設定がありません")
            try:
                return validate_arguments(self._catalog.require("project.settings.update", 1), resolved)
            except CatalogError as exc:
                raise OperationError("invalid_arguments", str(exc)) from exc
        if request.operation_id != _ADJUST:
            return arguments
        resolved = {"value": project.subtitle_font_size + arguments["delta"]}
        try:
            return validate_arguments(self._catalog.require(_SET, 1), resolved)
        except CatalogError as exc:
            raise OperationError("invalid_arguments", str(exc)) from exc

    def readiness(self, db: Session, request: OperationRequest) -> ReadinessResult:
        """Read-only preview; execution validates again under a writer lock."""
        replay = find_replay(db, request, canonical_request(request))
        if replay is not None:
            return ReadinessResult(operation_id=replay.operation_id, readiness=Readiness.ready,
                                   project_id=replay.project_id, state_revision=replay.state_revision,
                                   revision=replay.revision)
        definition, arguments = self._definition_and_arguments(request)
        result = self._current_readiness(db, definition, request)
        if result.readiness == Readiness.ready:
            project = resolve_project(db, request.target)
            if project is not None:
                self._resolve_arguments(project, request, arguments)
        return result

    @staticmethod
    def _require_ready(result: ReadinessResult) -> None:
        if result.readiness != Readiness.ready:
            raise OperationError(
                result.reason_code or "not_ready",
                f"operation is not ready: {result.readiness.value}",
                readiness=result.readiness,
                result=result,
            )

    def execute(self, db: Session, request: OperationRequest,
                *, before_dispatch: Callable[[Session], None] | None = None) -> OperationResult:
        """Commit settings, receipt and optional pending job as one SQLite write."""
        identity = canonical_request(request)
        try:
            with atomic_write(db):
                replay = find_replay(db, request, identity)
                if replay is not None:
                    return replay
                if before_dispatch is not None:
                    before_dispatch(db)
                definition, arguments = self._definition_and_arguments(request)
                self._require_ready(self._current_readiness(db, definition, request))
                project = resolve_project(db, request.target)
                if project is None:
                    raise OperationError("target_not_found", "project target disappeared")
                base_revision = project.revision
                resolved = self._resolve_arguments(project, request, arguments)
                self._require_ready(self._current_readiness(db, definition, request))
                handler = self._registry.require(definition.handler_key)
                result = handler(db, project, resolved).model_copy(update={
                    "operation_id": request.operation_id,
                    "base_revision": base_revision, "resolved_arguments": resolved,
                })
                result = save_receipt(db, request, identity, result,
                                      base_revision=base_revision, resolved_arguments=resolved)
            return result
        except WriteBusyError as exc:
            raise OperationError("database_busy", str(exc)) from exc
        except UnresolvedExternalWorkError as exc:
            raise OperationError("external_outcome_unknown", str(exc)) from exc
