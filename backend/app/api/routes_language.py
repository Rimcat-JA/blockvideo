"""Development HTTP entry for All Tools interpretation and core execution."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
import unicodedata

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.transport import ModelMessage
from app.language_operations.contracts import LanguageError, LanguageExecution, LanguageInput, LanguageResponse
from app.language_operations.service import LanguageOperationService
from app.language_operations.candidate_state import refresh_recorded_candidates
from app.operations.contracts import CandidateReadinessSnapshot
from app.operations.bootstrap import operation_service
from app.semantic_interpretation.runtime import configured_semantic

_INVALID_REQUEST_DETAIL = {
    "reason_code": "invalid_request",
    "message": "入力の形式・値が正しくありません。要求ID、本文、対象を確認してください。",
}


def _contains_invalid_text(value: object) -> bool:
    if isinstance(value, str):
        return any(unicodedata.category(character) in {"Cc", "Cs"} for character in value)
    if isinstance(value, dict):
        return any(
            _contains_invalid_text(key) or _contains_invalid_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_invalid_text(item) for item in value)
    return False


class _LanguageRoute(APIRoute):
    """Do not echo untrusted utterances in HTTP validation errors."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def safe_handler(request: Request) -> Response:
            try:
                try:
                    payload = await request.json()
                except ValueError:
                    payload = None
                if _contains_invalid_text(payload):
                    raise HTTPException(status_code=422, detail=_INVALID_REQUEST_DETAIL)
                return await handler(request)
            except RequestValidationError:
                raise HTTPException(status_code=422, detail=_INVALID_REQUEST_DETAIL) from None

        return safe_handler


router = APIRouter(prefix="/language/requests", route_class=_LanguageRoute)


class _LocalAdapter:
    """Defer configuration validation/connection until a new interpretation."""

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        settings = get_settings()
        if not settings.language_model:
            raise InterpretationError("configuration_error")
        async with LocalChatAdapter(settings.language_base_url, settings.language_model,
                                    reasoning_effort=settings.language_reasoning_effort) as adapter:
            return await adapter.complete(messages, schema)


def language_service() -> LanguageOperationService:
    """Replay needs no working model connection or valid endpoint configuration."""
    settings = get_settings()
    semantic = configured_semantic(settings.language_retrieval_index, settings.language_retrieval_profile,
        settings.language_embedding_assets, settings.language_embedding_base_url,
        allow_all_tools=settings.language_retrieval_all_tools) if settings.language_retrieval_index else None
    return LanguageOperationService(operation_service, _LocalAdapter() if settings.language_model else None,
                                    review_all=settings.language_review_all, semantic=semantic,
                                    readiness_annotations=settings.language_retrieval_readiness,
                                    yolo_enabled=settings.language_yolo_enabled)


def _http_error(exc: LanguageError) -> None:
    raise HTTPException(status_code=exc.status_code, detail={"reason_code": exc.code, "message": str(exc)}) from None


@router.post("/prepare", response_model=LanguageResponse)
async def prepare_request(
    request: LanguageInput, db: Session = Depends(get_db),
    service: LanguageOperationService = Depends(language_service),
) -> LanguageResponse:
    try:
        return await service.prepare(db, request)
    except LanguageError as exc:
        _http_error(exc)
        raise AssertionError("unreachable")


@router.post("", response_model=LanguageResponse)
async def submit_request(
    request: LanguageInput, db: Session = Depends(get_db),
    service: LanguageOperationService = Depends(language_service),
) -> LanguageResponse:
    try:
        return await service.submit(db, request)
    except LanguageError as exc:
        _http_error(exc)
        raise AssertionError("unreachable")


@router.get("/{request_id}", response_model=LanguageResponse)
def get_request(request_id: str, db: Session = Depends(get_db)) -> LanguageResponse:
    try:
        return LanguageOperationService(operation_service).get(db, request_id)
    except LanguageError as exc:
        _http_error(exc)
        raise AssertionError("unreachable")


@router.post("/{request_id}/execute", response_model=LanguageResponse)
def execute_request(
    request_id: str, confirmation: LanguageExecution, db: Session = Depends(get_db),
) -> LanguageResponse:
    try:
        return LanguageOperationService(operation_service).execute(db, request_id, confirmation)
    except LanguageError as exc:
        _http_error(exc)
        raise AssertionError("unreachable")


@router.get("/{request_id}/candidate-readiness", response_model=CandidateReadinessSnapshot)
def get_candidate_readiness(request_id: str, db: Session = Depends(get_db)) -> CandidateReadinessSnapshot:
    """Current observation only; original response and replay remain immutable."""
    try:
        stored = LanguageOperationService(operation_service).get(db, request_id)
        return refresh_recorded_candidates(db, operation_service, stored)
    except LanguageError as exc:
        _http_error(exc)
        raise AssertionError("unreachable")
