"""One real language/core path, fresh DB per mode; no media dispatcher or labels to model."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.interpretation.transport import StructuredAdapter
from app.language_operations.contracts import Continuation, LanguageExecution, LanguageInput
from app.language_operations.service import LanguageOperationService
from app.operations.bootstrap import operation_service
from app.operations.contracts import OperationTarget
from app.retrieval.serialization import canonical, digest
from app.semantic_interpretation.service import SemanticInterpreter
from evaluation.comparison import ComparisonInterpreter
from evaluation.comparison_fixture import TrialDatabase, observe, prompt_state
from evaluation.comparison_scoring import score_submit
from evaluation.contracts import Case
from evaluation.corpus import case_digest

_TRIAL_LOCK = threading.Lock()


@contextmanager
def isolated_media(directory: Path) -> Iterator[None]:
    """The bulk runner is sequential; concurrent workers must be separate processes."""
    if not _TRIAL_LOCK.acquire(blocking=False):
        raise ValueError("parallel trials in one process are unsupported")
    settings = get_settings()
    previous = settings.storage_root
    try:
        settings.storage_root = (directory / "media").resolve()
        yield
    finally:
        settings.storage_root = previous
        _TRIAL_LOCK.release()


async def run_trial(case: Case, mode: str, semantic: SemanticInterpreter | None,
                    adapter: StructuredAdapter, directory: Path) -> dict[str, Any]:
    if case.split != "development":
        raise ValueError("comparison runner refuses held-out cases")
    fixture = TrialDatabase(directory, case)
    try:
        with isolated_media(directory), fixture.sessions() as db:
            before = observe(db, case.initial.project_id)
            selector = ComparisonInterpreter(mode, semantic, prompt_state(before, case))
            service = LanguageOperationService(operation_service, adapter, semantic=selector, readiness_annotations=True)
            request = LanguageInput(request_id=case.request.request_id, text=case.request.text,
                target=OperationTarget(project_id=case.request.target_project_id), base_revision=case.request.base_revision,
                continuation=Continuation.model_validate(case.request.continuation.model_dump()) if case.request.continuation else None)
            response = await service.submit(db, request)
            after = observe(db, case.initial.project_id)
            record = {"case_id": case.case_id, "case_sha256": case_digest(case), "mode": mode,
                "initial_state_sha256": digest(canonical(before)), "before": before, "after_submit": after,
                "response": response.model_dump(mode="json"), "candidate_audit": selector.audit,
                "calls": selector.calls, "score": score_submit(case, response, before, after, len(selector.calls)),
                "event": {"kind": case.event.kind, "status": "not_measured" if case.event.kind != "none" else "not_applicable"},
                "media_rendered": False}
            # Small confirmation exercise on explicitly labelled generation events;
            # all other D24 events stay outside this submit-only comparison metric.
            if case.event.kind in {"confirm_generation", "confirm_twice"}:
                if response.status == "ready" and response.requires_confirmation:
                    permission = LanguageExecution(confirmation_token=response.confirmation_token, confirm_generation=True)
                    confirmed = service.execute(db, request.request_id, permission)
                    duplicate = service.execute(db, request.request_id, permission) if case.event.kind == "confirm_twice" else None
                    record["event"] = {"kind": case.event.kind, "status": "observed_unscored",
                        "response": confirmed.model_dump(mode="json"), "duplicate_equal": duplicate == confirmed if duplicate else None,
                        "after": observe(db, case.initial.project_id)}
            # Every trial checks durable identical resend without executing new work.
            count_before_replay = len(selector.calls)
            replay_before = observe(db, case.initial.project_id)
            replay = await service.submit(db, request)
            record["replay"] = {"model_calls": len(selector.calls) - count_before_replay,
                "db_unchanged": observe(db, case.initial.project_id) == replay_before,
                "same_request_id": replay.request_id == response.request_id,
                "status": replay.status}
            (directory / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            return record
    finally:
        fixture.close()
