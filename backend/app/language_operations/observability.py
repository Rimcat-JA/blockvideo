"""Allowlisted, pseudonymous diagnostics; never serialize arbitrary result data."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Any, Literal

from app.core.logging import log
from app.interpretation.contracts import OperationProposal
from app.language_operations.contracts import LanguageResponse
from app.operations.policies import load_policies

_PROCESS_KEY = secrets.token_bytes(32)
# Catalog operations (every one carries a reviewed policy) are safe identifiers to log.
_OPERATIONS = frozenset(load_policies().operations)
_REASONS = frozenset({
    "model_not_configured", "configuration_error", "timeout", "connection_failed", "http_error",
    "model_mismatch", "invalid_input", "invalid_json", "invalid_output", "invalid_response", "invalid_arguments", "candidate_not_offered",
    "incomplete_response", "response_too_large", "refused", "interpretation_failed",
    "interpretation_interrupted", "target_required", "target_conflict", "target_not_found",
    "stale_state", "project_busy", "not_ready", "dialogue_superseded", "job_not_found",
    "job_not_retryable", "external_result_unknown", "history_not_found", "artifact_not_found",
    "artifact_unavailable",
})
_NUMBERS = {
    "value": (16, 120), "delta": (-104, 104), "subtitle_font_size": (16, 120),
    "subtitle_font_size_delta": (-104, 104), "voicevox_speed_scale": (0.5, 2.0), "artifact_id": (1, 10**12),
    "voicevox_speaker_id": (0, 100000), "revision": (1, 10**12),
}


def identifier(kind: str, value: object) -> str | None:
    if value is None:
        return None
    return hmac.new(_PROCESS_KEY, f"{kind}:{value}".encode("utf-8", errors="replace"), hashlib.sha256).hexdigest()[:24]


def safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Free text, lists, unknown names and values never pass into logs."""
    result: dict[str, Any] = {}
    for name, value in arguments.items():
        if name in _NUMBERS:
            lower, upper = _NUMBERS[name]
            result[name] = value if type(value) in (int, float) and lower <= value <= upper else "redacted"
        elif name == "job_id":
            result[name] = identifier("job", value)
        elif name == "settings" and isinstance(value, dict):
            # Only a single settings wrapper is valid; never recurse into arbitrary trees.
            result[name] = safe_arguments({key: item for key, item in value.items() if key != "settings"})
        elif name in {"pronunciation_overrides", "subtitle_mode", "kind", "block_index"}:
            result[name] = "redacted"
        else:
            result["other_fields_redacted"] = True
    return result


def diagnostic_event(response: LanguageResponse, event: Literal["prepared", "executed", "replay_or_precheck"]) -> dict[str, Any]:
    proposed = response.interpretation.proposal if response.interpretation else None
    operation = proposed.operation_id if isinstance(proposed, OperationProposal) else None
    result = response.generation_result or response.result
    failure = response.failure or (response.interpretation.failure if response.interpretation else None)
    retrieval = response.diagnostics.retrieval
    return {
        "event": event, "request": identifier("request", response.request_id),
        "core_request": identifier("request", response.core_request_id),
        "project": identifier("project", response.project_id),
        "base_revision": response.base_revision, "status": response.status,
        "candidates": [{"id": item.operation_id, "version": item.operation_version}
                       for item in response.diagnostics.candidates if item.operation_id in _OPERATIONS],
        "proposal_kind": proposed.kind if proposed else None,
        "operation": operation if operation in _OPERATIONS else None,
        "extracted": safe_arguments(proposed.arguments) if isinstance(proposed, OperationProposal) else {},
        "resolved": safe_arguments(result.resolved_arguments) if result else {},
        "failure_code": failure.reason_code if failure and failure.reason_code in _REASONS else "other" if failure else None,
        "http_status": failure.http_status if failure else None,
        "guard_code": response.diagnostics.guard_code,
        "confirmation_required": response.requires_confirmation,
        "result_revision": result.revision if result else None,
        "job": identifier("job", result.job_id) if result else None,
        "attempts": response.interpretation.attempts if response.interpretation else 0,
        "interpretation_ms": response.diagnostics.interpretation_ms,
        "execution_ms": response.diagnostics.execution_ms,
        "generation_execution_ms": response.diagnostics.generation_execution_ms,
        "retrieval": None if retrieval is None else {
            "reason": retrieval.reason, "embedding_calls": retrieval.embedding_calls,
            "embedding_ms": retrieval.embedding_ms, "chat_calls": retrieval.chat_calls,
            "expansions": retrieval.expansion_count, "all_tools": retrieval.all_tools_count,
            "stages": [{"name": s.name, "candidate_count": len(s.candidates), "result": s.result,
                "chat_calls": s.chat_calls, "elapsed_ms": s.elapsed_ms,
                "request_bytes": s.request_bytes, "response_bytes": s.response_bytes} for s in retrieval.stages],
        },
    }


def record(response: LanguageResponse, event: Literal["prepared", "executed", "replay_or_precheck"]) -> None:
    # Diagnostics must never turn a committed operation into an HTTP failure.
    try:
        log.info("language_event {event}", event=json.dumps(diagnostic_event(response, event), ensure_ascii=True, allow_nan=False))
    except Exception:
        pass
