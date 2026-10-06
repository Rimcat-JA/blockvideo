"""D29 experiment-only modes. No labels, DB writes or alternative executor here."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.interpretation.contracts import CandidateRef, ClarificationProposal, InterpretationInput, InterpretationOutcome
from app.interpretation.errors import InterpretationError
from app.interpretation.service import Interpreter
from app.interpretation.transport import ModelMessage, StructuredAdapter
from app.operations.catalog import OperationCatalog
from app.operations.contracts import CandidateReadinessSnapshot
from app.retrieval.serialization import canonical, digest
from app.semantic_interpretation.contracts import SearchStage, SearchTrace, SemanticOutcome
from app.semantic_interpretation.service import SemanticInterpreter
from evaluation.readiness_filter import hard_filter_candidates


@dataclass(frozen=True)
class Mode:
    semantic: bool
    readiness: bool
    hard_filter: bool = False


MODES: dict[str, Mode] = {
    "B0": Mode(False, False), "B1": Mode(True, False),
    "B2": Mode(True, True, True), "P1": Mode(True, True), "B0+": Mode(False, True),
}
DEADLINE_SECONDS = 180
CHAT_CALL_LIMIT = 4
POLICY = {
    "deadline_seconds": DEADLINE_SECONDS, "chat_call_limit": CHAT_CALL_LIMIT,
    "schema_repairs": 1, "repair_at": "full_allowed_pool_only",
    "top_k": 5, "expanded_k": 8, "full_pool_fallback": True,
    "filter": "known_blocked_or_unsupported_only; keep_needs_input",
    "snapshot": "frozen_before_interpretation; final_core_checks_current_state",
    "snapshot_clock": "synthetic fixture time 0 in model hints; actual observation retained in audit",
    "candidate_order": "operation_id_then_version", "ranking": "max_document_cosine",
}


class AuditAdapter:
    """Same allowlisted raw state for every mode, and exact synthetic call evidence."""

    def __init__(self, adapter: StructuredAdapter, state: dict[str, Any]) -> None:
        self.adapter, self.state = adapter, deepcopy(state)
        self.calls: list[dict[str, Any]] = []

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        if len(self.calls) >= CHAT_CALL_LIMIT:
            raise InterpretationError("configuration_error")
        # The common interpreter's state remains unchanged. This extra raw context
        # gives the All Tools baselines the same available DB facts as other modes.
        payload = json.loads(messages[1].content)
        payload["execution_state"] = self.state
        messages = (messages[0], ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False)), *messages[2:])
        record: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages], "schema": schema,
            "payload_sha256": digest(canonical(payload)), "schema_sha256": digest(canonical(schema)),
            "state_sha256": digest(canonical({"state": payload["state"], "execution_state": self.state})),
            "request_bytes": sum(len(m.content.encode("utf-8")) for m in messages) + len(json.dumps(schema, ensure_ascii=False).encode("utf-8")),
        }
        self.calls.append(record)
        started = perf_counter()
        try:
            result = await self.adapter.complete(messages, schema)
            record.update(response=result, response_bytes=len(result.encode("utf-8")))
            return result
        except InterpretationError as exc:
            record["error_code"] = exc.code
            raise
        finally:
            record["elapsed_ms"] = round((perf_counter() - started) * 1000)


class ComparisonInterpreter:
    """Injected through the same candidate-interpreter interface in all five modes."""

    def __init__(self, mode: str, semantic: SemanticInterpreter | None,
                 raw_state: dict[str, Any]) -> None:
        if mode not in MODES or (MODES[mode].semantic and semantic is None):
            raise ValueError("unknown mode or missing semantic runner")
        self.mode, self.policy, self.semantic = mode, MODES[mode], semantic
        self.raw_state = deepcopy(raw_state)
        self.calls: list[dict[str, Any]] = []
        self.audit: dict[str, Any] = {}

    async def preview(self, catalog: OperationCatalog, adapter: StructuredAdapter,
                      request: InterpretationInput, *,
                      readiness_provider: Callable[[tuple[CandidateRef, ...]], CandidateReadinessSnapshot] | None = None,
                      ) -> SemanticOutcome:
        if readiness_provider is None:
            raise ValueError("all comparison modes require the common state provider")
        # Canonical ordering and the same snapshot source even for non-annotated modes.
        refs = tuple(sorted(request.candidates, key=lambda r: (r.operation_id, r.operation_version)))
        snapshot = readiness_provider(refs)
        observed_at = snapshot.observed_at
        snapshot = snapshot.model_copy(update={"observed_at": 0.0})
        InterpretationInput.model_validate(request.model_copy(update={"candidates": refs, "candidate_state": snapshot}).model_dump())
        allowed = hard_filter_candidates(refs, snapshot) if self.policy.hard_filter else refs
        self.audit = {"mode": self.mode, "input_candidates": [r.model_dump() for r in refs],
            "allowed_candidates": [r.model_dump() for r in allowed], "snapshot": snapshot.model_dump(mode="json"),
            "raw_state_sha256": digest(canonical(self.raw_state)), "observed_at_actual": observed_at, "policy": POLICY}
        meter = AuditAdapter(adapter, self.raw_state)
        self.calls = meter.calls
        if not allowed:
            return SemanticOutcome(interpretation=InterpretationOutcome(status="needs_input",
                proposal=ClarificationProposal(kind="clarification", question="現在の対象で候補を確認できません。対象や操作内容を確認してください。",
                                                missing_fields=["intent"])),
                trace=SearchTrace(reason="no_candidates"))

        def hints(offered: tuple[CandidateRef, ...]) -> CandidateReadinessSnapshot:
            keys = {(r.operation_id, r.operation_version) for r in offered}
            if not keys <= {(r.operation_id, r.operation_version) for r in allowed}:
                raise ValueError("candidate pool escaped comparison scope")
            return snapshot.model_copy(update={"candidates": tuple(r for r in snapshot.candidates
                if (r.operation_id, r.operation_version) in keys)})

        started = perf_counter()
        try:
            async with asyncio.timeout(DEADLINE_SECONDS):
                if self.policy.semantic:
                    result = await self.semantic.preview(catalog, meter, request.model_copy(update={"candidates": allowed}),
                        readiness_provider=hints if self.policy.readiness else None)
                    offset, stages = 0, []
                    for stage in result.trace.stages:
                        calls = meter.calls[offset:offset + stage.chat_calls]
                        stages.append(stage.model_copy(update={"request_bytes": sum(c["request_bytes"] for c in calls),
                            "response_bytes": sum(c.get("response_bytes", 0) for c in calls)}))
                        offset += stage.chat_calls
                    return result.model_copy(update={"trace": result.trace.model_copy(update={"stages": tuple(stages)})})
                state = hints(allowed) if self.policy.readiness else None
                result = await Interpreter(catalog, meter, timeout_seconds=DEADLINE_SECONDS, max_attempts=2).preview(
                    request.model_copy(update={"candidates": allowed, "candidate_state": state}))
        except TimeoutError:
            self.audit["failure_category"] = "interpretation_deadline_exceeded"
            for call in meter.calls:
                if "response" not in call and "error_code" not in call:
                    call["error_code"] = "timeout"
            return SemanticOutcome(interpretation=InterpretationOutcome(status="error",
                failure=InterpretationError("timeout").as_view()), trace=SearchTrace(
                policy="semantic-5-8-all-v1" if self.policy.semantic else "all-tools-v1",
                reason="deadline", chat_calls=len(meter.calls), elapsed_ms=round((perf_counter() - started) * 1000)))
        elapsed = round((perf_counter() - started) * 1000)
        return SemanticOutcome(interpretation=result, trace=SearchTrace(policy="all-tools-v1",
            chat_calls=len(meter.calls), elapsed_ms=elapsed, stages=(SearchStage(name="all_tools",
                candidates=allowed, candidate_state=state, result=result.status, chat_calls=len(meter.calls),
                elapsed_ms=elapsed, request_bytes=sum(c["request_bytes"] for c in meter.calls),
                response_bytes=sum(c.get("response_bytes", 0) for c in meter.calls)),)))
