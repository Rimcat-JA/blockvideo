"""One query, at most one expansion and one full-scope fallback; no executor."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from time import perf_counter
from typing import Any, Protocol

from app.interpretation.contracts import CandidateRef, ClarificationProposal, FailureView, InterpretationInput, InterpretationOutcome, OperationProposal
from app.interpretation.service import Interpreter
from app.interpretation.transport import ModelMessage, StructuredAdapter
from app.operations.catalog import OperationCatalog
from app.operations.contracts import CandidateReadinessSnapshot
from app.operations.limits import MAX_PROMPT_CANDIDATES
from app.operations.policies import load_policies
from app.retrieval.contracts import OperationRef, SearchScope
from app.retrieval.ranking import rank_operations
from app.retrieval.reader import VerifiedIndex, check_sources
from app.retrieval.serialization import RetrievalError, canonical, digest
from app.retrieval.sources import IndexSources
from app.semantic_interpretation.contracts import SearchStage, SearchTrace, SemanticOutcome


class QueryEncoder(Protocol):
    async def embed_query(self, text: str) -> tuple[float, ...]: ...


def retrieval_query(request: InterpretationInput) -> str:
    """Same bounded dialogue as interpretation; long input falls back, not truncates."""
    return "\n".join([turn.text for turn in request.dialogue] + [request.text])


class _Meter:
    def __init__(self, adapter: StructuredAdapter) -> None:
        self.adapter = adapter
        self.calls = self.request_bytes = self.response_bytes = 0

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        self.calls += 1
        # UTF-8 content bytes, explicitly not model tokens or a monetary charge.
        self.request_bytes += sum(len(m.content.encode("utf-8")) for m in messages)
        self.request_bytes += len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
        content = await self.adapter.complete(messages, schema)
        self.response_bytes += len(content.encode("utf-8"))
        return content


def _question() -> InterpretationOutcome:
    return InterpretationOutcome(status="needs_input", proposal=ClarificationProposal(kind="clarification",
        question="操作の候補を十分に確認できませんでした。変更する設定や、確認・生成などの目的を具体的に指定してください。通常の設定画面も使えます。",
        missing_fields=["intent"]))


def _failure(code: str) -> InterpretationOutcome:
    return InterpretationOutcome(status="error", failure=FailureView(reason_code=code,
        message="操作候補の確認を完了できませんでした。設定は変更していません。検索の設定を確認するか、通常の設定画面を使ってください。"))


class SemanticInterpreter:
    """Host-owned scope and loaders only; model cannot select paths or capabilities."""

    def __init__(self, load: Callable[[IndexSources], VerifiedIndex], sources: Callable[[], IndexSources],
                 encoder: QueryEncoder, *, scope: SearchScope | None = None,
                 allow_all_tools: bool = True, timeout_seconds: float = 180) -> None:
        if not 0 < timeout_seconds <= 180:
            raise ValueError("semantic deadline must be within180 seconds")
        self._load, self._sources, self._encoder = load, sources, encoder
        self._scope, self._allow_all = scope, allow_all_tools
        self._timeout = timeout_seconds

    async def preview(self, catalog: OperationCatalog, adapter: StructuredAdapter,
                      request: InterpretationInput, *,
                      readiness_provider: Callable[[tuple[CandidateRef, ...]], CandidateReadinessSnapshot] | None = None,
                      ) -> SemanticOutcome:
        started = perf_counter()
        trace = SearchTrace()
        meter = _Meter(adapter)
        outcome = _question()
        try:
            async with asyncio.timeout(self._timeout):
                current = await asyncio.to_thread(self._sources)
                ordered = sorted(catalog.definitions, key=lambda d: (d.operation_id, d.operation_version))
                if digest(canonical([d.model_dump(mode="json") for d in ordered])) != current.catalog_semantic_sha256:
                    raise RetrievalError("catalog_mismatch")
                index = await asyncio.to_thread(self._load, current)
                trace = trace.model_copy(update={"index_sha256": index.manifest.bundle_sha256})
                keys = {(r.operation_id, r.operation_version) for r in request.candidates}
                if len(keys) != len(request.candidates):
                    raise RetrievalError("invalid_scope")
                scope = self._scope or SearchScope(app_id=current.app_id,
                    capabilities=tuple(sorted({cap for d in current.documents for cap in d.required_capabilities})),
                    operations=tuple(OperationRef(operation_id=k[0], operation_version=k[1]) for k in sorted(keys)))
                # A request may narrow host scope, never expand it.
                scope = scope.model_copy(update={"operations": tuple(r for r in scope.operations if r.key in keys)})
                eligible = index.eligible_documents(scope, await asyncio.to_thread(self._sources))
                all_refs = tuple(CandidateRef(operation_id=k[0], operation_version=k[1]) for k in sorted({d.key for d in eligible}))
                if not all_refs:
                    trace = trace.model_copy(update={"reason": "no_candidates"})
                    return SemanticOutcome(interpretation=outcome,
                        trace=trace.model_copy(update={"elapsed_ms": round((perf_counter() - started) * 1000)}))
                embed_started = perf_counter()
                ranking = ()
                try:
                    query = retrieval_query(request)
                    if not query.strip() or len(query.encode("utf-8")) > 1200:
                        raise RetrievalError("query_too_long_or_empty")
                    trace = trace.model_copy(update={"embedding_calls": 1})
                    vector = await self._encoder.embed_query(query)
                except RetrievalError:
                    trace = trace.model_copy(update={"reason": "search_unavailable"})
                else:
                    # Integrity failure here must NOT become an encoder fallback.
                    ranking = rank_operations(index, vector, scope, await asyncio.to_thread(self._sources), query_text=query)
                    trace = trace.model_copy(update={"ranking": ranking})
                finally:
                    trace = trace.model_copy(update={"embedding_ms": round((perf_counter() - embed_started) * 1000)})
                ranked_refs = tuple(CandidateRef(operation_id=r.operation_id, operation_version=r.operation_version) for r in ranking)
                stages: list[tuple[str, tuple[CandidateRef, ...]]] = []
                unsupported_stages = 0
                if ranked_refs:
                    stages.append(("initial", ranked_refs[:5]))
                    if len(ranked_refs) > 5:
                        stages.append(("expanded", ranked_refs[:8]))
                if self._allow_all and len(all_refs) <= MAX_PROMPT_CANDIDATES and (not stages or len(stages[-1][1]) < len(all_refs)):
                    stages.append(("all_tools", all_refs))
                elif self._allow_all and len(ranked_refs) > 8:
                    # Too many operations to show at once: the widest ranked window stands in.
                    stages.append(("wide", ranked_refs[:MAX_PROMPT_CANDIDATES]))
                for name, refs in stages:
                    # Retrieval chooses membership; preserve canonical grammar branch
                    # order (including v1 before v2), as in the All Tools baseline.
                    refs = tuple(sorted(refs, key=lambda r: (r.operation_id, r.operation_version)))
                    full = len(refs) == len(all_refs)
                    check_sources(index.manifest, await asyncio.to_thread(self._sources))
                    if name == "expanded":
                        trace = trace.model_copy(update={"expansion_count": 1})
                    if name == "all_tools":
                        trace = trace.model_copy(update={"all_tools_count": 1})
                    stage_started = perf_counter()
                    before = (meter.calls, meter.request_bytes, meter.response_bytes)
                    candidate_state = readiness_provider(refs) if readiness_provider is not None else None
                    try:
                        outcome = await Interpreter(catalog, meter,
                            timeout_seconds=max(0.001, self._timeout - (perf_counter() - started)),
                            max_attempts=2 if full else 1).preview(request.model_copy(update={
                                "candidates": refs, "candidate_state": candidate_state}))
                    finally:
                        trace = trace.model_copy(update={"stages": (*trace.stages, SearchStage(name=name,
                            candidates=refs, candidate_state=candidate_state,
                            result=outcome.status if meter.response_bytes > before[2] else "error",
                            chat_calls=meter.calls - before[0], request_bytes=meter.request_bytes - before[1],
                            response_bytes=meter.response_bytes - before[2], elapsed_ms=round((perf_counter() - stage_started) * 1000)))})
                    check_sources(index.manifest, await asyncio.to_thread(self._sources))
                    follow_up = load_policies().follow_up_generation.operation_id
                    if (isinstance(outcome.proposal, OperationProposal) and outcome.proposal.generate_after_save
                            and not any(r.operation_id == follow_up for r in all_refs)):
                        # Follow-up generation must not escape the host capability scope.
                        outcome = _question()
                        trace = trace.model_copy(update={"reason": "fallback_unavailable"})
                        break
                    if outcome.status == "error" and (not outcome.failure or outcome.failure.reason_code not in {
                        "invalid_json", "invalid_output", "invalid_arguments", "candidate_not_offered"}):
                        trace = trace.model_copy(update={"reason": "model_failure"})
                        break
                    if full or outcome.status in {"proposed", "dismissed"}:
                        break
                    unsupported_stages += outcome.status == "unsupported"
                    trace = trace.model_copy(update={"reason": "candidate_insufficient"})
                else:
                    if (len(all_refs) > MAX_PROMPT_CANDIDATES and stages and stages[-1][0] == "wide"
                            and unsupported_stages == len(stages)):
                        # A catalog too large to show at once has no full-scope check; every stage,
                        # up to the widest ranked window, agreeing on "unsupported" is the strongest answer.
                        trace = trace.model_copy(update={"reason": "unsupported_without_full_scope"})
                    else:
                        # Missing candidates or an unavailable full-scope check is not unsupported.
                        outcome = _question()
                        trace = trace.model_copy(update={"reason": "fallback_unavailable"})
        except TimeoutError:
            trace = trace.model_copy(update={"reason": "deadline"})
            outcome = _failure("retrieval_deadline")
        except RetrievalError:
            trace = trace.model_copy(update={"reason": "integrity_failure"})
            outcome = _failure("retrieval_integrity_failed")
        except Exception:
            # Do not expose adapter exceptions, paths, query or private model output.
            trace = trace.model_copy(update={"reason": "search_unavailable"})
            outcome = _failure("retrieval_unavailable")
        return SemanticOutcome(interpretation=outcome, trace=trace.model_copy(update={
            "chat_calls": meter.calls, "elapsed_ms": round((perf_counter() - started) * 1000)}))
