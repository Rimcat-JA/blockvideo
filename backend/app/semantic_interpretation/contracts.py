"""Bounded diagnostics, not executable operations or a confidence score."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.interpretation.contracts import CandidateRef, InterpretationOutcome, StrictValue
from app.retrieval.contracts import Digest
from app.retrieval.ranking import RankedCandidate
from app.operations.contracts import CandidateReadinessSnapshot
from app.operations.limits import MAX_CATALOG_OPERATIONS, MAX_PROMPT_CANDIDATES


class SearchStage(StrictValue):
    name: Literal["initial", "expanded", "wide", "all_tools"]
    candidates: tuple[CandidateRef, ...] = Field(max_length=MAX_PROMPT_CANDIDATES, strict=False)
    candidate_state: CandidateReadinessSnapshot | None = None
    result: Literal["proposed", "needs_input", "unsupported", "dismissed", "error"]
    chat_calls: int = Field(ge=0, le=2)
    elapsed_ms: int = Field(ge=0)
    request_bytes: int = Field(ge=0)
    response_bytes: int = Field(ge=0)


class SearchTrace(StrictValue):
    policy: Literal["semantic-3-6-all-v1", "semantic-5-8-all-v1", "all-tools-v1"] = "semantic-5-8-all-v1"
    index_sha256: Digest | None = None
    ranking: tuple[RankedCandidate, ...] = Field(default=(), max_length=MAX_CATALOG_OPERATIONS, strict=False)
    stages: tuple[SearchStage, ...] = Field(default=(), max_length=3, strict=False)
    expansion_count: int = Field(default=0, ge=0, le=1)
    all_tools_count: int = Field(default=0, ge=0, le=1)
    embedding_calls: int = Field(default=0, ge=0, le=1)
    embedding_ms: int = Field(default=0, ge=0)
    chat_calls: int = Field(default=0, ge=0, le=4)
    elapsed_ms: int = Field(default=0, ge=0)
    reason: Literal["ranked", "candidate_insufficient", "search_unavailable", "no_candidates",
                    "integrity_failure", "fallback_unavailable", "deadline", "model_failure",
                    "unsupported_without_full_scope"] = "ranked"


class SemanticOutcome(StrictValue):
    interpretation: InterpretationOutcome
    trace: SearchTrace

    @property
    def candidates(self) -> tuple[CandidateRef, ...]:
        return self.trace.stages[-1].candidates if self.trace.stages else ()
