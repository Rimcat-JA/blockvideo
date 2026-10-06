"""Structural injection boundary; the application never imports experiment code."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from app.interpretation.contracts import CandidateRef, InterpretationInput
from app.interpretation.transport import StructuredAdapter
from app.operations.catalog import OperationCatalog
from app.operations.contracts import CandidateReadinessSnapshot
from app.semantic_interpretation.contracts import SemanticOutcome


class CandidateInterpreter(Protocol):
    async def preview(self, catalog: OperationCatalog, adapter: StructuredAdapter,
                      request: InterpretationInput, *,
                      readiness_provider: Callable[[tuple[CandidateRef, ...]], CandidateReadinessSnapshot] | None = None,
                      ) -> SemanticOutcome: ...
