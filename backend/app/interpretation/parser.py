"""Strict JSON and semantic candidate validation, with no salvage or dispatch."""
from __future__ import annotations

import json
import math
from typing import Any

from pydantic import ValidationError

from app.interpretation.contracts import OperationProposal, PlanProposal, ProposalEnvelope
from app.interpretation.errors import InterpretationError
from app.operations.catalog import CatalogError, validate_arguments
from app.operations.contracts import OperationDefinition
from app.operations.policies import load_policies

MAX_MODEL_TEXT_BYTES = 65_536


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, child in pairs:
        if name in value:
            raise ValueError("duplicate JSON key")
        value[name] = child
    return value


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError("JSON nesting limit exceeded")
    if isinstance(value, dict):
        for child in value.values():
            _check_depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    elif isinstance(value, str):
        value.encode("utf-8")


def strict_json(text: str) -> Any:
    """Reject ambiguous duplicate keys and non-standard numeric literals."""
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _check_depth(value)
    return value


def parse_proposal(
    text: str, definitions: tuple[OperationDefinition, ...],
) -> ProposalEnvelope:
    """Return a schema-validated proposal; never claim readiness or execution."""
    if not isinstance(text, str):
        raise InterpretationError("invalid_json")
    try:
        if len(text.encode("utf-8")) > MAX_MODEL_TEXT_BYTES:
            raise InterpretationError("response_too_large")
        raw = strict_json(text)
    except (ValueError, RecursionError, OverflowError) as exc:
        if isinstance(exc, InterpretationError):
            raise
        raise InterpretationError("invalid_json") from None
    try:
        envelope = ProposalEnvelope.model_validate(raw)
    except ValidationError:
        raise InterpretationError("invalid_output") from None
    proposal = envelope.result
    if isinstance(proposal, OperationProposal):
        _validate_operation(proposal, definitions)
    elif isinstance(proposal, PlanProposal):
        for step in proposal.steps:
            if step.generate_after_save:
                raise InterpretationError("invalid_arguments")
            _validate_operation(step, definitions)
    return envelope


def _validate_operation(proposal: OperationProposal, definitions: tuple[OperationDefinition, ...]) -> None:
    if proposal.generate_after_save and not load_policies().get(proposal.operation_id).allows_generate_after_save:
        raise InterpretationError("invalid_arguments")
    definition = next((item for item in definitions if
                       (item.operation_id, item.operation_version) ==
                       (proposal.operation_id, proposal.operation_version)), None)
    if definition is None:
        raise InterpretationError("candidate_not_offered")
    try:
        validate_arguments(definition, proposal.arguments)
    except (CatalogError, OverflowError, RecursionError):
        raise InterpretationError("invalid_arguments") from None
