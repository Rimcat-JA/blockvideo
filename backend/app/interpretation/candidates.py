"""Snapshot offered definitions and build a constrained response JSON Schema."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.interpretation.contracts import CandidateRef
from app.interpretation.errors import InterpretationError
from app.operations.catalog import CatalogError, OperationCatalog
from app.operations.contracts import OperationDefinition
from app.operations.limits import MAX_PLAN_STEPS
from app.operations.policies import load_policies


def select_candidates(
    catalog: OperationCatalog, refs: tuple[CandidateRef, ...],
) -> tuple[OperationDefinition, ...]:
    """Resolve exact versions once; freeze independent copies before inference."""
    keys = [(ref.operation_id, ref.operation_version) for ref in refs]
    if not keys or len(set(keys)) != len(keys):
        raise InterpretationError("invalid_input")
    try:
        return tuple(catalog.require(*key).model_copy(deep=True) for key in keys)
    except CatalogError:
        raise InterpretationError("invalid_input") from None


def candidate_payload(definition: OperationDefinition) -> dict[str, Any]:
    """Only public operation metadata, excluding callable/internal policy keys.

    Japanese annotations shape retrieval (which candidates are offered), not this
    payload: with the development model, adding annotation notes or example requests
    here made negated and ambiguous requests more often turn into operations.
    """
    return {
        "operation_id": definition.operation_id,
        "operation_version": definition.operation_version,
        "description": definition.description,
        "examples": list(definition.examples),
        "arguments_schema": model_argument_schema(definition.input_schema),
    }


def model_argument_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Stable property presentation for order-sensitive local JSON grammars.

    Ordering is not a JSON Schema constraint. Keep every field, type, bound,
    enum and required rule; the catalog and the parser's validation stay intact.
    Apply the same order to the prompt and the constrained output schema.
    """
    def ordered(value: Any) -> Any:
        if isinstance(value, list):
            return [ordered(item) for item in value]
        if isinstance(value, dict):
            return {key: {name: ordered(item[name]) for name in sorted(item)}
                    if key == "properties" and isinstance(item, dict) else ordered(item)
                    for key, item in value.items()}
        return deepcopy(value)

    return ordered(schema)


def constrained_arguments(definition: OperationDefinition) -> dict[str, Any]:
    """Equivalent settings schemas allow common key orders in local grammars."""
    ordered = model_argument_schema(definition.input_schema)
    view = load_policies().settings_view(definition.operation_id, definition.operation_version)
    if view is None or view.settings_at is None:
        return ordered
    alternate = deepcopy(ordered)
    settings = alternate if view.settings_at == "" else alternate["properties"][view.settings_at]
    fields = settings["properties"]
    leading = ("subtitle_font_size", "voicevox_speed_scale", "voicevox_speaker_id", "subtitle_mode", "pronunciation_overrides")
    settings["properties"] = {key: fields[key] for key in (*leading, *fields) if key in fields}
    # Both alternatives describe exactly the same allowed values. The ordinary
    # parser still validates against the unchanged catalog after generation.
    return {"anyOf": [ordered, alternate, deepcopy(definition.input_schema)]}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def response_schema(definitions: tuple[OperationDefinition, ...]) -> dict[str, Any]:
    """Bind every operation branch to its exact ID, version and argument schema."""
    branches = [
        _object({
            "kind": {"type": "string", "enum": ["operation"]},
            "operation_id": {"type": "string", "enum": [item.operation_id]},
            "operation_version": {"type": "integer", "enum": [item.operation_version]},
            "arguments": constrained_arguments(item),
            "generate_after_save": {"type": "boolean", **(
                {} if load_policies().get(item.operation_id).allows_generate_after_save else {"enum": [False]})},
        }) for item in definitions
    ]
    # A plan step is an ordinary operation branch whose generation flag is always false
    # (generation is its own explicit step).
    step_branches = [
        _object({
            "kind": {"type": "string", "enum": ["operation"]},
            "operation_id": {"type": "string", "enum": [item.operation_id]},
            "operation_version": {"type": "integer", "enum": [item.operation_version]},
            "arguments": constrained_arguments(item),
            "generate_after_save": {"type": "boolean", "enum": [False]},
        }) for item in definitions
    ]
    text_rule = {"type": "string", "minLength": 1, "maxLength": 240, "pattern": r"\S"}
    branches.extend([
        _object({
            "kind": {"type": "string", "enum": ["clarification"]},
            "question": deepcopy(text_rule),
            "missing_fields": {"type": "array", "minItems": 1, "maxItems": 3,
                               "items": {"type": "string", "enum": ["target", "arguments", "intent"]}},
        }),
        _object({"kind": {"type": "string", "enum": ["unsupported"]},
                 "reason": deepcopy(text_rule)}),
        _object({"kind": {"type": "string", "enum": ["no_operation"]},
                 "reason": deepcopy(text_rule)}),
        _object({"kind": {"type": "string", "enum": ["plan"]},
                 "steps": {"type": "array", "minItems": 2, "maxItems": MAX_PLAN_STEPS,
                           "items": {"anyOf": step_branches}}}),
    ])
    return _object({"result": {"anyOf": branches}})
