"""Load and validate the Git-managed operation definition catalog."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.operations.contracts import OperationDefinition


from app.operations.schema_validation import CatalogError as CatalogError, validate_schema, validate_value


class OperationCatalog(BaseModel):
    """Validated collection indexed by operation ID and version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    definitions: tuple[OperationDefinition, ...]

    def require(self, operation_id: str, operation_version: int) -> OperationDefinition:
        """Return an exact operation version or raise a bounded error."""
        for definition in self.definitions:
            if (
                definition.operation_id == operation_id
                and definition.operation_version == operation_version
            ):
                return definition
        raise CatalogError(f"unknown operation: {operation_id} version {operation_version}")


_KNOWN_PRECONDITIONS = {"project_exists", "project_editable"}
_KNOWN_POSTCONDITIONS = {"state_unchanged", "subtitle_font_size_saved", "settings_saved", "job_queued", "cancellation_requested", "settings_restored",
                         "artifact_restored"}
_KNOWN_ARTIFACTS = {"video"}


def _validate_input_schema(schema: dict[str, Any]) -> None:
    validate_schema(schema)


def load_catalog(path: Path) -> OperationCatalog:
    """Read a UTF-8 JSON catalog and reject ambiguous definitions."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        operations = raw["operations"]
        if set(raw) != {"operations"} or not isinstance(operations, list):
            raise CatalogError("catalog root must contain only an operations list")
        definitions = [OperationDefinition.model_validate(item) for item in operations]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise CatalogError(f"invalid operation catalog: {exc}") from exc
    seen: set[tuple[str, int]] = set()
    for definition in definitions:
        key = (definition.operation_id, definition.operation_version)
        if key in seen:
            raise CatalogError(f"duplicate operation definition: {key}")
        seen.add(key)
        if definition.schema_version != 1:
            raise CatalogError(f"unsupported schema_version: {definition.schema_version}")
        _validate_input_schema(definition.input_schema)
        if any(not example.strip() for example in definition.examples):
            raise CatalogError("examples must contain non-empty strings")
        if definition.precondition_key not in _KNOWN_PRECONDITIONS:
            raise CatalogError(f"unknown precondition: {definition.precondition_key}")
        if definition.postcondition_key not in _KNOWN_POSTCONDITIONS:
            raise CatalogError(f"unknown postcondition: {definition.postcondition_key}")
        unknown_artifacts = set(definition.affected_artifacts) - _KNOWN_ARTIFACTS
        if unknown_artifacts:
            raise CatalogError(f"unknown affected artifacts: {sorted(unknown_artifacts)}")
    return OperationCatalog(
        definitions=tuple(sorted(definitions, key=lambda item: item.operation_id))
    )


def validate_arguments(
    definition: OperationDefinition, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate the catalog's strict JSON-schema subset without coercion."""
    validate_value(definition.input_schema, arguments)
    return deepcopy(arguments)
