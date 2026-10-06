"""Construct the immutable process-wide operation service."""
from __future__ import annotations

from pathlib import Path

from app.operations.catalog import load_catalog
from app.operations.handlers import get_project_status, set_subtitle_font_size
from app.operations.annotations import load_annotations
from app.operations.policies import load_policies
from app.operations.control_handlers import (
    update_settings, start_generation, cancel_generation, retry_generation, restore_settings,
    restore_artifact,
)
from app.operations.registry import HandlerRegistry, RegistryError
from app.operations.service import OperationService

_DEFAULT_CATALOG = Path(__file__).with_name("definitions.json")
_HANDLER_POLICIES = {
    "project.update_settings_v2": ("project.settings.update", 2, "project_editable", "settings_saved", frozenset({"video"})),
    "project.update_settings": ("project.settings.update", 1, "project_editable", "settings_saved", frozenset({"video"})),
    "project.start_generation": ("project.generation.start", 1, "project_editable", "job_queued", frozenset({"video"})),
    "project.cancel_generation": ("project.generation.cancel", 1, "project_exists", "cancellation_requested", frozenset({"video"})),
    "project.retry_generation": ("project.generation.retry", 1, "project_editable", "job_queued", frozenset({"video"})),
    "project.restore_settings": ("project.settings.restore", 1, "project_editable", "settings_restored", frozenset({"video"})),
    "project.restore_artifact": ("project.artifact.restore", 1, "project_editable", "artifact_restored", frozenset({"video"})),
    "project.adjust_subtitle_font_size": (
        "project.subtitle-font-size.adjust", 1, "project_editable",
        "subtitle_font_size_saved", frozenset({"video"}),
    ),
    "project.get_status": (
        "project.status.get",
        1,
        "project_exists",
        "state_unchanged",
        frozenset(),
    ),
    "project.set_subtitle_font_size": (
        "project.subtitle-font-size.set",
        1,
        "project_editable",
        "subtitle_font_size_saved",
        frozenset({"video"}),
    ),
}


def build_operation_service(catalog_path: Path = _DEFAULT_CATALOG) -> OperationService:
    """Build and cross-check catalog metadata against handler safety policy."""
    registry = HandlerRegistry()
    registry.register("project.update_settings", update_settings)
    registry.register("project.update_settings_v2", update_settings)
    registry.register("project.start_generation", start_generation)
    registry.register("project.cancel_generation", cancel_generation)
    registry.register("project.retry_generation", retry_generation)
    registry.register("project.restore_settings", restore_settings)
    registry.register("project.restore_artifact", restore_artifact)
    registry.register("project.get_status", get_project_status)
    registry.register("project.set_subtitle_font_size", set_subtitle_font_size)
    # Relative arguments are resolved and validated before the same absolute setter.
    registry.register("project.adjust_subtitle_font_size", set_subtitle_font_size)
    catalog = load_catalog(catalog_path)
    for definition in catalog.definitions:
        registry.require(definition.handler_key)
        actual = (
            definition.operation_id,
            definition.operation_version,
            definition.precondition_key,
            definition.postcondition_key,
            frozenset(definition.affected_artifacts),
        )
        expected = _HANDLER_POLICIES[definition.handler_key]
        if actual != expected:
            raise RegistryError(
                f"definition violates handler policy: {definition.handler_key}"
            )
    # Safety and confirmation facts must exist for every operation before startup.
    operations = {(definition.operation_id, definition.operation_version) for definition in catalog.definitions}
    load_policies().require_catalog_coverage(operations)
    load_annotations().require_known_operations(operations)
    return OperationService(catalog, registry)


operation_service = build_operation_service()
