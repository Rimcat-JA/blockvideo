"""Operation catalog and registered-dispatch contract tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.operations.bootstrap import build_operation_service
from app.operations.catalog import CatalogError, load_catalog, validate_arguments
from app.operations.registry import HandlerRegistry, RegistryError


def _write_catalog(path: Path, definitions: list[dict]) -> Path:
    path.write_text(json.dumps({"operations": definitions}), encoding="utf-8")
    return path


def _definition(**changes) -> dict:
    value = {
        "schema_version": 1,
        "operation_id": "project.subtitle-font-size.set",
        "operation_version": 1,
        "description": "Set subtitle size",
        "examples": ["字幕を56pxにする"],
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": 16, "maximum": 120}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "handler_key": "project.set_subtitle_font_size",
        "affected_artifacts": ["video"],
        "precondition_key": "project_editable",
        "postcondition_key": "subtitle_font_size_saved",
    }
    value.update(changes)
    return value


def test_package_catalog_loads_versioned_operations() -> None:
    path = Path(__file__).parents[1] / "app" / "operations" / "definitions.json"
    catalog = load_catalog(path)
    assert [item.operation_id for item in catalog.definitions] == [
        "project.artifact.restore",
        "project.generation.cancel",
        "project.generation.retry",
        "project.generation.start",
        "project.settings.restore",
        "project.settings.update",
        "project.settings.update",  # v2 keeps the v1 contract available.
        "project.status.get",
        "project.subtitle-font-size.adjust",
        "project.subtitle-font-size.set",
    ]
    assert all(item.operation_version == 1 or (item.operation_id == "project.settings.update" and item.operation_version == 2)
               for item in catalog.definitions)


def test_duplicate_operation_id_and_version_is_rejected(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path / "duplicate.json", [_definition(), _definition()])
    with pytest.raises(CatalogError, match="duplicate"):
        load_catalog(path)


def test_unsupported_schema_keyword_is_rejected(tmp_path: Path) -> None:
    definition = _definition()
    definition["input_schema"]["properties"]["value"]["multipleOf"] = 2
    path = _write_catalog(tmp_path / "schema.json", [definition])
    with pytest.raises(CatalogError, match="multipleOf"):
        load_catalog(path)


@pytest.mark.parametrize("arguments", [{}, {"value": 15}, {"value": 121}, {"value": True}, {"value": 48, "extra": 1}])
def test_argument_validation_is_strict(tmp_path: Path, arguments: dict) -> None:
    definition = load_catalog(_write_catalog(tmp_path / "one.json", [_definition()])).definitions[0]
    with pytest.raises(CatalogError):
        validate_arguments(definition, arguments)


def test_argument_validation_returns_a_copy(tmp_path: Path) -> None:
    definition = load_catalog(_write_catalog(tmp_path / "one.json", [_definition()])).definitions[0]
    original = {"value": 56}
    validated = validate_arguments(definition, original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"precondition_key": "project_editabel"}, "precondition"),
        ({"postcondition_key": "unknown_result"}, "postcondition"),
        ({"examples": [""]}, "examples"),
    ],
)
def test_catalog_rejects_unknown_or_empty_metadata(
    tmp_path: Path, change: dict, message: str
) -> None:
    path = _write_catalog(tmp_path / "metadata.json", [_definition(**change)])
    with pytest.raises(CatalogError, match=message):
        load_catalog(path)


@pytest.mark.parametrize(
    "rule_change",
    [
        {"minimum": "16"},
        {"minimum": 20, "maximum": 10},
        {"minLength": 1},
    ],
)
def test_catalog_rejects_malformed_integer_constraints(
    tmp_path: Path, rule_change: dict
) -> None:
    definition = _definition()
    definition["input_schema"]["properties"]["value"].update(rule_change)
    path = _write_catalog(tmp_path / "constraints.json", [definition])
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_bootstrap_rejects_definition_without_registered_handler(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path / "handler.json",
        [_definition(handler_key="project.missing_handler")],
    )
    with pytest.raises(RegistryError, match="not registered"):
        build_operation_service(path)


@pytest.mark.parametrize(
    "change",
    [
        {"precondition_key": "project_exists"},
        {"postcondition_key": "state_unchanged"},
        {"affected_artifacts": []},
        {"operation_id": "project.status.get"},
        {"handler_key": "project.get_status"},
    ],
)
def test_bootstrap_rejects_handler_policy_mismatch(tmp_path: Path, change: dict) -> None:
    path = _write_catalog(tmp_path / "policy.json", [_definition(**change)])
    with pytest.raises(RegistryError, match="policy"):
        build_operation_service(path)


def test_registry_rejects_duplicate_and_unknown_keys() -> None:
    registry = HandlerRegistry()

    def handler(*_args):
        raise AssertionError("not called")

    registry.register("known", handler)
    with pytest.raises(RegistryError, match="duplicate"):
        registry.register("known", handler)
    with pytest.raises(RegistryError, match="not registered"):
        registry.require("missing")
    assert registry.require("known") is handler
