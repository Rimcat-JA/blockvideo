"""Keep the canonical validation while presenting ordered argument properties."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from app.interpretation.candidates import candidate_payload, model_argument_schema, response_schema
from app.operations.catalog import load_catalog


def test_local_schema_order_preserves_every_canonical_constraint() -> None:
    catalog = load_catalog(Path(__file__).parents[1] / "app/operations/definitions.json")
    before = catalog.model_copy(deep=True)
    branches = response_schema(catalog.definitions)["properties"]["result"]["anyOf"]
    for definition, branch in zip(catalog.definitions, branches, strict=False):
        canonical = deepcopy(definition.input_schema)
        presented = candidate_payload(definition)["arguments_schema"]
        # Dictionary equality checks all constraints independently of key order.
        assert presented == canonical
        constraints = branch["properties"]["arguments"]
        for variant in constraints.get("anyOf", [constraints]):
            assert variant == canonical
        assert list(presented["properties"]) == sorted(canonical["properties"])
        if definition.operation_id == "project.settings.update" and definition.operation_version == 1:
            fields = list(presented["properties"])
            assert fields.index("subtitle_font_size") < fields.index("voicevox_speed_scale")
        if definition.operation_id == "project.generation.start":
            assert list(presented["properties"]) == ["block_index", "kind"]
    assert catalog == before


def test_ordering_does_not_share_nested_metadata() -> None:
    source = {"type": "object", "properties": {"z": {"type": "array", "items": {
        "type": "object", "properties": {"z": {"enum": [1, 2]}, "a": {"type": "string"}}}}}}
    result = model_argument_schema(source)
    nested = result["properties"]["z"]["items"]["properties"]
    assert list(nested) == ["a", "z"]
    nested["z"]["enum"].append(3)
    assert source["properties"]["z"]["items"]["properties"]["z"]["enum"] == [1, 2]
