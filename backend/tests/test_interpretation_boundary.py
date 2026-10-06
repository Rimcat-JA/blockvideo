"""D16 adversarial JSON/candidate/privacy boundary tests without inference."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.interpretation.candidates import response_schema, select_candidates
from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.interpretation.errors import InterpretationError
from app.interpretation.parser import parse_proposal
from app.interpretation.service import Interpreter
from app.interpretation.transport import ModelMessage
from app.operations.catalog import OperationCatalog, load_catalog

CATALOG_PATH = Path(__file__).resolve().parents[1] / "app/operations/definitions.json"
SET = "project.subtitle-font-size.set"
REFS = (CandidateRef(operation_id=SET),)


def operation(**updates: Any) -> dict[str, Any]:
    return {"result": {"kind": "operation", "operation_id": SET,
                       "operation_version": 1, "arguments": {"value": 56}, **updates}}


@pytest.fixture()
def catalog() -> OperationCatalog:
    return load_catalog(CATALOG_PATH)


class FakeAdapter:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[tuple[ModelMessage, ...], dict[str, Any]]] = []

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        self.calls.append((messages, schema))
        return self.response


@pytest.mark.parametrize("text", [
    "", "not JSON", "```json\n{}\n```", "Here is the result: {}", "{}{}",
    '{"result": null, "result": null}',
    '{"result":{"arguments":{"value":56,"value":57}}}',
    '{"result": NaN}', '{"result": Infinity}', '{"result": -Infinity}',
    "[" * 2000 + "]" * 2000, '"\ud800"',
], ids=["empty", "text", "fence", "prefix", "concatenated", "duplicate-root",
        "duplicate-nested", "nan", "infinity", "negative-infinity", "depth", "unicode"])
def test_malformed_json_rejected(catalog: OperationCatalog, text: str) -> None:
    with pytest.raises(InterpretationError, match="JSON") as caught:
        parse_proposal(text, select_candidates(catalog, REFS))
    assert caught.value.code == "invalid_json"


@pytest.mark.parametrize("payload", [
    None, [], {}, {"result": None}, {"result": [], "execute": True},
    {**operation(), "executed": True}, operation(target={"project_id": 99}),
    operation(request_id="model-controlled"), operation(base_revision=7),
    operation(generation_requested=True), operation(handler_key="os.system"),
    operation(operation_version="1"), operation(operation_version=True),
    operation(kind="execute"), operation(arguments=[]),
    {"result": {"kind": "clarification", "question": " ", "missing_fields": ["arguments"]}},
    {"result": {"kind": "clarification", "question": "どれ？", "missing_fields": ["password"]}},
    {"result": {"kind": "clarification", "question": "どれ？", "missing_fields": []}},
    {"result": {"kind": "unsupported", "reason": "未対応", "arguments": {}}},
])
def test_extra_or_invalid_shape_rejected(catalog: OperationCatalog, payload: Any) -> None:
    with pytest.raises(InterpretationError) as caught:
        parse_proposal(json.dumps(payload), select_candidates(catalog, REFS))
    assert caught.value.code == "invalid_output"


@pytest.mark.parametrize("updates", [
    {"operation_id": "project.delete"}, {"operation_id": "project.status.get", "arguments": {}},
    {"operation_id": "os.system"}, {"operation_version": 2},
])
def test_only_exact_offered_version_allowed(catalog: OperationCatalog, updates: dict) -> None:
    with pytest.raises(InterpretationError) as caught:
        parse_proposal(json.dumps(operation(**updates)), select_candidates(catalog, REFS))
    assert caught.value.code == "candidate_not_offered"


@pytest.mark.parametrize("arguments", [
    {}, {"value": 56, "execute": True}, {"value": "56"}, {"value": True},
    {"value": 56.0}, {"value": None}, {"value": 15}, {"value": 121},
    {"value": 10 ** 500},
])
def test_argument_types_bounds_and_missing_rejected(catalog: OperationCatalog, arguments: dict) -> None:
    text = json.dumps(operation(arguments=arguments)).replace("Infinity", "1e999")
    with pytest.raises(InterpretationError) as caught:
        parse_proposal(text, select_candidates(catalog, REFS))
    assert caught.value.code == "invalid_arguments"


@pytest.mark.parametrize("payload,status", [
    (operation(), "proposed"),
    ({"result": {"kind": "clarification", "question": "何pxにしますか？", "missing_fields": ["arguments"]}}, "needs_input"),
    ({"result": {"kind": "unsupported", "reason": "メール送信には対応していません。"}}, "unsupported"),
])
async def test_three_result_variants_never_claim_execution(catalog: OperationCatalog, payload: dict, status: str) -> None:
    adapter = FakeAdapter(json.dumps(payload))
    result = await Interpreter(catalog, adapter).preview(InterpretationInput(text="合成要求", candidates=REFS))
    assert result.status == status
    assert result.executed is False
    assert result.failure is None
    assert len(adapter.calls) == 1


async def test_minimal_payload_and_schema_are_candidate_specific(catalog: OperationCatalog) -> None:
    adapter = FakeAdapter(json.dumps(operation()))
    await Interpreter(catalog, adapter).preview(InterpretationInput(
        text="字幕を56pxにして", candidates=REFS,
        state=MinimalState(selected_project_id=101, revision=7, subtitle_font_size=48),
    ))
    messages, schema = adapter.calls[0]
    payload = json.loads(messages[1].content)
    assert payload["state"] == {"selected_project_id": 101, "revision": 7, "subtitle_font_size": 48}
    candidate = payload["candidates"][0]
    # Public metadata only; annotations shape retrieval, not the payload.
    assert set(candidate) == {"operation_id", "operation_version", "description", "examples", "arguments_schema"}
    # One operation, clarification, unsupported, no_operation and the plan branch.
    assert len(schema["properties"]["result"]["anyOf"]) == 5
    branch = schema["properties"]["result"]["anyOf"][0]
    assert branch["properties"]["arguments"] == catalog.require(SET, 1).input_schema
    assert branch["properties"]["operation_id"]["enum"] == [SET]
    for forbidden in ("source_script", "api_key", "output_video_path", "handler_key", "precondition_key"):
        assert forbidden not in messages[1].content


@pytest.mark.parametrize("field", ["source_script", "title", "api_key", "history", "provider", "voicevox_url"])
def test_context_rejects_private_or_unnecessary_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        MinimalState.model_validate({field: "private"})


@pytest.mark.parametrize("refs", [
    (), (CandidateRef(operation_id="unknown"),),
    (CandidateRef(operation_id=SET, operation_version=2),), REFS + REFS,
])
async def test_bad_candidate_input_never_contacts_model(catalog: OperationCatalog, refs: tuple) -> None:
    adapter = FakeAdapter("ignored")
    request = InterpretationInput.model_construct(text="要求", candidates=refs, state=MinimalState())
    result = await Interpreter(catalog, adapter).preview(request)
    assert result.status == "error"
    assert result.failure.reason_code == "invalid_input"
    assert not adapter.calls


async def test_catalog_snapshot_cannot_change_during_inference(catalog: OperationCatalog) -> None:
    class ChangingAdapter(FakeAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            catalog.require(SET, 1).input_schema["properties"]["value"]["maximum"] = 999
            return json.dumps(operation(arguments={"value": 999}))

    result = await Interpreter(catalog, ChangingAdapter("")).preview(InterpretationInput(text="合成要求", candidates=REFS))
    assert result.failure.reason_code == "invalid_arguments"


def test_oversized_output_is_bounded(catalog: OperationCatalog) -> None:
    with pytest.raises(InterpretationError) as caught:
        parse_proposal("あ" * 30_000, select_candidates(catalog, REFS))
    assert caught.value.code == "response_too_large"


@pytest.mark.parametrize("text", [
    '{"result":1e999}',
    '{"result":{"kind":"unsupported","reason":"\\ud800"}}',
])
def test_overflow_numbers_and_escaped_surrogates_rejected(catalog: OperationCatalog, text: str) -> None:
    with pytest.raises(InterpretationError) as caught:
        parse_proposal(text, select_candidates(catalog, REFS))
    assert caught.value.code == "invalid_json"


async def test_invalid_unicode_input_never_contacts_adapter(catalog: OperationCatalog) -> None:
    adapter = FakeAdapter("ignored")
    request = InterpretationInput.model_construct(text="\ud800", candidates=REFS, state=MinimalState())
    result = await Interpreter(catalog, adapter).preview(request)
    assert result.failure.reason_code == "invalid_input"
    assert not adapter.calls


def test_every_catalog_schema_can_be_offered_without_handlers(catalog: OperationCatalog) -> None:
    refs = tuple(CandidateRef(operation_id=item.operation_id, operation_version=item.operation_version)
                 for item in catalog.definitions)
    schema = response_schema(select_candidates(catalog, refs))
    assert len(schema["properties"]["result"]["anyOf"]) == len(catalog.definitions) + 4  # + plan
    assert "handler_key" not in json.dumps(schema)


def test_interpretation_imports_no_execution_capability() -> None:
    root = CATALOG_PATH.parents[1] / "interpretation"
    # Catalog-side metadata only (policies and limits hold no handlers or executors).
    allowed_operations = {"app.operations.catalog", "app.operations.contracts",
                          "app.operations.policies", "app.operations.limits", "app.operations.annotations"}
    for file in root.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            imports = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for name in imports:
                assert not name.startswith(("app.api", "app.db", "app.models", "app.services", "app.workers")), file
                if name.startswith("app.operations"):
                    assert name in allowed_operations, file
