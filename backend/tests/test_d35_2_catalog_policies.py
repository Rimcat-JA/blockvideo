"""D35.2: operation knowledge lives in catalog-side files, not in code paths."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.interpretation.contracts import CandidateRef, InterpretationInput, OperationProposal
from app.interpretation.service import Interpreter, system_prompt
from app.language_operations.intent_guard import negative_control_reason, only_negated_instructions
from app.language_operations.references import reference_question
from app.operations.catalog import CatalogError, OperationCatalog, load_catalog
from app.operations.limits import MAX_CATALOG_OPERATIONS, MAX_PROMPT_CANDIDATES
from app.operations.policies import OperationPolicies, load_policies, settings_values

CATALOG = load_catalog(Path(__file__).parents[1] / "app/operations/definitions.json")
# SHA-256 of the D35 All Tools system prompt, before rules moved to prompt_rules.json.
D35_SYSTEM_SHA256 = "0c328f623765c4d26946c957994bf3eac93a5e9d355a11063bfcb7347dd82710"


def test_full_catalog_prompt_is_byte_identical_to_the_all_tools_prompt() -> None:
    # Operations added after D35 bring only their own tagged rules.
    offered = {item.operation_id for item in CATALOG.definitions} - {"project.artifact.restore"}
    assert hashlib.sha256(system_prompt(offered, features=frozenset()).encode("utf-8")).hexdigest() == D35_SYSTEM_SHA256


def test_narrowed_candidates_receive_only_general_and_their_own_rules() -> None:
    status_only = system_prompt({"project.status.get"})
    assert len(status_only) < len(system_prompt({item.operation_id for item in CATALOG.definitions}))
    assert "subtitle-font-size.adjust" not in status_only
    assert "pronunciation_overrides" not in status_only
    assert "返答はJSONオブジェクト1個のみ" in status_only
    restore = system_prompt({"project.settings.restore"})
    assert "以前の設定に戻して" in restore and "字幕を少し大きくして" not in restore


def test_every_catalog_operation_has_an_explicit_policy() -> None:
    load_policies().require_catalog_coverage(
        {(item.operation_id, item.operation_version) for item in CATALOG.definitions})
    with pytest.raises(CatalogError, match="without a policy"):
        load_policies().require_catalog_coverage({("project.unreviewed.operation", 1)})


def test_unknown_operations_default_to_mutating_and_confirmed() -> None:
    policy = load_policies().get("project.unreviewed.operation")
    assert policy.mutates and policy.requires_confirmation and not policy.allows_generate_after_save
    assert negative_control_reason("何もしないで", "project.unreviewed.operation") == "explicit_negative_intent"


@pytest.mark.parametrize(("text", "expected"), [
    ("字幕は小さくしないで", True), ("ジョブ7は止めないで", True), ("第2版には戻さないで", True),
    ("何も変えなくていいです", True), ("字幕は変えないで動画だけ作り直して", False),
    ("字幕を64pxにして。動画は作らないで", False), ("字幕を小さくして", False),
])
def test_requests_made_only_of_negations_are_detected(text: str, expected: bool) -> None:
    assert only_negated_instructions(text) is expected
    reason = negative_control_reason(text, "project.subtitle-font-size.adjust")
    assert (reason == "explicit_negative_intent") is expected


def test_policy_driven_guards_keep_the_d35_decisions() -> None:
    assert negative_control_reason("再試行しないで", "project.generation.retry") == "explicit_negative_intent"
    assert negative_control_reason("再試行しないで新しく作り直して", "project.generation.start") is None
    assert negative_control_reason("何もしないで", "project.status.get") is None
    cancel = OperationProposal(kind="operation", operation_id="project.generation.cancel",
                               operation_version=1, arguments={"job_id": 7})
    assert reference_question("ジョブ7を止めて", cancel) is None
    assert reference_question("止めて", cancel) is not None
    restore = OperationProposal(kind="operation", operation_id="project.settings.restore",
                                operation_version=1, arguments={"revision": 2})
    assert reference_question("第2版に戻して", restore) is None
    policies = load_policies()
    assert {key for key, item in policies.operations.items() if item.requires_confirmation} == {
        "project.generation.start", "project.generation.retry"}
    assert settings_values(policies.settings_view("project.settings.update", 2),
                           {"settings": {"voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": 2}) == {
        "voicevox_speed_scale": 1.2, "subtitle_font_size_delta": 2}
    assert settings_values(policies.settings_view("project.subtitle-font-size.set", 1), {"value": 64}) == {
        "subtitle_font_size": 64}


def test_policy_file_rejects_unknown_reference_kinds() -> None:
    raw = json.loads((Path(__file__).parents[1] / "app/operations/operation_policies.json").read_text(encoding="utf-8"))
    raw["operations"]["project.generation.cancel"]["reference"]["kind"] = "unknown"
    with pytest.raises(ValueError, match="unknown reference kind"):
        OperationPolicies.model_validate(raw)


def test_catalog_scale_inputs_are_accepted_but_one_call_is_bounded() -> None:
    refs = tuple(CandidateRef(operation_id=f"project.synthetic-{index}.get", operation_version=1)
                 for index in range(MAX_PROMPT_CANDIDATES + 1))
    InterpretationInput(text="状態を見せて", candidates=refs)
    assert MAX_CATALOG_OPERATIONS > MAX_PROMPT_CANDIDATES
    template = CATALOG.require("project.status.get", 1)
    catalog = OperationCatalog(definitions=tuple(
        template.model_copy(update={"operation_id": ref.operation_id}) for ref in refs))

    class _NeverCalled:
        async def complete(self, messages: object, schema: object) -> str:
            raise AssertionError("the model must not be called above the prompt limit")

    import asyncio
    outcome = asyncio.run(Interpreter(catalog, _NeverCalled()).preview(
        InterpretationInput(text="状態を見せて", candidates=refs)))
    assert outcome.status == "error"
    assert outcome.failure is not None and outcome.failure.reason_code == "too_many_candidates"
