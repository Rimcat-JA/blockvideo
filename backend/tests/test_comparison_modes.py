"""D29 controls, isolated real core, candidate scope and non-execution symmetry."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from app.interpretation.contracts import InterpretationInput, MinimalState
from app.interpretation.errors import InterpretationError
from app.language_operations.candidate_state import current_candidates
from app.language_operations.contracts import LanguageInput
from app.language_operations.service import LanguageOperationService
from app.operations.bootstrap import operation_service
from app.operations.contracts import OperationRequest, OperationTarget
from evaluation.comparison import AuditAdapter, ComparisonInterpreter, MODES
from evaluation.comparison_fixture import TrialDatabase, observe
from evaluation.comparison_runner import run_trial
from evaluation.contracts import Case
from evaluation.corpus import load_cases
from tests.test_semantic_interpretation import CATALOG, REFS, QUESTION, Replies, setup as setup

CASES = {c.case_id: c for c in load_cases(Path(__file__).parents[2] / "evaluation/d24/development.jsonl")}
SET64 = {"kind": "operation", "operation_id": "project.subtitle-font-size.set", "operation_version": 1,
         "arguments": {"value": 64}}
START = {"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1,
         "arguments": {"kind": "full"}}


def busy_case() -> Case:
    case = CASES["D24-D001"].model_copy(deep=True)
    case.initial.project_status = "generating"
    case.initial.jobs.append({"id": 7, "project_id": 101, "status": "running", "input_revision": 5,
                              "input_settings": dict(case.initial.settings), "cancel_requested": False, "kind": "full"})
    return case


@pytest.mark.parametrize("mode", MODES)
async def test_all_modes_use_actual_core_and_replay_once(setup: Any, tmp_path: Path, temp_storage: Path, mode: str) -> None:
    semantic, *_ = setup
    record = await run_trial(CASES["D24-D001"], mode, semantic, Replies([SET64] * 3), tmp_path / mode)
    assert record["score"]["submit_effects_match"]
    assert record["score"]["revision_delta"] == record["score"]["new_receipts"] == 1
    assert record["replay"]["db_unchanged"] and record["replay"]["model_calls"] == 0
    assert record["calls"][0]["state_sha256"]
    assert (tmp_path / mode / "trial.db").is_file()


async def test_paired_modes_share_input_state_and_schema_definitions(setup: Any, tmp_path: Path, temp_storage: Path) -> None:
    semantic, encoder, *_ = setup
    records = [await run_trial(CASES["D24-D001"], m, semantic, Replies([SET64] * 3), tmp_path / m) for m in MODES]
    assert len({r["initial_state_sha256"] for r in records}) == 1
    assert len({c["state_sha256"] for r in records for c in r["calls"]}) == 1
    assert encoder.calls == 3  # B0 and B0+ do not even invoke embeddings.
    payloads = {r["mode"]: json.loads(r["calls"][0]["messages"][1]["content"]) for r in records}
    for left, right in (("B0", "B0+"), ("B1", "P1"), ("B1", "B2")):
        a, b = payloads[left], payloads[right]
        assert a["state"] == b["state"] and a["execution_state"] == b["execution_state"]
        assert [{k: v for k, v in c.items() if k != "readiness_hint"} for c in a["candidates"]] == [
            {k: v for k, v in c.items() if k != "readiness_hint"} for c in b["candidates"]]
        ra, rb = next(r for r in records if r["mode"] == left), next(r for r in records if r["mode"] == right)
        assert ra["calls"][0]["schema_sha256"] == rb["calls"][0]["schema_sha256"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("proposal,status", [(QUESTION, "needs_input"), ({"kind": "no_operation", "reason": "実行を希望していません。"}, "dismissed")])
async def test_every_mode_allows_question_and_no_op(setup: Any, tmp_path: Path, mode: str, proposal: dict, status: str) -> None:
    semantic, *_ = setup
    record = await run_trial(CASES["D24-D001"], mode, semantic, Replies([proposal] * 3), tmp_path / mode)
    assert record["response"]["status"] == status
    assert record["before"] == record["after_submit"]
    assert not record["score"]["submit_effects_match"]  # Safe refusal is not success for a valid save.


@pytest.mark.parametrize("mode", MODES)
async def test_busy_retention_vs_hard_filter_without_substitute_effect(setup: Any, tmp_path: Path, mode: str) -> None:
    semantic, *_ = setup
    record = await run_trial(busy_case(), mode, semantic, Replies([QUESTION if mode == "B2" else SET64] * 3), tmp_path / mode)
    assert record["before"] == record["after_submit"]
    assert record["response"]["status"] == ("needs_input" if mode == "B2" else "blocked")
    ids = {r["operation_id"] for r in record["candidate_audit"]["allowed_candidates"]}
    assert (SET64["operation_id"] in ids) == (mode != "B2")
    if mode == "B2":
        assert ids == {"project.status.get", "project.generation.cancel"}


@pytest.mark.parametrize("embedding_fails", [False, True])
async def test_filter_survives_all_fallbacks_and_keeps_unknown_arguments(setup: Any, tmp_path: Path, embedding_fails: bool) -> None:
    semantic, encoder, *_ = setup
    encoder.failure = embedding_fails
    record = await run_trial(busy_case(), "B2", semantic, Replies([QUESTION] * 3), tmp_path / "filtered")
    for call in record["calls"]:
        payload = json.loads(call["messages"][1]["content"])
        assert {r["operation_id"] for r in payload["candidates"]} == {"project.status.get", "project.generation.cancel"}
        cancel = next(c for c in payload["candidates"] if c["operation_id"].endswith("cancel"))
        assert cancel["readiness_hint"]["readiness"] == "needs_input"


@pytest.mark.parametrize("mode", MODES)
async def test_generation_needs_confirmation_and_duplicate_confirmation_is_one_job(setup: Any, tmp_path: Path, temp_storage: Path, mode: str) -> None:
    semantic, *_ = setup
    record = await run_trial(CASES["D24-D054"], mode, semantic, Replies([START] * 3), tmp_path / mode)
    assert record["score"]["submit_effects_match"] and record["score"]["new_jobs"] == 0
    assert len(record["event"]["after"]["jobs"]) == 1 and record["event"]["duplicate_equal"]
    assert record["replay"]["db_unchanged"]


@pytest.mark.parametrize("mode", MODES)
async def test_core_rechecks_busy_after_model_in_all_modes(setup: Any, tmp_path: Path, temp_storage: Path, mode: str) -> None:
    semantic, *_ = setup
    fixture = TrialDatabase(tmp_path / mode, CASES["D24-D001"])
    adapter = Replies([SET64] * 3)
    def race() -> None:
        with fixture.sessions() as db:
            operation_service.execute(db, OperationRequest(operation_id="project.generation.start", target=OperationTarget(project_id=101),
                arguments={"kind": "full"}, request_id="race", base_revision=5))
    adapter.on_call = race
    selector = ComparisonInterpreter(mode, semantic, {})
    service = LanguageOperationService(operation_service, adapter, semantic=selector, readiness_annotations=True)
    try:
        with fixture.sessions() as db:
            result = await service.submit(db, LanguageInput(request_id="trial", text="字幕を64pxにして", target=OperationTarget(project_id=101), base_revision=5))
            assert result.status == "blocked" and result.failure.reason_code == "project_busy"
            assert observe(db, 101)["revision"] == 5
    finally:
        fixture.close()


async def test_zero_candidates_no_model_or_embedding(setup: Any, tmp_path: Path) -> None:
    semantic, encoder, *_ = setup
    fixture = TrialDatabase(tmp_path / "empty", busy_case())
    selector, adapter = ComparisonInterpreter("B2", semantic, {}), Replies([])
    refs = tuple(r for r in REFS if r.operation_id == SET64["operation_id"])
    try:
        with fixture.sessions() as db:
            result = await selector.preview(CATALOG, adapter, InterpretationInput(text="字幕を64pxにして", candidates=refs,
                state=MinimalState(selected_project_id=101)), readiness_provider=lambda offered: current_candidates(db, operation_service, offered, 101))
        assert result.interpretation.status == "needs_input" and result.trace.reason == "no_candidates"
        assert not adapter.calls and not encoder.calls
    finally:
        fixture.close()


@pytest.mark.parametrize("mode", MODES)
async def test_transport_failure_is_terminal_and_not_counted_as_success(setup: Any, tmp_path: Path, mode: str) -> None:
    record = await run_trial(CASES["D24-D001"], mode, setup[0], Replies([InterpretationError("connection_failed")]), tmp_path / mode)
    assert record["response"]["status"] == "error" and len(record["calls"]) == 1
    assert record["calls"][0]["error_code"] == "connection_failed"
    assert record["before"] == record["after_submit"] and not record["score"]["submit_effects_match"]


async def test_one_repair_at_full_pool_and_four_total_calls(setup: Any, tmp_path: Path) -> None:
    for mode in MODES:
        record = await run_trial(CASES["D24-D001"], mode, setup[0], Replies(["broken"] * 4), tmp_path / mode)
        assert len(record["calls"]) == (4 if MODES[mode].semantic else 2)
        assert record["response"]["interpretation"]["attempts"] == 2
        assert len(record["response"]["interpretation"]["repair_codes"]) == 1


async def test_shared_total_deadline(setup: Any, tmp_path: Path, monkeypatch: Any) -> None:
    import evaluation.comparison as comparison
    monkeypatch.setattr(comparison, "DEADLINE_SECONDS", 0.01)
    class Slow:
        async def complete(self, messages: Any, schema: Any) -> str:
            await asyncio.sleep(10)
            return "{}"
    for mode in MODES:
        record = await run_trial(CASES["D24-D001"], mode, setup[0], Slow(), tmp_path / mode)
        assert record["response"]["status"] == "error"
        assert record["before"] == record["after_submit"]
        from scripts.compare_modes import trial_stop_reason
        assert trial_stop_reason(record) == "interpretation_deadline_exceeded"


def test_invalid_configuration_and_fixture_do_not_silently_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ComparisonInterpreter("P2", None, {})
    with pytest.raises(ValueError):
        ComparisonInterpreter("B2", None, {})
    fixture = TrialDatabase(tmp_path / "unique", CASES["D24-D001"])
    fixture.close()
    with pytest.raises(FileExistsError):
        TrialDatabase(tmp_path / "unique", CASES["D24-D001"])
    invalid = busy_case()
    invalid.initial.jobs[0]["input_revision"] = 2
    with pytest.raises(ValueError, match="unreachable"):
        TrialDatabase(tmp_path / "bad-state", invalid)


@pytest.mark.parametrize("case_id,proposal", [("D24-D016", {"kind": "operation", "operation_id": "project.subtitle-font-size.adjust", "operation_version": 1, "arguments": {"delta": -2}}),
    ("D24-D081", {"kind": "operation", "operation_id": "project.settings.restore", "operation_version": 1, "arguments": {"revision": 2}})])
async def test_dialogue_current_revision_and_historical_restore_share_core(setup: Any, tmp_path: Path, case_id: str, proposal: dict) -> None:
    for mode in MODES:
        record = await run_trial(CASES[case_id], mode, setup[0], Replies([proposal] * 3), tmp_path / mode)
        assert record["score"]["submit_effects_match"]


async def test_audit_adapter_rejects_fifth_call() -> None:
    from app.interpretation.transport import ModelMessage
    meter = AuditAdapter(Replies([QUESTION] * 5), {})
    messages = (ModelMessage("system", "same"), ModelMessage("user", '{"state":{},"candidates":[]}'))
    for _ in range(4):
        await meter.complete(messages, {})
    with pytest.raises(InterpretationError):
        await meter.complete(messages, {})
    assert len(meter.calls) == 4


async def test_runner_refuses_held_out_without_creating_database(tmp_path: Path) -> None:
    # Synthetic development copy tests the boundary; no held-out file is opened.
    rejected = CASES["D24-D001"].model_copy(update={"split": "held_out"})
    with pytest.raises(ValueError, match="held-out"):
        await run_trial(rejected, "B0", None, Replies([]), tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


async def test_cli_rejects_unapproved_selection_before_loading_model(tmp_path: Path, monkeypatch: Any) -> None:
    from argparse import Namespace
    from scripts import compare_modes
    monkeypatch.setattr(compare_modes, "approved_development", lambda *_: ([CASES["D24-D001"]], {}))
    args = Namespace(output=tmp_path / "no-output", cases=Path("unused"), human=Path("unused"), ai=Path("unused"),
                     case_id=["D24-D060"])
    with pytest.raises(ValueError, match="approvals"):
        await compare_modes.run(args)
    assert not args.output.exists()


def test_media_scope_is_restored_and_disallows_in_process_overlap(tmp_path: Path) -> None:
    from app.core.config import get_settings
    from evaluation.comparison_runner import isolated_media
    previous = get_settings().storage_root
    with isolated_media(tmp_path / "one"):
        assert get_settings().storage_root == (tmp_path / "one/media").resolve()
        with pytest.raises(ValueError, match="parallel"):
            with isolated_media(tmp_path / "two"):
                pass
    assert get_settings().storage_root == previous
