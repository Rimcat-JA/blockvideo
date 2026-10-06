"""D29 review regressions: honest denominators, partial reports and effect checks."""
from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from app.language_operations.contracts import LanguageResponse
from evaluation.comparison_scoring import score_submit
from evaluation.comparison_runner import run_trial
from scripts import compare_modes
from tests.test_comparison_modes import CASES, SET64, busy_case
from tests.test_semantic_interpretation import Replies, setup as setup


def record(milliseconds: int | None, *, called: bool = True, mode: str = "B0") -> dict[str, Any]:
    return {"case_id": "D24-D001", "mode": mode, "initial_state_sha256": "same",
        "calls": [{"request_bytes": 10, "response_bytes": 5}] if called else [],
        "score": {"submit_effects_match": True, "proposal_match": called},
        "response": {"status": "completed", "diagnostics": {"interpretation_ms": milliseconds}},
        "replay": {"model_calls": 0, "db_unchanged": True}}


def test_summary_excludes_missing_time_and_host_only_trials() -> None:
    rows = [record(ms) for ms in (4800, 4900, 5000, None)]
    rows += [record(None, called=False), record(1, called=False)]
    summary = compare_modes.summarize(rows)["B0"]
    assert summary["median_interpretation_ms"] == 4900
    assert summary["interpretation_timing_samples"] == 3
    assert summary["model_called_trials"] == summary["model_proposals_match"] == 4
    assert summary["model_not_called_trials"] == 2
    assert summary["timing_comparable_between_modes"] is False


def test_summary_keeps_measured_zero_but_empty_timing_is_unknown() -> None:
    assert compare_modes.summarize([record(0)])["B0"]["median_interpretation_ms"] == 0
    for rows in ([], [record(None)], [record(0, called=False)]):
        summary = compare_modes.summarize(rows)["B0"]
        assert summary["median_interpretation_ms"] is None
        assert summary["interpretation_timing_samples"] == 0


@pytest.mark.parametrize("code,expected", [
    ("timeout", "interpretation_deadline_exceeded"),
    ("connection_failed", "local_model_transport_failure"),
    ("model_mismatch", "local_model_transport_failure"),
    ("invalid_arguments", None),
])
def test_stop_reason_separates_delay_from_transport(code: str, expected: str | None) -> None:
    row = record(100)
    row["calls"][0]["error_code"] = code
    assert compare_modes.trial_stop_reason(row) == expected


def test_outer_and_retrieval_deadlines_are_terminal_without_http_error() -> None:
    for details in ({"failure": {"reason_code": "timeout"}},
                    {"interpretation": {"failure": {"reason_code": "retrieval_deadline"}}},
                    {"diagnostics": {"retrieval": {"reason": "deadline"}}}):
        row = record(100)
        row["response"].update(details)
        assert compare_modes.trial_stop_reason(row) == "interpretation_deadline_exceeded"


class LocalContext:
    async def __aenter__(self) -> LocalContext:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def embed_query(self, text: str) -> list[float]:
        return [1.0]


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, asyncio.CancelledError])
async def test_partial_report_retained_for_runner_failure_or_interruption(
    tmp_path: Path, monkeypatch: Any, failure: type[BaseException],
) -> None:
    async def trial(case: Any, mode: str, *args: Any) -> dict[str, Any]:
        if mode == "B1":
            raise failure("private details must not enter the report")
        return record(10)
    monkeypatch.setattr(compare_modes, "run_trial", trial)
    monkeypatch.setattr(compare_modes, "source_hashes", lambda: {"fixed": "hash"})
    args = Namespace(output=tmp_path, repeats=1, modes=["B0", "B1", "P1"])
    manifest = {"source_sha256": {"fixed": "hash"}, "expected_trials": 3}
    call = compare_modes.execute_trials(args, [CASES["D24-D001"]], manifest, None, LocalContext(), LocalContext())
    if failure is asyncio.CancelledError:
        with pytest.raises(asyncio.CancelledError):
            await call
    else:
        await call
    result = json.loads((tmp_path / "report.json").read_text())
    assert result["completed_trials"] == 1 and not result["complete"]
    assert result["stopped_reason"] == ("interrupted" if failure is asyncio.CancelledError else "runner_error")
    assert result["failure_context"] == {"trial": {"case_id": "D24-D001", "mode": "B1", "repeat": 1},
                                          "exception_type": failure.__name__}
    assert "private details" not in (tmp_path / "report.json").read_text()


async def test_warmup_failure_still_records_zero_completed_trials(tmp_path: Path, monkeypatch: Any) -> None:
    class FailedWarmup(LocalContext):
        async def embed_query(self, text: str) -> list[float]:
            raise ValueError("not loaded")
    monkeypatch.setattr(compare_modes, "source_hashes", lambda: {})
    result = await compare_modes.execute_trials(Namespace(output=tmp_path, repeats=1, modes=["B0"]),
        [CASES["D24-D001"]], {"source_sha256": {}, "expected_trials": 1}, None, FailedWarmup(), LocalContext())
    assert result["completed_trials"] == 0 and result["stopped_reason"] == "runner_error"
    assert result["failure_context"]["trial"] is None and not result["complete"]
    assert (tmp_path / "report.json").exists()


async def test_completed_deadline_trial_stops_later_trials(tmp_path: Path, monkeypatch: Any) -> None:
    async def trial(case: Any, mode: str, *args: Any) -> dict[str, Any]:
        assert mode == "B0"
        row = record(180000)
        row["response"].update(status="error", failure={"reason_code": "timeout"})
        return row
    monkeypatch.setattr(compare_modes, "run_trial", trial)
    monkeypatch.setattr(compare_modes, "source_hashes", lambda: {})
    result = await compare_modes.execute_trials(Namespace(output=tmp_path, repeats=1, modes=["B0", "B1"]),
        [CASES["D24-D001"]], {"source_sha256": {}, "expected_trials": 2}, None, LocalContext(), LocalContext())
    assert result["completed_trials"] == 1 and not result["complete"]
    assert result["stopped_reason"] == "interpretation_deadline_exceeded"


async def test_history_and_status_corruption_fail_even_when_settings_match(
    setup: Any, tmp_path: Path, temp_storage: Path,
) -> None:
    case = CASES["D24-D001"]
    row = await run_trial(case, "B0", setup[0], Replies([SET64]), tmp_path / "save")
    assert row["score"]["submit_effects_match"]
    corruptions = [deepcopy(row["after_submit"]) for _ in range(4)]
    corruptions[0]["status"] = "failed"
    corruptions[1]["history"].pop()
    corruptions[2]["history"][0]["settings"]["subtitle_font_size"] = 99
    corruptions[3]["history"][-1]["settings"]["subtitle_font_size"] = 99
    for after in corruptions:
        score = score_submit(case, LanguageResponse.model_validate(row["response"]), row["before"], after, 1)
        assert score["checks"]["settings_delta"] and not score["submit_effects_match"]


async def test_pending_job_cancel_has_expected_project_status(setup: Any, tmp_path: Path) -> None:
    case = busy_case()
    case.initial.jobs[0]["status"] = "pending"
    case.request.text = "ジョブ7の生成を停止して"
    case.expected.submit = case.expected.submit.model_copy(update={"outcome": "cancelled", "revision_delta": 0,
        "settings_delta": {}, "job_assertions": {"job_id": 7, "status": "cancelled", "cancel_requested": True}})
    proposal = {"kind": "operation", "operation_id": "project.generation.cancel", "operation_version": 1,
                "arguments": {"job_id": 7}}
    row = await run_trial(case, "B0", setup[0], Replies([proposal]), tmp_path / "cancel")
    assert row["after_submit"]["status"] == "cancelled"
    assert row["score"]["submit_effects_match"]


async def test_dismiss_is_excluded_from_model_agreement_without_relabelling(setup: Any, tmp_path: Path) -> None:
    row = await run_trial(CASES["D24-D055"], "B0", setup[0], Replies([]), tmp_path / "dismiss")
    summary = compare_modes.summarize([row])["B0"]
    assert row["score"]["submit_effects_match"] and not row["score"]["proposal_match"]
    assert summary["model_called_trials"] == summary["model_proposals_match"] == 0
    assert summary["model_not_called_trials"] == 1


async def test_running_job_cancel_ignores_the_after_event_publication_promise(setup: Any, tmp_path: Path) -> None:
    case = CASES["D24-D061"]
    assert case.expected.submit.job_assertions.get("no_future_publication") is True
    proposal = {"kind": "operation", "operation_id": "project.generation.cancel", "operation_version": 1,
                "arguments": {"job_id": 7}}
    row = await run_trial(case, "B0", setup[0], Replies([proposal]), tmp_path / "cancel-running")
    assert row["after_submit"]["jobs"][0]["cancel_requested"] is True
    assert row["score"]["submit_effects_match"], row["score"]["checks"]
