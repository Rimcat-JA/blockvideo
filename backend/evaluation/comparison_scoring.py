"""Narrow submit-effect agreement, separate from semantic or final task scoring."""
from __future__ import annotations

from typing import Any

from app.language_operations.contracts import LanguageResponse
from evaluation.contracts import Case
from evaluation.development_probe import proposal_matches


def score_submit(case: Case, response: LanguageResponse, before: dict[str, Any], after: dict[str, Any],
                 model_calls: int) -> dict[str, Any]:
    expected = case.expected.submit
    settings = {k: v for k, v in after["settings"].items() if before["settings"][k] != v}
    new_jobs = [j for j in after["jobs"] if j["id"] not in {old["id"] for old in before["jobs"]}]
    states = {
        "saved": {"completed"}, "saved_awaiting_confirmation": {"ready"},
        "awaiting_confirmation": {"ready"}, "queried": {"completed"},
        "needs_input": {"needs_input"}, "unsupported": {"unsupported"}, "dismissed": {"dismissed"},
        "blocked": {"blocked"}, "replayed": {"completed", "ready"},
        "cancel_requested": {"completed"}, "cancelled": {"completed"}, "unchanged": {"completed"},
        "generation_queued": {"completed"},
    }
    job_assertions = dict(expected.job_assertions)
    job_id = job_assertions.pop("job_id", None)
    # An after-event promise (no later publication), not a submit-time job column.
    job_assertions.pop("no_future_publication", None)
    job = next((j for j in after["jobs"] if j["id"] == job_id), {})
    # Any unlabelled job mutation (for example cancelling instead of changing a
    # setting while busy) is a failure even if the intended settings were untouched.
    expected_jobs = [dict(j) for j in before["jobs"]]
    if job_id is not None:
        for j in expected_jobs:
            if j["id"] == job_id:
                j.update(job_assertions)
    expected_history = list(before["history"])
    if expected.revision_delta:
        expected_history.append({"revision": before["revision"] + expected.revision_delta,
                                 "settings": {**before["settings"], **expected.settings_delta}})
    expected_status = "generating" if expected.new_jobs else before["status"]
    initial_job = next((j for j in before["jobs"] if j["id"] == job_id), {})
    if (job_assertions.get("status") == "cancelled" and initial_job.get("status") == "pending"
            and initial_job.get("project_id") == before["project_id"]
            and initial_job.get("input_revision") in (None, before["revision"])):
        expected_status = "cancelled"
    checks = {
        "status_class": response.status in states[expected.outcome],
        "project_status": after["status"] == expected_status,
        "settings_delta": settings == expected.settings_delta,
        "revision_delta": after["revision"] - before["revision"] == expected.revision_delta,
        "new_jobs": len(new_jobs) == expected.new_jobs,
        "confirmation": response.requires_confirmation == expected.confirmation_required,
        "job_assertions": all(job.get(k) == v for k, v in job_assertions.items()),
        "no_unexpected_job_changes": [j for j in after["jobs"] if j not in new_jobs] == expected_jobs,
        "artifacts_preserved": after["artifacts"] == before["artifacts"],
        "settings_history": after["history"] == expected_history,
    }
    proposal = response.interpretation.proposal if response.interpretation else None
    return {"score_schema_version": 2, "submit_effects_match": all(checks.values()), "checks": checks,
        "settings_delta": settings, "revision_delta": after["revision"] - before["revision"],
        "new_jobs": len(new_jobs), "new_receipts": len(set(after["receipts"]) - set(before["receipts"])),
        "proposal_match": proposal_matches(case, proposal) if case.expected.interpretation != "not_called" else model_calls == 0,
        "expected_model_not_called": case.expected.interpretation == "not_called",
        "scope": "submit effects only; no question-quality, event, receipt-semantics or final-task score"}
