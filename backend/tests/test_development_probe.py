"""The D25 interpreter probe cannot silently include pending or held-out labels."""
from __future__ import annotations

import json

import pytest

from app.interpretation.contracts import OperationProposal
from evaluation.corpus import pending_ledger
from evaluation.development_probe import approved_development, operation_meaning, proposal_matches
from scripts.build_d24_development import build


def files(tmp_path, *, approve: bool = True, held: bool = False):
    cases = build()[:2]
    if held:
        for index, case in enumerate(cases):
            case.split = "held_out"
            case.case_id = f"D24-H{index + 1:03}"
            case.group_id = "D24-HG01"
            case.provenance.kind = "new_synthetic"
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases), encoding="utf-8")
    paths = []
    for role in ("human", "independent_ai"):
        ledger = pending_ledger(cases, role)
        if approve:
            ledger.reviewer, ledger.reviewed_at = "synthetic-test", "2026-09-20T00:00:00Z"
            for entry in ledger.entries if role == "independent_ai" else ledger.entries[:1]:
                entry.decision = "approved"
        target = tmp_path / f"{role}.json"
        target.write_text(ledger.model_dump_json(), encoding="utf-8")
        paths.append(target)
    return path, *paths


def test_only_both_approved_cases_are_probed(tmp_path) -> None:
    cases, gate = approved_development(*files(tmp_path))
    assert len(cases) == gate["eligible_count"] == 1
    assert gate["total"] == 2 and gate["excluded_count"] == 1


@pytest.mark.parametrize("kwargs", [{"approve": False}, {"held": True}])
def test_empty_approval_or_held_out_is_rejected(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError):
        approved_development(*files(tmp_path, **kwargs))


def test_changed_case_cannot_reuse_approval(tmp_path) -> None:
    paths = files(tmp_path)
    values = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
    values[0]["situation"] += "changed"
    paths[0].write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")
    with pytest.raises(ValueError):
        approved_development(*paths)


def test_equivalent_settings_encoding_does_not_erase_generation_intent() -> None:
    case = build()[0]
    accepted = OperationProposal(kind="operation", **case.expected.operations[0].model_dump(mode="json"))
    equivalent = OperationProposal(kind="operation", operation_id="project.settings.update", operation_version=1,
        arguments={"subtitle_font_size": accepted.arguments["value"]})
    assert operation_meaning(accepted) == operation_meaning(equivalent)
    assert proposal_matches(case, equivalent)
    assert not proposal_matches(case, equivalent.model_copy(update={"generate_after_save": True}))
