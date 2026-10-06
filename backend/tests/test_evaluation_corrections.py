"""D24 data corrections must retain real revision and human-approval semantics."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.models.project import Project
from app.models.settings_revision import SettingsRevision
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import OperationRequest
from evaluation.approval import carry_forward_review, validate_ledger_manifest
from evaluation.contracts import Case
from evaluation.corpus import case_digest, corpus_digest, load_cases, pending_ledger
from evaluation.effect_comparison import effect_signature
from evaluation.review import review_html
from scripts.build_d24_development import build


def test_same_value_restore_label_matches_existing_core(temp_storage) -> None:
    case = build()[85]
    with get_session_factory()() as db:
        project = Project(id=case.initial.project_id, title="D24 synthetic restore",
                          source_script="合成テスト用", revision=case.initial.revision,
                          use_fake_providers=True, **case.initial.settings)
        db.add(project)
        db.commit()
        # Record the same-value historical snapshot through the shared save core.
        service = build_operation_service()
        service.execute(db, OperationRequest(operation_id="project.settings.update",
            target={"project_id": project.id}, arguments={"subtitle_font_size": 48},
            base_revision=5, request_id="d24-seed-current-history"))
        result = service.execute(db, OperationRequest(operation_id="project.settings.restore",
            target={"project_id": project.id}, arguments={"revision": 5},
            base_revision=5, request_id="d24-same-value-restore"))
        assert result.revision - 5 == case.expected.submit.revision_delta == 1
        assert not result.changed and case.expected.submit.settings_delta == {}
        assert result.job_id is None and case.expected.submit.new_jobs == 0
        history = db.scalar(select(SettingsRevision).where(SettingsRevision.project_id == project.id,
                                                          SettingsRevision.revision == 6))
        assert history is not None and history.restored_from_revision == 5


def test_ordinary_same_value_save_cannot_claim_extra_revision() -> None:
    value = build()[3].model_dump(mode="json")
    value["expected"]["submit"]["revision_delta"] = 1
    with pytest.raises(ValueError, match="only a recorded restore"):
        Case.model_validate(value)


def test_successful_restore_cannot_claim_no_revision() -> None:
    value = build()[85].model_dump(mode="json")
    value["expected"]["submit"]["revision_delta"] = 0
    with pytest.raises(ValueError, match="completed restore"):
        Case.model_validate(value)


@pytest.mark.parametrize("damage", ["revision", "snapshot", "status"])
def test_live_fixture_errors_are_rejected(tmp_path: Path, damage: str) -> None:
    case = build()[6]
    if damage == "revision":
        case.initial.jobs[0]["input_revision"] = 2
    elif damage == "snapshot":
        case.initial.jobs[0]["input_settings"] = {"subtitle_font_size": 48}
    else:
        case.initial.project_status = "completed"
    path = tmp_path / "cases.jsonl"
    path.write_text(case.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid case at row 1"):
        load_cases(path)


def test_older_terminal_snapshot_is_not_rewritten_or_rejected(tmp_path: Path) -> None:
    case = build()[70]
    assert case.initial.jobs[0]["input_revision"] == 2
    assert case.initial.revision == 5
    path = tmp_path / "cases.jsonl"
    path.write_text(case.model_dump_json(), encoding="utf-8")
    assert len(load_cases(path)) == 1


@pytest.mark.parametrize("damage", ["later_artifact", "missing_input_revision"])
def test_unresolved_job_fixture_errors_are_rejected(tmp_path: Path, damage: str) -> None:
    case = build()[56]
    assert case.initial.jobs[0]["status"] == "unknown"
    if damage == "later_artifact":
        case.initial.jobs[0]["input_revision"] = 2
        case.initial.artifact_revisions = [4]
    else:
        case.initial.jobs[0]["input_revision"] = None
    path = tmp_path / "cases.jsonl"
    path.write_text(case.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid case at row 1"):
        load_cases(path)


def test_unresolved_job_allows_later_settings_without_later_artifact(tmp_path: Path) -> None:
    case = build()[56]
    case.initial.jobs[0]["input_revision"] = 2
    case.initial.artifact_revisions = [1, 2]
    assert case.initial.revision == 5
    path = tmp_path / "cases.jsonl"
    path.write_text(case.model_dump_json(), encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_carry_preserves_only_identical_full_case_and_records_source() -> None:
    original = build()[:3]
    previous = pending_ledger(original)
    previous.reviewer = "synthetic-test-human"
    previous.reviewed_at = "2026-09-20T00:00:00Z"
    previous.entries[0].decision = "approved"
    previous.entries[1].decision = "approved"
    previous.entries[2].decision = "rejected"
    current = copy.deepcopy(original)
    current[1].situation += "変更した説明"
    carried = carry_forward_review(previous, current)
    assert [entry.decision for entry in carried.entries] == ["approved", "pending", "rejected"]
    assert carried.entries[0].case_sha256 == previous.entries[0].case_sha256
    assert carried.entries[1].case_sha256 == case_digest(current[1])
    assert previous.corpus_sha256 in carried.entries[0].note
    assert carried.corpus_sha256 == corpus_digest(current) != previous.corpus_sha256
    validate_ledger_manifest(carried)


def test_carry_rejects_old_ledger_with_replaced_hashes() -> None:
    cases = build()[:2]
    old = pending_ledger(cases)
    old.entries[0].case_sha256 = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        carry_forward_review(old, cases)


def test_seeded_review_keeps_changed_cases_pending() -> None:
    import json

    cases = build()[:2]
    review = pending_ledger(cases)
    review.reviewer = "synthetic-test-human"
    review.reviewed_at = "2026-09-20T00:00:00Z"
    review.entries[0].decision = "approved"
    html = review_html(cases, review=review)
    embedded = json.loads(html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0])
    assert [e["decision"] for e in embedded["ledger"]["entries"]] == ["approved", "pending"]
    assert "$('filter').value='pending'" in html
    cases[1].expected.rationale += "change"
    with pytest.raises(ValueError, match="bound"):
        review_html(cases, review=review)


@pytest.mark.parametrize("index,alias", [(3, "saved"), (17, "unchanged"), (53, "replayed")])
def test_only_narrow_aliases_with_identical_effects_compare_equal(index: int, alias: str) -> None:
    case = build()[index]
    phase = "after_event" if case.expected.after_event else "submit"
    effect = case.expected.after_event or case.expected.submit
    alternate = effect.model_copy(update={"outcome": alias})
    kwargs = {"interpretation": case.expected.interpretation, "event_kind": case.event.kind, "phase": phase}
    assert effect_signature(effect, **kwargs) == effect_signature(alternate, **kwargs)


def test_safe_early_block_is_not_same_as_awaiting_confirmation() -> None:
    case = build()[56]
    ready = case.expected.submit
    blocked = case.expected.after_event
    assert ready.new_jobs == blocked.new_jobs == 0
    kwargs = {"interpretation": "operation", "event_kind": "confirm_generation", "phase": "submit"}
    assert effect_signature(ready, **kwargs) != effect_signature(blocked, **kwargs)


def test_alias_does_not_hide_different_receipt_or_settings() -> None:
    case = build()[3]
    changed = case.expected.submit.model_copy(update={"outcome": "saved", "receipt_rule": "first_result"})
    kwargs = {"interpretation": "operation", "event_kind": "none", "phase": "submit"}
    assert effect_signature(changed, **kwargs) != effect_signature(case.expected.submit, **kwargs)


def test_same_value_restore_revision_is_not_an_ordinary_noop() -> None:
    restore = build()[85].expected.submit
    assert effect_signature(restore, interpretation="operation", event_kind="none", phase="submit")["effects"]["revision_delta"] == 1
