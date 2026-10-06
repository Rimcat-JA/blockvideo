"""Approval and separation failures must not silently become final scores."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evaluation.contracts import Case, ReviewLedger
from evaluation.corpus import (
    case_digest, corpus_digest, eligibility, load_cases, load_review,
    pending_ledger, split_summary,
)
from evaluation.review import review_html
from scripts.build_d24_development import build
from scripts.evaluation_cases import main, write_new


@pytest.fixture
def cases() -> list[Case]:
    return build()[:2]


def approved(cases: list[Case], role: str) -> ReviewLedger:
    data = pending_ledger(cases, role).model_dump(mode="json")
    data.update(reviewer="synthetic-test-reviewer", reviewed_at="2026-09-20T00:00:00Z")
    for entry in data["entries"]:
        entry["decision"] = "approved"
    return ReviewLedger.model_validate(data)


def write_cases(path: Path, cases: list[Case]) -> None:
    path.write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")


def test_development_counts_and_catalog(tmp_path: Path) -> None:
    cases = build()
    file = tmp_path / "development.jsonl"
    write_cases(file, cases)
    loaded = load_cases(file)
    assert len(loaded) == 100
    assert len({case.group_id for case in loaded}) == 10
    assert {case.provenance.kind for case in loaded} == {"prior_development"}
    assert all(sum(c.group_id == group for c in loaded) == 10 for group in {c.group_id for c in loaded})


def test_ai_approval_never_becomes_human_approval(cases: list[Case]) -> None:
    result = eligibility(cases, pending_ledger(cases), approved(cases, "independent_ai"))
    assert result["eligible_count"] == 0
    assert result["score"] is None
    assert result["human_approved"] == 0


def test_only_double_approved_subset_eligible(cases: list[Case]) -> None:
    human = approved(cases, "human")
    human.entries[1].decision = "rejected"
    result = eligibility(cases, human, approved(cases, "independent_ai"))
    assert result["eligible_case_ids"] == [cases[0].case_id]
    assert result["excluded_count"] == 1


@pytest.mark.parametrize("field", ["request", "expected", "source_request", "initial"])
def test_editing_any_case_content_invalidates_approval(cases: list[Case], field: str) -> None:
    old = approved(cases, "human")
    changed = copy.deepcopy(cases)
    if field == "request":
        changed[0].request.text += "。"
    elif field == "expected":
        changed[0].expected.rationale += "確認"
    elif field == "source_request":
        changed[0].source_request += "。"
    else:
        changed[0].initial.settings["subtitle_font_size"] = 50
    with pytest.raises(ValueError, match="bound"):
        eligibility(changed, old, None)


@pytest.mark.parametrize("damage", ["duplicate", "missing", "unknown", "wrong_hash", "wrong_role"])
def test_invalid_review_is_rejected(tmp_path: Path, cases: list[Case], damage: str) -> None:
    value = approved(cases, "human").model_dump(mode="json")
    if damage == "duplicate":
        value["entries"][1] = value["entries"][0]
    elif damage == "missing":
        value["entries"].pop()
    elif damage == "unknown":
        value["entries"][0]["case_id"] = "D24-D999"
    elif damage == "wrong_hash":
        value["entries"][0]["case_sha256"] = "0" * 64
    else:
        value["role"] = "independent_ai"
    path = tmp_path / "review.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_review(path, cases, "human")


def test_case_digest_checked_even_with_matching_corpus_header(cases: list[Case]) -> None:
    human = approved(cases, "human")
    human.entries[0].case_sha256 = "0" * 64
    assert human.corpus_sha256 == corpus_digest(cases)
    with pytest.raises(ValueError, match="bound"):
        eligibility(cases, human, None)


def as_held(case: Case) -> Case:
    data = case.model_dump(mode="json")
    data.update(case_id="D24-H001", group_id="D24-HG01", split="held_out",
                provenance={"kind": "new_synthetic", "reference": "test-only"})
    return Case.model_validate(data)


def test_prior_development_cannot_be_relabelled_held_out(cases: list[Case]) -> None:
    held = as_held(cases[0]).model_dump(mode="json")
    held["provenance"]["kind"] = "prior_development"
    with pytest.raises(ValueError):
        Case.model_validate(held)


def test_duplicate_origin_rejected_without_leaking_text(cases: list[Case]) -> None:
    held = as_held(cases[0])
    with pytest.raises(ValueError, match="split leakage") as error:
        split_summary(cases, [held])
    assert held.source_request not in str(error.value)


def test_normalized_request_overlap_detected_even_with_different_origin(cases: list[Case]) -> None:
    held = as_held(cases[0])
    held.source_request = "独立した起点のつもり"
    held.request.text = held.request.text.replace("64", "６４") + "！"
    with pytest.raises(ValueError, match="requests=1"):
        split_summary(cases, [held])


def test_short_answer_can_be_used_in_independent_dialogue(cases: list[Case]) -> None:
    dev = cases[0].model_copy(deep=True)
    dev.request.text = "はい"
    dev.initial.prior_turns = [{"text": "字幕を変更する確認"}]
    held = as_held(dev)
    held.source_request = "別の元の依頼"
    held.initial.prior_turns = [{"text": "動画を生成する確認"}]
    assert split_summary([dev], [held])["shared_normalized_requests"] == 0


def test_one_origin_per_group(tmp_path: Path, cases: list[Case]) -> None:
    cases[1].source_request += "別の起点"
    path = tmp_path / "data.jsonl"
    write_cases(path, cases)
    with pytest.raises(ValueError, match="multiple origins"):
        load_cases(path)


def test_schema_error_does_not_echo_secret_like_example(tmp_path: Path, cases: list[Case]) -> None:
    value = cases[0].model_dump(mode="json")
    value["request"]["text"] = "sensitive-placeholder-" * 200
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_cases(path)
    assert "sensitive-placeholder" not in str(error.value)


def test_review_html_escapes_data_and_starts_pending(cases: list[Case]) -> None:
    cases[0].request.text = '</script><script>fetch("https://invalid.example/")</script>'
    html = review_html(cases)
    assert cases[0].request.text not in html
    assert "\\u003c/script\\u003e" in html
    assert "connect-src 'none'" in html
    blob = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    embedded = json.loads(blob)
    assert all(entry["decision"] == "pending" for entry in embedded["ledger"]["entries"])
    assert embedded["ledger"]["role"] == "human"
    assert embedded["ledger"]["entries"][0]["case_sha256"] == case_digest(cases[0])


def test_no_silent_file_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_new(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_new(path, {"value": 2})
    assert json.loads(path.read_text())["value"] == 1


def test_prior_dialogue_must_fit_application_contract(tmp_path: Path, cases: list[Case]) -> None:
    cases[0].initial.prior_turns = [{"request_id": "prior", "text": "確認したい", "status": "needs_input",
        "settings_saved": False, "proposal": {"kind": "clarification", "question": "何pxですか？",
        "missing_fields": ["not_an_application_field"]}}]
    path = tmp_path / "cases.jsonl"
    write_cases(path, cases)
    with pytest.raises(ValueError, match="invalid case at row 1"):
        load_cases(path)


@pytest.mark.parametrize("damage", ["status", "value", "missing"])
def test_initial_fixture_must_fit_application_contract(tmp_path: Path, cases: list[Case], damage: str) -> None:
    if damage == "status":
        cases[0].initial.project_status = "not_a_project_status"
    elif damage == "value":
        cases[0].initial.settings["voicevox_speed_scale"] = 5.0
    else:
        del cases[0].initial.settings["subtitle_font_size"]
    path = tmp_path / "cases.jsonl"
    write_cases(path, cases)
    with pytest.raises(ValueError, match="invalid case at row 1"):
        load_cases(path)


def test_cli_refuses_final_aggregate_without_human_approval(
    tmp_path: Path, cases: list[Case], monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, human, ai, output = [tmp_path / name for name in ("cases.jsonl", "human.json", "ai.json", "eligible.json")]
    write_cases(corpus, cases)
    write_new(human, pending_ledger(cases).model_dump(mode="json"))
    write_new(ai, approved(cases, "independent_ai").model_dump(mode="json"))
    monkeypatch.setattr("sys.argv", ["evaluation_cases", "eligible", "--cases", str(corpus),
                                  "--human", str(human), "--ai", str(ai), "--output", str(output)])
    assert main() == 2
    assert not output.exists()
