"""D37 projection accepts the partial D24 states the development seed accepts."""
from __future__ import annotations

import json

from evaluation.blinded_runner import case_to_unlabeled
from evaluation.contracts import Case
from evaluation.projection_diagnostics import projection_failure
from tests.test_d37_blinded_runner import _score_case


def _with_initial(case: Case, **changes: object) -> Case:
    return case.model_copy(update={"initial": case.initial.model_copy(update=changes)})


def test_partial_job_and_history_settings_complete_from_owner_settings() -> None:
    base = _score_case("none")
    case = _with_initial(
        base,
        revision=3,
        jobs=[{"id": 7, "project_id": 1, "status": "failed", "input_revision": 2,
               "cancel_requested": False, "input_settings": {"subtitle_font_size": 40}}],
        history=[{"revision": 2, "settings": {"subtitle_font_size": 44}}],
    )

    projected = case_to_unlabeled(case)

    current = projected.initial.settings
    job = projected.initial.jobs[0]
    assert job.input_settings.subtitle_font_size == 40
    assert job.input_settings.voicevox_speed_scale == current.voicevox_speed_scale
    assert job.input_settings.narration_sentence_pause_seconds == 0.2
    assert job.kind == "full"
    history = projected.initial.history[0].settings
    assert history.subtitle_font_size == 44
    assert history.voicevox_speaker_id == current.voicevox_speaker_id


def test_other_project_jobs_seed_a_default_project_like_the_development_seed() -> None:
    case = _with_initial(
        _score_case("none"),
        jobs=[{"id": 9, "project_id": 5, "status": "running", "input_revision": 4,
               "cancel_requested": False, "kind": "full",
               "input_settings": {"subtitle_font_size": 60}}],
    )

    projected = case_to_unlabeled(case)

    (other,) = projected.initial.additional_projects
    assert (other.project_id, other.revision, other.project_status) == (5, 4, "generating")
    assert other.settings.subtitle_font_size == 60
    assert other.settings.voicevox_speed_scale == 1.0
    assert projected.initial.jobs[0].input_settings == other.settings


def test_prior_turn_identity_defaults_to_the_selected_project() -> None:
    case = _with_initial(
        _score_case("none"),
        revision=2,
        prior_turns=[
            {"request_id": "turn-1", "text": "synthetic earlier turn", "status": "needs_input",
             "question": "どのくらいにしますか？"},
            {"request_id": "turn-2", "text": "synthetic answer", "status": "completed",
             "settings_saved": True, "result_revision": 2},
        ],
    )

    first, second = case_to_unlabeled(case).initial.prior_turns

    assert (first.project_id, first.base_revision, first.settings_saved) == (1, 2, False)
    assert (second.project_id, second.base_revision, second.settings_saved) == (1, 1, True)


def test_projection_failure_report_is_structural_and_content_free() -> None:
    secret = "秘密の依頼文"
    base = _score_case("revision_race")
    case = base.model_copy(update={
        "event": base.event.model_copy(update={"details": {"competing": secret}}),
        "initial": base.initial.model_copy(update={"prior_turns": [
            {"request_id": "turn-1", "text": secret, "status": secret}
        ]}),
    })

    report = projection_failure(case)

    assert report is not None
    assert report["error"] == "missing_key"
    assert report["missing_key"] == "external_revision"
    assert report["event_detail_keys"] == ["competing"]
    assert report["prior_turns"][0]["status"] == "<redacted>"
    assert secret not in json.dumps(report, ensure_ascii=False)
    assert projection_failure(_score_case("none")) is None


def _with_details(case: Case, details: dict[str, object]) -> Case:
    return case.model_copy(update={"event": case.event.model_copy(update={"details": details})})


def test_author_spellings_of_event_details_map_to_wire_names() -> None:
    race = case_to_unlabeled(_with_details(_score_case("revision_race"), {
        "competing_revision": 2, "competing_settings": {"subtitle_font_size": 52},
        "phase": "before_confirmation", "then": "synthetic"}))
    assert (race.event.external_revision, race.event.external_settings.subtitle_font_size) == (2, 52)

    switch = case_to_unlabeled(_with_details(_score_case("switch_target"), {
        "switched_project_id": 202, "original_project_id": 1}))
    assert (switch.event.selected_project_id_after, switch.event.action) == (202, "read_original_request")

    for changed, target in (("別の依頼", None), ({"text": "別の依頼", "target_project_id": 1}, 1)):
        body = case_to_unlabeled(_with_details(_score_case("same_id_different_body"),
                                               {"changed_request": changed}))
        assert (body.event.replacement_text, body.event.replacement_target_project_id) == ("別の依頼", target)


def test_prior_turn_author_forms_project_into_the_wire_contract() -> None:
    base = _score_case("none")
    case = _with_initial(
        base,
        revision=2,
        settings={**base.initial.settings, "voicevox_pitch_scale": 0.1},
        prior_turns=[
            {"request_id": "turn-1", "text": "synthetic pitch change", "status": "completed",
             "target_project_id": 1, "base_revision": None, "result_revision": 2,
             "settings_saved": True,
             "proposal": {"kind": "operation", "operation_id": "project.settings.update",
                          "operation_version": 1, "arguments": {"voicevox_pitch_scale": 0.1},
                          "generate_after_save": False}},
            {"request_id": "turn-2", "text": "synthetic unsupported turn", "status": "unsupported",
             "proposal": {"kind": "unsupported", "reason": "synthetic reason"}},
            {"request_id": "turn-3", "text": "synthetic dismissed turn", "status": "dismissed",
             "proposal": {"kind": "no_operation", "reason": "synthetic reason"},
             "parent_request_id": None, "successor_request_id": base.request.request_id},
        ],
    )

    projected = case_to_unlabeled(case)

    first, second, third = projected.initial.prior_turns
    assert projected.initial.settings.voicevox_pitch_scale == 0.1
    assert (first.project_id, first.base_revision) == (1, 1)
    assert first.proposal.arguments.voicevox_pitch_scale == 0.1
    assert (second.proposal.kind, third.proposal.kind) == ("unsupported", "no_operation")
    assert third.successor_request_id is None


def test_competing_full_snapshot_reduces_to_the_changed_modeled_settings() -> None:
    base = _score_case("revision_race")
    snapshot = {**base.initial.settings, "subtitle_font_size": 52, "title": "synthetic",
                "subtitle_position": "bottom", "voicevox_volume_scale": 1.0}
    projected = case_to_unlabeled(_with_details(base, {
        "competing_revision": 2, "competing_settings": snapshot, "phase": "synthetic"}))

    assert projected.event.external_settings.model_dump(exclude_unset=True) == {
        "subtitle_font_size": 52
    }


def test_references_to_unseeded_projects_and_turns_become_representable() -> None:
    base = _score_case("none")
    case = _with_initial(base, prior_turns=[
        {"request_id": "turn-1", "text": "synthetic other project", "status": "dismissed",
         "target_project_id": 8, "base_revision": 3, "settings_saved": False,
         "successor_request_id": "turn-outside-case",
         "proposal": {"kind": "clarification", "question": "どれですか？",
                      "missing_fields": ["target"]}},
    ])

    projected = case_to_unlabeled(case)

    (other,) = projected.initial.additional_projects
    assert (other.project_id, other.revision, other.project_status) == (8, 3, "completed")
    assert projected.initial.prior_turns[0].successor_request_id is None

    body = case_to_unlabeled(_with_details(_score_case("same_id_different_body"), {
        "changed_request": {"text": "別の依頼", "target_project_id": 9}}))
    assert [item.project_id for item in body.initial.additional_projects] == [9]


def test_target_less_clarification_belongs_to_the_answering_request_target() -> None:
    base = _score_case("none")
    request = type(base.request).model_validate({
        **base.request.model_dump(), "target_project_id": 4, "base_revision": 2,
        "continuation": {"parent_request_id": "turn-1", "relation": "answer"},
    })
    case = _with_initial(base, prior_turns=[
        {"request_id": "turn-1", "text": "synthetic which project", "status": "needs_input",
         "target_project_id": None, "base_revision": 2, "settings_saved": False,
         "proposal": {"kind": "clarification", "question": "どれですか？",
                      "missing_fields": ["target"]}},
    ]).model_copy(update={"request": request})

    projected = case_to_unlabeled(case)

    assert projected.initial.prior_turns[0].project_id == 4
    assert [item.project_id for item in projected.initial.additional_projects] == [4]


def test_failure_report_names_race_timing_by_label_and_offset() -> None:
    report = projection_failure(_with_details(_score_case("revision_race"), {
        "competing_revision": 3, "competing_settings": {"subtitle_font_size": 52},
        "original_confirmation_revision": 2, "phase": "after_save_before_confirm",
        "then": "秘密の説明"}))

    assert report is not None
    assert report["event_timing"] == {
        "labels": {"phase": "after_save_before_confirm", "then": "<redacted>"},
        "revision_offsets": {"competing_revision": 2, "original_confirmation_revision": 1},
    }
