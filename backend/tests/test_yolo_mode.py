"""Unattended (YOLO) requests: no confirmation or clarification, guesses reported."""
from __future__ import annotations

import json

from app.interpretation.service import system_prompt
from app.models.job import GenerationJob
from app.operations.bootstrap import operation_service
from tests.test_language_operations import count, create, harness, language_input, submit  # noqa: F401

ALL_OPERATIONS = {item.operation_id for item in operation_service.list_definitions()}


def _reply(adapter, operation_id: str, arguments: dict, *, generate: bool = False, version: int = 1) -> None:
    adapter.response = json.dumps({"result": {"kind": "operation", "operation_id": operation_id,
                                              "operation_version": version, "arguments": arguments,
                                              "generate_after_save": generate}})


def test_yolo_prompt_adds_guessing_rules_without_changing_normal_mode() -> None:
    normal, yolo = system_prompt(ALL_OPERATIONS), system_prompt(ALL_OPERATIONS, "yolo")
    assert "【自動実行モード】" in yolo and "【自動実行モード】" not in normal
    assert yolo.startswith(normal)


def test_yolo_starts_generation_without_confirmation(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.generation.start", {"kind": "full"})
    before = count(GenerationJob)

    response = submit(client, language_input(project_id, text="動画を作り直して", mode="yolo"))

    assert response["status"] == "completed" and response["executed"] is True
    assert response["execution_mode"] == "yolo"
    assert response["yolo_report"]["auto_confirmed"] == ["project.generation.start"]
    assert count(GenerationJob) == before + 1
    assert adapter.calls and "【自動実行モード】" not in json.dumps(adapter.calls[0]["prompt"], ensure_ascii=False)


def test_yolo_saves_then_generates_in_one_request(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.subtitle-font-size.set", {"value": 64}, generate=True)
    before = count(GenerationJob)

    response = submit(client, language_input(project_id, text="字幕を64pxにして動画も作り直して", mode="yolo"))

    assert response["status"] == "completed", response
    assert response["result"] is not None and response["generation_result"] is not None
    assert response["yolo_report"]["auto_confirmed"] == ["project.subtitle-font-size.set", "project.generation.start"]
    assert count(GenerationJob) == before + 1
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 64


def test_yolo_guesses_where_normal_mode_asks(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.subtitle-font-size.adjust", {"delta": 4})

    normal = submit(client, language_input(project_id, request_id="nl-normal", text="字幕を大きくして"))
    assert normal["status"] == "needs_input" and normal["diagnostics"]["guard_code"] == "subtitle_value"

    response = submit(client, language_input(project_id, request_id="nl-yolo", text="字幕を大きくして", mode="yolo"))
    assert response["status"] == "completed"
    assert response["yolo_report"]["bypassed_guards"] == ["subtitle_value"]
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 52


def test_yolo_never_overrides_an_explicit_negation(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.generation.start", {"kind": "full"})
    before = count(GenerationJob)

    response = submit(client, language_input(project_id, text="動画は生成しないで", mode="yolo"))

    assert response["status"] == "dismissed"
    assert count(GenerationJob) == before


def test_normal_mode_still_requires_generation_confirmation(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.generation.start", {"kind": "full"})
    before = count(GenerationJob)

    response = submit(client, language_input(project_id, text="動画を作り直して"))

    assert response["status"] == "ready" and response["requires_confirmation"] is True
    assert response["execution_mode"] == "normal" and response["yolo_report"] is None
    assert count(GenerationJob) == before


def test_yolo_can_be_disabled_by_the_server(harness) -> None:  # noqa: F811
    client, adapter, service = harness
    service.yolo_enabled = False
    project_id = create(client)
    _reply(adapter, "project.generation.start", {"kind": "full"})

    response = submit(client, language_input(project_id, text="動画を作り直して", mode="yolo"))

    assert response["status"] == "error" and response["failure"]["reason_code"] == "yolo_disabled"
    assert not adapter.calls


def test_negation_only_requests_are_dismissed_before_clarifying(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    # The model picks a different operation and an unstated revision; neither asks nor runs.
    for mode in ("normal", "yolo"):
        _reply(adapter, "project.settings.restore", {"revision": 5})
        response = submit(client, language_input(project_id, request_id=f"nl-neg-{mode}",
                                                 text="第2版には戻さないで", mode=mode))
        assert response["status"] == "dismissed", response
        assert response["clarification"] is None and response["executed"] is False
    _reply(adapter, "project.generation.cancel", {"job_id": 101})
    response = submit(client, language_input(project_id, request_id="nl-neg-cancel", text="動画は作り直さないで"))
    assert response["status"] == "dismissed" and count(GenerationJob) == 0


def test_negated_follow_up_generation_is_dropped_in_both_modes(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    before = count(GenerationJob)
    for mode in ("normal", "yolo"):
        _reply(adapter, "project.subtitle-font-size.set", {"value": 60}, generate=True)
        response = submit(client, language_input(project_id, request_id=f"nl-nogen-{mode}",
                                                 text="字幕を60pxにして、動画は生成しないで", mode=mode))
        assert response["status"] == "completed", response
        assert response["generation_request"] is None and response["generate_after_save"] is False
        if mode == "yolo":
            assert response["yolo_report"]["dropped_steps"] == ["project.generation.start"]
    assert count(GenerationJob) == before
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 60


def test_yolo_never_overrides_a_value_the_user_wrote(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.subtitle-font-size.set", {"value": 80})
    response = submit(client, language_input(project_id, request_id="nl-conflict",
                                             text="字幕を64pxにして", mode="yolo"))
    assert response["status"] == "needs_input" and response["executed"] is False
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] != 80


def test_numbered_negation_with_another_request_still_blocks_the_negated_operation(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.artifact.restore", {"artifact_id": 3})
    response = submit(client, language_input(project_id, request_id="nl-neg-restore",
                                             text="動画3には戻さないで、状態だけ教えて"))
    assert response["status"] == "dismissed" and response["executed"] is False


def test_model_question_about_a_declined_operation_is_dismissed(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    adapter.response = json.dumps({"result": {"kind": "clarification", "question": "戻したい版を指定してください。",
                                              "missing_fields": ["arguments"]}})
    response = submit(client, language_input(project_id, request_id="nl-neg-question", text="第2版には戻さないで"))
    assert response["status"] == "dismissed" and response["clarification"] is None


def test_swapped_stated_values_are_questioned_in_normal_mode_too(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _reply(adapter, "project.settings.update", {"voicevox_speed_scale": 0.8, "voicevox_volume_scale": 1.2})
    response = submit(client, language_input(project_id, request_id="nl-swapped",
                                             text="話速を1.2倍、音量を0.8倍にして"))
    assert response["status"] == "needs_input" and response["executed"] is False
    assert response["diagnostics"]["guard_code"] in {"settings_value", "explicit_value"}
