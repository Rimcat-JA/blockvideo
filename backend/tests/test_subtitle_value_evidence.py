"""Real D23 misinterpretations must ask before any partial save or job creation."""
from __future__ import annotations

import json
from typing import Any

import pytest
import httpx

from app.interpretation.contracts import DialogueContextTurn, OperationProposal
from app.language_operations.subtitle_values import subtitle_question
from app.interpretation.local_chat import LocalChatAdapter
from app.models.job import GenerationJob
from app.models.operation_request import OperationReceipt
from tests.test_language_operations import count, create, harness as harness, language_input, submit


def proposal(arguments: dict, version: int = 2) -> OperationProposal:
    return OperationProposal(kind="operation", operation_id="project.settings.update",
                             operation_version=version, arguments=arguments, generate_after_save=True)


@pytest.mark.parametrize("text,delta,allowed", [
    ("字幕を大きくして、速度を1.2倍にして", 2, False),
    ("字幕を読みやすくして", 2, False),
    ("字幕を大きくして、速度を少し上げて", 2, False),
    ("字幕を少し大きくして、速度を1.2倍に", 2, True),
    ("字幕を少し小さくして", -2, True), ("違う、少し小さく", -2, True),
    ("字幕を少し小さくして", 2, False),
    ("字幕を2px大きくして", 2, True), ("字幕を３ピクセル下げて", -3, True),
    ("字幕を12px大きくして", 2, False), ("字幕を1.2px大きくして", 2, False),
    ("字幕を2px小さくして", 2, False), ("字幕を0px上げて", 0, False),
])
def test_relative_quantity_and_direction(text: str, delta: int, allowed: bool) -> None:
    value = proposal({"settings": {"voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": delta})
    assert (subtitle_question(text, (), None, value) is None) == allowed


@pytest.mark.parametrize("text,value,allowed", [("字幕を６４ｐｘにして", 64, True),
    ("字幕を64ピクセルに", 64, True), ("字幕を大きくして", 64, False),
    ("速度を1.64倍に", 64, False), ("字幕を64pxに", 48, False)])
def test_absolute_evidence(text: str, value: int, allowed: bool) -> None:
    assert (subtitle_question(text, (), None, proposal({"subtitle_font_size": value}, 1)) is None) == allowed


def test_saved_history_cannot_supply_new_values_and_pending_answer_cannot_drop_font() -> None:
    old = DialogueContextTurn(text="字幕を64pxにして、APIの読み方を登録して", status="needs_input",
                              question="APIは何と読みますか？")
    reading = {"pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": None}]}
    assert subtitle_question("エーピーアイ", (old,), "answer", proposal(reading, 1))
    assert subtitle_question("エーピーアイ", (old,), "answer", proposal({**reading, "subtitle_font_size": 64}, 1)) is None
    saved = old.model_copy(update={"status": "ready", "settings_saved": True})
    assert subtitle_question("違う、少し小さく", (saved,), "correction",
                             proposal({"settings": {"subtitle_font_size": 64}, "subtitle_font_size_delta": -2}))
    size_question = old.model_copy(update={"question": "字幕は何pxにしますか？"})
    assert subtitle_question("60", (size_question,), "answer", proposal({"subtitle_font_size": 60}, 1)) is None


def test_hallucinated_delta_saves_nothing_then_explicit_answer_saves_everything(harness: Any) -> None:
    client, adapter, _ = harness
    project = create(client)
    wrong = proposal({"settings": {"voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": 2})
    adapter.response = json.dumps({"result": wrong.model_dump()})
    text = "字幕を大きくして、速度を1.2倍にして、動画を作って"
    first = submit(client, language_input(project, text=text))
    assert first["status"] == "needs_input" and first["interpretation"]["proposal"] == wrong.model_dump()
    assert count(OperationReceipt) == count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project}").json()["revision"] == 1
    # The model can still see and complete the original pending settings.
    good = proposal({"subtitle_font_size": 64, "voicevox_speed_scale": 1.2}, 1)
    adapter.response = json.dumps({"result": good.model_dump()})
    answer = submit(client, language_input(project, request_id="answer", text="64px", continuation={
        "parent_request_id": first["request_id"], "relation": "answer"}))
    assert answer["status"] == "ready" and answer["executed"] and answer["requires_confirmation"]
    assert answer["result"]["revision"] == 2
    assert count(OperationReceipt) == 1 and count(GenerationJob) == 0
    assert submit(client, language_input(project, request_id="answer", text="64px", continuation={
        "parent_request_id": first["request_id"], "relation": "answer"})) == answer


def test_dropped_pending_font_never_partially_saves_reading(harness: Any) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.response = json.dumps({"result": {"kind": "clarification", "question": "APIは何と読みますか？", "missing_fields": ["arguments"]}})
    first = submit(client, language_input(project, text="字幕を64pxにして、APIの読み方を登録して"))
    adapter.response = json.dumps({"result": proposal({"pronunciation_overrides": [
        {"surface": "API", "reading": "エーピーアイ", "accent": None}]}, 1).model_dump()})
    result = submit(client, language_input(project, request_id="answer", text="エーピーアイ", continuation={
        "parent_request_id": first["request_id"], "relation": "answer"}))
    assert result["status"] == "needs_input" and count(OperationReceipt) == count(GenerationJob) == 0


@pytest.mark.parametrize("omit", ["speed", "generation"])
def test_answer_cannot_drop_other_pending_setting_or_generation(harness: Any, omit: str) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.response = json.dumps({"result": proposal({"settings": {"voicevox_speed_scale": 1.2},
        "subtitle_font_size_delta": 2}).model_dump()})
    first = submit(client, language_input(project, text="字幕を大きくして、速度を1.2倍にして、動画を作って"))
    assert first["status"] == "needs_input"
    value = proposal({"subtitle_font_size": 64, **({"voicevox_speed_scale": 1.2} if omit == "generation" else {})}, 1)
    if omit == "generation":
        value = value.model_copy(update={"generate_after_save": False})
    adapter.response = json.dumps({"result": value.model_dump()})
    result = submit(client, language_input(project, request_id="answer", text="64px", continuation={
        "parent_request_id": first["request_id"], "relation": "answer"}))
    assert result["status"] == "needs_input" and count(OperationReceipt) == count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project}").json()["revision"] == 1


async def test_substituted_model_never_reaches_the_writer(harness: Any) -> None:
    client, _, service = harness
    project = create(client)
    calls = []
    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"model": "unexpected", "choices": [{"finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"result": proposal({"subtitle_font_size": 64}, 1).model_dump()})}}]})
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "chosen", transport=httpx.MockTransport(respond)) as adapter:
        from app.db import get_session_factory
        from app.language_operations.contracts import LanguageInput
        service.adapter = adapter
        with get_session_factory()() as db:
            result = await service.submit(db, LanguageInput.model_validate(language_input(project, text="字幕を64pxにして")))
    assert result.status == "error" and result.failure.reason_code == "model_mismatch"
    assert len(calls) == 1 and count(OperationReceipt) == count(GenerationJob) == 0


@pytest.mark.parametrize("settings", [
    {"subtitle_font_size": 72, "narration_sentence_pause_seconds": 0.5},
    {"subtitle_font_size": 72, "voicevox_speed_scale": 1.2, "narration_sentence_pause_seconds": 0.5},
    {"subtitle_font_size": 72},
])
def test_unrequested_number_or_missing_speed_cannot_partially_save(harness: Any, settings: dict) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.response = json.dumps({"result": proposal(settings, 1).model_dump()})
    result = submit(client, language_input(project, text="字幕を72pxにして、速度を1.2倍にして。動画を作り直して"))
    assert result["status"] == "needs_input"
    assert count(OperationReceipt) == count(GenerationJob) == 0
    assert client.get(f"/api/projects/{project}").json()["revision"] == 1
