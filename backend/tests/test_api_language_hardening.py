"""Request validation, guards and failure reporting that used to break or mislead."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes_language import language_service
from app.core.provider_errors import ProviderError
from app.db import get_session_factory
from app.interpretation.contracts import OperationProposal
from app.language_operations.pronunciation import reading_question
from app.language_operations.service import LanguageOperationService
from app.main import create_app
from app.models.block import Block
from app.models.project import Project
from app.operations.bootstrap import operation_service
from app.services import voice
from app.workers.job_runner import _failure_message
from tests.test_language_operations import ReplyAdapter, create, language_input


@pytest.fixture()
def harness(temp_storage: Path):
    adapter = ReplyAdapter()
    app = create_app()
    app.dependency_overrides[language_service] = lambda: LanguageOperationService(operation_service, adapter)
    client = TestClient(app, raise_server_exceptions=True)
    try:
        yield client, adapter
    finally:
        client.close()


def test_multi_line_language_request_is_accepted(harness: Any) -> None:
    client, _ = harness
    project_id = create(client)
    response = client.post("/api/language/requests",
                           json=language_input(project_id, text="字幕を56pxにして\n話速はそのままで"))
    assert response.status_code == 200, response.text


def test_control_characters_other_than_whitespace_are_still_rejected(harness: Any) -> None:
    client, _ = harness
    project_id = create(client)
    response = client.post("/api/language/requests", json=language_input(project_id, text="字幕\x00を56pxに"))
    assert response.status_code == 422


def _settings_proposal(arguments: dict[str, Any]) -> OperationProposal:
    return OperationProposal(kind="operation", operation_id="project.settings.update", operation_version=1,
                             arguments=arguments)


def test_empty_reading_list_asks_instead_of_erasing() -> None:
    question = reading_question(["APIの読み方の登録を消して"], _settings_proposal({"pronunciation_overrides": []}))
    assert question is not None


def test_accent_written_with_ha_counts_as_stated() -> None:
    proposal = _settings_proposal({"pronunciation_overrides": [
        {"surface": "API", "reading": "エーピーアイ", "accent": 0}]})
    assert reading_question(["APIの読み方はエーピーアイ、アクセントは0にして"], proposal) is None


def test_block_patch_rejects_wrong_types_and_nan(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)
    with get_session_factory()() as db:
        block = Block(project_id=project_id, index=0, source_text="本文。", tts_text="本文。")
        db.add(block)
        db.commit()
        block_id = block.id
    bad_type = client.patch(f"/api/blocks/{block_id}",
                            json={"visual_plan": {"visual_type": "text_slide", "visual_summary": 123}})
    assert bad_type.status_code == 422
    nan = client.patch(f"/api/blocks/{block_id}", content=json.dumps(
        {"visual_plan": {"visual_type": "text_slide", "heading": "x", "font_scale": float("nan")}}),
        headers={"Content-Type": "application/json"})
    assert nan.status_code == 422


def test_huge_integer_argument_is_invalid_not_a_crash(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)
    body = ('{"operation_id": "project.settings.update", "operation_version": 1, '
            f'"target": {{"project_id": {project_id}}}, "arguments": {{"voicevox_speed_scale": 1{"0" * 400}}}}}')
    response = client.post("/api/operations/readiness", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422, response.text


def test_render_only_operation_refuses_stale_media(temp_storage: Path) -> None:
    client = TestClient(create_app())
    project_id = create(client)
    with get_session_factory()() as db:
        db.add(Block(project_id=project_id, index=0, source_text="本文。", tts_text="本文。"))
        db.commit()
        revision = db.get(Project, project_id).revision
    response = client.post("/api/operations/execute", json={
        "operation_id": "project.generation.start", "operation_version": 1, "request_id": "op-stale",
        "target": {"project_id": project_id}, "base_revision": revision, "arguments": {"kind": "rerender"}})
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["reason_code"] == "stale_media"


def test_safe_provider_errors_reach_the_job_message() -> None:
    message = _failure_message(ProviderError("VOICEVOX Engineに接続できません (http://x) sk-abcdefghijklmnopqrstuvwxyz123456",
                                             safe=True))
    assert "VOICEVOX Engineに接続できません" in message
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in message
    assert _failure_message(ValueError("internal detail")) == "生成処理に失敗しました (ValueError)"


def test_voicevox_permanent_4xx_is_not_retried() -> None:
    calls = 0

    async def failing() -> bytes:
        nonlocal calls
        calls += 1
        raise ProviderError("VOICEVOX audio_query失敗 (status 422)", safe=True)

    with pytest.raises(ProviderError):
        asyncio.run(voice._with_retries(failing, max_attempts=3))
    assert calls == 1
