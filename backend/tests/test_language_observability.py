"""D25 diagnostics preserve replay/effects and exclude free-form log data."""
from __future__ import annotations

import asyncio
import json
import logging

import pytest
import httpx
from fastapi.testclient import TestClient
from loguru import logger

from app.api.routes_language import language_service
from app.core.access_logging import RequestPathFilter
from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import LocalChatAdapter
from app.language_operations import observability
from app.language_operations import repository
from app.db import get_session_factory
from app.language_operations.contracts import LanguageResponse
from app.language_operations.service import LanguageOperationService
from app.main import create_app
from app.operations.bootstrap import operation_service


class Adapter:
    def __init__(self) -> None:
        self.calls = 0
        self.proposal = {"kind": "operation", "operation_id": "project.subtitle-font-size.set",
                         "operation_version": 1, "arguments": {"value": 56}}

    async def complete(self, messages, schema) -> str:
        self.calls += 1
        await asyncio.sleep(0.015)
        return json.dumps({"result": self.proposal})


@pytest.fixture()
def harness(temp_storage):
    adapter = Adapter()
    service = LanguageOperationService(operation_service, adapter)
    app = create_app()
    app.dependency_overrides[language_service] = lambda: service
    client = TestClient(app)
    try:
        project = client.post("/api/projects", json={"title": "PRIVATE-TITLE", "source_script": "PRIVATE-SCRIPT",
            "subtitle_font_size": 48, "use_fake_providers": True}).json()
        yield client, adapter, project["id"]
    finally:
        client.close()


def payload(project: int, request_id: str = "PRIVATE-ID") -> dict:
    return {"request_id": request_id, "text": "PRIVATE-UTTERANCE 字幕を56pxにして",
            "target": {"selected_project_id": project}, "base_revision": 1}


def test_timings_candidates_and_guard_survive_exact_replay(harness) -> None:
    client, adapter, project = harness
    original = client.post("/api/language/requests", json=payload(project)).json()
    assert original["status"] == "completed"
    metrics = original["diagnostics"]
    assert metrics["started_at"] > 0 and metrics["interpretation_ms"] >= 10
    assert metrics["execution_ms"] >= 0 and metrics["generation_execution_ms"] is None
    assert len(metrics["candidates"]) == 10
    replay = client.post("/api/language/requests", json=payload(project)).json()
    lookup = client.get("/api/language/requests/PRIVATE-ID").json()
    assert replay["diagnostics"] == lookup["diagnostics"] == metrics
    assert replay["result"] == original["result"] and adapter.calls == 1


def test_guard_is_distinct_from_original_model_proposal(harness) -> None:
    client, _, project = harness
    request = {**payload(project), "text": "字幕を大きくして"}
    result = client.post("/api/language/requests", json=request).json()
    assert result["status"] == "needs_input" and result["result"] is None
    assert result["interpretation"]["proposal"]["arguments"] == {"value": 56}
    assert result["diagnostics"]["guard_code"] == "subtitle_value"
    assert result["diagnostics"]["execution_ms"] is None


def test_compound_save_and_confirm_keep_separate_durable_timings(harness) -> None:
    client, adapter, project = harness
    adapter.proposal = {"kind": "operation", "operation_id": "project.settings.update", "operation_version": 1,
        "arguments": {"subtitle_font_size": 56}, "generate_after_save": True}
    request = {**payload(project), "text": "字幕を56pxにして動画を作り直して"}
    saved = client.post("/api/language/requests", json=request).json()
    assert saved["status"] == "ready" and saved["result"]["revision"] == 2
    assert saved["diagnostics"]["execution_ms"] is not None
    assert saved["diagnostics"]["generation_execution_ms"] is None
    confirmation = {"confirmation_token": saved["confirmation_token"], "confirm_generation": True}
    result = client.post("/api/language/requests/PRIVATE-ID/execute", json=confirmation).json()
    repeated = client.post("/api/language/requests/PRIVATE-ID/execute", json=confirmation).json()
    assert result["generation_result"]["job_id"] is not None
    assert result["diagnostics"]["interpretation_ms"] == saved["diagnostics"]["interpretation_ms"]
    assert result["diagnostics"]["execution_ms"] == saved["diagnostics"]["execution_ms"]
    assert result["diagnostics"]["generation_execution_ms"] is not None
    assert repeated["diagnostics"] == result["diagnostics"] and adapter.calls == 1


def test_late_acknowledgement_cannot_replace_a_recorded_duration(harness) -> None:
    client, _, project = harness
    response = LanguageResponse.model_validate(client.post("/api/language/requests", json=payload(project)).json())
    later = response.model_copy(update={"diagnostics": response.diagnostics.model_copy(update={"execution_ms": 999999})})
    with get_session_factory()() as db:
        acknowledged = repository.acknowledge(db, response.request_id, later)
    assert acknowledged.diagnostics == response.diagnostics
    assert client.get("/api/language/requests/PRIVATE-ID").json()["diagnostics"] == response.diagnostics.model_dump(mode="json")


def test_logged_request_is_pseudonymous_and_text_is_absent(harness) -> None:
    client, _, project = harness
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        result = client.post("/api/language/requests", json=payload(project)).json()
    finally:
        logger.remove(sink)
    events = [json.loads(message.split("language_event ", 1)[1]) for message in messages if "language_event " in message]
    assert {event["event"] for event in events} == {"prepared", "executed"}
    text = json.dumps(events)
    for secret in ("PRIVATE-ID", "PRIVATE-UTTERANCE", "PRIVATE-TITLE", "PRIVATE-SCRIPT", result["confirmation_token"]):
        assert secret not in text
    assert events[0]["request"] == events[1]["request"]
    assert events[-1]["resolved"] == {"value": 56}


def test_sensitive_nested_and_unknown_values_never_enter_event() -> None:
    values = {"settings": {"pronunciation_overrides": [{"surface": "SECRET", "reading": "秘密"}],
        "voicevox_speed_scale": "TOKEN", "private-key": "PASSWORD", "settings": {"settings": "DEEP"}},
        "job_id": "CLIENT-SECRET", "kind": "SECRET-KIND", "secret-name": 99}
    text = json.dumps(observability.safe_arguments(values), ensure_ascii=False)
    for secret in ("SECRET", "秘密", "TOKEN", "PASSWORD", "private-key", "DEEP", "CLIENT", "secret-name"):
        assert secret not in text


def test_logging_failure_cannot_turn_committed_save_into_failure(harness, monkeypatch) -> None:
    client, _, project = harness
    def broken(*args, **kwargs):
        raise RuntimeError("SECRET-SINK-FAILURE")
    monkeypatch.setattr(observability.log, "info", broken)
    result = client.post("/api/language/requests", json=payload(project)).json()
    assert result["status"] == "completed" and result["result"]["revision"] == 2


def test_legacy_response_loads_without_inventing_measurements() -> None:
    response = LanguageResponse.model_validate({"request_id": "old", "core_request_id": "old-core", "status": "completed"})
    assert response.diagnostics.interpretation_ms is None
    assert response.diagnostics.execution_ms is None
    assert response.diagnostics.started_at is None


def test_untrusted_failure_message_and_reason_are_not_logged() -> None:
    response = LanguageResponse(request_id="x", core_request_id="y", status="error",
        failure={"reason_code": "TOKEN-as-code", "message": "PRIVATE-FAILURE"})
    event = observability.diagnostic_event(response, "prepared")
    assert event["failure_code"] == "other"
    assert "TOKEN" not in json.dumps(event) and "PRIVATE-FAILURE" not in json.dumps(event)


def test_access_log_removes_client_request_id_and_query() -> None:
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:123", "POST", "/api/language/requests/PRIVATE-ID/execute?token=SECRET", "1.1", 200), None)
    assert RequestPathFilter().filter(record)
    assert "/api/language/requests/:request/execute" in record.getMessage()
    assert "PRIVATE-ID" not in record.getMessage() and "SECRET" not in record.getMessage()


def test_unknown_access_log_format_is_not_forwarded_with_raw_url() -> None:
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, "PRIVATE-URL", (), None)
    assert not RequestPathFilter().filter(record)


@pytest.mark.asyncio
async def test_http_failure_exposes_only_status_without_server_body() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503, text="SECRET-UPSTREAM-BODY"))
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic-test", transport=transport) as adapter:
        with pytest.raises(InterpretationError) as error:
            await adapter.complete((), {})
    view = error.value.as_view()
    assert view.http_status == 503 and view.reason_code == "http_error"
    assert "SECRET" not in view.model_dump_json()
