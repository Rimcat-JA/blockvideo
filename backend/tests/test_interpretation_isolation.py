"""Model proposals cannot change projects, durable receipts, jobs or artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from app.db import Base, get_session_factory
from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.service import Interpreter
from app.interpretation.transport import ModelMessage
from app.main import create_app
from app.operations.catalog import load_catalog
from app.operations.contracts import OperationRequest


def database_snapshot() -> dict[str, list[dict[str, Any]]]:
    with get_session_factory()() as db:
        return {table.name: [dict(row) for row in db.execute(select(table)).mappings()]
                for table in Base.metadata.sorted_tables}


def file_snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("operation_id,arguments", [
    ("project.subtitle-font-size.set", {"value": 56}),
    ("project.generation.start", {"kind": "full"}),
    ("project.settings.restore", {"revision": 1}),
    ("project.generation.cancel", {"job_id": 1}),
])
async def test_receiving_valid_mutation_and_generation_proposals_never_writes(
    temp_storage: Path, operation_id: str, arguments: dict[str, Any],
) -> None:
    client = TestClient(create_app())
    project = client.post("/api/projects", json={
        "title": "Synthetic boundary test", "source_script": "合成原稿・送信禁止",
        "use_fake_providers": True, "subtitle_font_size": 48,
    }).json()
    proposal = {"kind": "operation", "operation_id": operation_id,
                "operation_version": 1, "arguments": arguments}
    sent: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(request.content)
        return httpx.Response(200, json={"model": "synthetic", "choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps({"result": proposal}),
        }}]})

    catalog = load_catalog(Path(__file__).resolve().parents[1] / "app/operations/definitions.json")
    db_before = database_snapshot()
    files_before = file_snapshot(temp_storage)
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=httpx.MockTransport(respond)) as adapter:
        result = await Interpreter(catalog, adapter).preview(InterpretationInput(
            text="合成要求", candidates=(CandidateRef(operation_id=operation_id),),
            state=MinimalState(selected_project_id=project["id"], revision=1, subtitle_font_size=48),
        ))
    assert result.status == "proposed" and result.executed is False
    assert database_snapshot() == db_before
    assert file_snapshot(temp_storage) == files_before
    assert "合成原稿・送信禁止" not in sent[0].decode()
    with pytest.raises(ValidationError):
        OperationRequest.model_validate(result.proposal.model_dump())
    response = client.post("/api/operations/execute", json=result.model_dump(mode="json"))
    assert response.status_code == 422
    assert database_snapshot() == db_before


async def test_adapter_replacement_and_failure_display_need_no_core_changes() -> None:
    class OtherAdapter:
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            return '{"result":{"kind":"unsupported","reason":"合成の未対応要求です。"}}'

    catalog = load_catalog(Path(__file__).resolve().parents[1] / "app/operations/definitions.json")
    request = InterpretationInput(text="合成要求", candidates=(CandidateRef(operation_id="project.status.get"),))
    other_result = await Interpreter(catalog, OtherAdapter()).preview(request)
    assert other_result.status == "unsupported" and other_result.executed is False

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private echoed source", request=request)

    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=httpx.MockTransport(fail)) as adapter:
        failed = await Interpreter(catalog, adapter).preview(request)
    assert failed.status == "error" and failed.executed is False
    assert failed.failure.reason_code == "timeout"
    assert "制限時間" in failed.failure.message
    assert "private" not in failed.model_dump_json()
