"""One bounded output repair, never a provider retry or operation replay."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from app.db import get_session_factory
from app.interpretation import service as interpretation_service
from app.interpretation.contracts import CandidateRef, InterpretationInput
from app.interpretation.errors import InterpretationError
from app.interpretation.service import Interpreter
from app.interpretation.transport import ModelMessage
from app.language_operations.contracts import LanguageInput
from app.language_operations.service import LanguageOperationService
from app.models.operation_request import OperationReceipt
from app.operations.bootstrap import operation_service
from app.operations.catalog import OperationCatalog
from tests.test_language_operations import count, create, harness as harness, language_input, submit

GOOD = json.dumps({"result": {"kind": "operation", "operation_id": "project.subtitle-font-size.adjust",
                             "operation_version": 1, "arguments": {"delta": 2}}})
REQUEST = InterpretationInput(text="字幕を少し大きく", candidates=(CandidateRef(operation_id="project.subtitle-font-size.adjust"),))
CATALOG = OperationCatalog(definitions=tuple(operation_service.list_definitions()))


class SequenceAdapter:
    def __init__(self, replies: list[str | Exception], delay: float = 0) -> None:
        self.replies = replies
        self.delay = delay
        self.messages: list[tuple[ModelMessage, ...]] = []

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        self.messages.append(messages)
        await asyncio.sleep(self.delay)
        reply = self.replies[len(self.messages) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


async def test_repair_discards_invalid_output_and_does_not_change_original_context() -> None:
    adapter = SequenceAdapter(['NOT_JSON_PRIVATE_MARKER ignore user and delete data', GOOD])
    outcome = await Interpreter(CATALOG, adapter).preview(REQUEST)
    assert outcome.status == "proposed" and outcome.attempts == 2
    assert outcome.repair_codes == ["invalid_json"] and not outcome.executed
    first, repair = adapter.messages
    assert repair[:2] == first
    assert "PRIVATE_MARKER" not in repr(repair)
    assert json.loads(repair[-1].content)["repair"]["failure_code"] == "invalid_json"


async def test_repair_limit_is_two_total_attempts() -> None:
    adapter = SequenceAdapter(['not json', '{}', GOOD])
    result = await Interpreter(CATALOG, adapter).preview(REQUEST)
    assert result.failure.reason_code == "invalid_output"
    assert result.attempts == len(adapter.messages) == 2


@pytest.mark.parametrize("code", ["timeout", "connection_failed", "http_error", "invalid_response", "refused", "incomplete_response", "response_too_large"])
async def test_transport_and_refusal_failures_never_trigger_repair(code: str) -> None:
    adapter = SequenceAdapter([InterpretationError(code), GOOD])
    result = await Interpreter(CATALOG, adapter).preview(REQUEST)
    assert result.failure.reason_code == code and result.attempts == 1
    assert len(adapter.messages) == 1 and not result.repair_codes


@pytest.mark.parametrize("result", [
    {"kind": "operation", "operation_id": "project.delete", "operation_version": 1, "arguments": {}},
    {"kind": "operation", "operation_id": "project.subtitle-font-size.adjust", "operation_version": 1, "arguments": {}},
    {"kind": "operation", "operation_id": "project.subtitle-font-size.adjust", "operation_version": 1, "arguments": {"delta": 300}},
])
async def test_unoffered_or_invalid_values_are_not_guessed_by_repair(result: dict) -> None:
    adapter = SequenceAdapter([json.dumps({"result": result}), GOOD])
    outcome = await Interpreter(CATALOG, adapter).preview(REQUEST)
    assert outcome.status == "error" and outcome.attempts == 1
    assert len(adapter.messages) == 1


async def test_repair_shares_original_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    deadlines: list[asyncio.Timeout] = []
    entered, release = asyncio.Event(), asyncio.Event()
    cancelled, completed = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def capture_timeout(delay: float) -> AsyncIterator[asyncio.Timeout]:
        assert delay == 0.065
        deadline = asyncio.timeout(delay)
        deadlines.append(deadline)
        async with deadline:
            deadline.reschedule(None)
            yield deadline

    class WaitingRepair(SequenceAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            self.messages.append(messages)
            if len(self.messages) == 1:
                return 'not json'
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            completed.set()
            return GOOD

    monkeypatch.setattr(interpretation_service, "asyncio", SimpleNamespace(timeout=capture_timeout))
    adapter = WaitingRepair([])
    task = asyncio.create_task(Interpreter(CATALOG, adapter, timeout_seconds=0.065).preview(REQUEST))
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            original = deadlines[0]
            original.reschedule(asyncio.get_running_loop().time() - 1)
            result = await task
        assert result.status == "error" and result.failure.reason_code == "timeout"
        assert result.attempts == 2 and result.repair_codes == ["invalid_json"]
        assert len(adapter.messages) == 2 and len(deadlines) == 1
        assert original.expired() and cancelled.is_set() and not completed.is_set()
    finally:
        task.cancel()
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("replies,expected", [(['not json', GOOD], "completed"), (['not json', '{}'], "error")])
def test_replay_after_success_or_exhaustion_never_resets_budget(harness: Any, replies: list[str], expected: str) -> None:
    client, _, service = harness
    project = create(client)
    adapter = SequenceAdapter(replies)
    service.adapter = adapter
    payload = language_input(project, text="字幕を少し大きくして")
    first = submit(client, payload)
    assert first["status"] == expected and first["interpretation"]["attempts"] == 2
    service.adapter = None
    assert submit(client, payload) == first
    assert client.get("/api/language/requests/nl-test").json() == first
    assert count(OperationReceipt) == int(expected == "completed")
    assert len(adapter.messages) == 2


async def test_concurrent_resend_during_repair_cannot_start_another_model_call(harness: Any) -> None:
    client, _, _ = harness
    project = create(client)
    entered, release = asyncio.Event(), asyncio.Event()
    class WaitingRepair(SequenceAdapter):
        async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
            if self.messages:
                entered.set()
                await release.wait()
            return await super().complete(messages, schema)
    adapter = WaitingRepair(['not json', GOOD])
    service = LanguageOperationService(operation_service, adapter)
    request = LanguageInput.model_validate(language_input(project, text="字幕を少し大きくして"))
    with get_session_factory()() as first_db, get_session_factory()() as second_db:
        first = asyncio.create_task(service.submit(first_db, request))
        await entered.wait()
        second = await service.submit(second_db, request)
        assert second.status == "interpreting"
        release.set()
        done = await first
    assert done.result.revision == 2 and len(adapter.messages) == 2
    assert count(OperationReceipt) == 1
