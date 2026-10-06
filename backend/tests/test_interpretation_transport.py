"""Local transport verifies actual wire shape and fails closed on HTTP errors."""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import LocalChatAdapter, local_base_url
from app.interpretation.transport import ModelMessage

MESSAGES = (ModelMessage("system", "合成の指示"), ModelMessage("user", "合成の要求"))
SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def envelope(content: Any = "{}", **message_fields: Any) -> dict[str, Any]:
    return {"model": "synthetic", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": content, **message_fields,
    }}]}


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "http://192.168.1.10:1234/v1",
    "http://127.0.0.1.evil.example:1234/v1", "http://127.0.0.1:1234/v1?api_key=private",
    "http://user:private@127.0.0.1:1234/v1", "http://127.0.0.1:1234/v1#private",
    "http://127.0.0.1:1234/other", "http://127.0.0.1/v1", "http://127.0.0.1:0/v1",
    "http://127.0.0.1:70000/v1", "http://[::1:1234/v1", "file:///v1", "",
])
def test_nonlocal_or_ambiguous_endpoints_never_connect(url: str) -> None:
    with pytest.raises(InterpretationError) as caught:
        local_base_url(url)
    assert caught.value.code == "configuration_error"
    assert url not in str(caught.value) or not url


@pytest.mark.parametrize("url,expected", [
    ("http://localhost:1234/v1/", "http://127.0.0.1:1234/v1"),
    ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
    ("http://[::1]:1234/v1", "http://[::1]:1234/v1"),
])
def test_loopback_urls_are_normalized(url: str, expected: str) -> None:
    assert local_base_url(url) == expected


@pytest.mark.parametrize("reasoning_effort", [None, "none"])
async def test_wire_schema_strict_completion_and_no_environment_secrets(
    monkeypatch: pytest.MonkeyPatch, reasoning_effort: str | None,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "DO-NOT-SEND")
    monkeypatch.setenv("HTTP_PROXY", "http://secret-proxy.invalid")
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=envelope('{"result":{}}'))

    async with LocalChatAdapter("http://localhost:1234/v1", "synthetic", reasoning_effort=reasoning_effort,
                                transport=httpx.MockTransport(respond)) as adapter:
        assert await adapter.complete(MESSAGES, SCHEMA) == '{"result":{}}'
    assert len(seen) == 1
    request = seen[0]
    payload = json.loads(request.content)
    assert str(request.url) == "http://127.0.0.1:1234/v1/chat/completions"
    assert payload["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "blockvideo_proposal", "strict": True, "schema": SCHEMA,
    }}
    assert payload["temperature"] == 0 and payload["stream"] is False
    assert payload["max_tokens"] == 768
    if reasoning_effort is None:
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == "none"
    assert "tools" not in payload and "Authorization" not in request.headers
    assert "DO-NOT-SEND" not in request.content.decode()


def test_unknown_reasoning_mode_is_rejected_before_client_creation() -> None:
    with pytest.raises(InterpretationError) as caught:
        LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", reasoning_effort="unsupported")
    assert caught.value.code == "configuration_error"


@pytest.mark.parametrize("status", [301, 307, 400, 401, 429, 500, 503])
async def test_http_failure_is_sanitized_no_retry_or_redirect(status: int) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="private-echo max_tokens max_completion_tokens",
                              headers={"location": "https://remote.invalid/v1"})

    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=httpx.MockTransport(respond)) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == "http_error"
    assert "private-echo" not in str(caught.value)
    assert calls == 1


@pytest.mark.parametrize("exc_type,code", [
    (httpx.ConnectError, "connection_failed"), (httpx.RemoteProtocolError, "connection_failed"),
    (httpx.ReadTimeout, "timeout"), (httpx.ConnectTimeout, "timeout"),
])
async def test_network_error_is_sanitized_without_retry(exc_type: type, code: str) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exc_type("private credential or request echo", request=request)

    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=httpx.MockTransport(respond)) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == code
    assert "private" not in str(caught.value)
    assert calls == 1


@pytest.mark.parametrize("body,code", [
    ({}, "invalid_response"), ({"choices": []}, "invalid_response"),
    ({"choices": [None]}, "invalid_response"),
    (envelope(None), "invalid_response"), (envelope([]), "invalid_response"),
    (envelope({}, role="assistant"), "invalid_response"),
    (envelope("{}", role="tool"), "invalid_response"),
    (envelope("{}", tool_calls=[{"function": {"name": "execute"}}]), "invalid_response"),
    (envelope("{}", function_call={"name": "execute"}), "invalid_response"),
    (envelope("{}", refusal="private refusal"), "refused"),
    ({"choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": "length"}]}, "incomplete_response"),
    ({"choices": [{"message": {"role": "assistant", "content": "{}"}, "finish_reason": None}]}, "incomplete_response"),
    ({"choices": envelope()["choices"] * 2}, "invalid_response"),
])
async def test_envelope_must_have_one_complete_text_choice(body: Any, code: str) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=transport) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == code


@pytest.mark.parametrize("content,code", [
    (b"not-json-private", "invalid_response"), (b"\xff", "invalid_response"),
    (b'{"choices":[],"choices":[]}', "invalid_response"),
    (b"X" * 131_073, "response_too_large"),
], ids=["text", "utf8", "duplicate", "oversized"])
async def test_malformed_and_oversized_wire_body(content: bytes, code: str) -> None:
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=content),
    )) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == code


async def test_whole_request_deadline_stops_slow_drip_response() -> None:
    closed = False

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(20):
                await asyncio.sleep(0.15)
                yield b" "

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=SlowStream()))
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic", timeout_seconds=1, transport=transport) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == "timeout"
    assert closed


@pytest.mark.parametrize("reported", [None, "different-local-model", 42, "synthetic-alias"])
async def test_server_side_model_substitution_is_rejected(reported: Any) -> None:
    body = {**envelope('{"result":{}}'), "model": reported}
    async with LocalChatAdapter("http://127.0.0.1:1234/v1", "synthetic",
                                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as adapter:
        with pytest.raises(InterpretationError) as caught:
            await adapter.complete(MESSAGES, SCHEMA)
    assert caught.value.code == "model_mismatch"
