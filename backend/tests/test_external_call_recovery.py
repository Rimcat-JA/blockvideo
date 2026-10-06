"""D14 synthetic HTTP/crash tests: unknown remote work never gets blindly replayed."""
from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.models.external_call import ExternalCall
from app.providers.image_openai import OpenAIImageProvider
from app.providers.llm import LLMMessage, LLMRequest, ProviderError, safe_chat_json
from app.providers.llm_openai import OpenAICompatibleProvider
from app.providers.voicevox import VoicevoxClient
from app.services import external_calls, splitter, visual_planner
from app.services.external_calls import (
    ExternalOutcomeUnknown,
    has_unresolved_calls,
    job_call_context,
    journaled_post,
    mark_interrupted_calls,
)
from app.services.transactions import atomic_write


URL = "https://synthetic.invalid/v1/chat/completions"
PAYLOAD = {"model": "synthetic", "messages": [{"role": "user", "content": "synthetic private source"}]}


@pytest.mark.asyncio
async def test_success_cache_survives_new_client_and_commits_before_network(temp_storage) -> None:
    requests = []

    async def send(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        with get_session_factory()() as db, atomic_write(db):
            call = db.scalar(select(ExternalCall))
            assert call.status == "in_flight"
            assert call.response_body is None
        return httpx.Response(200, json={"answer": "saved"}, headers={"x-request-id": "provider-id"})

    for _ in range(2):
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            with job_call_context(101):
                response = await journaled_post(client, URL, provider="chat", json=PAYLOAD,
                                                headers={"Authorization": "Bearer synthetic-secret"})
                assert response.json() == {"answer": "saved"}
    assert len(requests) == 1
    with get_session_factory()() as db:
        call = db.scalar(select(ExternalCall))
        assert call.status == "succeeded"
        assert call.provider_response_id == "provider-id"
        assert call.attempts == 1
        assert not has_unresolved_calls(db, 101)
        assert "synthetic-secret" not in repr(vars(call))
        assert "synthetic private source" not in repr(vars(call))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "disconnect", "http_408", "http_500"])
async def test_unknown_remote_outcome_is_persisted_and_not_repeated(temp_storage, failure: str) -> None:
    calls = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("contains synthetic-secret", request=request)
        if failure == "disconnect":
            raise httpx.RemoteProtocolError("disconnected", request=request)
        return httpx.Response(int(failure.removeprefix("http_")), text="synthetic private source")

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        for payload in [PAYLOAD, PAYLOAD, {"model": "changed-after-unknown"}]:
            with job_call_context(102), pytest.raises(ExternalOutcomeUnknown) as exc:
                await journaled_post(client, URL, provider="chat", json=payload)
            assert "synthetic-secret" not in str(exc.value)
            assert "synthetic private source" not in str(exc.value)
    assert calls == 1
    with get_session_factory()() as db:
        call = db.scalar(select(ExternalCall))
        assert call.status == "unknown"
        assert call.response_body is None
        assert has_unresolved_calls(db, 102)


@pytest.mark.asyncio
async def test_missing_success_cache_is_unknown_instead_of_repeating_paid_request(temp_storage) -> None:
    calls = []

    def send(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"saved": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with job_call_context(103):
            await journaled_post(client, URL, provider="chat", json=PAYLOAD)
        with get_session_factory()() as db:
            db.scalar(select(ExternalCall)).response_body = None
            db.commit()
        with job_call_context(103), pytest.raises(ExternalOutcomeUnknown):
            await journaled_post(client, URL, provider="chat", json=PAYLOAD)
    assert len(calls) == 1
    with get_session_factory()() as db:
        assert has_unresolved_calls(db, 103)
        assert db.scalar(select(ExternalCall)).error_code == "cached_response_missing"


@pytest.mark.asyncio
async def test_concurrent_identical_calls_share_one_network_result(temp_storage) -> None:
    calls = 0

    async def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return httpx.Response(200, json={"saved": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with job_call_context(104):
            results = await asyncio.gather(*(
                journaled_post(client, URL, provider="chat", json=PAYLOAD) for _ in range(6)
            ))
    assert calls == 1
    assert all(response.json() == {"saved": True} for response in results)


@pytest.mark.asyncio
async def test_local_voice_computation_can_retry_after_transport_failure(temp_storage) -> None:
    calls = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("local timeout", request=request)
        return httpx.Response(200, content=b"synthetic-wav")

    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with job_call_context(105):
            with pytest.raises(httpx.ReadTimeout):
                await journaled_post(client, "http://127.0.0.1:50021/synthesis", provider="voicevox",
                                     remote_side_effect=False, json={"query": "synthetic"})
            result = await journaled_post(client, "http://127.0.0.1:50021/synthesis", provider="voicevox",
                                          remote_side_effect=False, json={"query": "synthetic"})
    assert result.content == b"synthetic-wav"
    assert calls == 2
    with get_session_factory()() as db:
        call = db.scalar(select(ExternalCall))
        assert call.attempts == 2
        assert call.status == "succeeded"
        assert not has_unresolved_calls(db, 105)


def test_startup_distinguishes_remote_unknown_from_local_retryable(temp_storage) -> None:
    with get_session_factory()() as db:
        for remote, job_id in [(True, 106), (False, 107)]:
            db.add(ExternalCall(job_id=job_id, fingerprint=str(job_id), provider="synthetic",
                                endpoint=URL, remote_side_effect=remote, status="in_flight"))
        db.commit()
        with atomic_write(db):
            assert mark_interrupted_calls(db) == 2
        assert has_unresolved_calls(db, 106)
        assert not has_unresolved_calls(db, 107)
        assert {call.status for call in db.scalars(select(ExternalCall))} == {"unknown", "failed"}
        assert mark_interrupted_calls(db) == 0


@pytest.mark.asyncio
async def test_voicevox_adapter_can_retry_known_local_server_failure(temp_storage) -> None:
    calls = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls == 1 else httpx.Response(200, content=b"synthetic-wav")

    voice = VoicevoxClient("http://127.0.0.1:50021")
    await voice._client.aclose()
    voice._client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    try:
        with job_call_context(114):
            with pytest.raises(ProviderError):
                await voice.synthesis({"synthetic": 1}, 1)
            assert await voice.synthesis({"synthetic": 1}, 1) == b"synthetic-wav"
    finally:
        await voice.aclose()
    assert calls == 2
    with get_session_factory()() as db:
        call = db.scalar(select(ExternalCall))
        assert call.attempts == 2
        assert not call.remote_side_effect


@pytest.mark.asyncio
async def test_nonlocal_voicevox_timeout_is_conservatively_unknown(temp_storage) -> None:
    calls = 0

    def send(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("synthetic remote voice timeout", request=request)

    voice = VoicevoxClient("https://synthetic.invalid/voice")
    await voice._client.aclose()
    voice._client = httpx.AsyncClient(transport=httpx.MockTransport(send))
    try:
        with job_call_context(115):
            for _ in range(2):
                with pytest.raises(ExternalOutcomeUnknown):
                    await voice.synthesis({"synthetic": 1}, 1)
    finally:
        await voice.aclose()
    assert calls == 1
    with get_session_factory()() as db:
        assert has_unresolved_calls(db, 115)


@pytest.mark.asyncio
async def test_response_save_failure_keeps_ambiguous_call_visible(temp_storage, monkeypatch) -> None:
    def fail_save(*args, **kwargs) -> None:
        raise OSError("synthetic full disk")

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text="done"))) as client:
        monkeypatch.setattr(external_calls, "_finish", fail_save)
        with job_call_context(108), pytest.raises(ExternalOutcomeUnknown):
            await journaled_post(client, URL, provider="chat", json=PAYLOAD)
    with get_session_factory()() as db:
        assert has_unresolved_calls(db, 108)
        assert db.scalar(select(ExternalCall)).status == "in_flight"


@pytest.mark.asyncio
async def test_chat_known_compatibility_rejection_and_success_are_both_cached(temp_storage) -> None:
    calls = []

    def send(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if b'"max_tokens"' in request.content:
            return httpx.Response(400, text="use max_completion_tokens instead of max_tokens")
        return httpx.Response(200, json={"choices": [{"message": {"content": "saved"}}]})

    for _ in range(2):
        client = httpx.AsyncClient(transport=httpx.MockTransport(send))
        provider = OpenAICompatibleProvider("synthetic-secret", "https://synthetic.invalid/v1", "test", client=client)
        try:
            with job_call_context(109):
                result = await provider.chat(LLMRequest([LLMMessage("user", "synthetic")], max_tokens=10))
                assert result.content == "saved"
        finally:
            await provider.aclose()
    assert len(calls) == 2
    with get_session_factory()() as db:
        assert {call.status for call in db.scalars(select(ExternalCall))} == {"succeeded", "failed"}


@pytest.mark.asyncio
async def test_image_response_can_restore_local_file_without_repeating_generation(temp_storage, tmp_path) -> None:
    calls = []

    def send(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(b"synthetic-png").decode()}]})

    provider = OpenAIImageProvider("synthetic-secret", client=httpx.AsyncClient(transport=httpx.MockTransport(send)))
    try:
        for filename in ["first.png", "restored.png"]:
            with job_call_context(110):
                output = await provider.generate_image("synthetic", 100, 80, tmp_path / filename)
                assert output.read_bytes() == b"synthetic-png"
    finally:
        await provider.aclose()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_upstream_rejection_does_not_expose_body_in_public_error(temp_storage) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(400, text="synthetic-secret synthetic private source"),
    ))
    provider = OpenAICompatibleProvider("synthetic-secret", "https://synthetic.invalid", "test", client=client)
    try:
        with job_call_context(111), pytest.raises(ProviderError) as exc:
            await provider.chat(LLMRequest([LLMMessage("user", "synthetic")]))
        assert "400" in str(exc.value)
        assert "synthetic" not in str(exc.value)
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_fallback_catches_propagate_unknown_instead_of_retrying_or_downgrading(monkeypatch) -> None:
    class UnknownProvider:
        calls = 0

        async def chat_json(self, request):
            self.calls += 1
            raise ExternalOutcomeUnknown("synthetic unknown")

    provider = UnknownProvider()
    with pytest.raises(ExternalOutcomeUnknown):
        await visual_planner._design_diagram(provider, kind="pointer_diagram", schema={}, block_index=0,
                                              tts_text="synthetic", source_text="synthetic", heading="test")
    assert provider.calls == 1
    monkeypatch.setattr(splitter, "narration_has_gaps", lambda *args: True)
    from app.core.config import Settings
    from app.services.stage_schemas import SplitBlock
    with pytest.raises(ExternalOutcomeUnknown):
        await splitter.repair_narration_gaps([SplitBlock(index=0, source_text="synthetic", tts_text="short")],
                                            provider, Settings())
    assert provider.calls == 2
    with pytest.raises(ExternalOutcomeUnknown):
        await safe_chat_json(provider, LLMRequest([]))


@pytest.mark.parametrize("crash", ["before_send", "after_send", "after_save"])
def test_actual_process_death_retains_call_boundary_evidence(temp_storage, crash: str) -> None:
    script = r'''
import asyncio, os, sys
import httpx
from app.db import init_db
from app.services.external_calls import job_call_context, journaled_post
init_db()
async def main():
    def send(request):
        if sys.argv[1] == "before_send":
            os._exit(71)
        with open(sys.argv[2], "a", encoding="utf-8") as marker:
            marker.write("sent\n")
        if sys.argv[1] == "after_send":
            os._exit(72)
        return httpx.Response(200, json={"synthetic": "success"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
        with job_call_context(112):
            await journaled_post(client, "https://synthetic.invalid/test", provider="synthetic", json={"x": 1})
    os._exit(73)
asyncio.run(main())
'''
    marker = Path(temp_storage) / "network-marker.txt"
    result = subprocess.run([sys.executable, "-c", script, crash, str(marker)],
                            capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert result.returncode == {"before_send": 71, "after_send": 72, "after_save": 73}[crash], result.stderr
    with get_session_factory()() as db, atomic_write(db):
        mark_interrupted_calls(db)
    with get_session_factory()() as db:
        call = db.scalar(select(ExternalCall).where(ExternalCall.job_id == 112))
        assert call.status == ("succeeded" if crash == "after_save" else "unknown")
        assert (call.response_body is not None) == (crash == "after_save")
    assert marker.exists() == (crash != "before_send")


def test_voicevox_speaker_zero_is_preserved_instead_of_global_fallback(temp_storage) -> None:
    from app.core.config import Settings
    from app.models.project import Project
    from app.services.provider_factory import build_voicevox_settings

    with get_session_factory()() as db:
        project = Project(title="Synthetic voice zero", source_script="合成。", voicevox_speaker_id=0)
        db.add(project)
        db.commit()
        settings = build_voicevox_settings(project, Settings(voicevox_speaker_id=3))
        assert settings.speaker_id == 0
