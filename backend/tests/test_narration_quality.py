"""Output-quality regression tests at the VOICEVOX HTTP boundary (no engine needed)."""
from __future__ import annotations

import copy
import json
import re

import httpx
import pytest

from app.providers.llm import ProviderError
from app.providers.voicevox import FakeVoicevoxClient, VoicevoxClient, VoicevoxSettings
from app.services.narration import build_narration_plan, sentence_pause_for, split_sentences_with_offsets
from app.services.pronunciation import apply_readings, normalize_overrides, pronunciation_query
from app.services.voice import synthesize_audio, synthesize_block


def _query(text: str) -> dict:
    kana = re.findall(r"[ァ-ヴ][ァィゥェォャュョヮ]?|[^\s。！？!?、，．「」]", text)
    return {
        "accent_phrases": [{
            "moras": [{"text": ch, "consonant": None, "consonant_length": None,
                       "vowel": "a", "vowel_length": 0.1, "pitch": 5.0} for ch in kana],
            "accent": 1, "pause_mora": None, "is_interrogative": text.endswith("？"),
        }] if kana else [],
        "speedScale": 1.0, "prePhonemeLength": 0.1, "postPhonemeLength": 0.1,
        "pauseLength": 4.0, "pauseLengthScale": 3.0,
    }


@pytest.fixture
async def engine():
    calls: list[tuple[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.params.get("speaker") == "1"
        if request.url.path == "/audio_query":
            text = request.url.params["text"]
            calls.append(("query", text))
            return httpx.Response(200, json=_query(text))
        body = json.loads(request.content)
        if request.url.path == "/mora_data":
            calls.append(("mora", copy.deepcopy(body)))
            for phrase in body:
                for index, mora in enumerate(phrase["moras"]):
                    mora["vowel_length"] = 0.2
                    mora["pitch"] = 6.0 if index < phrase["accent"] else 4.0
            return httpx.Response(200, json=body)
        if request.url.path == "/synthesis":
            calls.append(("synthesis", body))
            return httpx.Response(200, content=b"mock-wave")
        raise AssertionError(f"Unexpected endpoint (dictionary writes are forbidden): {request.url}")

    client = VoicevoxClient("http://voicevox.test")
    await client.aclose()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    try:
        yield client, calls
    finally:
        await client.aclose()


@pytest.mark.parametrize("speed", [0.5, 1.0, 1.8])
async def test_adaptive_pauses_are_wall_clock_seconds(engine, speed):
    client, _ = engine
    plan = await build_narration_plan(
        "説明します。図の矢印を見てください。重要なのはこの向きです。次です。",
        VoicevoxSettings(speed_scale=speed), client,
        sentence_pause_seconds=9.0, pacing_mode="adaptive",
    )
    assert plan is not None
    phrases = plan.query["accent_phrases"]
    assert [p["pause_mora"]["vowel_length"] / speed for p in phrases[:-1]] == pytest.approx([0.6, 1.0, 1.5])
    assert phrases[-1]["pause_mora"] is None
    assert plan.query["pauseLength"] is None
    assert plan.query["pauseLengthScale"] == 1.0
    for index, phrase in enumerate(phrases[:-1]):
        speech_ms = sum(m["vowel_length"] for m in phrase["moras"]) / speed * 1000
        measured_ms = plan.spans[index].end_ms - plan.spans[index].start_ms
        assert measured_ms - speech_ms == pytest.approx([600, 1000, 1500][index], abs=1)


@pytest.mark.parametrize("speed", [0.7, 1.7])
async def test_fixed_mode_preserves_scalar_pause(engine, speed):
    client, _ = engine
    plan = await build_narration_plan(
        "重要なのはここです。図を見てください。", VoicevoxSettings(speed_scale=speed),
        client, sentence_pause_seconds=1.23, pacing_mode="fixed",
    )
    assert plan is not None
    assert plan.query["accent_phrases"][0]["pause_mora"]["vowel_length"] / speed == pytest.approx(1.23)


def test_visual_focus_changes_and_dense_sentences_get_reading_time():
    assert sentence_pause_for("説明します。", pacing_mode="adaptive", fixed_seconds=8,
                              focus_terms=["キー"], next_focus_terms=["矢印"]) == 1.0
    assert sentence_pause_for("説明します。", pacing_mode="adaptive", fixed_seconds=8,
                              focus_terms=["キー"], next_focus_terms=["キー"]) == 0.6
    assert sentence_pause_for("説明" * 35, pacing_mode="adaptive", fixed_seconds=8) == 1.5
    assert sentence_pause_for("説明します。", pacing_mode="adaptive", fixed_seconds=8,
                              next_text="次に別の操作です。") == 1.0


async def test_source_offsets_survive_longer_readings_and_whitespace(engine):
    client, calls = engine
    source = "  APIを説明します。 \n次です！  "
    plan = await build_narration_plan(
        source, VoicevoxSettings(), client, sentence_pause_seconds=0.6,
        pronunciation_overrides=[{"surface": "API", "reading": "エーピーアイ"}],
    )
    assert plan is not None
    assert "".join(span.text for span in plan.spans) == source
    assert all(source[span.char_start:span.char_end] == span.text for span in plan.spans)
    assert plan.spans[0].char_start == 0
    assert plan.spans[-1].char_end == len(source)
    assert ("query", "  エーピーアイを説明します。") in calls


def test_longest_literal_matches_are_bounded_and_do_not_cascade():
    overrides = normalize_overrides([
        {"surface": "API", "reading": "キー"},
        {"surface": "API key", "reading": "エーピーアイキー"},
        {"surface": "キー", "reading": "カギ"},
    ])
    assert apply_readings("API key / API / APIs / XAPI / api / キーとキーコード", overrides) == (
        "エーピーアイキー / キー / APIs / XAPI / api / カギとキーコード"
    )
    assert apply_readings("APIキー", normalize_overrides([{"surface": "API", "reading": "エーピーアイ"}])) == "エーピーアイキー"
    assert apply_readings("API keys", overrides) == "キー keys"
    technical = normalize_overrides([{"surface": "insert!", "reading": "インサート"}])
    assert apply_readings("insert!を使う / insert!item", technical) == "インサートを使う / insert!item"


async def test_technical_identifier_bang_is_allowed_and_keeps_display_text(engine):
    client, calls = engine
    source = "insert!を使います。"
    plan = await build_narration_plan(
        source, VoicevoxSettings(), client, sentence_pause_seconds=0.6,
        pronunciation_overrides=[{"surface": "insert!", "reading": "インサート"}],
    )
    assert plan is not None
    assert "".join(span.text for span in plan.spans) == source
    assert ("query", "インサートを使います。") in calls
    assert len(plan.spans) == 1


@pytest.mark.parametrize("source", ["insert!を使います。", "map?が結果を返します。",
                                     "「insert!」を実行します。", "get?itemを指定します。"])
def test_technical_suffix_does_not_insert_an_internal_sentence_hold(source):
    pieces = split_sentences_with_offsets(source)
    assert len(pieces) == 1
    assert pieces[0].text == source
    assert pieces[0].char_end == len(source)


@pytest.mark.parametrize("source, count", [("Hello! Next sentence.", 2),
                                             ("Hello!はじめまして。", 2),
                                             ("Really? 次です。", 2),
                                             ("こんにちは！次です。", 2)])
def test_ordinary_prose_retains_sentence_breaks(source, count):
    pieces = split_sentences_with_offsets(source)
    assert len(pieces) == count
    assert "".join(piece.text for piece in pieces) == source


@pytest.mark.parametrize("reading, accent", [("キャット", 0), ("キャット", 3), ("エーピーアイ", None)])
def test_valid_readings_and_mora_accent_bounds(reading, accent):
    assert normalize_overrides([{"surface": "word", "reading": reading, "accent": accent}])


@pytest.mark.parametrize("reading, accent", [("キャット", 4), ("キー", -1), ("キー", True),
                                           ("キー", 1.5), ("かな", None), ("ィー", None)])
def test_invalid_readings_and_accents_are_rejected(reading, accent):
    with pytest.raises(ProviderError):
        normalize_overrides([{"surface": "word", "reading": reading, "accent": accent}])


@pytest.mark.parametrize("accent, expected", [(1, 1), (0, 3), (3, 3)])
async def test_accent_edits_recalculate_real_query_pitch_and_lengths(engine, accent, expected):
    client, calls = engine
    query = await pronunciation_query("catです。", 1, client, normalize_overrides([
        {"surface": "cat", "reading": "キャット", "accent": accent},
    ]))
    sent = next(body for kind, body in calls if kind == "mora")
    assert sent[0]["accent"] == expected
    assert len(sent[0]["moras"]) == 3
    assert query["accent_phrases"][0]["moras"][0]["vowel_length"] == 0.2
    assert query["accent_phrases"][0]["moras"][0]["pitch"] == 6.0
    assert all(kind in {"query", "mora"} for kind, _ in calls)
    assert query["kana"] is None


async def test_reading_only_override_keeps_full_sentence_analysis(engine):
    client, calls = engine
    await pronunciation_query("catです。", 1, client, normalize_overrides([
        {"surface": "cat", "reading": "キャット"},
    ]))
    assert calls == [("query", "キャットです。")]


async def test_accent_question_punctuation_is_preserved(engine):
    client, _ = engine
    query = await pronunciation_query("cat？", 1, client, normalize_overrides([
        {"surface": "cat", "reading": "キャット", "accent": 1},
    ]))
    assert query["accent_phrases"][-1]["is_interrogative"] is True


async def test_accent_measurement_uses_recalculated_duration_with_original_offsets(engine):
    client, _ = engine
    source = "catです。次です。"
    plan = await build_narration_plan(
        source, VoicevoxSettings(), client, sentence_pause_seconds=0.6,
        pronunciation_overrides=[{"surface": "cat", "reading": "キャット", "accent": 1}],
    )
    assert plan is not None
    assert plan.spans[0].text == "catです。"
    assert source[plan.spans[0].char_start:plan.spans[0].char_end] == "catです。"
    # First phrase has 3 overridden moras + 2 context moras at the recalculated
    # 200 ms each, then the independently configured 600 ms sentence hold.
    assert plan.spans[0].end_ms - plan.spans[0].start_ms == 1600


@pytest.fixture
def mock_duration(monkeypatch):
    async def measure(_path):
        return 1000
    monkeypatch.setattr("app.services.voice.ffprobe_duration_ms", measure)


async def test_fake_fallback_receives_readings(tmp_path, mock_duration):
    client = FakeVoicevoxClient()
    result = await synthesize_block(
        "APIを使う。", VoicevoxSettings(), tmp_path / "fake.wav", client=client,
        pacing_mode="adaptive", pronunciation_overrides=[{"surface": "API", "reading": "エーピーアイ", "accent": 0}],
    )
    assert client.calls == [("エーピーアイを使う。", 1)]
    assert result.spans == []
    assert result.path.exists()


async def test_live_fallback_reapplies_accent_instead_of_losing_override(engine, tmp_path, mock_duration, monkeypatch):
    client, calls = engine
    async def cannot_plan(*_args, **_kwargs):
        return None
    monkeypatch.setattr("app.services.voice.build_narration_plan", cannot_plan)
    await synthesize_block(
        "catです。", VoicevoxSettings(), tmp_path / "fallback.wav", client=client,
        pronunciation_overrides=[{"surface": "cat", "reading": "キャット", "accent": 1}],
    )
    synthesized = next(body for kind, body in calls if kind == "synthesis")
    assert synthesized["accent_phrases"][0]["accent"] == 1
    assert synthesized["accent_phrases"][0]["moras"][0]["pitch"] == 6.0


async def test_accent_failure_never_falls_back_to_unmodified_speech(engine, tmp_path, mock_duration, monkeypatch):
    client, calls = engine
    async def failed_mora(*_args):
        raise ProviderError("accent endpoint unavailable", safe=True)
    monkeypatch.setattr(client, "mora_data", failed_mora)
    with pytest.raises(ProviderError, match="accent endpoint unavailable"):
        await synthesize_audio(
            "catです。", VoicevoxSettings(), tmp_path / "never.wav", client=client,
            max_attempts=1,
            pronunciation_overrides=[{"surface": "cat", "reading": "キャット", "accent": 1}],
        )
    assert not any(kind == "synthesis" for kind, _ in calls)
    assert not (tmp_path / "never.wav").exists()
