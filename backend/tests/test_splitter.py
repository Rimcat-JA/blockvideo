"""Unit tests for the splitter stage.

These do not require any network access.
"""
from __future__ import annotations

import pytest

from app.core import config
from app.providers.llm_fake import FakeLLMProvider
from app.services.splitter import (
    normalize_for_comparison,
    split_joined_source,
    split_script,
)
from app.services.stage_schemas import SplitPayload


SAMPLE_SCRIPT = (
    "みなさん、こんにちは。本日は関数型プログラミングにおける高階関数について解説します。"
    "まず、高階関数とは何かということですが、これは関数を引数として受け取る関数、あるいは"
    "関数を戻り値として返す関数のことを指します。"
    "次に、map関数を見てみましょう。mapはリストのそれぞれの要素に対して関数を適用する関数です。"
    "たとえば、リストのそれぞれの数値を二倍にしたい場合は、mapと無名関数を組み合わせて実現できます。"
    "続いて、reduce関数について説明します。reduceはリストを一つの値に集約する関数で、累積計算に向いています。"
    "本章のまとめです。高階関数を使うと、繰り返しのパターンを抽象化できます。"
)


@pytest.mark.asyncio
async def test_fake_split_returns_blocks() -> None:
    config.reset_settings_cache()
    from app.core.config import get_settings

    settings = get_settings()
    provider = FakeLLMProvider()
    result = await split_script(SAMPLE_SCRIPT, provider, settings)
    assert result.blocks, "fake should produce at least one block"
    assert not result.used_fallback
    joined = split_joined_source(result.blocks)
    assert normalize_for_comparison(joined) == normalize_for_comparison(SAMPLE_SCRIPT)


@pytest.mark.asyncio
async def test_fake_split_keys_present() -> None:
    config.reset_settings_cache()
    from app.core.config import get_settings

    settings = get_settings()
    provider = FakeLLMProvider()
    result = await split_script(SAMPLE_SCRIPT, provider, settings)
    payload = SplitPayload.model_validate({"blocks": [b.model_dump() for b in result.blocks]})
    assert payload.blocks
    indices = [b.index for b in payload.blocks]
    assert indices == list(range(len(payload.blocks)))


@pytest.mark.asyncio
async def test_deterministic_fallback_when_invalid_payload() -> None:
    """A provider returning malformed JSON should fall back to the deterministic splitter."""

    class BadProvider:
        name = "bad"

        async def chat(self, request):
            from app.providers.llm import LLMResponse

            # return XML instead of JSON
            return LLMResponse(content="<nope />", raw={"nope": True})

        async def chat_json(self, request):
            raise ValueError("boom")

    config.reset_settings_cache()
    from app.core.config import get_settings

    settings = get_settings()
    result = await split_script(SAMPLE_SCRIPT, BadProvider(), settings)
    assert result.used_fallback is True
    assert result.blocks


@pytest.mark.asyncio
async def test_deterministic_fallback_when_join_mismatch() -> None:
    class MisjoinProvider:
        name = "misjoin"

        async def chat(self, request):
            from app.providers.llm import LLMResponse

            bad = (
                '{"blocks":[{"index":0,"source_text":"テスト","tts_text":"テスト"},'
                '{"index":1,"source_text":"追加された文字列","tts_text":"追加された文字列"}]}'
            )
            return LLMResponse(content=bad, raw={})

        async def chat_json(self, request):
            import json


            bad = (
                '{"blocks":[{"index":0,"source_text":"テスト","tts_text":"テスト"},'
                '{"index":1,"source_text":"追加された文字列","tts_text":"追加された文字列"}]}'
            )
            return json.loads(bad)

    config.reset_settings_cache()
    from app.core.config import get_settings

    settings = get_settings()
    result = await split_script("テスト", MisjoinProvider(), settings)
    assert result.used_fallback is True


def test_normalize_for_comparison() -> None:
    assert normalize_for_comparison("あ  いう\nえお") == normalize_for_comparison("あ いう えお")
    assert normalize_for_comparison("") == ""
