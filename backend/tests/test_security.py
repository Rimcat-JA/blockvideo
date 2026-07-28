"""API key masking and redaction."""
from __future__ import annotations

from app.core.security import mask, redact, is_likely_key


def test_mask_short_returns_asterisks() -> None:
    assert mask("") == ""
    assert mask("ab") == "***"
    assert mask("abcd") == "***"
    assert mask("abcdefgh") == "***"


def test_mask_long_shows_first_and_last_two() -> None:
    masked = mask("sk-proj-abcdefghijklmnop")
    assert masked.startswith("sk-p")
    assert "***" in masked
    assert "len=" in masked


def test_is_likely_key_true_for_sk_prefix() -> None:
    assert is_likely_key("sk-proj-abcdefghijklmnop")
    assert is_likely_key("AIzaSyA-abcdefghijklmnopqrstuvwxyz1234")


def test_is_likely_key_false_for_normal_text() -> None:
    assert not is_likely_key("hello world")
    assert not is_likely_key("")
    assert not is_likely_key("123")


def test_redact_strips_api_keys_in_text() -> None:
    text = "error contacting api with sk-proj-abcdefghijklmnop and body=ok"
    out = redact(text)
    assert "sk-proj-abcdefghijklmnop" not in out
    assert "***" in out


def test_redact_passes_through_safe_text() -> None:
    assert redact("hello world") == "hello world"
    assert redact("") == ""