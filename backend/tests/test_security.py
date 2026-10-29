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

def test_redact_keeps_ordinary_words_and_long_paths() -> None:
    from app.core.security import redact

    assert redact("task-scheduler failed") == "task-scheduler failed"
    path = "C:/Users/x/a_very_long_folder_name_without_any_digits_here_ok/file"
    assert redact(path) == path
    assert "sk-abcdefghijklmnop123" not in redact("key sk-abcdefghijklmnop123 end")


def test_host_and_cross_site_guards() -> None:
    from fastapi.testclient import TestClient

    from app.core.security import host_allowed, plain_http_url
    from app.main import create_app

    allowed = ["localhost", "127.0.0.1", "::1"]
    assert host_allowed("localhost:8000", allowed)
    assert host_allowed("[::1]:8000", allowed)
    assert not host_allowed("rebind.attacker.example:8000", allowed)
    assert not host_allowed(None, allowed)
    assert plain_http_url("http://host.docker.internal:50021")
    for url in ("http://192.168.0.1/cgi-bin?x=", "http://h/#frag", "ftp://h", "http://u:p@h", "http://h:99999"):
        assert not plain_http_url(url)

    client = TestClient(create_app())
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health", headers={"host": "rebind.attacker.example"}).status_code == 400
    assert client.get("/api/health", headers={"origin": "http://localhost:5173"}).status_code == 200
    assert client.post("/api/projects/1/generate", headers={"origin": "https://attacker.example"}).status_code == 403
    assert client.post("/api/projects/1/generate", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.get("/api/voicevox/speakers", params={"url": "http://192.168.0.1/admin?x="}).status_code == 422


def test_deeply_nested_language_request_is_422(temp_storage) -> None:
    from fastapi.testclient import TestClient

    from app.main import create_app

    body = "[" * 100_000 + "]" * 100_000
    with TestClient(create_app()) as client:
        response = client.post("/api/language/requests", content=body,
                               headers={"content-type": "application/json"})
    assert response.status_code == 422
