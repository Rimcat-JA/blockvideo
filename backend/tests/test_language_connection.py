"""Connection inspection cannot send project content or select a fallback."""
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes_language_connection as routes
from app.core.config import Settings
from app.interpretation.connection import inspect_connection


@pytest.mark.parametrize("url", ["https://example.com/v1", "http://user:secret@127.0.0.1:1234/v1",
                               "http://127.0.0.1:1234/v1?key=secret"])
async def test_invalid_settings_are_not_echoed_or_contacted(url: str) -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("must not contact any server")
    view = await inspect_connection(url, "model", check=True, transport=httpx.MockTransport(forbidden))
    assert view.status == "invalid_configuration" and view.base_url is None and view.model is None


async def test_configuration_read_and_missing_model_make_no_network_request() -> None:
    def forbidden(request: httpx.Request) -> httpx.Response:
        pytest.fail("metadata read must not contact the server")
    transport = httpx.MockTransport(forbidden)
    assert (await inspect_connection("http://localhost:1234/v1", "m", transport=transport)).status == "configured"
    assert (await inspect_connection("bad", None, check=True, transport=transport)).status == "unconfigured"


@pytest.mark.parametrize("data,status", [
    ({"data": [{"id": "chosen"}, {"id": "other"}]}, "listed"),
    ({"data": [{"id": "other"}]}, "model_missing"),
    ({"data": []}, "model_missing"), ({"data": {}}, "invalid_response"),
    ({"data": [None]}, "invalid_response"), (None, "invalid_response"),
])
async def test_exact_model_match_and_bounded_display(data: Any, status: str, monkeypatch: Any) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    seen = []
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=data)
    view = await inspect_connection("http://localhost:1234/v1", "chosen", check=True,
                                    transport=httpx.MockTransport(respond))
    assert view.status == status and view.model == "chosen" and not view.cloud_fallback
    assert len(seen) == 1 and seen[0].method == "GET" and seen[0].content == b""
    assert str(seen[0].url) == "http://127.0.0.1:1234/v1/models"
    assert "Authorization" not in seen[0].headers
    assert "other" not in view.model_dump_json()


@pytest.mark.parametrize("mode,status", [("redirect", "unreachable"), ("refused", "unreachable"),
    ("timeout", "unreachable"), ("large", "invalid_response"), ("malformed", "invalid_response")])
async def test_failure_is_terminal_without_fallback(mode: str, status: str) -> None:
    seen = []
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if mode == "refused":
            raise httpx.ConnectError("sensitive exception")
        if mode == "timeout":
            raise httpx.ReadTimeout("sensitive exception")
        if mode == "redirect":
            return httpx.Response(302, headers={"location": "https://remote.invalid"})
        return httpx.Response(200, content=b"x" * (131_073 if mode == "large" else 10))
    view = await inspect_connection("http://127.0.0.1:1234/v1", "chosen", check=True,
                                    transport=httpx.MockTransport(respond))
    assert view.status == status and len(seen) == 1
    assert "sensitive" not in view.model_dump_json()


def test_route_exposes_only_allowlisted_config(monkeypatch: Any) -> None:
    settings = Settings(_env_file=None, language_model="chosen", llm_api_key="never-expose-this")
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    body = TestClient(app).get("/api/language/connection").json()
    assert body == {"status": "configured", "model": "chosen",
                    "base_url": "http://127.0.0.1:1234/v1", "cloud_fallback": False, "operation_mode": "all_tools",
                    "yolo_enabled": True}


@pytest.mark.parametrize("index,readiness,mode", [(None, True, "all_tools"),
    ("private-not-loaded-index", False, "semantic"), ("private-not-loaded-index", True, "stateful")])
def test_mode_display_uses_host_config_without_loading_index(monkeypatch: Any, index: str | None,
                                                            readiness: bool, mode: str) -> None:
    settings = Settings(_env_file=None, language_model="chosen", language_retrieval_index=index,
                        language_retrieval_readiness=readiness)
    monkeypatch.setattr(routes, "get_settings", lambda: settings)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    response = TestClient(app).get("/api/language/connection?operation_mode=all_tools")
    assert response.json()["operation_mode"] == mode
    assert "private-not-loaded-index" not in response.text
