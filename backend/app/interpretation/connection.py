"""Read-only local configuration inspection; model listing is not inference."""
from __future__ import annotations

import asyncio
from typing import Literal

import httpx
from pydantic import BaseModel

from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import MAX_HTTP_RESPONSE_BYTES, local_base_url
from app.interpretation.parser import strict_json


class ConnectionView(BaseModel):
    status: Literal["unconfigured", "configured", "invalid_configuration", "listed",
                    "model_missing", "unreachable", "invalid_response"]
    model: str | None = None
    base_url: str | None = None
    cloud_fallback: Literal[False] = False
    operation_mode: Literal["all_tools", "semantic", "stateful"] = "all_tools"
    # Whether the server accepts unattended (YOLO) requests from the UI toggle.
    yolo_enabled: bool = False


async def inspect_connection(
    base_url: str, model: str | None, *, check: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ConnectionView:
    """Validate before displaying; an explicit check sends no input or inference."""
    if model is None or model == "":
        return ConnectionView(status="unconfigured")
    try:
        normalized = local_base_url(base_url)
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise InterpretationError("configuration_error")
    except InterpretationError:
        return ConnectionView(status="invalid_configuration")
    view = ConnectionView(status="configured", model=model, base_url=normalized)
    if not check:
        return view
    try:
        async with asyncio.timeout(5):
            async with httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False,
                                         transport=transport) as client:
                async with client.stream("GET", f"{normalized}/models") as response:
                    if response.status_code != 200:
                        return view.model_copy(update={"status": "unreachable"})
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_HTTP_RESPONSE_BYTES:
                            return view.model_copy(update={"status": "invalid_response"})
        body = strict_json(data.decode("utf-8"))
        entries = body["data"]
        if not isinstance(entries, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in entries
        ):
            raise ValueError("invalid model list")
        status = "listed" if any(item["id"] == model for item in entries) else "model_missing"
        return view.model_copy(update={"status": status})
    except (httpx.HTTPError, TimeoutError):
        return view.model_copy(update={"status": "unreachable"})
    except (ValueError, TypeError, KeyError, RecursionError):
        return view.model_copy(update={"status": "invalid_response"})
