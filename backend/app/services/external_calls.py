"""Journal actual HTTP attempts and reuse known results without blind paid replay.

    A process can die between recording in-flight and actually sending. Such a
    remote call remains unknown conservatively. Response caches permit local
    postprocessing after a restart, but cannot reconcile an unknown upstream call.
"""
from __future__ import annotations

import asyncio
import hashlib
import json as json_module
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from weakref import WeakKeyDictionary

import httpx
from sqlalchemy import ColumnElement, and_, select, update
from sqlalchemy.orm import Session

from app.db import get_session_factory
from app.models.external_call import ExternalCall
from app.services.transactions import atomic_write


class ExternalOutcomeUnknown(RuntimeError):
    """An upstream effect may exist; no automatic or same-job replay is allowed."""


_job_id: ContextVar[int | None] = ContextVar("external_call_job_id", default=None)
# Per-loop (job, fingerprint) locks with a user count, dropped once unused.
_locks: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[int, str], tuple[asyncio.Lock, int]]
] = WeakKeyDictionary()
# Failures raised before any request byte left the process. They prove the
# provider never saw the call, so it is recorded as failed and may be re-sent.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
                    httpx.UnsupportedProtocol, httpx.InvalidURL)
_NOT_SENT_PREFIX = "not_sent:"
_UNKNOWN_MESSAGE = "外部処理の結果が未確定です。重複実行を避けるため自動再送を停止しました。"


@contextmanager
def job_call_context(job_id: int) -> Iterator[None]:
    """Associate nested provider calls and inherited asyncio tasks with one job."""
    token = _job_id.set(job_id)
    try:
        yield
    finally:
        _job_id.reset(token)


def unresolved_remote_side_effect_predicate() -> ColumnElement[bool]:
    """Match only unresolved calls that may have produced a remote side effect."""
    return and_(
        ExternalCall.remote_side_effect.is_(True),
        ExternalCall.status.in_(["in_flight", "unknown"]),
    )


def has_unresolved_calls(db: Session, job_id: int) -> bool:
    """Return whether a job has an in-flight or unresolved potentially paid call."""
    return db.scalar(select(ExternalCall.id).where(
        ExternalCall.job_id == job_id,
        unresolved_remote_side_effect_predicate(),
    ).limit(1)) is not None


def clear_call_bodies(db: Session, job_id: int) -> None:
    """Drop cached reply bodies of a finished job; caller owns commit.

    Bodies only serve resuming the same job, and a retry is always a new job.
    Rows stay, because status and attempts are still needed for reconciliation.
    """
    db.execute(update(ExternalCall).where(ExternalCall.job_id == job_id).values(
        response_body=None, response_content_type=None))


def mark_interrupted_calls(db: Session) -> int:
    """Classify abandoned calls at single-server startup; caller owns commit.

    Remote work is unknown. Local VOICEVOX computation has no billed remote
    generation identity and can be retried, so its interrupted call is failed.
    """
    calls = list(db.scalars(select(ExternalCall).where(ExternalCall.status == "in_flight")))
    for call in calls:
        call.status = "unknown" if call.remote_side_effect else "failed"
        call.error_code = "process_interrupted"
        call.finished_at = datetime.now(timezone.utc)
    return len(calls)


def _identity(provider: str, url: str, payload: Any, params: Any) -> str:
    value = json_module.dumps(
        {"provider": provider, "url": url, "payload": payload, "params": params},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_endpoint(url: str) -> str:
    parts = urlsplit(url)
    # Do not save URL credentials, query values, or fragments.
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))[:512]


def _response(call: ExternalCall, url: str) -> httpx.Response:
    if call.response_status is None or call.response_body is None:
        raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
    headers = {"Content-Type": call.response_content_type} if call.response_content_type else {}
    return httpx.Response(
        call.response_status, content=call.response_body, headers=headers,
        request=httpx.Request("POST", _safe_endpoint(url)),
    )


def _claim(job_id: int, fingerprint: str, provider: str, url: str,
           remote: bool) -> tuple[int, httpx.Response | None]:
    missing_cache = False
    with get_session_factory()() as db, atomic_write(db):
        call = db.scalar(select(ExternalCall).where(
            ExternalCall.job_id == job_id, ExternalCall.fingerprint == fingerprint,
        ))
        if call is not None:
            if call.status == "unknown" or (call.status == "in_flight" and call.remote_side_effect):
                raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
            if call.status == "succeeded" or (
                call.status == "failed" and call.response_body is not None and call.remote_side_effect
            ):
                if call.response_status is not None and call.response_body is not None:
                    return call.id, _response(call, url)
                call.status = "unknown"
                call.error_code = "cached_response_missing"
                missing_cache = True
            elif call.remote_side_effect and not (
                call.status == "failed" and (call.error_code or "").startswith(_NOT_SENT_PREFIX)
            ):
                # Even malformed/missing cached metadata cannot authorize another
                # remote execution. An unproven old failure is not a safe retry;
                # only a failure proven to have happened before sending is.
                raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
            else:
                call.status = "in_flight"
                call.attempts += 1
                call.error_code = None
                call.response_status = None
                call.response_body = None
                call.response_content_type = None
                call.provider_response_id = None
                call.started_at = datetime.now(timezone.utc)
                call.finished_at = None
        else:
            if remote and db.scalar(select(ExternalCall.id).where(
                ExternalCall.job_id == job_id, ExternalCall.remote_side_effect.is_(True),
                ExternalCall.status == "unknown",
            ).limit(1)) is not None:
                raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
            call = ExternalCall(job_id=job_id, fingerprint=fingerprint, provider=provider,
                                endpoint=_safe_endpoint(url), remote_side_effect=remote,
                                status="in_flight")
            db.add(call)
            db.flush()
        call_id = call.id
    if missing_cache:
        raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
    return call_id, None


def _finish(call_id: int, status: str, *, response: httpx.Response | None = None,
            error_code: str | None = None) -> None:
    with get_session_factory()() as db, atomic_write(db):
        call = db.get(ExternalCall, call_id)
        if call is None:
            raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
        call.status = status
        call.error_code = error_code
        call.finished_at = datetime.now(timezone.utc)
        if response is not None and status != "unknown":
            call.response_status = response.status_code
            call.response_body = response.content
            call.response_content_type = response.headers.get("content-type", "")[:128]
            # IDs are diagnostic references only; no adapter supports lookup.
            call.provider_response_id = response.headers.get("x-request-id", "")[:128] or None


async def journaled_post(
    client: httpx.AsyncClient, url: str, *, provider: str,
    remote_side_effect: bool = True, json: Any = None,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST once in a job context, durably caching replies before returning.

    Authentication headers are neither fingerprinted nor persisted. Without a job
    context, existing standalone provider behavior is preserved. Transport errors,
    HTTP 408, and server errors may occur after paid execution and remain unknown,
    except connection failures raised before anything was sent, which are failed.
    A known 4xx rejection is cached as failed and cannot be retried by same-job POST.
    """
    job_id = _job_id.get()
    if job_id is None:
        return await client.post(url, json=json, params=params, headers=headers)
    fingerprint = _identity(provider, url, json, params)
    loop_locks = _locks.setdefault(asyncio.get_running_loop(), {})
    key = (job_id, fingerprint)
    lock, users = loop_locks.get(key, (asyncio.Lock(), 0))
    loop_locks[key] = (lock, users + 1)
    try:
        async with lock:
            return await _journaled_post_locked(
                client, url, job_id=job_id, fingerprint=fingerprint, provider=provider,
                remote_side_effect=remote_side_effect, json=json, params=params, headers=headers)
    finally:
        lock, users = loop_locks[key]
        if users <= 1:
            del loop_locks[key]
        else:
            loop_locks[key] = (lock, users - 1)


async def _journaled_post_locked(
    client: httpx.AsyncClient, url: str, *, job_id: int, fingerprint: str, provider: str,
    remote_side_effect: bool, json: Any, params: Mapping[str, Any] | None,
    headers: Mapping[str, str] | None,
) -> httpx.Response:
    call_id, cached = _claim(job_id, fingerprint, provider, url, remote_side_effect)
    if cached is not None:
        return cached
    try:
        response = await client.post(url, json=json, params=params, headers=headers)
    except BaseException as exc:
        not_sent = isinstance(exc, _NOT_SENT_ERRORS)
        error_code = f"{_NOT_SENT_PREFIX}{type(exc).__name__}" if not_sent else type(exc).__name__
        try:
            _finish(call_id, "unknown" if remote_side_effect and not not_sent else "failed",
                    error_code=error_code)
        except Exception:
            if remote_side_effect:
                raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE) from exc
            raise
        if remote_side_effect and not not_sent:
            raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE) from exc
        raise
    if remote_side_effect and (response.status_code == 408 or response.status_code >= 500):
        _finish(call_id, "unknown", error_code=f"http_{response.status_code}")
        raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE)
    try:
        _finish(call_id, "failed" if response.status_code >= 400 else "succeeded", response=response)
    except Exception as exc:
        if remote_side_effect:
            raise ExternalOutcomeUnknown(_UNKNOWN_MESSAGE) from exc
        raise
    return response
