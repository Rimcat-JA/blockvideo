"""Durable external-call outcomes; request secrets and source inputs are not stored."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ExternalCall(Base):
    """One job-local HTTP request fingerprint and its durable response cache.

    Cached response bytes are private application data, never an API/log payload.
    There is deliberately no provider lookup claim: these synchronous adapters
    cannot query an unknown remote result without submitting another request.
    """

    __tablename__ = "external_calls"
    __table_args__ = (UniqueConstraint("job_id", "fingerprint"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    remote_side_effect: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    response_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    response_content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_response_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
