"""Durable operation receipts; history is retained after target deletion."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class OperationReceipt(Base):
    """One committed request and its immutable response, also a job outbox."""

    __tablename__ = "operation_requests"

    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    canonical_request: Mapped[str] = mapped_column(Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    operation_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # Deliberately not cascading FKs: a retry after deletion must not re-execute.
    project_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    base_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    result_revision: Mapped[int] = mapped_column(Integer, nullable=False)
    resolved_arguments: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_requested: Mapped[bool] = mapped_column(Boolean, nullable=False)
    job_id: Mapped[int | None] = mapped_column(Integer, nullable=True, unique=True)
    result_ref: Mapped[str] = mapped_column(String(256), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False,
    )
