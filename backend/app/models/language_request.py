"""Durable interpretation identity; committed effects live in operation receipts."""
from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LanguageRequestRecord(Base):
    __tablename__ = "language_requests"

    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    input_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    core_request_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    base_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_token: Mapped[str] = mapped_column(String(64), nullable=False)
    lease_until: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    request_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
