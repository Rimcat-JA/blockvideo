"""Local bounded dialogue context; immutable receipts remain in the operation ledger."""
from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LanguageTurn(Base):
    __tablename__ = "language_turns"

    request_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    parent_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    relation: Mapped[str | None] = mapped_column(String(16), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    successor_request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
