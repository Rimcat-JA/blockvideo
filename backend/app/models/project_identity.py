"""Permanent ID reservations prevent deleted project URLs/storage from being reused."""
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class ProjectIdentity(Base):
    __tablename__ = "project_identities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
