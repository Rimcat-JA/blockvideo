"""Common HTTP utilities."""
from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.core.config import get_settings


def validate_artifact_path(rel: str) -> Path:
    """Resolve a DB-relative artifact path and ensure it stays under storage."""
    settings = get_settings()
    abs_path = (settings.storage_root / rel).resolve()
    storage_root = settings.storage_root.resolve()
    if not str(abs_path).startswith(str(storage_root)):
        raise HTTPException(status_code=400, detail="不正なパスです")
    return abs_path