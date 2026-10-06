"""Database-independent startup status endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.startup_status import StartupStatus, get_startup_status


router = APIRouter()


@router.get("/startup", response_model=StartupStatus)
def startup_status() -> StartupStatus:
    """Return the current bounded process startup snapshot."""
    return get_startup_status()
