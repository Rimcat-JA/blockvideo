"""Process-local bounded startup state and database availability error."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StartupStatus(BaseModel):
    """Immutable public snapshot of application startup progress."""

    model_config = ConfigDict(frozen=True)

    status: Literal["starting", "ready", "migration_failed"]
    reason_code: str | None
    message: str
    schema_version: int | None
    backup_available: bool


class StartupUnavailableError(RuntimeError):
    """Raised before session creation while database startup is unavailable."""

    reason_code: Literal["startup_unavailable"] = "startup_unavailable"

    def __init__(self) -> None:
        super().__init__(self.reason_code)


_STARTING = StartupStatus(
    status="starting",
    reason_code=None,
    message="起動処理中です。",
    schema_version=None,
    backup_available=False,
)
_status = _STARTING


def get_startup_status() -> StartupStatus:
    """Return the current immutable startup snapshot."""
    return _status


def set_startup_status(status: StartupStatus) -> None:
    """Atomically replace the process-local startup snapshot."""
    global _status
    _status = status


def reset_startup_status_for_tests() -> None:
    """Restore the pre-lifespan state for isolated tests."""
    set_startup_status(_STARTING)
