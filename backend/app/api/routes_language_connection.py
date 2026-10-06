"""Expose only allowlisted language configuration, without project data."""
from fastapi import APIRouter

from app.core.config import get_settings
from app.interpretation.connection import ConnectionView, inspect_connection

router = APIRouter(prefix="/language")


@router.get("/connection", response_model=ConnectionView)
async def language_connection(check: bool = False) -> ConnectionView:
    settings = get_settings()
    view = await inspect_connection(settings.language_base_url, settings.language_model, check=check)
    mode = ("stateful" if settings.language_retrieval_readiness else "semantic") if settings.language_retrieval_index else "all_tools"
    return view.model_copy(update={"operation_mode": mode, "yolo_enabled": settings.language_yolo_enabled})
