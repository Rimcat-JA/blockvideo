"""Health and provider-discovery endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import __version__
from app.core.config import get_settings
from app.providers.llm import ProviderError
from app.providers.voicevox import VoicevoxClient
from app.schemas import HealthResponse, SpeakersEnvelope
from app.services.ffmpeg_runner import ffmpeg_available, ffprobe_available


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        ffmpeg_available=ffmpeg_available(),
        ffprobe_available=ffprobe_available(),
    )


@router.get("/voicevox/speakers", response_model=SpeakersEnvelope)
async def voicevox_speakers(url: str | None = None) -> SpeakersEnvelope:
    """Fetch speakers from a running VOICEVOX Engine.

    The ``url`` query param overrides the project default; useful when the
    user wants to test a different engine instance.
    """
    settings = get_settings()
    base_url = (url or settings.voicevox_url).rstrip("/")
    client = VoicevoxClient(base_url)
    try:
        speakers = await client.speakers()
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await client.aclose()
    return SpeakersEnvelope(url=base_url, speakers=speakers)