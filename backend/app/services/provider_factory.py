"""Build configured LLM / Image / Voicevox instances for a project.

Resolution order:
    1. In-memory secrets stored per project (BYOK).
    2. Environment-level fallback (only used if explicitly configured).
    3. ``FakeLLMProvider`` / ``FakeImageProvider`` if the project has
       ``use_fake_providers = True``.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.security import SecretBundle, secret_store
from app.models.project import Project
from app.providers.image import ImageProvider
from app.providers.image_fake import FakeImageProvider
from app.providers.image_openai import OpenAIImageProvider
from app.providers.llm import LLMProvider
from app.providers.llm_fake import FakeLLMProvider
from app.providers.llm_openai import OpenAICompatibleProvider
from app.providers.voicevox import FakeVoicevoxClient, VoicevoxClient, VoicevoxSettings


@dataclass
class ProviderBundle:
    llm: LLMProvider
    image: ImageProvider | None
    voicevox: VoicevoxClient | FakeVoicevoxClient
    use_fake: bool
    # High-volume per-block planning provider. Same object as ``llm`` unless
    # ``llm_model_planner`` selects a different (cheaper/faster) model.
    llm_planner: LLMProvider | None = None

    @property
    def planner(self) -> LLMProvider:
        return self.llm_planner or self.llm


def _openrouter_headers(base_url: str, settings: Settings) -> dict[str, str]:
    """OpenRouter attribution headers. No-op for other providers."""
    if "openrouter.ai" not in (base_url or "").lower():
        return {}
    headers: dict[str, str] = {}
    if settings.openrouter_referer:
        headers["HTTP-Referer"] = settings.openrouter_referer
    if settings.openrouter_title:
        headers["X-Title"] = settings.openrouter_title
    return headers


def get_settings_for(project: Project) -> Settings:
    return get_settings()


def build_providers_for_project(project: Project) -> ProviderBundle:
    secrets: SecretBundle | None = (
        secret_store.get(project.id) if not project.use_fake_providers else None
    )
    settings = get_settings()

    use_fake = bool(project.use_fake_providers)
    if use_fake:
        return ProviderBundle(
            llm=FakeLLMProvider(),
            image=FakeImageProvider(),
            voicevox=FakeVoicevoxClient(),
            use_fake=True,
        )

    llm_api_key = (secrets.llm_api_key if secrets else None) or settings.llm_api_key
    llm_base_url = (secrets.llm_base_url if secrets else None) or settings.llm_base_url
    llm_model = (secrets.llm_model if secrets else None) or settings.llm_model

    if not (llm_api_key and llm_base_url and llm_model):
        raise RuntimeError(
            "LLM API設定が不足しています。プロジェクトのAPIキーノート画面で "
            "API key / base URL / model を入力してください。"
        )

    extra_headers = _openrouter_headers(llm_base_url, settings)
    llm = OpenAICompatibleProvider(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        extra_headers=extra_headers,
    )

    # Optional cheaper/faster model for the 1-call-per-block planning stage.
    planner_model = settings.llm_model_planner
    llm_planner: LLMProvider | None = None
    if planner_model and planner_model != llm_model:
        llm_planner = OpenAICompatibleProvider(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=planner_model,
            extra_headers=extra_headers,
        )

    image_api_key = (secrets.image_api_key if secrets else None) or settings.image_api_key
    image_base_url = (secrets.image_base_url if secrets else None) or "https://api.openai.com/v1"
    image_model = (secrets.image_model if secrets else None) or settings.image_model or "gpt-image-1"

    image: ImageProvider | None = None
    if image_api_key:
        try:
            image = OpenAIImageProvider(
                api_key=image_api_key, model=image_model, base_url=image_base_url
            )
        except Exception:  # pragma: no cover - defensive
            image = None

    voicevox = VoicevoxClient(project.voicevox_url)
    return ProviderBundle(
        llm=llm,
        image=image,
        voicevox=voicevox,
        use_fake=False,
        llm_planner=llm_planner,
    )


def build_voicevox_settings(project: Project, settings: Settings | None = None) -> VoicevoxSettings:
    s = settings or get_settings()
    return VoicevoxSettings(
        base_url=project.voicevox_url or s.voicevox_url,
        speaker_id=project.voicevox_speaker_id or s.voicevox_speaker_id,
        speed_scale=project.voicevox_speed_scale or s.voicevox_speed_scale,
        pitch_scale=project.voicevox_pitch_scale,
        intonation_scale=project.voicevox_intonation_scale,
        volume_scale=project.voicevox_volume_scale,
    )
