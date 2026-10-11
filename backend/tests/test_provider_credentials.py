"""Provider keys are only sent to the endpoint of the same owner."""
from __future__ import annotations

from app.core.config import Settings
from app.core.security import SecretBundle
from app.services.provider_credentials import resolve_image_credential, resolve_llm_credential


def _settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)


def test_server_llm_key_is_not_sent_to_a_project_base_url() -> None:
    settings = _settings(llm_api_key="sk-server", llm_base_url="https://api.openai.com/v1", llm_model="m")
    resolved = resolve_llm_credential(SecretBundle(llm_base_url="https://attacker.example/v1"), settings)
    assert resolved.api_key is None
    assert resolved.base_url == "https://attacker.example/v1"
    assert resolved.origin == "missing"


def test_project_url_equal_to_the_server_url_may_use_the_server_key() -> None:
    settings = _settings(llm_api_key="sk-server", llm_base_url="https://api.openai.com/v1", llm_model="m")
    resolved = resolve_llm_credential(SecretBundle(llm_base_url="https://API.openai.com/v1/"), settings)
    assert resolved.api_key == "sk-server"
    assert resolved.origin == "global"


def test_project_key_goes_with_the_project_url() -> None:
    settings = _settings(llm_api_key="sk-server", llm_base_url="https://api.openai.com/v1", llm_model="m")
    bundle = SecretBundle(llm_api_key="sk-project", llm_base_url="https://openrouter.ai/api/v1")
    resolved = resolve_llm_credential(bundle, settings)
    assert (resolved.api_key, resolved.base_url, resolved.origin) == (
        "sk-project", "https://openrouter.ai/api/v1", "project")


def test_image_base_url_setting_is_used_with_the_server_image_key() -> None:
    settings = _settings(image_api_key="sk-img", image_base_url="https://openrouter.ai/api/v1")
    resolved = resolve_image_credential(None, settings)
    assert resolved.api_key == "sk-img"
    assert resolved.base_url == "https://openrouter.ai/api/v1"
    assert resolved.model == "gpt-image-1"


def test_image_defaults_to_openai_without_settings() -> None:
    resolved = resolve_image_credential(None, _settings())
    assert resolved.api_key is None
    assert resolved.base_url == "https://api.openai.com/v1"
