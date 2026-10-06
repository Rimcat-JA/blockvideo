"""Non-secret generation inputs and stable identities, independent of transport."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.config import get_settings
from app.core.security import secret_store
from app.models.project import Project


class GenerationCancelled(RuntimeError):
    """Cooperative cancellation reached a safe boundary before publication."""


class StaleGenerationInput(RuntimeError):
    """A job's inputs changed outside its guarded generation boundary."""


PROJECT_INPUT_FIELDS = (
    "title", "source_script", "llm_provider", "llm_base_url", "llm_model",
    "image_provider", "image_model", "voicevox_url", "voicevox_speaker_id",
    "voicevox_speed_scale", "voicevox_pitch_scale", "voicevox_intonation_scale",
    "voicevox_volume_scale", "subtitle_enabled", "subtitle_font_size",
    "subtitle_position", "subtitle_text_color", "subtitle_outline_color",
    "subtitle_background", "subtitle_max_chars_per_line", "visual_focus_enabled",
    "subtitle_mode", "narration_pacing_mode", "pronunciation_overrides",
    "narration_sentence_pause_seconds", "max_slides_per_block",
    "pre_margin_seconds", "post_margin_seconds", "min_display_seconds",
    "use_fake_providers",
)
RUNTIME_INPUT_FIELDS = (
    "output_width", "output_height", "output_fps", "crossfade_seconds",
    "subtitle_band_height", "splitter_min_chars", "splitter_max_chars",
    "splitter_target_chars", "splitter_segment_chars", "narration_repair_enabled",
    "llm_base_url", "llm_model", "llm_model_planner", "image_base_url", "image_model",
)
FROZEN_SNAPSHOT_KEYS = ("schema_version", "project_id", "project", "runtime", "resolved_providers")


def capture_inputs(project: Project) -> dict[str, Any]:
    """Copy user-editable inputs; omit progress, generated paths and raw secrets.

    Block plans are editable inputs once established. The runner separately
    records the resulting input snapshot after its own split/plan stages.
    """
    settings = get_settings()
    secrets = secret_store.get(project.id) if not project.use_fake_providers else None
    payload = {
        "schema_version": 1,
        "project_id": project.id,
        "global_visual_style": project.global_visual_style,
        "project": {key: getattr(project, key) for key in PROJECT_INPUT_FIELDS},
        "runtime": {key: getattr(settings, key) for key in RUNTIME_INPUT_FIELDS},
        "resolved_providers": {
            "fake": project.use_fake_providers,
            "llm_base_url": (secrets.llm_base_url if secrets else None) or settings.llm_base_url,
            "llm_model": (secrets.llm_model if secrets else None) or settings.llm_model,
            "planner_model": settings.llm_model_planner or (secrets.llm_model if secrets else None) or settings.llm_model,
            "llm_credential_origin": ("project" if secrets and secrets.llm_api_key else
                                      "global" if settings.llm_api_key else "missing"),
            "image_base_url": (secrets.image_base_url if secrets else None) or "https://api.openai.com/v1",
            "image_model": (secrets.image_model if secrets else None) or settings.image_model or "gpt-image-1",
            "image_credential_origin": ("project" if secrets and secrets.image_api_key else
                                        "global" if settings.image_api_key else "missing"),
        },
        "blocks": [{
            "id": block.id,
            "index": block.index,
            "source_text": block.source_text,
            "tts_text": block.tts_text,
            "visual_type": block.visual_type.value if block.visual_type else None,
            "visual_plan": block.visual_plan_json,
            "image_prompt": block.image_prompt,
        } for block in sorted(project.blocks, key=lambda row: row.index)],
    }
    # Detach nested SQLAlchemy JSON dictionaries from later in-place mutation.
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def fingerprint_inputs(snapshot: dict[str, Any]) -> str:
    """SHA-256 of canonical complete content, never Python's process-local hash."""
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
