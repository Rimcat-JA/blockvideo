"""Stage schemas for validation of LLM outputs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.block import VisualType


class SplitBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    source_text: str = Field(min_length=1, max_length=4000)
    # May legitimately be empty: a block that is only a fenced code listing
    # has nothing to read aloud. The splitter fills these with a spoken cue
    # before the block reaches the voice stage.
    tts_text: str = Field(min_length=0, max_length=4000)


class SplitPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[SplitBlock] = Field(min_length=1, max_length=500)

    @field_validator("blocks")
    @classmethod
    def _validate_indices(cls, blocks: list[SplitBlock]) -> list[SplitBlock]:
        for i, block in enumerate(blocks):
            if block.index != i:
                raise ValueError(
                    f"block index mismatch at position {i}: expected {i}, got {block.index}"
                )
        return blocks


class VisualPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_type: VisualType
    heading: str | None = Field(default=None, max_length=120)
    visual_summary: str | None = Field(default=None, max_length=400)
    image_prompt: str | None = Field(default=None, max_length=2000)
    code: str | None = Field(default=None, max_length=8000)
    # Slide content supplied by the script author, drawn exactly as written.
    # Never requested from a model — a model asked to design a diagram has to
    # invent identifiers for whatever the script left implicit, and does.
    verbatim: str | None = Field(default=None, max_length=8000)
    language: str | None = Field(default=None, max_length=40)
    formula: str | None = Field(default=None, max_length=2000)
    diagram: str | None = Field(default=None, max_length=20000)
    # Structured specs consumed by app.services.diagram_renderer. Kept as
    # free-form dicts here because the renderer already normalises loose
    # shapes; the JSON schema sent to the model is the strict contract.
    pointer_diagram: dict[str, Any] | None = None
    env_diagram: dict[str, Any] | None = None
    left_panel: dict[str, Any] | None = None
    right_panel: dict[str, Any] | None = None

    @field_validator("left_panel", "right_panel")
    @classmethod
    def _restrict_panels(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        # Reject unknown / unwanted keys explicitly.
        allowed = {"title", "content"}
        for key in list(value.keys()):
            if key not in allowed:
                value.pop(key)
        if "title" in value and not isinstance(value["title"], str):
            raise ValueError("panel.title must be a string")
        if "content" in value and not isinstance(value["content"], str):
            raise ValueError("panel.content must be a string")
        return value


class VisualPlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: VisualPlan


class GlobalVisualStylePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_visual_style: str = Field(min_length=4, max_length=2000)
