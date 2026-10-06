"""Strict contracts for the deterministic D36 candidate freeze."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evaluation.tool_attestation import FileFingerprint

CANDIDATE_COMMIT_SUBJECT = "[DONE] Mission 35 Add recovery-oriented operational UI"
# The authorized successor fixes only D39 release-verification blockers of D35
# (credential-shaped test literals, build-time config rewrite, README prerequisites).
SUCCESSOR_COMMIT_SUBJECT = "[DONE] Mission 35.1 Fix release-candidate verification blockers"
CANDIDATE_COMMIT_SUBJECTS: tuple[str, ...] = (CANDIDATE_COMMIT_SUBJECT, SUCCESSOR_COMMIT_SUBJECT)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictFreezeModel(BaseModel):
    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def validate_schema_primitive(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("freeze schema version must be an integer")
        return value

    @field_validator("git_tree_clean", mode="before", check_fields=False)
    @classmethod
    def validate_clean_primitive(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("freeze clean flag must be a boolean")
        return value


class CandidateControl(_StrictFreezeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    git_commit_subject: Literal[
        "[DONE] Mission 35 Add recovery-oriented operational UI",
        "[DONE] Mission 35.1 Fix release-candidate verification blockers",
    ]
    git_tree_clean: Literal[True]


class CompletionMarker(_StrictFreezeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    files: list[FileFingerprint]

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[FileFingerprint]) -> list[FileFingerprint]:
        if [item.path for item in value] != [
            "d36-tool-attestation.json",
            "freeze-manifest.json",
        ]:
            raise ValueError("completion marker files must match the canonical artifacts")
        return value


class FreezeManifest(_StrictFreezeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1]
    candidate_id: Annotated[str, Field(pattern=r"^[0-9a-f]{16}-[0-9a-f]{12}$")]
    git_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    git_tree_clean: Literal[True]
    candidate_control_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    created_at: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
    runtime: dict[str, str]
    schema_version_number: Annotated[int, Field(strict=True, ge=1)]
    mode_configuration: dict[str, object]
    files: list[FileFingerprint]
    aggregate_sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[FileFingerprint]) -> list[FileFingerprint]:
        paths = [item.path for item in value]
        if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("candidate files must be non-empty, unique, and sorted")
        return value
