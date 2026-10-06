"""Per-operation safety and argument-layout policy, kept beside the catalog.

Guards read these facts instead of naming operation IDs in code, so adding an
operation means adding its definition and its policy entry. The file is separate
from definitions.json, whose bytes bind the retrieval index.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.operations.schema_validation import CatalogError

DEFAULT_POLICIES = Path(__file__).with_name("operation_policies.json")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceKind(_Strict):
    """How an opaque numeric reference (a job, a saved revision) is evidenced."""

    patterns: tuple[str, ...] = Field(min_length=1)
    marker: str = Field(min_length=1)
    answer_prefix: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def patterns_compile(self) -> ReferenceKind:
        for pattern in (*self.patterns, self.marker):
            re.compile(pattern)
        return self


class ReferenceBinding(_Strict):
    argument: str = Field(min_length=1)
    kind: str = Field(min_length=1)


class SettingsView(_Strict):
    """Where settings values sit in one operation version's arguments.

    ``settings_at`` is "" when arguments are the settings object itself, a key when
    they are nested, or None when the operation carries no settings object.
    ``aliases`` maps an argument name to the setting it expresses.
    """

    settings_at: str | None = None
    aliases: dict[str, str] = Field(default_factory=dict)


class OperationPolicy(_Strict):
    mutates: bool
    requires_confirmation: bool
    allows_generate_after_save: bool = False
    requires_arguments: bool = False
    negative_phrases: tuple[str, ...] = ()
    # Regexes for negations that name a target ("動画3には戻さない"), matched on normalized text.
    negative_patterns: tuple[str, ...] = ()
    reference: ReferenceBinding | None = None
    settings_views: dict[int, SettingsView] = Field(default_factory=dict)


class FollowUpGeneration(_Strict):
    operation_id: str = Field(min_length=1)
    operation_version: int = Field(ge=1)
    arguments: dict[str, Any]


class OperationPolicies(_Strict):
    schema_version: int
    global_negative_phrases: tuple[str, ...]
    # A request made only of negated instructions ("…しないで") asks for no change, whichever
    # operation the model proposed: removing every negated clause leaves only the residue
    # pattern (punctuation, polite endings). Both are matched on normalized text.
    negated_clause_pattern: str | None = None
    negation_residue_pattern: str | None = None
    # Where a stated number for a setting is found: after a keyword, or in the regex's group.
    setting_keywords: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    follow_up_generation: FollowUpGeneration
    references: dict[str, ReferenceKind]
    operations: dict[str, OperationPolicy]

    @model_validator(mode="after")
    def references_are_known(self) -> OperationPolicies:
        if self.schema_version != 1:
            raise ValueError("unsupported policy schema_version")
        for policy in self.operations.values():
            if policy.reference is not None and policy.reference.kind not in self.references:
                raise ValueError("operation policy references an unknown reference kind")
        for pattern in (self.negated_clause_pattern, self.negation_residue_pattern,
                        *(item for policy in self.operations.values() for item in policy.negative_patterns),
                        *(item for items in self.setting_keywords.values() for item in items)):
            if pattern is not None:
                re.compile(pattern)
        if self.follow_up_generation.operation_id not in self.operations:
            raise ValueError("follow-up generation operation has no policy")
        return self

    # An operation without a policy is treated as mutating and confirmed, never as safe.
    def get(self, operation_id: str) -> OperationPolicy:
        return self.operations.get(operation_id) or OperationPolicy(mutates=True, requires_confirmation=True)

    def settings_view(self, operation_id: str, operation_version: int) -> SettingsView | None:
        return self.get(operation_id).settings_views.get(operation_version)

    def require_catalog_coverage(self, operations: set[tuple[str, int]]) -> None:
        """Every catalog operation must carry an explicit policy (fail closed)."""
        missing = sorted({operation_id for operation_id, _ in operations} - set(self.operations))
        if missing:
            raise CatalogError(f"operations without a policy: {missing}")
        follow = self.follow_up_generation
        if (follow.operation_id, follow.operation_version) not in operations:
            raise CatalogError("follow-up generation operation is not in the catalog")


@lru_cache(maxsize=4)
def load_policies(path: Path = DEFAULT_POLICIES) -> OperationPolicies:
    try:
        return OperationPolicies.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError, re.error) as exc:
        raise CatalogError(f"invalid operation policies: {exc}") from exc


def settings_base(view: SettingsView | None, arguments: dict[str, Any]) -> dict[str, Any] | None:
    """The settings object an operation carries, or None when it carries none."""
    if view is None or view.settings_at is None:
        return None
    return arguments if view.settings_at == "" else arguments.get(view.settings_at, {})


def settings_values(view: SettingsView | None, arguments: dict[str, Any]) -> dict[str, Any]:
    """Settings object plus aliased arguments, keyed by setting name."""
    if view is None:
        return {}
    values = dict(settings_base(view, arguments) or {})
    values.update({setting: arguments[name] for name, setting in view.aliases.items() if name in arguments})
    return values
