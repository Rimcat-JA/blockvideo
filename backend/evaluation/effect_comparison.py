"""Narrow outcome aliases; phase and every effect assertion remain observable."""
from __future__ import annotations

from typing import Any

from evaluation.contracts import Effects


def effect_signature(effect: Effects, *, interpretation: str, event_kind: str, phase: str) -> dict[str, Any]:
    """Metadata for future scorers, not an inference runner or a passing score.

    Call separately at submit and after_event. Never collapse an early rejection
    into a confirmation request just because both avoided a write.
    """
    value = effect.model_dump(mode="json")
    outcome = effect.outcome
    no_revision_write = not effect.settings_delta and effect.revision_delta == 0
    no_job_write = effect.new_jobs == 0 and not effect.job_assertions
    if (outcome in {"saved", "unchanged"} and interpretation == "operation"
            and no_revision_write and no_job_write and not effect.confirmation_required):
        outcome = "unchanged_settings"
    elif (outcome in {"dismissed", "unchanged"} and interpretation == "no_operation"
            and no_revision_write and no_job_write and not effect.confirmation_required):
        outcome = "no_operation"
    elif (outcome in {"generation_queued", "replayed"} and interpretation == "operation"
            and phase == "after_event" and event_kind == "confirm_twice"
            and effect.new_jobs == 1 and not effect.confirmation_required):
        outcome = "one_confirmed_job"
    value["outcome"] = outcome
    return {"phase": phase, "interpretation": interpretation, "event_kind": event_kind, "effects": value}
