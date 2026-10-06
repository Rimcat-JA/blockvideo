"""Describe D37 projection failures by structure only, never by held-out content."""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from evaluation.blinded_runner import case_to_unlabeled
from evaluation.contracts import Case

_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _label(value: object) -> str:
    """Return identifier-shaped labels; anything else may be authored content."""
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else "<redacted>"


def _keys(value: object) -> list[str]:
    return sorted(_label(key) for key in value) if isinstance(value, dict) else []


def _location(parts: tuple[Any, ...]) -> str:
    return ".".join(str(part) if isinstance(part, int) else _label(part) for part in parts)


def _relation(case: Case, project_id: object) -> str:
    """Name a project reference by its role, never by its value."""
    if project_id is None:
        return "null"
    if project_id == case.initial.project_id:
        return "selected"
    if project_id == case.request.target_project_id:
        return "request_target"
    return "other" if type(project_id) is int else "<redacted>"


def _event_timing(case: Case) -> dict[str, object]:
    """Timing labels and revision offsets relative to the initial revision."""
    details = case.event.details
    offsets = {
        key: details[key] - case.initial.revision
        for key in ("competing_revision", "external_revision", "original_confirmation_revision")
        if type(details.get(key)) is int
    }
    labels = {key: _label(details[key]) for key in ("phase", "when", "then") if key in details}
    return {"labels": labels, "revision_offsets": offsets}


def projection_failure(case: Case) -> dict[str, object] | None:
    """Return a content-free failure description, or None when the case projects."""
    try:
        case_to_unlabeled(case)
    except KeyError as error:
        problem: dict[str, object] = {
            "error": "missing_key",
            "missing_key": _label(error.args[0] if error.args else None),
        }
    except ValidationError as error:
        problem = {
            "error": "validation",
            "fields": sorted({
                f"{_location(item['loc'])}:{item['type']}" for item in error.errors()
            }),
            # Model validators raise fixed messages; they never echo the input.
            "messages": sorted({
                str(item["ctx"]["error"]) for item in error.errors()
                if item["type"] == "value_error" and "ctx" in item
            }),
        }
    except (TypeError, ValueError) as error:
        problem = {"error": type(error).__name__}
    else:
        return None
    return {
        "case_id": case.case_id,
        "event_timing": _event_timing(case),
        "request_target": _relation(case, case.request.target_project_id),
        "event_kind": case.event.kind,
        "event_detail_keys": _keys(case.event.details),
        "job_keys": [_keys(job) for job in case.initial.jobs],
        "job_project_is_selected": [
            job.get("project_id") == case.initial.project_id for job in case.initial.jobs
        ],
        "prior_turns": [
            {
                "keys": _keys(turn),
                "status": _label(turn.get("status")),
                "target": _relation(case, turn.get("target_project_id", turn.get("project_id"))),
                "is_continuation_parent": case.request.continuation is not None
                and case.request.continuation.parent_request_id == turn.get("request_id"),
                "proposal_kind": _label(
                    turn["proposal"].get("kind")
                    if isinstance(turn.get("proposal"), dict) else None
                ),
            }
            for turn in case.initial.prior_turns
        ],
        **problem,
    }
