"""Reject impossible normal-state evaluation fixtures without executing jobs."""
from __future__ import annotations

from evaluation.contracts import Case


def validate_fixture(case: Case) -> None:
    state = case.initial
    for job in state.jobs:
        if job.get("project_id") == state.project_id and job.get("status") == "unknown":
            input_revision = job.get("input_revision")
            if type(input_revision) is not int or input_revision < 1:
                raise ValueError("unresolved job needs its recorded input revision")
            if any(revision > input_revision for revision in state.artifact_revisions):
                raise ValueError("unresolved job cannot precede a later successfully generated revision")
        if job.get("project_id") != state.project_id or job.get("status") not in {"pending", "running"}:
            continue
        if job.get("input_revision") != state.revision:
            raise ValueError("active job must use current revision under the settings lock")
        snapshot = job.get("input_settings")
        if not isinstance(snapshot, dict) or snapshot != state.settings:
            raise ValueError("active job fixture needs the full current settings snapshot")
        if state.project_status not in {"generating", "splitting", "planning", "rendering"}:
            raise ValueError("active job conflicts with project lifecycle state")
