"""Approved development-only interpreter inputs; not an end-to-end scorer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.interpretation.contracts import CandidateRef, DialogueContextTurn, InterpretationInput, MinimalState, OperationProposal
from evaluation.contracts import Case
from evaluation.corpus import eligibility, load_cases, load_review


def unreviewed_development(cases_path: Path) -> tuple[list[Case], dict[str, Any]]:
    """Every development case, for unofficial regression runs when approvals are stale.

    The result is never an approved score: callers must label it unofficial.
    """
    cases = load_cases(cases_path)
    if any(case.split != "development" for case in cases):
        raise ValueError("development probe refuses held-out material")
    return cases, {"unofficial": True, "approval_gate": "skipped", "total": len(cases)}


def approved_development(cases_path: Path, human_path: Path, ai_path: Path) -> tuple[list[Case], dict[str, Any]]:
    cases = load_cases(cases_path)
    if any(case.split != "development" for case in cases):
        raise ValueError("development probe refuses held-out material")
    human = load_review(human_path, cases, "human")
    ai = load_review(ai_path, cases, "independent_ai")
    gate = eligibility(cases, human, ai)
    ids = set(gate["eligible_case_ids"])
    if not ids:
        raise ValueError("no approved development cases")
    return [case for case in cases if case.case_id in ids], gate


def interpretation_input(case: Case, candidates: tuple[CandidateRef, ...]) -> InterpretationInput:
    return InterpretationInput(text=case.request.text, candidates=candidates,
        state=MinimalState(selected_project_id=case.request.target_project_id, revision=case.initial.revision,
            subtitle_font_size=case.initial.settings["subtitle_font_size"], status=case.initial.project_status),
        dialogue=tuple(DialogueContextTurn.model_validate({key: value for key, value in turn.items()
            if key in DialogueContextTurn.model_fields}) for turn in case.initial.prior_turns))


def operation_meaning(proposal: OperationProposal) -> dict[str, Any]:
    """Equivalent settings encodings; no absolute-value inference or text repair."""
    name, arguments = proposal.operation_id, proposal.arguments
    if name == "project.subtitle-font-size.set":
        fields = {"subtitle_font_size": arguments["value"]}
    elif name == "project.subtitle-font-size.adjust":
        fields = {"subtitle_font_size_delta": arguments["delta"]}
    elif name == "project.settings.update":
        fields = dict(arguments if proposal.operation_version == 1 else arguments["settings"])
        if proposal.operation_version == 2 and arguments["subtitle_font_size_delta"] is not None:
            fields["subtitle_font_size_delta"] = arguments["subtitle_font_size_delta"]
    else:
        return proposal.model_dump(mode="json")
    if "pronunciation_overrides" in fields:
        fields["pronunciation_overrides"] = [{"accent": None, **item} for item in fields["pronunciation_overrides"]]
    return {"kind": "settings", "fields": fields, "generate_after_save": proposal.generate_after_save}


def proposal_matches(case: Case, proposal: object) -> bool:
    if getattr(proposal, "kind", None) != case.expected.interpretation:
        return False
    if not isinstance(proposal, OperationProposal):
        # Wording/which question is useful is explicitly outside this narrow metric.
        return True
    return any(operation_meaning(proposal) == operation_meaning(OperationProposal(
        kind="operation", **expected.model_dump(mode="json"))) for expected in case.expected.operations)
