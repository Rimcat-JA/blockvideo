"""Read and validate data without leaking held-out content in diagnostics."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.interpretation.contracts import DialogueContextTurn, MinimalState
from app.operations.catalog import CatalogError, load_catalog, validate_arguments
from evaluation.contracts import Case, ReviewEntry, ReviewLedger
from evaluation.fixtures import validate_fixture


def digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def case_digest(case: Case) -> str:
    return digest(case.model_dump(mode="json"))


def corpus_digest(cases: list[Case]) -> str:
    return digest(sorted((case.case_id, case_digest(case)) for case in cases))


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def load_cases(path: Path) -> list[Case]:
    cases = []
    catalog = load_catalog(Path(__file__).parents[1] / "app/operations/definitions.json")
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case.model_validate_json(line)
            validate_fixture(case)
            if not {"subtitle_font_size", "voicevox_speed_scale"} <= case.initial.settings.keys():
                raise ValueError("required fixture settings are missing")
            MinimalState(selected_project_id=case.initial.project_id, revision=case.initial.revision,
                         subtitle_font_size=case.initial.settings["subtitle_font_size"],
                         status=case.initial.project_status)
            validate_arguments(catalog.require("project.settings.update", 1), case.initial.settings)
            for turn in case.initial.prior_turns:
                context = {key: value for key, value in turn.items()
                           if key in DialogueContextTurn.model_fields}
                DialogueContextTurn.model_validate(context)
            for operation in case.expected.operations:
                definition = catalog.require(operation.operation_id, operation.operation_version)
                validate_arguments(definition, operation.arguments)
                if operation.generate_after_save and operation.operation_id not in {
                    "project.settings.update", "project.subtitle-font-size.set", "project.subtitle-font-size.adjust"
                }:
                    raise ValueError("only settings support follow-up generation")
            cases.append(case)
        except (ValidationError, CatalogError, ValueError) as exc:
            # Validation errors can embed the input. Return field locations only.
            locations = [".".join(map(str, e["loc"])) for e in exc.errors()] if isinstance(exc, ValidationError) else []
            raise ValueError(f"invalid case at row {number}; fields={locations}") from None
    if not cases:
        raise ValueError("empty corpus")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    groups: dict[str, tuple[str, str]] = {}
    for case in cases:
        origin = (case.split, case.source_request)
        if case.group_id in groups and groups[case.group_id] != origin:
            raise ValueError("one group has multiple origins or splits")
        groups[case.group_id] = origin
    return cases


def split_summary(development: list[Case], held_out: list[Case]) -> dict[str, Any]:
    if any(case.split != "development" for case in development) or any(case.split != "held_out" for case in held_out):
        raise ValueError("wrong split in corpus")
    shared_groups = {c.group_id for c in development} & {c.group_id for c in held_out}
    shared_origins = {normalized(c.source_request) for c in development} & {normalized(c.source_request) for c in held_out}
    # Short answers such as a bare number can legitimately recur in independent
    # dialogue. Compare them with their preceding dialogue, not in isolation.
    def request_key(case: Case) -> str:
        texts = [str(turn.get("text", "")) for turn in case.initial.prior_turns] + [case.request.text]
        return digest([normalized(text) for text in texts])
    shared_requests = {request_key(c) for c in development} & {request_key(c) for c in held_out}
    if shared_groups or shared_origins or shared_requests:
        raise ValueError(f"split leakage: groups={len(shared_groups)}, origins={len(shared_origins)}, requests={len(shared_requests)}")
    return {"development": summary(development), "held_out": summary(held_out),
            "shared_groups": 0, "shared_normalized_origins": 0, "shared_normalized_requests": 0,
            "semantic_independence": "requires independent reviewer; string checks are insufficient"}


def summary(cases: list[Case]) -> dict[str, Any]:
    return {"cases": len(cases), "groups": len({c.group_id for c in cases}),
            "corpus_sha256": corpus_digest(cases),
            "tags": dict(sorted(Counter(tag for case in cases for tag in set(case.tags)).items())),
            "interpretations": dict(sorted(Counter(c.expected.interpretation for c in cases).items()))}


def pending_ledger(cases: list[Case], role: str = "human") -> ReviewLedger:
    return ReviewLedger(schema_version=1, corpus_sha256=corpus_digest(cases), role=role,
                        reviewer="", reviewed_at="", entries=[ReviewEntry(case_id=c.case_id,
                        case_sha256=case_digest(c), decision="pending", note="") for c in cases])


def load_review(path: Path, cases: list[Case], role: str) -> ReviewLedger:
    try:
        ledger = ReviewLedger.model_validate_json(path.read_text(encoding="utf-8-sig"))
    except (ValidationError, ValueError):
        raise ValueError("invalid review ledger") from None
    if ledger.role != role or ledger.corpus_sha256 != corpus_digest(cases):
        raise ValueError("review role or corpus hash mismatch")
    hashes = {case.case_id: case_digest(case) for case in cases}
    ids = [entry.case_id for entry in ledger.entries]
    if len(ids) != len(set(ids)) or set(ids) != set(hashes):
        raise ValueError("review must contain every case exactly once")
    if any(entry.case_sha256 != hashes[entry.case_id] for entry in ledger.entries):
        raise ValueError("review case hash mismatch")
    return ledger


def eligibility(cases: list[Case], human: ReviewLedger | None, independent: ReviewLedger | None) -> dict[str, Any]:
    def decisions(review: ReviewLedger | None, role: str) -> dict[str, str]:
        if review is None:
            return {}
        hashes = {c.case_id: case_digest(c) for c in cases}
        ids = [e.case_id for e in review.entries]
        if (review.role != role or review.corpus_sha256 != corpus_digest(cases)
                or len(ids) != len(set(ids)) or set(ids) != set(hashes)
                or any(e.case_sha256 != hashes[e.case_id] for e in review.entries)):
            raise ValueError("review is not bound to this corpus and role")
        return {e.case_id: e.decision for e in review.entries}
    human_decisions = decisions(human, "human")
    ai_decisions = decisions(independent, "independent_ai")
    eligible = [c.case_id for c in cases if human_decisions.get(c.case_id) == "approved"
                and ai_decisions.get(c.case_id) == "approved"]
    return {"total": len(cases), "human_approved": sum(v == "approved" for v in human_decisions.values()),
            "independent_ai_approved": sum(v == "approved" for v in ai_decisions.values()),
            "eligible_count": len(eligible), "excluded_count": len(cases) - len(eligible),
            "eligible_case_ids": eligible,
            "score": None, "reason": "D24 contains labels only; no inference scores"}
