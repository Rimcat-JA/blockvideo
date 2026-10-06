"""Deterministic negative-control intent guard driven by operation policies."""
from __future__ import annotations

import re
import unicodedata

from app.language_operations.clauses import clauses
from app.operations.policies import load_policies

_APOSTROPHE_TRANSLATION = str.maketrans({"‘": "'", "’": "'", "＇": "'"})


def normalized_intent(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.translate(_APOSTROPHE_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def only_negated_instructions(text: str) -> bool:
    """True when the request contains a negated instruction and nothing else to do."""
    policies = load_policies()
    negated, residue = policies.negated_clause_pattern, policies.negation_residue_pattern
    if negated is None or residue is None:
        return False
    remainder, removed = re.subn(negated, "", normalized_intent(text))
    return removed > 0 and re.fullmatch(residue, remainder) is not None


def negative_control_reason(text: str, operation_id: str) -> str | None:
    policies = load_policies()
    policy = policies.get(operation_id)
    if not policy.mutates:
        return None
    normalized = normalized_intent(text)
    if any(phrase in normalized for phrase in policies.global_negative_phrases):
        return "explicit_negative_intent"
    if any(phrase in normalized for phrase in policy.negative_phrases):
        return "explicit_negative_intent"
    # A named negation must sit in one clause: "設定を第2版に戻して動画は戻さないで" negates only the video.
    if any(re.search(pattern, clause) for pattern in policy.negative_patterns for clause in clauses(text)):
        return "explicit_negative_intent"
    if only_negated_instructions(text):
        return "explicit_negative_intent"
    return None
