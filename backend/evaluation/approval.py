"""Carry recorded decisions only when the complete case content is identical."""
from __future__ import annotations

from evaluation.contracts import Case, ReviewLedger
from evaluation.corpus import case_digest, digest, pending_ledger


def validate_ledger_manifest(ledger: ReviewLedger) -> None:
    pairs = [(entry.case_id, entry.case_sha256) for entry in ledger.entries]
    if len({key for key, _ in pairs}) != len(pairs) or digest(sorted(pairs)) != ledger.corpus_sha256:
        raise ValueError("review entries do not match the recorded corpus fingerprint")


def carry_forward_review(previous: ReviewLedger, current: list[Case]) -> ReviewLedger:
    """Do not re-approve changed data or require obsolete labels to pass new rules.

    This checks a previously recorded manifest, not the identity of its human
    author. New approval must still come from an actual, recorded user decision.
    """
    validate_ledger_manifest(previous)
    result = pending_ledger(current, previous.role)
    result.reviewer = previous.reviewer
    result.reviewed_at = previous.reviewed_at
    old = {entry.case_id: entry for entry in previous.entries}
    for entry, case in zip(result.entries, current, strict=True):
        prior = old.get(entry.case_id)
        if prior and prior.case_sha256 == case_digest(case):
            entry.decision = prior.decision
            entry.note = prior.note + f"\n同一のケース全文ハッシュを照合して移行。元corpus: {previous.corpus_sha256}"
        elif prior:
            entry.note = f"ケース内容が変わったため再確認が必要。元ケースhash: {prior.case_sha256}"
    return result
