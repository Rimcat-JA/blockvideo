"""Require explicit opaque job/history references before proposing execution."""
from __future__ import annotations

import re
import unicodedata

from app.interpretation.contracts import ClarificationProposal, OperationProposal
from app.operations.policies import load_policies


def reference_question(text: str, proposal: OperationProposal) -> ClarificationProposal | None:
    """Minimal state is not a source of job IDs or a chosen historical version.

    This is a conservative reference binding, not a replacement language parser.
    Names, implicit 'previous' and multi-reference corrections need later dialogue.
    Which argument is a reference, and how it is evidenced, comes from the policy.
    """
    policies = load_policies()
    binding = policies.get(proposal.operation_id).reference
    if binding is None:
        return None
    kind = policies.references[binding.kind]
    text = unicodedata.normalize("NFKC", text)
    references = {int(value) for pattern in kind.patterns for value in re.findall(pattern, text, re.IGNORECASE)}
    if references == {proposal.arguments[binding.argument]}:
        return None
    return ClarificationProposal(kind="clarification", question=kind.question, missing_fields=["arguments"])
