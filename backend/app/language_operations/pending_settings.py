"""Prevent a short answer from silently dropping pending compound intent."""
from __future__ import annotations

from typing import Literal

from app.interpretation.contracts import ClarificationProposal, DialogueContextTurn, OperationProposal
from app.operations.policies import load_policies, settings_base


def pending_settings_question(
    turns: tuple[DialogueContextTurn, ...], relation: Literal["answer", "correction", "dismiss"] | None,
    proposal: OperationProposal,
) -> ClarificationProposal | None:
    """Previous proposals can veto partial saves, but never supply executable values."""
    policies = load_policies()
    view = policies.settings_view(proposal.operation_id, proposal.operation_version)
    if relation != "answer" or view is None:
        return None
    current = settings_base(view, proposal.arguments)
    for turn in reversed(turns):
        if turn.settings_saved or turn.status == "completed":
            break
        prior = turn.proposal
        fields = (settings_base(policies.settings_view(prior.operation_id, prior.operation_version), prior.arguments)
                  if isinstance(prior, OperationProposal) else None)
        if fields is None:
            continue
        required = set(fields) - {"subtitle_font_size"}
        present = set(current) if current is not None else set()
        if required - present or (prior.generate_after_save and not proposal.generate_after_save):
            return ClarificationProposal(kind="clarification", missing_fields=["arguments", "intent"],
                question="前の依頼の設定や生成希望が今回の提案から抜けています。字幕サイズ・ほかの設定・生成の希望をまとめて、もう一度指定してください。まだ保存していません。")
    return None
