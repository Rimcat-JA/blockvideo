"""Bounded linked intent context and pending-request invalidation."""
from __future__ import annotations

import re
import unicodedata

from sqlalchemy.orm import Session

from app.interpretation.contracts import DialogueContextTurn, MinimalState, OperationProposal
from app.language_operations.contracts import LanguageError, LanguageInput, LanguageResponse
from app.models.language_request import LanguageRequestRecord
from app.models.language_turn import LanguageTurn
from app.models.operation_request import OperationReceipt
from app.operations.errors import OperationError
from app.operations.policies import load_policies
from app.operations.receipts import canonical_request


def context(db: Session, request: LanguageInput) -> tuple[DialogueContextTurn, ...]:
    """No project scripts, titles, provider secrets or whole state snapshots."""
    request_id = request.continuation.parent_request_id if request.continuation else None
    turns: list[DialogueContextTurn] = []
    length = 0
    while request_id:
        turn = db.get(LanguageTurn, request_id)
        record = db.get(LanguageRequestRecord, request_id)
        if turn is None or record is None:
            raise LanguageError("dialogue_unavailable", "この依頼には対話の記録がありません。依頼全体を新しく入力してください。")
        response = LanguageResponse.model_validate(record.response_json)
        proposal = response.interpretation.proposal if response.interpretation else None
        question = response.clarification.question if response.clarification else None
        receipt = db.get(OperationReceipt, record.core_request_id) if response.generate_after_save else None
        saved = bool(receipt and response.prepared_request
                     and receipt.canonical_request == canonical_request(response.prepared_request))
        item = DialogueContextTurn(text=turn.text, status=response.status, proposal=proposal,
                                   question=question, relation=turn.relation, settings_saved=saved)
        length += len(item.model_dump_json())
        if len(turns) >= 8 or length + len(request.text) > 6000:
            raise LanguageError("dialogue_limit", "対話が長くなりました。現在の希望を一つの新しい依頼として入力してください。")
        turns.append(item)
        request_id = turn.parent_request_id
    return tuple(reversed(turns))


def require_not_superseded(db: Session, request_id: str) -> LanguageTurn | None:
    turn = db.get(LanguageTurn, request_id)
    if turn is not None and turn.successor_request_id:
        raise LanguageError("dialogue_superseded", "この質問には既に回答または訂正があります。最新の依頼を確認してください。")
    return turn


def attach(db: Session, request: LanguageInput, state: MinimalState | None,
           parent: LanguageResponse | None) -> None:
    """Called only inside the claim writer transaction, before inference."""
    continuation = request.continuation
    if continuation:
        if parent is None:
            raise LanguageError("parent_not_found", "回答先の要求が見つかりません。")
        turn = require_not_superseded(db, parent.request_id)
        if turn is None:
            raise LanguageError("dialogue_unavailable", "以前の依頼には対話の記録がありません。依頼全体を新しく入力してください。")
        if parent.project_id is not None and parent.project_id != request.target.resolved_id:
            raise LanguageError("dialogue_target_mismatch", "この回答は別のプロジェクトへの依頼に属しています。元の対象に戻ってください。")
        allowed = {"answer": {"needs_input"}, "correction": {"needs_input", "ready", "completed"},
                   "dismiss": {"needs_input", "ready"}}[continuation.relation]
        if parent.status not in allowed:
            raise LanguageError("dialogue_not_pending", "この依頼への回答・訂正は受け付けられません。別の操作なら新しく依頼してください。")
        expected = parent.result.revision if parent.result else parent.base_revision
        if continuation.relation != "dismiss" and expected is not None and (state is None or state.revision != expected):
            raise LanguageError("dialogue_stale", "この質問の後に設定が変わりました。現在の設定を確認し、希望を新しく依頼してください。")
        context(db, request)  # Validate bounds before consuming the predecessor.
        turn.successor_request_id = request.request_id
    db.add(LanguageTurn(request_id=request.request_id, text=request.text,
                        parent_request_id=continuation.parent_request_id if continuation else None,
                        relation=continuation.relation if continuation else None))


def require_current(db: Session, request_id: str) -> None:
    """Run under the core writer lock so old-confirm versus correction is atomic."""
    turn = db.get(LanguageTurn, request_id)
    if turn and turn.successor_request_id:
        raise OperationError("dialogue_superseded", "この依頼は回答・訂正・取り下げによって置き換えられています。")


def reference_text(request: LanguageInput, turns: tuple[DialogueContextTurn, ...], proposal: OperationProposal) -> str:
    """Bind opaque references to explicit dialogue text, never state guesses."""
    text = unicodedata.normalize("NFKC", request.text).strip()
    policies = load_policies()
    binding = policies.get(proposal.operation_id).reference
    if not turns or binding is None:
        return text
    kind = policies.references[binding.kind]
    marker = kind.marker
    if re.search(marker, text, re.IGNORECASE):
        return text
    question = turns[-1].question or ""
    if re.fullmatch(r"\d+", text) and re.search(marker, question, re.IGNORECASE):
        return kind.answer_prefix + text
    for turn in reversed(turns):
        if re.search(marker, turn.text, re.IGNORECASE):
            return turn.text + "\n" + text
    return text
