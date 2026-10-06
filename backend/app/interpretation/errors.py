"""Stable failure codes whose display text never echoes model/HTTP payloads."""
from __future__ import annotations

from typing import Literal

from app.interpretation.contracts import FailureView

FailureCode = Literal[
    "invalid_input", "invalid_json", "invalid_output", "candidate_not_offered",
    "invalid_arguments", "connection_failed", "timeout", "http_error",
    "invalid_response", "refused", "incomplete_response", "response_too_large",
    "configuration_error", "model_mismatch", "too_many_candidates",
]

_MESSAGES: dict[FailureCode, str] = {
    "invalid_input": "解釈に渡す要求・候補・状態が不正です。設定は変更していません。",
    "invalid_json": "モデルの返答を厳密なJSONとして読めませんでした。設定は変更していません。",
    "invalid_output": "モデルの返答が所定の形式と一致しません。設定は変更していません。",
    "candidate_not_offered": "提示していない操作または版が返されました。設定は変更していません。",
    "invalid_arguments": "提案された値が操作の定義に合いません。設定は変更していません。",
    "connection_failed": "ローカルモデルに接続できません。サーバーの起動状態を確認してください。",
    "timeout": "解釈全体の制限時間を超えました。設定は変更せず、これ以上の自動試行は行いません。",
    "http_error": "モデルサーバーが要求を受け付けませんでした。接続条件を確認してください。",
    "invalid_response": "モデルサーバーの応答形式が不正です。設定は変更していません。",
    "refused": "モデルが回答を拒否しました。設定は変更していません。",
    "incomplete_response": "モデルの返答が完了していません。設定は変更していません。",
    "response_too_large": "モデルの返答がサイズ上限を超えました。設定は変更していません。",
    "configuration_error": "ローカルモデルの接続設定が不正です。通信は行っていません。",
    "model_mismatch": "設定したモデルと応答元のモデルが一致しません。モデル名を確認してください。設定は変更していません。",
    "too_many_candidates": "操作の候補が多すぎて一度に解釈できません。検索による絞り込みを有効にしてください。設定は変更していません。",
}


class InterpretationError(ValueError):
    """A boundary rejection with bounded, non-sensitive public information."""

    def __init__(self, code: FailureCode, *, http_status: int | None = None) -> None:
        self.code = code
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        super().__init__(_MESSAGES[code])

    def as_view(self) -> FailureView:
        return FailureView(reason_code=self.code, message=_MESSAGES[self.code], http_status=self.http_status)
