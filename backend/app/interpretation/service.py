"""Interpret a request without receiving a database, handler or executor."""
from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.interpretation.candidates import candidate_payload, response_schema, select_candidates
from app.interpretation.contracts import InterpretationInput, InterpretationOutcome
from app.interpretation.errors import InterpretationError
from app.interpretation.parser import parse_proposal
from app.interpretation.transport import ModelMessage, StructuredAdapter
from app.operations.catalog import OperationCatalog
from app.operations.limits import MAX_PROMPT_CANDIDATES

_RULES_PATH = Path(__file__).with_name("prompt_rules.json")


@lru_cache(maxsize=2)
def _rules(path: Path = _RULES_PATH) -> tuple[tuple[frozenset[str], frozenset[str], str | None, str], ...]:
    """Ordered prompt rules; a rule tagged with operations applies only when one is
    offered, a rule tagged with modes only in those modes (untagged: every mode), and
    a rule tagged with a feature only while that feature is enabled."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("rules"), list):
        raise ValueError("invalid prompt rules")
    return tuple((frozenset(item["operations"]), frozenset(item.get("modes", ("normal", "yolo"))),
                  item.get("feature"), item["text"]) for item in raw["rules"])


DEFAULT_FEATURES: frozenset[str] = frozenset({"plan"})


def system_prompt(offered: set[str], mode: str = "normal",
                  features: frozenset[str] = DEFAULT_FEATURES) -> str:
    """General rules plus the rules of offered operations, in their fixed order.

    With every catalog operation offered, normal mode and no features this is
    exactly the D35 All Tools prompt; the plan feature and YOLO mode append rules.
    """
    return "".join(text + "\n" for operations, modes, feature, text in _rules()
                   if mode in modes and (feature is None or feature in features)
                   and (not operations or operations & offered))


_READINESS_SYSTEM = """
候補のreadiness_hintは引数を確定する前の状態の観測です。実行許可ではありません。
readinessと意味の一致は別です。まず依頼の意味に合う操作と値を選んでください。
blockedでも正しい操作は同じoperationとして提案し、アプリ側の実行前検査に渡してください。
生成中の設定変更を、生成停止・再試行・別の設定・状態照会へ勝手に置き換えてはいけません。
生成が終わるまで待つ予約もできません。blockedは未対応を意味しません。
arguments_uncheckedは引数が未検査という意味です。入力の値が明確なら抽出し、不足なら質問します。
stateや候補の観測は後で変わり得ます。対象番号やjob_id、戻す版を観測から推測してはいけません。
"""


class Interpreter:
    """A transport and approved metadata are the only injected capabilities."""

    def __init__(self, catalog: OperationCatalog, adapter: StructuredAdapter,
                 *, timeout_seconds: float = 120, max_attempts: int = 2) -> None:
        if not 0 < timeout_seconds <= 180:
            raise ValueError("interpretation deadline must be within 180 seconds")
        self._catalog = catalog.model_copy(deep=True)
        self._adapter = adapter
        self._timeout_seconds = timeout_seconds
        if type(max_attempts) is not int or max_attempts not in (1, 2):
            raise ValueError("interpretation attempts must be one or two")
        self._max_attempts = max_attempts

    async def preview(self, request: InterpretationInput) -> InterpretationOutcome:
        """At most two calls under one deadline, before any execution exists."""
        attempts = 0
        repair_codes: list[str] = []
        try:
            # Revalidate a snapshot even if a caller bypassed Pydantic construction.
            try:
                request = InterpretationInput.model_validate(request.model_dump())
            except (ValidationError, AttributeError, TypeError):
                raise InterpretationError("invalid_input") from None
            definitions = select_candidates(self._catalog, request.candidates)
            if len(definitions) > MAX_PROMPT_CANDIDATES:
                # The catalog outgrew one prompt; retrieval must narrow it first.
                raise InterpretationError("too_many_candidates")
            payload = {
                "request": request.text,
                "state": request.state.model_dump(exclude_none=True),
                "candidates": [candidate_payload(item) for item in definitions],
            }
            if request.candidate_state is not None:
                hints = {(r.operation_id, r.operation_version): r for r in request.candidate_state.candidates}
                for item in payload["candidates"]:
                    item["readiness_hint"] = hints[(item["operation_id"], item["operation_version"])].model_dump(
                        mode="json", exclude={"operation_id", "operation_version"})
                payload["candidate_state_observed_at"] = request.candidate_state.observed_at
            if request.dialogue:
                payload["dialogue"] = [turn.model_dump(mode="json", exclude_none=True) for turn in request.dialogue]
            prompt = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            try:
                prompt.encode("utf-8")
            except UnicodeError:
                raise InterpretationError("invalid_input") from None
            base = system_prompt({item.operation_id for item in definitions},
                                 "yolo" if request.guess_missing else "normal")
            system = base + _READINESS_SYSTEM if request.candidate_state is not None else base
            messages = (ModelMessage("system", system), ModelMessage("user", prompt))
            async with asyncio.timeout(self._timeout_seconds):
                for attempt in range(self._max_attempts):
                    attempts += 1
                    # Transport failures are never retried as output repair.
                    content = await self._adapter.complete(messages, response_schema(definitions))
                    try:
                        proposal = parse_proposal(content, definitions).result
                        break
                    except InterpretationError as exc:
                        if attempt + 1 == self._max_attempts or exc.code not in {"invalid_json", "invalid_output"}:
                            raise
                        repair_codes.append(exc.code)
                        messages = (*messages, ModelMessage("user", json.dumps({
                            "repair": {"failure_code": exc.code,
                                "instruction": "元の依頼を所定のJSON形式で再提出してください。値の推測・部分実行は禁止です。不足はclarificationにしてください。"},
                        }, ensure_ascii=False)))
            status = {"operation": "proposed", "plan": "proposed", "clarification": "needs_input",
                      "unsupported": "unsupported", "no_operation": "dismissed"}[proposal.kind]
            return InterpretationOutcome(status=status, proposal=proposal, attempts=attempts,
                                         repair_codes=repair_codes)
        except TimeoutError:
            return InterpretationOutcome(status="error", failure=InterpretationError("timeout").as_view(),
                                         attempts=attempts, repair_codes=repair_codes)
        except InterpretationError as exc:
            return InterpretationOutcome(status="error", failure=exc.as_view(), attempts=attempts,
                                         repair_codes=repair_codes)
