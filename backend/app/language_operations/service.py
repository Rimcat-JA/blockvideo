"""All Tools orchestration; only the existing operation core may mutate state."""
from __future__ import annotations

import asyncio
import hmac
from time import perf_counter

from sqlalchemy.orm import Session

from app.interpretation.contracts import (
    CandidateRef, ClarificationProposal, FailureView, InterpretationInput, OperationProposal, PlanProposal,
)
from app.interpretation.service import Interpreter
from app.interpretation.transport import StructuredAdapter
from app.language_operations import dialogue, observability, repository
from app.language_operations.candidate_state import current_candidates
from app.language_operations.contracts import (
    LanguageError, LanguageExecution, LanguageInput, LanguageResponse, YoloReport,
)
from app.language_operations.explicit_values import explicit_conflict, plan_drops_stated
from app.language_operations.intent_guard import negative_control_reason, only_negated_instructions
from app.language_operations.references import reference_question
from app.language_operations.pronunciation import merged_arguments, reading_question
from app.language_operations.subtitle_values import subtitle_question
from app.language_operations.pending_settings import pending_settings_question
from app.language_operations.settings_values import settings_value_question
from app.operations.catalog import OperationCatalog
from app.operations.contracts import OperationRequest, OperationTarget, Readiness
from app.operations.errors import OperationError
from app.operations.policies import load_policies
from app.operations.service import OperationService
from app.semantic_interpretation.interface import CandidateInterpreter



_STATED_VALUE_QUESTION = ClarificationProposal(
    kind="clarification", missing_fields=["arguments"],
    question="依頼に書かれた値や番号と、提案された内容が一致しません。値をもう一度指定してください。まだ何も変更していません。")


class LanguageOperationService:
    def __init__(
        self, core: OperationService, adapter: StructuredAdapter | None = None,
        *, review_all: bool = False, semantic: CandidateInterpreter | None = None,
        readiness_annotations: bool = False, yolo_enabled: bool = True,
    ) -> None:
        self.core = core
        self.adapter = adapter
        self.review_all = review_all
        self.semantic = semantic
        self.readiness_annotations = readiness_annotations
        self.yolo_enabled = yolo_enabled

    async def prepare(self, db: Session, request: LanguageInput) -> LanguageResponse:
        """Freeze a model proposal; this method never calls execute."""
        response, owner, state = repository.claim(db, request)
        if owner is None or state is None:
            observability.record(response, "replay_or_precheck")
            return response
        started = perf_counter()
        guard_code = None
        yolo = request.mode == "yolo"
        bypassed: list[str] = []
        dropped: list[str] = []
        unresolved: str | None = None
        try:
            if yolo and not self.yolo_enabled:
                response = response.model_copy(update={"status": "error", "failure": FailureView(
                    reason_code="yolo_disabled",
                    message="確認なしの自動実行（YOLO）はこのサーバーで無効です。設定は変更していません。")})
            elif request.continuation and request.continuation.relation == "dismiss":
                response = response.model_copy(update={"status": "dismissed"})
            elif self.adapter is None:
                response = response.model_copy(update={"status": "error", "failure": FailureView(
                    reason_code="model_not_configured", message="自然言語用のローカルモデルが設定されていません。")})
            else:
                definitions = tuple(self.core.list_definitions())
                catalog = OperationCatalog(definitions=definitions)
                refs = [CandidateRef(operation_id=item.operation_id, operation_version=item.operation_version) for item in definitions]
                response = response.model_copy(update={"diagnostics": response.diagnostics.model_copy(update={"candidates": refs})})
                turns = dialogue.context(db, request)
                db.rollback()
                db.expire_all()
                interpretation_input = InterpretationInput(
                    text=request.text, state=state, dialogue=turns,
                    candidates=tuple(CandidateRef(operation_id=item.operation_id,
                                                  operation_version=item.operation_version) for item in definitions),
                    guess_missing=yolo,
                )
                if self.semantic is None:
                    outcome = await Interpreter(catalog, self.adapter).preview(interpretation_input)
                else:
                    provider = (lambda offered: current_candidates(db, self.core, offered, response.project_id)) if self.readiness_annotations else None
                    semantic = await self.semantic.preview(catalog, self.adapter, interpretation_input,
                                                           readiness_provider=provider)
                    outcome = semantic.interpretation
                    response = response.model_copy(update={"mode": "all_tools" if semantic.trace.policy == "all-tools-v1" else "semantic", "diagnostics":
                        response.diagnostics.model_copy(update={"retrieval": semantic.trace,
                            "candidates": list(semantic.candidates)})})
                response = response.model_copy(update={"interpretation": outcome})
                question = None
                relation = request.continuation.relation if request.continuation else None

                # Values the user stated: this request and, for an answer, the unsaved request it completes.
                stated_texts = [request.text]
                if relation == "answer":
                    for turn in reversed(turns):
                        if turn.settings_saved or turn.status == "completed":
                            break
                        stated_texts.append(turn.text)

                def vet(proposed: OperationProposal, *, whole_request: bool = True,
                        ) -> tuple[ClarificationProposal | None, str | None]:
                    """The first clarifying guard that fires; in YOLO it is recorded and skipped.

                    A contradiction of a stated value or reference is never skipped. A plan
                    step is checked for contradictions only: later steps may set the rest.
                    """
                    empty = ClarificationProposal(kind="clarification", question="どの設定を、どの値に変更しますか？",
                                                  missing_fields=["arguments"])
                    checks = (
                        ("reference", lambda: reference_question(
                            dialogue.reference_text(request, turns, proposed), proposed)),
                        ("subtitle_value", lambda: subtitle_question(request.text, turns, relation, proposed)),
                        ("settings_value", lambda: settings_value_question(request.text, turns, relation, proposed)),
                        ("pending_settings", lambda: pending_settings_question(turns, relation, proposed)),
                        ("reading", lambda: reading_question([turn.text for turn in turns] + [request.text], proposed)),
                        ("empty_settings", lambda: empty if load_policies().get(proposed.operation_id).requires_arguments
                         and not proposed.arguments else None),
                        ("explicit_value", lambda: _STATED_VALUE_QUESTION if explicit_conflict(
                            stated_texts, proposed, require_all=whole_request,
                            current_subtitle=state.subtitle_font_size) else None),
                    )
                    for code, check in checks:
                        found = check()
                        if found is None:
                            continue
                        if yolo and code != "explicit_value":
                            # Unattended: a guessed (missing) value stands; the bypass is reported.
                            bypassed.append(code)
                            continue
                        return found, code
                    return None, None

                plan_steps: list[OperationProposal] = []
                negated_plan = False
                negative_reason = None
                if isinstance(outcome.proposal, OperationProposal):
                    # A request made only of negations is dismissed before any clarifying
                    # question: asking "which revision?" for "第2版には戻さないで" is wrong.
                    if only_negated_instructions(request.text):
                        negative_reason = negative_control_reason(request.text, outcome.proposal.operation_id)
                    if negative_reason is None:
                        question, guard_code = vet(outcome.proposal)
                    follow_up = load_policies().follow_up_generation.operation_id
                    if (negative_reason is None and outcome.proposal.generate_after_save
                            and negative_control_reason(request.text, follow_up) is not None):
                        # "…して、動画は生成しないで": save only, never the negated generation.
                        outcome = outcome.model_copy(update={"proposal": outcome.proposal.model_copy(
                            update={"generate_after_save": False})})
                        if yolo:
                            dropped.append(follow_up)
                elif isinstance(outcome.proposal, PlanProposal):
                    for step in outcome.proposal.steps:
                        if negative_control_reason(request.text, step.operation_id) is not None:
                            if yolo:
                                dropped.append(step.operation_id)
                                continue
                            negated_plan = True
                            break
                        question, guard_code = vet(step, whole_request=False)
                        if question is not None:
                            break
                        plan_steps.append(step)
                    if question is None and plan_steps and plan_drops_stated(stated_texts, plan_steps):
                        # The plan as a whole must carry every value the user stated.
                        question, guard_code = _STATED_VALUE_QUESTION, "explicit_value"
                if question is None and negative_reason is None and isinstance(outcome.proposal, OperationProposal):
                    negative_reason = negative_control_reason(request.text, outcome.proposal.operation_id)
                if question is None and isinstance(outcome.proposal, PlanProposal) and (negated_plan or not plan_steps):
                    negative_reason = "explicit_negative_intent"
                if isinstance(outcome.proposal, ClarificationProposal) and only_negated_instructions(request.text):
                    # Asking for details of an operation the user just declined is pointless.
                    negative_reason = "explicit_negative_intent"
                if question is not None:
                    # Retain the original structured interpretation for audit;
                    # the application asks instead of executing a guessed ID.
                    response = response.model_copy(update={"status": "needs_input", "clarification": question})
                elif negative_reason is not None:
                    guard_code = "negative_intent"
                    response = response.model_copy(update={"status": "dismissed"})
                elif isinstance(outcome.proposal, PlanProposal):
                    prepared_steps = [OperationRequest(
                        operation_id=step.operation_id, operation_version=step.operation_version,
                        target=OperationTarget(project_id=response.project_id),
                        arguments=merged_arguments(db, response.project_id, step),
                        request_id=f"{response.core_request_id}-s{index}",
                        # Later steps are bound at execution to the previous step's revision.
                        base_revision=response.base_revision if index == 1 else None,
                        generation_requested=False,
                    ) for index, step in enumerate(plan_steps, 1)]
                    confirmation = repository.digest({"request_id": request.request_id,
                                                       "plan": [step.model_dump(mode="json") for step in prepared_steps]})
                    response = response.model_copy(update={
                        "status": "ready", "plan": prepared_steps, "confirmation_token": confirmation,
                        # A multi-step plan is always confirmed once as a whole, except unattended.
                        "requires_confirmation": self.review_all or not yolo,
                    })
                    db.rollback()
                    db.expire_all()
                    readiness = self.core.readiness(db, prepared_steps[0])
                    if readiness.readiness != Readiness.ready:
                        response = response.model_copy(update={"status": "blocked", "requires_confirmation": False,
                                                               "failure": FailureView(
                            reason_code=readiness.reason_code or "not_ready",
                            message="対象の状態が変わったか、この操作を現在実行できません。最新の状態を確認してください。")})
                elif not isinstance(outcome.proposal, OperationProposal):
                    clarification = outcome.proposal if isinstance(outcome.proposal, ClarificationProposal) else None
                    if yolo and outcome.status in {"needs_input", "unsupported"}:
                        unresolved = "自動実行モードでも、実行できる操作を推測できませんでした。"
                    if clarification and clarification.question.strip().rstrip("。.!?？") == request.text.strip().rstrip("。.!?？"):
                        clarification = clarification.model_copy(update={"question": "操作する対象を選択してください。" if clarification.missing_fields == ["target"]
                            else "変更する値や読み方など、不足している内容を指定してください。"})
                    response = response.model_copy(update={"status": outcome.status, "failure": outcome.failure,
                        "clarification": clarification})
                else:
                    proposal = outcome.proposal
                    prepared = OperationRequest(
                        operation_id=proposal.operation_id, operation_version=proposal.operation_version,
                        target=OperationTarget(project_id=response.project_id), arguments=merged_arguments(db, response.project_id, proposal),
                        request_id=response.core_request_id, base_revision=response.base_revision,
                        generation_requested=False,
                    )
                    confirmation = repository.digest({"request_id": request.request_id,
                                                       "prepared_request": prepared.model_dump(mode="json")})
                    response = response.model_copy(update={
                        "status": "ready", "prepared_request": prepared,
                        "generate_after_save": proposal.generate_after_save,
                        "confirmation_token": confirmation,
                        # The server-wide review setting still wins over an unattended request.
                        "requires_confirmation": self.review_all or (not yolo and (
                            request.review_all or load_policies().get(proposal.operation_id).requires_confirmation)),
                    })
                    # Expire read snapshots before asking the core about the latest state.
                    db.rollback()
                    db.expire_all()
                    readiness = self.core.readiness(db, prepared)
                    if readiness.readiness != Readiness.ready:
                        # Nothing can be confirmed: the user has to ask again once the state allows it.
                        response = response.model_copy(update={"status": "blocked", "requires_confirmation": False,
                                                               "failure": FailureView(
                            reason_code=readiness.reason_code or "not_ready",
                            message="対象の状態が変わったか、この操作を現在実行できません。最新の状態を確認してください。")})
        except asyncio.CancelledError:
            response = response.model_copy(update={"status": "error", "failure": FailureView(
                reason_code="interpretation_interrupted", message="解釈処理が中断されました。設定は変更していません。")})
            repository.finish_interpretation(db, request.request_id, owner, response)
            raise
        except OperationError as exc:
            response = response.model_copy(update={"status": "blocked", "requires_confirmation": False, "failure": FailureView(
                reason_code=exc.reason_code, message=str(exc))})
        except Exception:
            # A transport/plugin may violate the safe exception contract. Never
            # expose its exception string or raw request/response in public logs.
            response = response.model_copy(update={"status": "error", "failure": FailureView(
                reason_code="interpretation_failed", message="解釈処理に失敗しました。設定は変更していません。")})
        response = response.model_copy(update={"diagnostics": response.diagnostics.model_copy(update={
            "interpretation_ms": round((perf_counter() - started) * 1000), "guard_code": guard_code,
        })})
        if yolo:
            response = response.model_copy(update={"execution_mode": "yolo", "yolo_report": YoloReport(
                guessing_allowed=self.yolo_enabled, bypassed_guards=bypassed, dropped_steps=dropped,
                unresolved=unresolved)})
        response = repository.finish_interpretation(db, request.request_id, owner, response)
        observability.record(response, "prepared")
        return response

    async def submit(self, db: Session, request: LanguageInput) -> LanguageResponse:
        """Execute an unambiguous non-confirming proposal using the common core.

        An unattended (YOLO) request confirms on the user's behalf, including the
        follow-up generation after a settings save, and reports what it confirmed.
        """
        response = await self.prepare(db, request)
        if response.status != "ready" or response.requires_confirmation or response.result is not None:
            return response
        yolo = response.execution_mode == "yolo"
        confirmed = ([response.prepared_request.operation_id] if response.prepared_request
                     else [step.operation_id for step in response.plan or []])
        response = await asyncio.to_thread(self.execute, db, request.request_id, LanguageExecution(
            confirmation_token=response.confirmation_token, confirm_generation=yolo,
        ))
        if not yolo:
            return response
        if (response.status == "ready" and response.generation_request is not None
                and response.generation_result is None and response.confirmation_token is not None):
            confirmed.append(response.generation_request.operation_id)
            response = await asyncio.to_thread(self.execute, db, request.request_id, LanguageExecution(
                confirmation_token=response.confirmation_token, confirm_generation=True,
            ))
        report = (response.yolo_report or YoloReport()).model_copy(update={"auto_confirmed": confirmed})
        return repository.acknowledge(db, request.request_id, response.model_copy(update={"yolo_report": report}))

    def get(self, db: Session, request_id: str) -> LanguageResponse:
        return repository.lookup(db, request_id)

    def execute(self, db: Session, request_id: str, confirmation: LanguageExecution) -> LanguageResponse:
        response = repository.lookup(db, request_id)
        if response.plan:
            return self._execute_plan(db, request_id, response, confirmation)
        if response.result is not None and (not response.generate_after_save or response.generation_result is not None):
            return response
        if response.result is not None and response.generation_request is not None:
            save_token = repository.digest({"request_id": request_id,
                "prepared_request": response.prepared_request.model_dump(mode="json")})
            if hmac.compare_digest(confirmation.confirmation_token, save_token):
                # Replaying the save confirmation must never confirm the next phase.
                return response
        if response.status != "ready" or response.prepared_request is None or response.confirmation_token is None:
            raise LanguageError("request_not_ready", "この要求は実行できる状態ではありません。")
        if not hmac.compare_digest(confirmation.confirmation_token, response.confirmation_token):
            raise LanguageError("confirmation_mismatch", "確認内容が保存済みの提案と一致しません。")
        prepared = response.generation_request or response.prepared_request
        if load_policies().get(prepared.operation_id).requires_confirmation and not confirmation.confirm_generation:
            raise LanguageError("generation_confirmation_required", "動画生成の対象と内容を確認してから開始してください。")
        started = perf_counter()
        try:
            result = self.core.execute(db, prepared,
                                       before_dispatch=lambda session: dialogue.require_current(session, request_id))
            response = response.model_copy(update={"status": "completed",
                                                   "generation_result" if response.generation_request else "result": result,
                                                   "executed": True, "failure": None,
                                                   "requires_confirmation": False})
        except OperationError as exc:
            response = response.model_copy(update={"status": "blocked", "failure": FailureView(
                reason_code=exc.reason_code, message=str(exc))})
        response = response.model_copy(update={"diagnostics": response.diagnostics.model_copy(update={
            "generation_execution_ms" if response.generation_request else "execution_ms": round((perf_counter() - started) * 1000),
        })})
        response = repository.acknowledge(db, request_id, response)
        observability.record(response, "executed")
        return response

    def _execute_plan(self, db: Session, request_id: str, response: LanguageResponse,
                      confirmation: LanguageExecution) -> LanguageResponse:
        """Run plan steps in order through the common core; stop at the first failure.

        Each step has its own core request ID, so a replayed confirmation resumes after
        the last committed step instead of repeating it.
        """
        plan = response.plan or []
        if response.status == "completed" or len(response.plan_results) == len(plan):
            return response
        if response.status != "ready" or response.confirmation_token is None:
            raise LanguageError("request_not_ready", "この要求は実行できる状態ではありません。")
        if not hmac.compare_digest(confirmation.confirmation_token, response.confirmation_token):
            raise LanguageError("confirmation_mismatch", "確認内容が保存済みの提案と一致しません。")
        if (any(load_policies().get(step.operation_id).requires_confirmation for step in plan)
                and not confirmation.confirm_generation):
            raise LanguageError("generation_confirmation_required", "動画生成の対象と内容を確認してから開始してください。")
        started = perf_counter()
        results = list(response.plan_results)
        base = results[-1].revision if results else response.base_revision
        failure: FailureView | None = None
        for index, step in enumerate(plan):
            if index < len(results):
                continue
            try:
                result = self.core.execute(db, step.model_copy(update={"base_revision": base}),
                                           before_dispatch=lambda session: dialogue.require_current(session, request_id))
            except OperationError as exc:
                failure = FailureView(reason_code=exc.reason_code,
                                      message=f"手順{index + 1}で停止しました。残りの手順は実行していません。{exc}")
                break
            results.append(result)
            base = result.revision
        response = response.model_copy(update={
            "plan_results": results, "executed": bool(results), "requires_confirmation": False,
            **({"status": "blocked", "failure": failure} if failure else {"status": "completed", "failure": None}),
            "diagnostics": response.diagnostics.model_copy(update={
                "execution_ms": round((perf_counter() - started) * 1000)}),
        })
        response = repository.acknowledge(db, request_id, response)
        observability.record(response, "executed")
        return response
