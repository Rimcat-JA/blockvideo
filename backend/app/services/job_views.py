"""Public job metadata, with safe retry guidance and unambiguous UTC timestamps."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.schemas import (
    JobSummary,
    ProjectGenerationRecovery,
    RecommendedAction,
    RecoveryCode,
)
from app.services.external_calls import unresolved_remote_side_effect_predicate
from app.services.generation_plan import STAGE_ORDER


def utc_timestamp(value: datetime | None) -> str | None:
    """SQLite drops timezone information; persisted application clocks are UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


_EXTERNAL_UNKNOWN_MESSAGE = (
    "外部処理の結果が未確定です。このアプリでは結果を照会できないため、"
    "外部サービス側の履歴を確認してください。"
)
_PROJECT_BLOCKED_MESSAGE = (
    "以前の外部処理の結果が未確定です。"
    "外部サービス側の履歴を確認できるまで再実行できません。"
)
_REFRESH_MESSAGE = "現在の状態を再取得してから再実行してください。"


@dataclass(frozen=True)
class RecoveryContext:
    """Project-wide durable facts required to authorize retry guidance."""

    project_id: int
    has_active_job: bool
    has_unknown_job: bool
    has_unresolved_remote_side_effect: bool

    @property
    def blocks_retry(self) -> bool:
        return self.has_unknown_job or self.has_unresolved_remote_side_effect


def build_recovery_contexts(
    db: Session, project_ids: Iterable[int]
) -> dict[int, RecoveryContext]:
    """Load project-wide retry blockers in one aggregate query."""
    ids = sorted(set(project_ids))
    if not ids:
        return {}
    rows = db.execute(
        select(
            GenerationJob.project_id,
            func.max(
                case(
                    (GenerationJob.status.in_([JobStatus.pending, JobStatus.running]), 1),
                    else_=0,
                )
            ),
            func.max(case((GenerationJob.status == JobStatus.unknown, 1), else_=0)),
            func.max(case((unresolved_remote_side_effect_predicate(), 1), else_=0)),
        )
        .outerjoin(ExternalCall, ExternalCall.job_id == GenerationJob.id)
        .where(GenerationJob.project_id.in_(ids))
        .group_by(GenerationJob.project_id)
    )
    contexts = {
        project_id: RecoveryContext(
            project_id=project_id,
            has_active_job=False,
            has_unknown_job=False,
            has_unresolved_remote_side_effect=False,
        )
        for project_id in ids
    }
    contexts.update({
        project_id: RecoveryContext(
            project_id=project_id,
            has_active_job=bool(has_active_job),
            has_unknown_job=bool(has_unknown_job),
            has_unresolved_remote_side_effect=bool(has_unresolved_call),
        )
        for project_id, has_active_job, has_unknown_job, has_unresolved_call in rows
    })
    return contexts


def project_generation_recovery(context: RecoveryContext) -> ProjectGenerationRecovery:
    """Map all persisted project blockers to a bounded public generation contract."""
    if context.has_active_job:
        return ProjectGenerationRecovery(code="busy", recommended_action="wait")
    if context.blocks_retry:
        return ProjectGenerationRecovery(
            code="external_outcome_unknown",
            recommended_action="check_provider",
        )
    return ProjectGenerationRecovery(code="ready", recommended_action="generate")


def _recovery_action(
    job: GenerationJob, context: RecoveryContext | None
) -> tuple[RecoveryCode, RecommendedAction, bool, str | None]:
    if job.status in {JobStatus.pending, JobStatus.running}:
        return "wait", "wait", False, None
    if job.status == JobStatus.unknown:
        return "external_outcome_unknown", "check_provider", False, _EXTERNAL_UNKNOWN_MESSAGE
    if job.status == JobStatus.completed:
        return "completed", "none", False, None
    if job.status in {JobStatus.failed, JobStatus.cancelled}:
        if context is None or context.project_id != job.project_id:
            return "refresh_required", "refresh", False, _REFRESH_MESSAGE
        if context.has_active_job:
            return "wait", "wait", False, None
        if context.blocks_retry:
            return "external_outcome_unknown", "check_provider", False, _PROJECT_BLOCKED_MESSAGE
        return "safe_retry", "retry_current", True, None
    return "failed", "none", False, None


def job_summary(
    job: GenerationJob, recovery_context: RecoveryContext | None = None
) -> JobSummary:
    """Expose control metadata without snapshots, provider responses or secrets."""
    recovery_code, recommended_action, retryable, blocked_reason = _recovery_action(
        job, recovery_context
    )
    raw_stages = (job.plan_json or {}).get("stages", [])
    stages = raw_stages if isinstance(raw_stages, list) else []
    plan = {"stages": [stage for stage in STAGE_ORDER if stage in stages]} if job.plan_json else None
    return JobSummary(
        id=job.id, project_id=job.project_id, current_stage=job.current_stage,
        status=job.status.value, progress=job.progress, stage_progress=job.stage_progress,
        started_at=utc_timestamp(job.started_at), finished_at=utc_timestamp(job.finished_at),
        error_message=job.error_message, cancel_requested=job.cancel_requested,
        input_revision=job.input_revision, parent_job_id=job.parent_job_id,
        recovery_message=job.recovery_message,
        retryable=retryable, retry_blocked_reason=blocked_reason,
        recovery_code=recovery_code, recommended_action=recommended_action, plan=plan,
    )
