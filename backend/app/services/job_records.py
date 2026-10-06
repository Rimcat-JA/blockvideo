"""Shared durable job identity and project-busy checks, independent of transport."""
from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.job import GenerationJob, JobStatus
from app.models.operation_request import OperationReceipt
from app.models.project import Project, ProjectStatus
from app.models.external_call import ExternalCall
from app.models.artifact import GenerationArtifact
from app.services.external_calls import unresolved_remote_side_effect_predicate
from app.services.generation_plan import build_generation_plan
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.services.settings_history import record_settings


class ProjectBusyError(RuntimeError):
    """A generation request raced with another job for the same project."""


class UnresolvedExternalWorkError(RuntimeError):
    """An earlier potentially billed request must be resolved before new generation."""


def has_active_project_job(db: Session, project_id: int, is_running: Callable[[int], bool]) -> bool:
    """Durable intents remain busy even before dispatch or across a restart."""
    rows = db.execute(
        select(GenerationJob.id, OperationReceipt.request_id)
        .outerjoin(OperationReceipt, OperationReceipt.job_id == GenerationJob.id)
        .where(GenerationJob.project_id == project_id,
               GenerationJob.status.in_([JobStatus.pending, JobStatus.running]))
    )
    return any(receipt_id is not None or db.get(GenerationJob, job_id).input_snapshot is not None
               or is_running(job_id) for job_id, receipt_id in rows)


def create_pending_job(db: Session, project_id: int, stage: str = "queued", *, kind: str | None = None,
                       block_index: int | None = None, parent_job_id: int | None = None) -> GenerationJob:
    """Allocate under the caller's writer lock; historical references reserve IDs.

    SQLite can reuse deleted integer primary keys. Receipts outlive project/job
    deletion, so neither legacy nor durable enqueue may reuse a referenced job ID.
    """
    active = db.scalar(select(GenerationJob.id).where(
        GenerationJob.project_id == project_id,
        GenerationJob.status.in_([JobStatus.pending, JobStatus.running]),
    ).limit(1))
    if active:
        raise ProjectBusyError("このプロジェクトは生成中です。完了後に再実行してください。")
    uncertain = db.scalar(select(GenerationJob.id).where(
        GenerationJob.project_id == project_id, GenerationJob.status == JobStatus.unknown).limit(1))
    unresolved_call = db.scalar(select(ExternalCall.id).join(GenerationJob, GenerationJob.id == ExternalCall.job_id)
                               .where(GenerationJob.project_id == project_id,
                                      unresolved_remote_side_effect_predicate()).limit(1))
    if uncertain or unresolved_call:
        raise UnresolvedExternalWorkError("以前の外部処理の結果が未確定です。確認できるまで生成を再送できません")
    project = db.get(Project, project_id)
    if project is None:
        raise ValueError("project not found")
    record_settings(db, project)
    if kind is None:
        kind = stage.split(":", 1)[0] if stage.startswith("block_") else ("rerender" if stage == "rerender" else "full")
        if stage.startswith("block_"):
            block_index = int(stage.split(":", 1)[1])
    snapshot = capture_inputs(project)
    plan = build_generation_plan(project, kind=kind, block_index=block_index)
    current_max = db.scalar(select(func.max(GenerationJob.id))) or 0
    historical_max = db.scalar(select(func.max(OperationReceipt.job_id))) or 0
    external_max = db.scalar(select(func.max(ExternalCall.job_id))) or 0
    artifact_max = db.scalar(select(func.max(GenerationArtifact.job_id))) or 0
    job = GenerationJob(id=max(current_max, historical_max, external_max, artifact_max) + 1, project_id=project_id,
                        current_stage=stage, status=JobStatus.pending,
                        kind=kind, block_index=block_index, input_revision=project.revision,
                        input_snapshot=snapshot, input_fingerprint=fingerprint_inputs(snapshot),
                        plan_json=plan, parent_job_id=parent_job_id)
    project.status = ProjectStatus.generating
    project.current_stage = "queued"
    project.progress = 0.0
    project.error_message = None
    db.add(job)
    db.flush()
    return job
