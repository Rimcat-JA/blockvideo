"""Disposable file databases for synthetic development trials, never production DB."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.db import Base
from app.interpretation.contracts import ClarificationProposal, InterpretationOutcome
from app.language_operations.contracts import LanguageResponse
from app.models import block as _block  # noqa: F401
from app.models.artifact import GenerationArtifact
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.language_request import LanguageRequestRecord
from app.models.language_turn import LanguageTurn
from app.models.operation_request import OperationReceipt
from app.models.project import Project, ProjectStatus
from app.models.settings_revision import SettingsRevision
from app.operations.contracts import OperationRequest, OperationResult, OperationTarget
from app.schemas import ProjectCreate
from app.services.generation_snapshots import capture_inputs, fingerprint_inputs
from app.services.settings_history import configuration
from evaluation.contracts import Case


def _project(project_id: int, revision: int, status: str, settings: dict[str, Any]) -> Project:
    values = ProjectCreate(title="D29 synthetic", source_script="比較試験用の合成原稿です。",
                           use_fake_providers=True, **settings).model_dump(exclude={"providers"})
    return Project(id=project_id, revision=revision, status=ProjectStatus(status), **values)


def seed(db: Session, case: Case) -> None:
    """Only initial fixture data is read here; expected answers never seed state."""
    initial = case.initial
    project = _project(initial.project_id, initial.revision, initial.project_status, initial.settings)
    db.add(project)
    other_ids = {j["project_id"] for j in initial.jobs} - {project.id}
    for other in other_ids:
        job = next(j for j in initial.jobs if j["project_id"] == other)
        db.add(_project(other, job["input_revision"], "generating" if job["status"] in {"pending", "running"} else "failed",
                        job.get("input_settings", {})))
    db.flush()
    defaults = configuration(project)
    for version in initial.history:
        db.add(SettingsRevision(project_id=project.id, revision=version["revision"],
            settings_json={**defaults, **version["settings"]}, changed_fields=[]))
    for index, revision in enumerate(initial.artifact_revisions, 1):
        # Manifest-only history stands in for bytes; bulk trials never play media.
        path = f"synthetic-history/{index}.mp4"
        db.add(GenerationArtifact(id=index, project_id=project.id, revision=revision,
                                 video_path=path, manifest_json={"synthetic_placeholder": True}))
        project.current_artifact_id, project.output_video_path = index, path
    for item in initial.jobs:
        owner = db.get(Project, item["project_id"])
        if item["status"] in {"pending", "running"} and item["input_revision"] != owner.revision:
            raise ValueError("unreachable active-job fixture")
        snapshot = capture_inputs(owner)
        snapshot["project"].update(item.get("input_settings", {}))
        db.add(GenerationJob(id=item["id"], project_id=owner.id, status=JobStatus(item["status"]),
            kind=item.get("kind", "full"), cancel_requested=item["cancel_requested"],
            input_revision=item["input_revision"], input_snapshot=snapshot,
            input_fingerprint=fingerprint_inputs(snapshot)))
    parent = None
    for turn in initial.prior_turns:
        proposal = turn.get("proposal")
        outcome = InterpretationOutcome.model_validate({"status": "proposed" if proposal and proposal["kind"] == "operation" else "needs_input",
                                                        "proposal": proposal})
        response = LanguageResponse(request_id=turn["request_id"], core_request_id="seed-" + turn["request_id"],
            project_id=turn["project_id"], base_revision=turn["base_revision"], status=turn["status"], interpretation=outcome,
            clarification=ClarificationProposal(kind="clarification", question=turn["question"], missing_fields=["arguments"])
                if turn.get("question") else None, dialogue_available=True)
        if turn.get("result_revision"):
            response = response.model_copy(update={"result": OperationResult(operation_id=proposal["operation_id"],
                project_id=turn["project_id"], changed=True, state_revision=str(turn["result_revision"]),
                revision=turn["result_revision"], data={}), "executed": True})
        if proposal and proposal["kind"] == "operation":
            response = response.model_copy(update={"prepared_request": OperationRequest(operation_id=proposal["operation_id"],
                operation_version=proposal["operation_version"], target=OperationTarget(project_id=turn["project_id"]),
                arguments=proposal["arguments"], base_revision=turn["base_revision"], request_id=response.core_request_id)})
        db.add(LanguageRequestRecord(request_id=response.request_id, core_request_id=response.core_request_id,
            input_fingerprint="0" * 64, project_id=response.project_id, base_revision=response.base_revision,
            status=response.status, owner_token="seed", lease_until=0, created_at=time.time(),
            response_json=response.model_dump(mode="json")))
        db.add(LanguageTurn(request_id=response.request_id, text=turn["text"], parent_request_id=parent,
                            relation=turn.get("relation")))
        parent = response.request_id
    db.commit()


class TrialDatabase:
    def __init__(self, directory: Path, case: Case) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "media").mkdir()
        self.path = directory / "trial.db"
        self.engine = create_engine(f"sqlite:///{self.path.as_posix()}", connect_args={"check_same_thread": False})
        self.sessions = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=True)
        Base.metadata.create_all(self.engine)
        try:
            with self.sessions() as db:
                seed(db, case)
        except BaseException:
            self.engine.dispose()
            raise

    def close(self) -> None:
        self.engine.dispose()


def observe(db: Session, project_id: int) -> dict[str, Any]:
    db.rollback()
    db.expire_all()
    project = db.get(Project, project_id)
    return {"project_id": project.id, "revision": project.revision, "status": project.status.value,
        "settings": configuration(project),
        "jobs": [{"id": j.id, "project_id": j.project_id, "status": j.status.value,
                  "input_revision": j.input_revision, "cancel_requested": j.cancel_requested,
                  "parent_job_id": j.parent_job_id} for j in db.scalars(select(GenerationJob).order_by(GenerationJob.id))],
        "history": [{"revision": r.revision, "settings": r.settings_json} for r in db.scalars(
            select(SettingsRevision).where(SettingsRevision.project_id == project_id).order_by(SettingsRevision.revision))],
        "artifacts": [{"id": a.id, "revision": a.revision, "video_path": a.video_path} for a in db.scalars(
            select(GenerationArtifact).where(GenerationArtifact.project_id == project_id).order_by(GenerationArtifact.id))],
        "receipts": sorted(db.scalars(select(OperationReceipt.request_id)).all()),
        "external_calls": [{
            "id": call.id,
            "job_id": call.job_id,
            "fingerprint": call.fingerprint,
            "provider": call.provider,
            "endpoint": call.endpoint,
            "remote_side_effect": call.remote_side_effect,
            "status": call.status,
            "attempts": call.attempts,
            "response_status": call.response_status,
            "response_body_sha256": (
                hashlib.sha256(call.response_body).hexdigest() if call.response_body is not None else None
            ),
            "response_content_type": call.response_content_type,
            "provider_response_id": call.provider_response_id,
            "error_code": call.error_code,
            "started_at": call.started_at.isoformat(),
            "finished_at": call.finished_at.isoformat() if call.finished_at is not None else None,
        } for call in db.scalars(select(ExternalCall).order_by(ExternalCall.id))]}


def prompt_state(state: dict[str, Any], case: Case) -> dict[str, Any]:
    """Available same-target facts only, with no labels, title, script or media paths."""
    if case.request.target_project_id != state["project_id"]:
        return {}
    return {"project_id": state["project_id"], "revision": state["revision"], "status": state["status"],
        "settings": {key: state["settings"][key] for key in case.initial.settings},
        "jobs": [j for j in state["jobs"] if j["project_id"] == state["project_id"]],
        "saved_revisions": [r["revision"] for r in state["history"]],
        "artifact_revisions": [a["revision"] for a in state["artifacts"]]}
