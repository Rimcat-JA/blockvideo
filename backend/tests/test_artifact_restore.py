"""A catalog-only operation addition: make an earlier verified video the current one."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.db import get_session_factory
from app.language_operations.intent_guard import negative_control_reason
from app.language_operations.references import reference_question
from app.interpretation.contracts import OperationProposal
from app.main import create_app
from app.models.artifact import GenerationArtifact
from app.models.project import Project
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import OperationRequest
from app.services import artifact_store as store
from app.services.generation_snapshots import capture_inputs
from tests.test_artifact_history import accept_synthetic_probe, candidate_for, make_job  # noqa: F401

RESTORE = "project.artifact.restore"


def _request(project_id: int, artifact_id: int) -> OperationRequest:
    with get_session_factory()() as db:
        revision = db.get(Project, project_id).revision
    return OperationRequest(operation_id=RESTORE, target={"project_id": project_id},
                            arguments={"artifact_id": artifact_id}, base_revision=revision,
                            request_id=str(uuid4()))


async def _two_artifacts() -> tuple[int, GenerationArtifact, GenerationArtifact]:
    with get_session_factory()() as db:
        project, job = make_job(db)
        project_id, first_id = project.id, job.id
        snapshot = capture_inputs(project)
    first = await store.publish_artifact(first_id, candidate_for(project_id, first_id), None,
                                         settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    with get_session_factory()() as db:
        _, job = make_job(db, db.get(Project, project_id))
        second_id = job.id
    second = await store.publish_artifact(second_id, candidate_for(project_id, second_id, b"second"), None,
                                          settled_inputs=snapshot, materials=[], cancel_check=lambda: False)
    return project_id, first, second


@pytest.mark.asyncio
async def test_restore_points_current_video_at_an_earlier_artifact(
    temp_storage, accept_synthetic_probe,  # noqa: F811
) -> None:
    project_id, first, second = await _two_artifacts()
    with get_session_factory()() as db:
        revision = db.get(Project, project_id).revision
    request = _request(project_id, first.id)
    with get_session_factory()() as db:
        result = build_operation_service().execute(db, request)
    assert result.changed and result.revision == revision and result.job_id is None
    assert result.data["artifact_id"] == first.id
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert project.current_artifact_id == first.id
        assert project.output_video_path == first.video_path
        assert project.revision == revision
        # Both videos stay in the immutable history.
        assert db.get(GenerationArtifact, second.id) is not None
    # A replay of the same request returns the stored receipt without re-running.
    with get_session_factory()() as db:
        assert build_operation_service().execute(db, request) == result


@pytest.mark.asyncio
async def test_restore_rejects_foreign_missing_or_unverified_artifacts(
    temp_storage, accept_synthetic_probe,  # noqa: F811
) -> None:
    project_id, first, _ = await _two_artifacts()
    client = TestClient(create_app())
    response = client.post("/api/operations/execute", json=_request(project_id, 999).model_dump(mode="json"))
    assert response.status_code == 404
    assert response.json()["detail"]["reason_code"] == "artifact_not_found"
    with get_session_factory()() as db:
        # A corrupt subtitle alone also makes the video unrestorable.
        row = db.get(GenerationArtifact, first.id)
        subtitle = Path(store.artifact_file_path(row).parent / "video.srt")
        subtitle.write_text("1", encoding="utf-8")
        row.subtitle_path = row.video_path.replace("video.mp4", "video.srt")
        row.manifest_json = {**row.manifest_json, "subtitle": {**store.file_identity(subtitle), "size": 999}}
        db.commit()
    response = client.post("/api/operations/execute", json=_request(project_id, first.id).model_dump(mode="json"))
    assert response.json()["detail"]["reason_code"] == "artifact_unavailable"
    with get_session_factory()() as db:
        row = db.get(GenerationArtifact, first.id)
        row.subtitle_path = None
        db.commit()
    Path(store.artifact_file_path(first)).write_bytes(b"tampered")
    response = client.post("/api/operations/execute", json=_request(project_id, first.id).model_dump(mode="json"))
    assert response.status_code == 422
    assert response.json()["detail"]["reason_code"] == "artifact_unavailable"


def test_restore_reference_and_negation_come_from_the_catalog() -> None:
    proposal = OperationProposal(kind="operation", operation_id=RESTORE, operation_version=1,
                                 arguments={"artifact_id": 3})
    assert reference_question("動画3に戻して", proposal) is None
    assert reference_question("前の動画に戻して", proposal) is not None
    assert reference_question("動画2に戻して", proposal) is not None
    assert negative_control_reason("動画3には戻さないで", RESTORE) == "explicit_negative_intent"
    assert negative_control_reason("動画3に戻して", RESTORE) is None
