"""D11 acceptance: durable identity, absolute resolution and atomic writes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db import get_session_factory
from app.main import create_app
from app.models.job import GenerationJob
from app.models.operation_request import OperationReceipt
from app.models.project import Project
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import OperationRequest, OperationResult
from app.operations.service import OperationError


def make_project() -> int:
    with get_session_factory()() as db:
        project = Project(title="D11 synthetic", source_script="合成テストです。",
                          subtitle_font_size=48, use_fake_providers=True)
        db.add(project)
        db.commit()
        return project.id


def adjustment(project_id: int, request_id: str = "adjust-1", revision: int = 1,
               delta: int = 2, generate: bool = False) -> OperationRequest:
    return OperationRequest(
        request_id=request_id, base_revision=revision,
        operation_id="project.subtitle-font-size.adjust",
        target={"project_id": project_id}, arguments={"delta": delta},
        generation_requested=generate,
    )


def execute(request: OperationRequest) -> OperationResult:
    with get_session_factory()() as db:
        return build_operation_service().execute(db, request)


def test_replay_returns_original_result_before_relative_resolution(temp_storage) -> None:
    project_id = make_project()
    request = adjustment(project_id)
    first = execute(request)
    execute(adjustment(project_id, "another-request", revision=2))
    replay = execute(request)
    assert replay.model_dump() == first.model_dump()
    assert first.resolved_arguments == {"value": 50}
    assert first.revision == 2
    with get_session_factory()() as db:
        assert db.get(Project, project_id).subtitle_font_size == 52
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 2


@pytest.mark.parametrize("changes", [
    {"arguments": {"delta": 4}}, {"base_revision": 2},
    {"generation_requested": True}, {"target": {"project_id": 999}},
    {"operation_id": "project.status.get", "arguments": {}},
])
def test_same_id_different_content_is_rejected(temp_storage, changes: dict) -> None:
    request = adjustment(make_project())
    execute(request)
    changed = OperationRequest.model_validate({**request.model_dump(), **changes})
    with pytest.raises(OperationError) as error:
        execute(changed)
    assert error.value.reason_code == "request_id_conflict"


def test_target_alias_and_default_fields_do_not_change_identity(temp_storage) -> None:
    project_id = make_project()
    request = adjustment(project_id)
    first = execute(request)
    equivalent = OperationRequest.model_validate({
        **request.model_dump(), "target": {"selected_project_id": project_id},
    })
    assert execute(equivalent) == first


def test_distinct_requests_with_same_base_revision_conflict(temp_storage) -> None:
    project_id = make_project()
    execute(adjustment(project_id))
    with pytest.raises(OperationError) as error:
        execute(adjustment(project_id, "new-but-stale"))
    assert error.value.reason_code == "stale_state"
    with get_session_factory()() as db:
        assert db.get(Project, project_id).subtitle_font_size == 50
        assert db.get(OperationReceipt, "new-but-stale") is None


def test_same_id_concurrent_delivery_applies_one_increment(temp_storage) -> None:
    project_id = make_project()
    request = adjustment(project_id, generate=True)
    barrier = Barrier(6)

    def send():
        barrier.wait(timeout=10)
        return execute(request).model_dump()

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: send(), range(6)))
    assert all(result == results[0] for result in results)
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (50, 2)
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 1
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 1


def test_rollback_covers_setting_revision_receipt_and_job(temp_storage, monkeypatch) -> None:
    from app.models.block import Block, BlockStatus

    project_id = make_project()
    with get_session_factory()() as db:
        db.add(Block(project_id=project_id, index=0, source_text="before", tts_text="before",
                     status_render=BlockStatus.completed))
        db.commit()
    with get_session_factory()() as db:
        def fail_commit() -> None:
            db.flush()
            raise RuntimeError("simulated commit failure")
        monkeypatch.setattr(db, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="simulated"):
            build_operation_service().execute(db, adjustment(project_id, generate=True))
    with get_session_factory()() as db:
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == (48, 1)
        assert project.blocks[0].status_render == BlockStatus.completed
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 0
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 0


def test_legacy_patch_advances_revision_and_rejects_stale_request(temp_storage) -> None:
    project_id = make_project()
    client = TestClient(create_app())
    response = client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 60})
    assert response.status_code == 200
    assert response.json()["revision"] == 2
    assert client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 60}).json()["revision"] == 2
    response = client.post("/api/operations/execute", json=adjustment(project_id).model_dump())
    assert response.status_code == 409
    assert response.json()["detail"]["reason_code"] == "stale_state"


def test_pending_generation_is_durable_and_blocks_edits(temp_storage) -> None:
    project_id = make_project()
    request = adjustment(project_id, generate=True)
    result = execute(request)
    assert result.job_id is not None
    assert execute(request) == result
    client = TestClient(create_app())
    assert client.get(result.result_ref).json() == result.model_dump(mode="json")
    assert client.patch(f"/api/projects/{project_id}", json={"subtitle_font_size": 60}).status_code == 409
    assert client.post(f"/api/projects/{project_id}/generate-all").status_code == 409
    with get_session_factory()() as db:
        receipt = db.get(OperationReceipt, request.request_id)
        assert receipt.base_revision == 1
        assert receipt.result_revision == 2
        assert receipt.resolved_arguments == {"value": 50}
        assert receipt.generation_requested is True
        assert receipt.job_id == result.job_id


@pytest.mark.parametrize("delta", [True, "2", 2.0, 105, -105, 73, -33])
def test_invalid_relative_value_never_reserves_request(temp_storage, delta) -> None:
    request = adjustment(make_project()).model_copy(update={"arguments": {"delta": delta}})
    with pytest.raises(OperationError) as error:
        execute(request)
    assert error.value.reason_code == "invalid_arguments"
    with get_session_factory()() as db:
        assert db.get(Project, request.target.project_id).subtitle_font_size == 48
        assert db.get(OperationReceipt, request.request_id) is None


@pytest.mark.parametrize("missing", ["request_id", "base_revision"])
def test_relative_request_requires_identity_and_revision(temp_storage, missing: str) -> None:
    request = adjustment(make_project()).model_copy(update={missing: None})
    with pytest.raises(OperationError) as error:
        execute(request)
    assert error.value.reason_code == "request_metadata_required"
    assert missing in error.value.result.missing_fields


def test_status_with_id_is_an_immutable_snapshot(temp_storage) -> None:
    project_id = make_project()
    request = OperationRequest(request_id="snapshot", operation_id="project.status.get",
                               target={"project_id": project_id})
    snapshot = execute(request)
    execute(adjustment(project_id))
    assert execute(request) == snapshot
    assert snapshot.revision == snapshot.base_revision == 1
    with get_session_factory()() as db:
        assert db.get(Project, project_id).revision == 2


def test_noop_can_explicitly_request_one_generation(temp_storage) -> None:
    request = adjustment(make_project(), delta=0, generate=True)
    result = execute(request)
    assert result.changed is False
    assert result.revision == 1
    assert result.job_id is not None
    assert execute(request) == result


def test_readiness_does_not_reserve_identity(temp_storage) -> None:
    request = adjustment(make_project())
    with get_session_factory()() as db:
        result = build_operation_service().readiness(db, request)
        assert result.revision == 1
        assert db.get(OperationReceipt, request.request_id) is None
    changed = request.model_copy(update={"arguments": {"delta": 4}})
    assert execute(changed).resolved_arguments == {"value": 52}


def test_database_itself_rejects_duplicate_request_id(temp_storage) -> None:
    request = adjustment(make_project())
    execute(request)
    with get_session_factory()() as db:
        row = db.get(OperationReceipt, request.request_id)
        values = {column.name: getattr(row, column.name) for column in OperationReceipt.__table__.columns}
        with pytest.raises(IntegrityError):
            db.execute(OperationReceipt.__table__.insert().values(**values))
        db.rollback()
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 1


def test_replay_after_target_deletion_still_returns_original(temp_storage) -> None:
    request = adjustment(make_project())
    result = execute(request)
    client = TestClient(create_app())
    assert client.delete(f"/api/projects/{result.project_id}").status_code == 204
    assert execute(request) == result
    assert client.get(result.result_ref).json() == result.model_dump(mode="json")
    assert client.get("/api/operations/requests/unknown").status_code == 404


def test_block_patch_advances_revision_only_for_changes(temp_storage) -> None:
    from app.models.block import Block

    project_id = make_project()
    with get_session_factory()() as db:
        block = Block(project_id=project_id, index=0, source_text="before", tts_text="before")
        db.add(block)
        db.commit()
        block_id = block.id
    client = TestClient(create_app())
    for _ in range(2):
        assert client.patch(f"/api/blocks/{block_id}", json={"tts_text": "after"}).status_code == 200
        assert client.get(f"/api/projects/{project_id}").json()["revision"] == 2
    with pytest.raises(OperationError, match="blocked"):
        execute(adjustment(project_id))
