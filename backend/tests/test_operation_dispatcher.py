"""D11 outbox identity with D14 classified recovery and D15 durable cancellation."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.main import create_app
from app.models.external_call import ExternalCall
from app.models.job import GenerationJob, JobStatus
from app.models.project import Project, ProjectStatus
from app.services.generation_snapshots import GenerationCancelled
from app.workers import operation_dispatcher as dispatcher
from app.workers.job_runner import JobRegistry
from tests.test_operation_durability import adjustment, execute, make_project


@pytest.mark.asyncio
async def test_two_registries_claim_one_persisted_job(temp_storage) -> None:
    result = execute(adjustment(make_project(), generate=True))
    registries = [JobRegistry(), JobRegistry()]
    calls = []

    async def work(cancel_check):
        calls.append(result.job_id)
        await asyncio.sleep(0)

    tasks = [registry.submit(result.job_id, work) for registry in registries]
    assert registries[0].submit(result.job_id, work) is tasks[0]
    await asyncio.gather(*tasks)
    assert calls == [result.job_id]
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).status == JobStatus.completed


def test_lifespan_recovers_committed_pending_job(temp_storage, monkeypatch) -> None:
    from tests.test_operation_processes import run_process

    request = adjustment(make_project(), generate=True)
    crashed = run_process(request.model_dump(mode="json"), crash="after")
    assert crashed.returncode == 92, crashed.stderr
    original = execute(request)
    called = Event()
    seen = []

    async def work(job_id, cancel_check):
        with get_session_factory()() as db:
            job = db.get(GenerationJob, job_id)
            assert job.id == original.job_id
            project = db.get(Project, job.project_id)
            seen.append((project.subtitle_font_size, project.revision))
        called.set()

    monkeypatch.setattr(dispatcher, "run_generation_job", work)
    monkeypatch.setattr(dispatcher, "job_registry", JobRegistry())
    with TestClient(create_app()) as client:
        assert called.wait(timeout=10)
        assert client.post("/api/operations/execute", json=request.model_dump()).json() == original.model_dump(mode="json")
        jobs = client.get(f"/api/projects/{original.project_id}/jobs").json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "completed"
    assert seen == [(50, 2)]


@pytest.mark.asyncio
async def test_shutdown_closes_admission_before_draining_accepted_tasks(temp_storage) -> None:
    accepted = execute(adjustment(make_project(), "accepted-before-shutdown", generate=True))
    rejected = execute(adjustment(make_project(), "rejected-after-shutdown", generate=True))
    registry = JobRegistry()
    work_started = asyncio.Event()
    cancellation_started = asyncio.Event()
    allow_cancellation = asyncio.Event()
    rejected_factory_called = False

    async def accepted_work(_cancel_check):
        work_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_started.set()
            await allow_cancellation.wait()
            raise

    async def rejected_work(_cancel_check):
        nonlocal rejected_factory_called
        rejected_factory_called = True

    accepted_task = registry.submit(accepted.job_id, accepted_work)
    await asyncio.wait_for(work_started.wait(), timeout=5)
    shutdown_task = asyncio.create_task(registry.shutdown())
    await asyncio.wait_for(cancellation_started.wait(), timeout=5)

    rejected_task = None
    try:
        with pytest.raises(RuntimeError, match="closing"):
            rejected_task = registry.submit(rejected.job_id, rejected_work)
    finally:
        allow_cancellation.set()
        await asyncio.wait_for(shutdown_task, timeout=5)
        if rejected_task is not None:
            rejected_task.cancel()
            await asyncio.gather(rejected_task, return_exceptions=True)
            await registry.shutdown()

    assert accepted_task.done()
    assert not rejected_factory_called
    assert registry._tasks == {}
    assert registry._cancel_flags == {}
    assert not registry.is_running(accepted.job_id)
    assert not registry.is_running(rejected.job_id)


def test_repeated_lifespans_reopen_registry_before_dispatcher(
    temp_storage, monkeypatch
) -> None:
    from app import main as main_module

    registry = JobRegistry()
    dispatcher_start_states: list[bool] = []

    async def controlled_dispatcher() -> None:
        dispatcher_start_states.append(registry.accepting)
        await asyncio.Event().wait()

    monkeypatch.setattr(main_module, "mark_interrupted_operation_jobs", lambda: 0)
    monkeypatch.setattr(main_module, "run_operation_dispatcher", controlled_dispatcher)
    monkeypatch.setattr(dispatcher, "job_registry", registry)
    application = create_app()

    async def run_lifespans() -> None:
        for _ in range(2):
            async with application.router.lifespan_context(application):
                await asyncio.sleep(0)
                assert registry.accepting
            assert not registry.accepting

    asyncio.run(run_lifespans())

    assert dispatcher_start_states == [True, True]


@pytest.mark.asyncio
async def test_pending_cancellation_survives_empty_registry(temp_storage, monkeypatch) -> None:
    result = execute(adjustment(make_project(), generate=True))
    registry = JobRegistry()
    assert registry.request_cancel(result.job_id)
    monkeypatch.setattr(dispatcher, "job_registry", registry)

    async def unexpected(*args, **kwargs):
        pytest.fail("a cancelled pending job must not call providers")

    monkeypatch.setattr(dispatcher, "run_generation_job", unexpected)
    # Cancellation before the running claim is now terminal immediately. It
    # never needs a dispatcher task to discover the saved cancellation flag.
    assert dispatcher.dispatch_pending_operation_jobs() == 0
    assert registry._tasks == {}
    with get_session_factory()() as db:
        job = db.get(GenerationJob, result.job_id)
        assert job.status == JobStatus.cancelled
        assert job.started_at is None
    assert dispatcher.dispatch_pending_operation_jobs() == 0


@pytest.mark.parametrize("ambiguous", [False, True])
def test_interrupted_running_job_resumes_only_known_safe_work(temp_storage, ambiguous: bool) -> None:
    request = adjustment(make_project(), generate=True)
    result = execute(request)
    with get_session_factory()() as db:
        job = db.get(GenerationJob, result.job_id)
        job.status = JobStatus.running
        if ambiguous:
            db.add(ExternalCall(job_id=job.id, fingerprint="synthetic-interruption", provider="synthetic",
                                endpoint="https://synthetic.invalid", remote_side_effect=True,
                                status="in_flight"))
        db.commit()
    assert dispatcher.mark_interrupted_operation_jobs() == 1
    assert dispatcher.mark_interrupted_operation_jobs() == 0
    if ambiguous:
        assert dispatcher.dispatch_pending_operation_jobs() == 0
    assert execute(request) == result
    with get_session_factory()() as db:
        jobs = list(db.scalars(select(GenerationJob)))
        assert len(jobs) == 1
        assert jobs[0].status == (JobStatus.unknown if ambiguous else JobStatus.pending)
        if ambiguous:
            assert "未確定" in jobs[0].error_message
        else:
            assert jobs[0].error_message is None
            assert "再開" in jobs[0].recovery_message
        assert db.get(Project, result.project_id).revision == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False])
async def test_deleted_jobs_cannot_alias_historical_result_references(temp_storage, monkeypatch, legacy: bool) -> None:
    from app.workers import job_runner

    request = adjustment(make_project(), generate=True)
    original = execute(request)
    client = TestClient(create_app())
    assert client.delete(f"/api/projects/{original.project_id}").status_code == 409
    with get_session_factory()() as db:
        db.get(GenerationJob, original.job_id).status = JobStatus.completed
        db.commit()
    assert client.delete(f"/api/projects/{original.project_id}").status_code == 204
    project_id = make_project()
    if legacy:
        monkeypatch.setattr(job_runner.job_registry, "submit", lambda *args: None)
        new_job = await job_runner.enqueue_full_pipeline(project_id)
        assert new_job.id > original.job_id
    else:
        result = execute(adjustment(project_id, "next-job", generate=True))
        assert result.job_id > original.job_id
    assert execute(request) == original
    with get_session_factory()() as db:
        assert db.get(GenerationJob, original.job_id) is None


@pytest.mark.asyncio
async def test_legacy_enqueue_rechecks_durable_intent_under_write_lock(temp_storage) -> None:
    from app.services.job_records import ProjectBusyError
    from app.workers.job_runner import enqueue_full_pipeline

    project_id = make_project()
    execute(adjustment(project_id, generate=True))
    with pytest.raises(ProjectBusyError):
        await enqueue_full_pipeline(project_id)


def test_pending_cancellation_updates_project_terminal_state(temp_storage) -> None:
    result = execute(adjustment(make_project(), generate=True))
    assert JobRegistry().request_cancel(result.job_id)
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).status == JobStatus.cancelled
        project = db.get(Project, result.project_id)
        assert project.status == ProjectStatus.cancelled
        assert (project.subtitle_font_size, project.revision) == (50, 2)


@pytest.mark.asyncio
async def test_running_cancellation_updates_project_terminal_state(temp_storage) -> None:
    result = execute(adjustment(make_project(), generate=True))
    registry = JobRegistry()
    started, release = asyncio.Event(), asyncio.Event()

    async def work(cancel_check):
        started.set()
        await release.wait()
        assert cancel_check()
        raise GenerationCancelled("synthetic safe boundary")

    task = registry.submit(result.job_id, work)
    await asyncio.wait_for(started.wait(), timeout=5)
    assert registry.request_cancel(result.job_id)
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).cancel_requested
        assert db.get(GenerationJob, result.job_id).status == JobStatus.running
    release.set()
    await task
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).status == JobStatus.cancelled
        project = db.get(Project, result.project_id)
        assert project.status == ProjectStatus.cancelled
        assert (project.subtitle_font_size, project.revision) == (50, 2)


@pytest.mark.asyncio
async def test_committed_completion_wins_over_late_cancel_even_if_callback_then_raises(temp_storage) -> None:
    from app.services.transactions import atomic_write

    result = execute(adjustment(make_project(), generate=True))
    registry = JobRegistry()

    async def work(cancel_check):
        # This is the same durable terminal boundary used by artifact publication.
        with get_session_factory()() as db, atomic_write(db):
            db.get(GenerationJob, result.job_id).status = JobStatus.completed
            db.get(Project, result.project_id).status = ProjectStatus.completed
        assert registry.request_cancel(result.job_id) is False
        raise GenerationCancelled("synthetic late callback")

    await registry.submit(result.job_id, work)
    with get_session_factory()() as db:
        assert db.get(GenerationJob, result.job_id).status == JobStatus.completed
        assert not db.get(GenerationJob, result.job_id).cancel_requested
        assert db.get(Project, result.project_id).status == ProjectStatus.completed


def test_deleted_project_cleanup_cannot_remove_newly_created_project_files(temp_storage, monkeypatch) -> None:
    import shutil

    from app.services.paths import project_dir

    project_id = make_project()
    old_directory = project_dir(project_id).resolve()
    assert old_directory.is_relative_to(temp_storage.resolve())
    old_directory.mkdir(parents=True, exist_ok=True)
    (old_directory / "old-synthetic.txt").write_text("old", encoding="utf-8")
    cleanup_started, release_cleanup, new_created = Event(), Event(), Event()
    remove_tree = shutil.rmtree

    def paused_cleanup(path, *args, **kwargs):
        if Path(path).resolve() == old_directory:
            cleanup_started.set()
            assert release_cleanup.wait(timeout=10)
        return remove_tree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", paused_cleanup)

    def delete_old():
        return TestClient(create_app()).delete(f"/api/projects/{project_id}")

    def create_new() -> Path:
        response = TestClient(create_app()).post("/api/projects", json={
            "title": "New synthetic project", "source_script": "別の合成台本。", "use_fake_providers": True,
        })
        assert response.status_code == 201
        marker = project_dir(response.json()["id"]) / "new-synthetic.txt"
        marker.write_text("new", encoding="utf-8")
        new_created.set()
        return marker

    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(delete_old)
        assert cleanup_started.wait(timeout=5)
        creation = pool.submit(create_new)
        # A sound deletion may retain its writer lock until path detachment.
        # In that case creation waits and proceeds when cleanup is released.
        new_created.wait(timeout=1)
        release_cleanup.set()
        assert deletion.result(timeout=10).status_code == 204
        marker = creation.result(timeout=10)
    assert marker.read_text(encoding="utf-8") == "new"
