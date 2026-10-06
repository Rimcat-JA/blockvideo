"""Use independent interpreters and real process death against one SQLite file."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest
from sqlalchemy import func, select

from app.db import get_session_factory
from app.models.job import GenerationJob
from app.models.operation_request import OperationReceipt
from app.models.project import Project
from tests.test_operation_durability import adjustment, execute, make_project


WORKER = """
import json, os, sys
from app.db import get_session_factory
from app.operations.bootstrap import build_operation_service
from app.operations.contracts import OperationRequest
from app.operations.errors import OperationError

payload = json.load(sys.stdin)
request = OperationRequest.model_validate(payload['request'])
with get_session_factory()() as db:
    commit = db.commit
    def crash_commit():
        db.flush()
        if payload['crash'] == 'before':
            os._exit(91)
        commit()
        os._exit(92)
    if payload['crash']:
        db.commit = crash_commit
    try:
        result = build_operation_service().execute(db, request)
        print(json.dumps(result.model_dump(mode='json')))
    except OperationError as error:
        print(json.dumps({'reason_code': error.reason_code}))
"""


def run_process(request: dict, crash: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", WORKER],
        input=json.dumps({"request": request, "crash": crash}),
        text=True, capture_output=True, timeout=40,
        cwd=Path(__file__).resolve().parents[1],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


@pytest.mark.parametrize("mode", ["same", "different-ids", "conflicting-content"])
def test_independent_processes_serialize_requests(temp_storage, mode: str) -> None:
    project_id = make_project()
    barrier = Barrier(3)

    def send(index: int) -> dict:
        request = adjustment(project_id, f"request-{index}" if mode == "different-ids" else "same",
                             delta=2 + index if mode == "conflicting-content" else 2)
        barrier.wait(timeout=10)
        process = run_process(request.model_dump(mode="json"))
        assert process.returncode == 0, process.stderr
        return json.loads(process.stdout)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(send, range(3)))
    if mode == "same":
        assert results[0] == results[1] == results[2]
    else:
        reason = "stale_state" if mode == "different-ids" else "request_id_conflict"
        assert sum(item.get("reason_code") == reason for item in results) == 2
        assert sum(item.get("revision") == 2 for item in results) == 1
    with get_session_factory()() as db:
        winner = next(item for item in results if item.get("revision") == 2)
        assert db.get(Project, project_id).subtitle_font_size == winner["resolved_arguments"]["value"]
        assert db.get(Project, project_id).revision == 2
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 1


@pytest.mark.parametrize("crash", ["before", "after"])
def test_process_death_before_or_after_commit_then_resend(temp_storage, crash: str) -> None:
    project_id = make_project()
    request = adjustment(project_id, generate=True)
    payload = request.model_dump(mode="json")
    process = run_process(payload, crash)
    assert process.returncode == (91 if crash == "before" else 92), process.stderr
    with get_session_factory()() as db:
        committed = crash == "after"
        project = db.get(Project, project_id)
        assert (project.subtitle_font_size, project.revision) == ((50, 2) if committed else (48, 1))
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == int(committed)
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == int(committed)
    # Both retries run in fresh interpreters, including the engine/service caches.
    first = run_process(payload)
    second = run_process(payload)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert json.loads(first.stdout) == json.loads(second.stdout)
    assert json.loads(first.stdout) == execute(request).model_dump(mode="json")
    with get_session_factory()() as db:
        assert db.get(Project, project_id).subtitle_font_size == 50
        assert db.scalar(select(func.count()).select_from(OperationReceipt)) == 1
        assert db.scalar(select(func.count()).select_from(GenerationJob)) == 1
