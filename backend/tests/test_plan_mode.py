"""Multi-step plans: each step is an ordinary proposal; the application runs them in order."""
from __future__ import annotations

import json

import pytest

from app.interpretation.candidates import response_schema
from app.interpretation.errors import InterpretationError
from app.interpretation.parser import parse_proposal
from app.interpretation.service import system_prompt
from app.models.job import GenerationJob
from app.models.operation_request import OperationReceipt
from app.operations.bootstrap import operation_service
from tests.test_language_operations import confirm, count, create, harness, language_input, submit  # noqa: F401

DEFINITIONS = tuple(operation_service.list_definitions())
SET_60 = {"kind": "operation", "operation_id": "project.subtitle-font-size.set", "operation_version": 1,
          "arguments": {"value": 60}, "generate_after_save": False}
START = {"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1,
         "arguments": {"kind": "full"}, "generate_after_save": False}
STATUS = {"kind": "operation", "operation_id": "project.status.get", "operation_version": 1,
          "arguments": {}, "generate_after_save": False}


def _plan(adapter, *steps: dict) -> None:
    adapter.response = json.dumps({"result": {"kind": "plan", "steps": list(steps)}})


def test_plan_grammar_parser_and_prompt() -> None:
    schema = response_schema(DEFINITIONS)
    plan_branch = next(branch for branch in schema["properties"]["result"]["anyOf"]
                       if branch["properties"]["kind"]["enum"] == ["plan"])
    assert plan_branch["properties"]["steps"]["maxItems"] == 4
    parsed = parse_proposal(json.dumps({"result": {"kind": "plan", "steps": [SET_60, START]}}), DEFINITIONS)
    assert [step.operation_id for step in parsed.result.steps] == ["project.subtitle-font-size.set", "project.generation.start"]
    with pytest.raises(InterpretationError):
        parse_proposal(json.dumps({"result": {"kind": "plan", "steps": [{**SET_60, "generate_after_save": True}, START]}}),
                       DEFINITIONS)
    with pytest.raises(InterpretationError):
        parse_proposal(json.dumps({"result": {"kind": "plan", "steps": [SET_60, START]}}), DEFINITIONS[:1])
    offered = {item.operation_id for item in DEFINITIONS}
    assert "【複数手順】" in system_prompt(offered) and "【複数手順】" not in system_prompt(offered, features=frozenset())


def test_normal_plan_is_confirmed_once_then_runs_in_order(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _plan(adapter, SET_60, START)
    before = count(GenerationJob)

    ready = submit(client, language_input(project_id, text="字幕を60pxにしてから動画を作り直して"))
    assert ready["status"] == "ready" and ready["requires_confirmation"] is True
    assert [step["operation_id"] for step in ready["plan"]] == ["project.subtitle-font-size.set", "project.generation.start"]
    assert count(GenerationJob) == before

    assert confirm(client, ready, generation=False).status_code == 409  # generation needs explicit confirmation
    done = confirm(client, ready, generation=True).json()
    assert done["status"] == "completed" and len(done["plan_results"]) == 2
    assert done["plan_results"][1]["base_revision"] == done["plan_results"][0]["revision"]
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 60
    assert count(GenerationJob) == before + 1

    receipts = count(OperationReceipt)
    again = confirm(client, ready, generation=True).json()
    assert again["status"] == "completed" and count(OperationReceipt) == receipts and count(GenerationJob) == before + 1


def test_plan_stops_at_the_first_failing_step(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _plan(adapter, SET_60, {"kind": "operation", "operation_id": "project.generation.retry", "operation_version": 1,
                            "arguments": {"job_id": 999}, "generate_after_save": False})

    ready = submit(client, language_input(project_id, text="字幕を60pxにしてからジョブ999を再試行して"))
    done = confirm(client, ready, generation=True).json()

    assert done["status"] == "blocked" and len(done["plan_results"]) == 1
    assert "手順2で停止" in done["failure"]["message"]
    assert client.get(f"/api/projects/{project_id}").json()["subtitle_font_size"] == 60


def test_yolo_plan_runs_unattended_and_drops_negated_steps(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _plan(adapter, SET_60, START)
    before = count(GenerationJob)

    done = submit(client, language_input(project_id, text="字幕を60pxにして。動画は生成しないで", mode="yolo"))

    assert done["status"] == "completed" and [s["operation_id"] for s in done["plan"]] == ["project.subtitle-font-size.set"]
    assert done["yolo_report"]["dropped_steps"] == ["project.generation.start"]
    assert done["yolo_report"]["auto_confirmed"] == ["project.subtitle-font-size.set"]
    assert count(GenerationJob) == before


def test_normal_plan_with_a_negated_step_is_dismissed(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _plan(adapter, SET_60, START)

    response = submit(client, language_input(project_id, text="字幕を60pxにして。動画は生成しないで"))

    assert response["status"] == "dismissed" and response["diagnostics"]["guard_code"] == "negative_intent"


def test_yolo_plan_with_generation_runs_to_the_end(harness) -> None:  # noqa: F811
    client, adapter, _ = harness
    project_id = create(client)
    _plan(adapter, SET_60, STATUS, START)
    before = count(GenerationJob)

    done = submit(client, language_input(project_id, text="字幕を60pxにして、状態を見て、動画を作り直して", mode="yolo"))

    assert done["status"] == "completed" and len(done["plan_results"]) == 3
    assert count(GenerationJob) == before + 1
