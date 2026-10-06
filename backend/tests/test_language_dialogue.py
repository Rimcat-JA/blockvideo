"""D19 continuation lineage, current-state fences and non-execution paths."""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from sqlalchemy import select

from app.db import get_session_factory
from app.language_operations.contracts import LanguageExecution, LanguageInput
from app.models.language_request import LanguageRequestRecord
from app.models.language_turn import LanguageTurn
from app.models.operation_request import OperationReceipt
from app.models.job import GenerationJob
from app.operations.bootstrap import operation_service
from tests.test_language_operations import confirm, count, create, language_input, submit
from tests.test_language_operations import harness as harness  # noqa: F401


def reply(parent: dict[str, Any], text: str, *, request_id: str = 'answer',
          relation: str = 'answer', project_id: int | None = None, revision: int | None = None) -> dict[str, Any]:
    return language_input(project_id if project_id is not None else parent['project_id'],
        request_id=request_id, text=text,
        base_revision=revision if revision is not None else (parent['result']['revision'] if parent['result'] else parent['base_revision']),
        continuation={'parent_request_id': parent['request_id'], 'relation': relation})


def ask(client, adapter, project_id: int, *, text: str = '字幕を大きくして', question: str = '何pxにしますか？',
        request_id: str = 'question') -> dict[str, Any]:
    adapter.response = json.dumps({'result': {'kind': 'clarification', 'question': question, 'missing_fields': ['arguments']}})
    return submit(client, language_input(project_id, request_id=request_id, text=text, base_revision=1))


def test_short_answer_has_linked_context_and_a_new_core_identity(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = ask(client, adapter, project)
    adapter.operation('project.subtitle-font-size.set', {'value': 56})
    payload = reply(parent, '56px')
    result = submit(client, payload)
    assert result['status'] == 'completed' and result['parent_request_id'] == parent['request_id']
    assert result['core_request_id'] != parent['core_request_id']
    assert result['result']['revision'] == 2
    prompt = adapter.calls[-1]['prompt']
    assert prompt['request'] == '56px'
    assert prompt['dialogue'][0]['text'] == '字幕を大きくして'
    assert prompt['dialogue'][0]['question'] == '何pxにしますか？'
    assert 'source_script' not in json.dumps(prompt) and 'title' not in prompt['state']
    assert submit(client, payload) == result
    assert len(adapter.calls) == 2 and count(OperationReceipt) == 1


def test_correction_uses_current_value_and_keeps_previous_receipt(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = submit(client, language_input(project))
    adapter.operation('project.subtitle-font-size.adjust', {'delta': -2})
    result = submit(client, reply(parent, '違う、少し小さく', relation='correction'))
    assert result['result']['resolved_arguments'] == {'value': 54}
    assert result['result']['revision'] == 3
    earlier = client.get('/api/language/requests/nl-test').json()
    assert earlier['result'] == parent['result'] and earlier['superseded_by'] == 'answer'
    assert count(OperationReceipt) == 2 and count(GenerationJob) == 0


@pytest.mark.parametrize('relation', ['answer', 'correction'])
def test_old_answer_cannot_consume_an_already_answered_question(harness, relation: str) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = ask(client, adapter, project)
    adapter.operation('project.subtitle-font-size.set', {'value': 56})
    submit(client, reply(parent, '56px'))
    result = submit(client, reply(parent, '80px', request_id='old-answer', relation=relation, revision=2))
    assert result['status'] == 'blocked' and result['failure']['reason_code'] == 'dialogue_superseded'
    assert len(adapter.calls) == 2 and count(OperationReceipt) == 1


def test_answer_to_other_project_is_rejected_before_interpretation(harness) -> None:
    client, adapter, _ = harness
    first, second = create(client), create(client)
    parent = ask(client, adapter, first)
    result = submit(client, reply(parent, '56px', project_id=second))
    assert result['failure']['reason_code'] == 'dialogue_target_mismatch'
    assert len(adapter.calls) == 1 and count(OperationReceipt) == 0
    for project in (first, second):
        assert client.get(f'/api/projects/{project}').json()['subtitle_font_size'] == 48


def test_missing_target_can_be_selected_without_guessing(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = submit(client, language_input(None, text='字幕を56pxにして'))
    assert parent['status'] == 'needs_input' and not adapter.calls
    result = submit(client, reply(parent, 'この対象です', project_id=project, revision=1))
    assert result['status'] == 'completed' and result['project_id'] == project
    assert len(adapter.calls) == 1


@pytest.mark.parametrize('completed', [False, True])
def test_changed_revision_rejects_answer_or_correction_before_model(harness, completed: bool) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = submit(client, language_input(project)) if completed else ask(client, adapter, project)
    current_revision = 2 if completed else 1
    result = client.post('/api/operations/execute', json={'operation_id': 'project.subtitle-font-size.set',
        'arguments': {'value': 70}, 'target': {'project_id': project}, 'request_id': 'other-setting', 'base_revision': current_revision})
    assert result.status_code == 200
    response = submit(client, reply(parent, '56px', revision=current_revision + 1,
        relation='correction' if completed else 'answer'))
    assert response['status'] == 'blocked' and response['failure']['reason_code'] == 'dialogue_stale'
    assert len(adapter.calls) == 1
    assert client.get(f'/api/projects/{project}').json()['subtitle_font_size'] == 70


def test_dismiss_invalidates_old_confirmation_without_model_or_job(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.operation('project.generation.start', {'kind': 'full'})
    parent = submit(client, language_input(project, text='動画を生成して'))
    result = submit(client, reply(parent, '取り下げる', relation='dismiss'))
    assert result['status'] == 'dismissed' and not result['executed']
    old = confirm(client, parent, generation=True).json()
    assert old['status'] == 'blocked' and old['failure']['reason_code'] == 'dialogue_superseded'
    assert len(adapter.calls) == 1 and count(GenerationJob) == 0


@pytest.mark.parametrize('text', ['やめて', '変更しないで', 'それはしない'])
def test_negative_continuation_never_undoes_or_runs_another_operation(harness, text: str) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = submit(client, language_input(project))
    adapter.response = json.dumps({'result': {'kind': 'no_operation', 'reason': '今回は操作しません。'}})
    result = submit(client, reply(parent, text, relation='correction'))
    assert result['status'] == 'dismissed' and result['result'] is None
    assert client.get(f'/api/projects/{project}').json()['subtitle_font_size'] == 56
    assert count(OperationReceipt) == 1


def test_missing_reading_preserves_surface_and_accepts_only_answered_value(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = ask(client, adapter, project, text='APIの読み方を登録して', question='APIは何と読みますか？')
    adapter.operation('project.settings.update', {'pronunciation_overrides': [{'surface': 'API', 'reading': 'エーピーアイ', 'accent': None}]})
    result = submit(client, reply(parent, 'エーピーアイ'))
    assert result['status'] == 'completed'
    assert result['result']['data']['settings']['pronunciation_overrides'][0]['reading'] == 'エーピーアイ'
    assert adapter.calls[-1]['prompt']['dialogue'][0]['text'] == 'APIの読み方を登録して'


@pytest.mark.parametrize('status', ['blocked', 'unsupported', 'error'])
def test_blocked_or_unavailable_turn_cannot_be_replaced_by_a_fallback(harness, status: str) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = ask(client, adapter, project)
    with get_session_factory()() as db:
        record = db.get(LanguageRequestRecord, parent['request_id'])
        record.status = status
        record.response_json = {**record.response_json, 'status': status}
        db.commit()
    adapter.operation('project.subtitle-font-size.set', {'value': 80})
    result = submit(client, reply(parent, 'では80px', relation='correction'))
    assert result['failure']['reason_code'] == 'dialogue_not_pending'
    assert count(OperationReceipt) == 0 and len(adapter.calls) == 1


def test_concurrent_answers_allow_only_one_successor(harness) -> None:
    client, adapter, service = harness
    project = create(client)
    parent = ask(client, adapter, project)
    adapter.operation('project.subtitle-font-size.set', {'value': 56})

    def send(request_id: str):
        with get_session_factory()() as db:
            return asyncio.run(service.submit(db, LanguageInput.model_validate(reply(parent, '56px', request_id=request_id))))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send, ['answer-a', 'answer-b']))
    assert sorted(item.status for item in results) == ['blocked', 'completed']
    assert count(OperationReceipt) == 1 and len(adapter.calls) == 2
    with get_session_factory()() as db:
        assert len(db.scalars(select(LanguageTurn).where(LanguageTurn.parent_request_id == parent['request_id'])).all()) == 1


def test_dismiss_between_lookup_and_dispatch_is_checked_inside_core_transaction(harness, monkeypatch) -> None:
    client, adapter, service = harness
    project = create(client)
    adapter.operation('project.generation.start', {'kind': 'full'})
    parent = submit(client, language_input(project, text='動画を生成して'))
    original = operation_service.execute

    def race(db, request, **kwargs):
        with get_session_factory()() as another:
            result = asyncio.run(service.submit(another, LanguageInput.model_validate(reply(parent, '取り下げ', relation='dismiss'))))
            assert result.status == 'dismissed'
        return original(db, request, **kwargs)

    monkeypatch.setattr(operation_service, 'execute', race)
    with get_session_factory()() as db:
        result = service.execute(db, parent['request_id'], LanguageExecution(confirmation_token=parent['confirmation_token'], confirm_generation=True))
    assert result.status == 'blocked' and result.failure.reason_code == 'dialogue_superseded'
    assert count(GenerationJob) == 0


def test_linked_answer_replay_survives_new_service_and_current_setting_change(harness) -> None:
    client, adapter, service = harness
    project = create(client)
    parent = ask(client, adapter, project)
    adapter.operation('project.subtitle-font-size.adjust', {'delta': 2})
    payload = reply(parent, '少し')
    first = submit(client, payload)
    service.adapter = None
    assert submit(client, payload) == first
    assert first['result']['resolved_arguments'] == {'value': 50}
    assert count(OperationReceipt) == 1


def test_guessed_reading_is_asked_even_if_model_supplies_schema_valid_value(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    adapter.operation('project.settings.update', {'pronunciation_overrides': [{'surface': 'API', 'reading': 'エーピーアイ'}]})
    result = submit(client, language_input(project, text='APIの読み方を登録して'))
    assert result['status'] == 'needs_input' and result['clarification']
    assert count(OperationReceipt) == 0


def test_new_reading_preserves_existing_entries_and_normalizes_hiragana(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    initial = client.post('/api/operations/execute', json={'operation_id': 'project.settings.update',
        'arguments': {'pronunciation_overrides': [{'surface': 'CPU', 'reading': 'シーピーユー', 'accent': None}]},
        'target': {'project_id': project}, 'request_id': 'initial-reading', 'base_revision': 1})
    assert initial.status_code == 200
    adapter.operation('project.settings.update', {'pronunciation_overrides': [{'surface': 'API', 'reading': 'エーピーアイ', 'accent': None}]})
    result = submit(client, language_input(project, text='APIはえーぴーあいと読んで', base_revision=2))
    assert result['status'] == 'completed'
    assert {item['surface'] for item in result['result']['data']['settings']['pronunciation_overrides']} == {'API', 'CPU'}


def test_long_context_rejection_does_not_consume_the_question(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = ask(client, adapter, project, text='字' * 2000)
    second = submit(client, reply(parent, '字' * 2000))
    third = submit(client, reply(second, '字' * 2000, request_id='too-long'))
    assert third['status'] == 'blocked' and third['failure']['reason_code'] == 'dialogue_limit'
    with get_session_factory()() as db:
        assert db.get(LanguageTurn, second['request_id']).successor_request_id is None


def test_empty_setting_correction_asks_instead_of_recording_a_success(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    parent = submit(client, language_input(project))
    adapter.operation('project.settings.update', {})
    result = submit(client, reply(parent, '違う', relation='correction'))
    assert result['status'] == 'needs_input' and not result['executed']
    assert count(OperationReceipt) == 1


def test_echo_is_not_presented_as_a_useful_clarification(harness) -> None:
    client, adapter, _ = harness
    project = create(client)
    result = ask(client, adapter, project, text='字幕を大きくして。', question='字幕を大きくして。')
    assert result['clarification']['question'] != '字幕を大きくして。'
    assert result['interpretation']['proposal']['question'] == '字幕を大きくして。'
    assert result['status'] == 'needs_input' and count(OperationReceipt) == 0
