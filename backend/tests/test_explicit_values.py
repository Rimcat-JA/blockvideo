"""YOLO may guess missing values but never contradict or drop stated ones."""
from __future__ import annotations

import pytest

from app.interpretation.contracts import OperationProposal
from app.language_operations.explicit_values import explicit_conflict, plan_drops_stated
from app.language_operations.intent_guard import negative_control_reason, only_negated_instructions

ADJUST, SET, UPDATE = "project.subtitle-font-size.adjust", "project.subtitle-font-size.set", "project.settings.update"


def _proposal(operation_id: str, arguments: dict) -> OperationProposal:
    return OperationProposal(kind="operation", operation_id=operation_id, operation_version=1, arguments=arguments)


@pytest.mark.parametrize(("texts", "operation_id", "arguments", "conflict"), [
    (["字幕を2px小さくして"], ADJUST, {"delta": 2}, True),
    (["字幕を2px小さくして"], ADJUST, {"delta": -2}, False),
    (["字幕を64pxにして"], ADJUST, {"delta": 64}, True),
    (["字幕を64pxにして"], SET, {"value": 64}, False),
    (["字幕を64pxにして"], SET, {"value": 80}, True),
    (["字幕を大きくして"], ADJUST, {"delta": 4}, False),
    (["話速を1.2倍、音量を0.8倍にして"], UPDATE, {"voicevox_speed_scale": 0.8, "voicevox_volume_scale": 1.2}, True),
    (["話速を1.2倍、音量を0.8倍にして"], UPDATE, {"voicevox_speed_scale": 1.2, "voicevox_volume_scale": 0.8}, False),
    (["話者を3にして、話速も少し速くして"], UPDATE, {"voicevox_speaker_id": 3, "voicevox_speed_scale": 1.1}, False),
    (["字幕を64pxにして、話速を1.2倍にして"], UPDATE, {"subtitle_font_size": 64}, True),
    (["字幕を64pxにして、話速を1.2倍にして"], SET, {"value": 64}, True),
    (["1.5倍速にして"], UPDATE, {"voicevox_speed_scale": 1.2}, True),
    (["動画3ではなく動画4に戻して"], "project.artifact.restore", {"artifact_id": 4}, False),
    (["動画3に戻して"], "project.artifact.restore", {"artifact_id": 4}, True),
    (["字幕を64pxに大きくして"], SET, {"value": 80}, True),
    (["字幕を64pxに大きくして"], SET, {"value": 64}, False),
    (["字幕を4px大きくして"], SET, {"value": 52}, False),
    (["字幕を64pxにして、話速を1.2倍にして"], UPDATE, {"voicevox_speed_scale": 1.2}, True),
    (["話速を1.2倍にしないで、音量だけ少し上げて"], UPDATE,
     {"voicevox_speed_scale": 1.2, "voicevox_volume_scale": 1.1}, True),
    (["話速を1.2倍にしないで、音量だけ少し上げて"], UPDATE, {"voicevox_volume_scale": 1.1}, False),
    (["1.2倍の話速にして"], UPDATE, {"voicevox_speed_scale": 1.5}, True),
    # Each statement is read within its own clause.
    (["話速を1.2倍にして動画は生成しないで"], UPDATE, {"voicevox_speed_scale": 1.2}, False),
    (["話速を1.2倍にして動画は生成しないで"], UPDATE, {"voicevox_speed_scale": 1.5}, True),
    (["話速を1.2倍から1.5倍にして"], UPDATE, {"voicevox_speed_scale": 1.5}, False),
    (["幅1920pxの動画です。話速を1.2倍にして"], UPDATE, {"voicevox_speed_scale": 1.2}, False),
    (["原稿の文字数は1000です。話速を1.2倍にして"], UPDATE, {"voicevox_speed_scale": 1.2}, False),
    (["「話速を1.2倍にして」とは言っていません。音量だけ0.8倍にして"], UPDATE,
     {"voicevox_speed_scale": 1.2, "voicevox_volume_scale": 0.8}, True),
    (["「話速を1.2倍にして」とは言っていません。音量だけ0.8倍にして"], UPDATE, {"voicevox_volume_scale": 0.8}, False),
    (["音量を調整して0.8倍にして"], UPDATE, {"voicevox_volume_scale": 1.2}, True),
    (["音量を調整して0.8倍にして"], UPDATE, {"voicevox_volume_scale": 0.8}, False),
    (["字幕を64pxにしないで、話速を1.2倍にして"], UPDATE, {"subtitle_font_size": 64, "voicevox_speed_scale": 1.2}, True),
    (["字幕を64pxにしないで、話速を1.2倍にして"], UPDATE, {"voicevox_speed_scale": 1.2}, False),
    (["字幕を2px変えて小さくして"], ADJUST, {"delta": -2}, False),
    (["字幕を2px変えて小さくして"], ADJUST, {"delta": 2}, True),
    # An answer's own reference settles the earlier ambiguity.
    (["動画4", "動画3か動画4に戻して"], "project.artifact.restore", {"artifact_id": 4}, False),
    # An answer completes the unsaved request: its stated values still bind.
    (["64px", "話速を1.2倍にして、字幕を大きくして"], UPDATE,
     {"subtitle_font_size": 64, "voicevox_speed_scale": 1.5}, True),
])
def test_explicit_conflicts(texts: list[str], operation_id: str, arguments: dict, conflict: bool) -> None:
    assert explicit_conflict(texts, _proposal(operation_id, arguments)) is conflict


@pytest.mark.parametrize("delta,conflict", [(2, True), (-2, False)])
def test_direction_is_read_from_the_subtitle_clause_only(delta: int, conflict: bool) -> None:
    proposal = OperationProposal(kind="operation", operation_id=UPDATE, operation_version=2, arguments={
        "settings": {"voicevox_volume_scale": 1.2}, "subtitle_font_size_delta": delta})
    assert explicit_conflict(["字幕を2px小さくして、音量を1.2倍に上げて"], proposal) is conflict


def test_a_plan_must_carry_every_stated_value() -> None:
    text = ["字幕を64pxにして、話速を1.2倍にして、状態を確認してから動画を作って"]
    steps = [_proposal(SET, {"value": 64}), _proposal("project.status.get", {}),
             _proposal("project.generation.start", {"kind": "full"})]
    assert plan_drops_stated(text, steps)
    assert not plan_drops_stated(text, [*steps[:2], _proposal(UPDATE, {"voicevox_speed_scale": 1.2}), steps[2]])


def test_plan_checks_every_stated_value_and_trusts_restores() -> None:
    twice = ["話速を1.2倍にして、状態を確認して、話速を1.5倍にして、動画を作って"]
    assert plan_drops_stated(twice, [_proposal(UPDATE, {"voicevox_speed_scale": 1.2}),
                                     _proposal("project.status.get", {}),
                                     _proposal("project.generation.start", {"kind": "full"})])
    restore = ["設定を第2版に戻して、字幕サイズが64pxになったことを確認して"]
    assert not plan_drops_stated(restore, [_proposal("project.settings.restore", {"revision": 2}),
                                           _proposal("project.status.get", {})])
    # "動画1" is a reference, never the volume.
    text = ["音量を0.8倍にしてほしいので動画1に戻して"]
    assert not explicit_conflict(text, _proposal(UPDATE, {"voicevox_volume_scale": 0.8}), require_all=False)
    assert explicit_conflict(text, _proposal(UPDATE, {"voicevox_volume_scale": 1}), require_all=False)


def test_a_plan_step_is_not_required_to_carry_values_of_later_steps() -> None:
    texts = ["字幕を64pxにして、状態を確認して、話速を1.2倍にして"]
    assert not explicit_conflict(texts, _proposal(SET, {"value": 64}), require_all=False)
    assert explicit_conflict(texts, _proposal(SET, {"value": 64}))


@pytest.mark.parametrize(("text", "expected"), [
    ("ジョブ7は止めていただかなくていい", True), ("ジョブ7は止めなくても大丈夫です", True),
    ("字幕を60pxにして動画は生成しないで", False), ("ジョブ7を止めて", False),
    ("字幕を60pxにしたうえで動画は生成しないで", False),
])
def test_polite_negations_are_negation_only(text: str, expected: bool) -> None:
    assert only_negated_instructions(text) is expected


@pytest.mark.parametrize(("text", "operation_id", "blocked"), [
    ("設定は戻さないで、動画3に戻して", "project.artifact.restore", False),
    ("動画は戻さないで、設定を第2版に戻して", "project.settings.restore", False),
    ("動画3には戻さないで、状態だけ教えて", "project.artifact.restore", True),
    ("第2版には戻さないで、状態を教えて", "project.settings.restore", True),
    ("動画3には絶対に戻さないで、状態だけ教えて", "project.artifact.restore", True),
    ("成果物3には戻さないで、状態だけ教えて", "project.artifact.restore", True),
    ("動画の旧版には戻さないで、設定を第2版に戻して", "project.settings.restore", False),
    ("設定を第2版に戻して動画は戻さないで", "project.settings.restore", False),
    ("動画3に戻して設定は戻さないで", "project.artifact.restore", False),
    ("動画3については戻さないで、状態だけ教えて", "project.artifact.restore", True),
])
def test_restore_negations_name_their_target(text: str, operation_id: str, blocked: bool) -> None:
    assert (negative_control_reason(text, operation_id) is not None) is blocked


def test_sixth_review_cases() -> None:
    restore_video = ["動画3に戻して、話速を1.2倍にしてから作り直して"]
    assert plan_drops_stated(restore_video, [_proposal("project.artifact.restore", {"artifact_id": 3}),
                                             _proposal("project.generation.start", {"kind": "full"})])
    assert explicit_conflict(["確認用に話速を1.2倍にして"], _proposal(UPDATE, {"voicevox_speed_scale": 1.5}))
    forbidden = ["字幕は64pxにしないで、少し大きくして"]
    assert explicit_conflict(forbidden, _proposal(ADJUST, {"delta": 2}), current_subtitle=62)
    assert not explicit_conflict(forbidden, _proposal(ADJUST, {"delta": 4}), current_subtitle=62)
    assert not explicit_conflict(["話速を少し上げて、もう1回生成して"], _proposal(UPDATE, {"voicevox_speed_scale": 1.1}))
    corrected = ["話速は1.2倍、いや、話速は1.5倍にして。状態も教えて"]
    assert not plan_drops_stated(corrected, [_proposal(UPDATE, {"voicevox_speed_scale": 1.5}),
                                             _proposal("project.status.get", {})])
    assert explicit_conflict(corrected, _proposal(UPDATE, {"voicevox_speed_scale": 1.2}))


def test_seventh_review_cases_keep_their_bindings() -> None:
    assert explicit_conflict(["話者を変更して、3番にして"], _proposal(UPDATE, {"voicevox_speaker_id": 4}))
    assert not explicit_conflict(["話者を変更して、3番にして"], _proposal(UPDATE, {"voicevox_speaker_id": 3}))
    assert explicit_conflict(["話速を1.2倍にして。いや、話速だけ変えて、ほかはそのままにして"],
                             _proposal(UPDATE, {"voicevox_speed_scale": 1.5}))
    assert explicit_conflict(["字幕を64pxにできるか教えて"], _proposal(SET, {"value": 80}))
