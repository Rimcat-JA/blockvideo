"""Build 100 synthetic DEVELOPMENT labels; never read/write held-out data."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluation.contracts import Case

ROOTS: list[str] = [
    "字幕サイズを64pxにして", "字幕を少し小さくして", "動画の状態を確認して",
    "字幕を64pxにして、APIをエーピーアイと読んで。作り直して",
    "字幕を72pxにして、速度を1.2倍にして。動画を作り直して",
    "今の設定で動画を作り直して", "ジョブ7を止めて", "ジョブ7を再試行して",
    "設定を第2版に戻して", "この動画をメールで送って",
]


def proposals(name: str, args: dict[str, Any], generate: bool = False) -> list[dict[str, Any]]:
    def one(operation: str, arguments: dict[str, Any], version: int = 1) -> dict[str, Any]:
        return {"operation_id": f"project.{operation}", "operation_version": version,
                "arguments": arguments, "generate_after_save": generate}
    if name == "set":
        return [one("subtitle-font-size.set", {"value": args["subtitle_font_size"]}),
                one("settings.update", args), one("settings.update", {"settings": args, "subtitle_font_size_delta": None}, 2)]
    if name == "adjust":
        return [one("subtitle-font-size.adjust", args),
                one("settings.update", {"settings": {}, "subtitle_font_size_delta": args["delta"]}, 2)]
    if name == "settings":
        variants = [args]
        if "pronunciation_overrides" in args:
            alternate = copy.deepcopy(args)
            for item in alternate["pronunciation_overrides"]:
                item["accent"] = None
            variants.append(alternate)
        return [item for value in variants for item in [one("settings.update", value),
                one("settings.update", {"settings": value, "subtitle_font_size_delta": None}, 2)]]
    if name == "relative_settings":
        return [one("settings.update", args, 2)]
    if name == "generation.start":
        return [one(name, {}), one(name, {"kind": "full"})]
    return [one(name, args)]


def effects(outcome: str, reason: str, delta: dict[str, Any] | None = None, *,
            ask: list[str] | None = None, confirm: bool = False, jobs: int = 0,
            job_assertions: dict[str, Any] | None = None, receipt: str = "new_request") -> dict[str, Any]:
    return {"outcome": outcome, "reason": reason, "question_for": ask or [],
            "settings_delta": delta or {}, "revision_delta": int(bool(delta)), "new_jobs": jobs,
            "confirmation_required": confirm, "job_assertions": job_assertions or {},
            "artifact_policy": "job_may_publish_on_success" if jobs else "preserve_all_no_new_publication",
            "receipt_rule": receipt}


def job(status: str = "running", *, owner: int = 101, cancelled: bool = False) -> dict[str, Any]:
    return {"id": 7, "project_id": owner, "status": status, "input_revision": 2,
            "cancel_requested": cancelled, "kind": "full", "input_settings": {"subtitle_font_size": 40}}


def turn(text: str, *, saved: bool = False, question: str | None = None,
         proposal: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"request_id": "prior-1", "text": text, "status": "completed" if saved else "needs_input",
            "settings_saved": saved, "question": question, "proposal": proposal,
            "project_id": 101, "base_revision": 4 if saved else 5, "result_revision": 5 if saved else None}


def build() -> list[Case]:
    rows: list[Case] = []

    def add(group: int, text: str, expected: dict[str, Any], *, name: str | None = None,
            args: dict[str, Any] | None = None, generate: bool = False, interpretation: str | None = None,
            tags: tuple[str, ...] = ("paraphrase",), settings: dict[str, Any] | None = None,
            status: str = "completed", jobs: list[dict[str, Any]] | None = None,
            target: int | None = 101, base: int | None = 5,
            prior: list[dict[str, Any]] | None = None, relation: str | None = None,
            event: str = "none", details: dict[str, Any] | None = None,
            after: dict[str, Any] | None = None, situation: str = "選択中はプロジェクト101、版5。成功動画は版4。",
            rationale: str = "明示された操作と値だけを採用し、他の設定・既存動画を保持する。",
            rules: tuple[str, ...] = ("R01", "R02"), limitation: str | None = None) -> None:
        current = {"subtitle_font_size": 48, "voicevox_speed_scale": 1.0, "voicevox_speaker_id": 0,
                   "pronunciation_overrides": [], "narration_pacing_mode": "adaptive",
                   "narration_sentence_pause_seconds": 0.2}
        current.update(settings or {})
        data = {"schema_version": 1, "case_id": f"D24-D{len(rows) + 1:03d}", "group_id": f"D24-DG{group:02d}",
                "split": "development", "source_request": ROOTS[group - 1],
                "provenance": {"kind": "prior_development", "reference": "D16–D23開発試験・承認済み製品ルールから作成した合成例。利用者の実データではない。"},
                "tags": list(tags), "situation": situation,
                "initial": {"project_id": 101, "revision": 5, "settings": current, "project_status": status,
                    "jobs": jobs or [], "history": [{"revision": 2, "settings": {**current, "subtitle_font_size": 40}},
                                                  {"revision": 5, "settings": current}],
                    "artifact_revisions": [2, 4], "prior_turns": prior or []},
                "request": {"request_id": f"request-d{len(rows) + 1:03d}", "text": text,
                    "target_project_id": target, "base_revision": base,
                    "continuation": {"parent_request_id": "prior-1", "relation": relation} if relation else None},
                "event": {"kind": event, "details": details or {}},
                "expected": {"interpretation": interpretation or ("operation" if name else "clarification"),
                    "operations": proposals(name, args or {}, generate) if name else [], "target_project_id": target,
                    "submit": expected, "after_event": after, "rationale": rationale, "rule_ids": list(rules)},
                "known_limitation": limitation}
        rows.append(Case.model_validate(data))

    set64 = effects("saved", "explicit_settings", {"subtitle_font_size": 64})
    blocked = effects("blocked", "project_busy", receipt="none")
    stale = effects("blocked", "stale_state", receipt="none")
    missing_target = effects("needs_input", "target_required", ask=["target"], receipt="none")
    for text in [ROOTS[0], "字幕の文字を64pxに設定してください", "字幕を６４ｐｘへ変更"]:
        add(1, text, set64, name="set", args={"subtitle_font_size": 64})
    add(1, ROOTS[0], effects("unchanged", "already_equal"), name="set", args={"subtitle_font_size": 64}, settings={"subtitle_font_size": 64}, tags=("boundary",))
    add(1, ROOTS[0], missing_target, target=None, interpretation="not_called", tags=("target",), rules=("R01",))
    add(1, "プロジェクト202の字幕を64pxにして", effects("blocked", "target_conflict", receipt="none"), interpretation="not_called", tags=("target",), rules=("R01",))
    add(1, ROOTS[0], blocked, name="set", args={"subtitle_font_size": 64}, jobs=[job()], status="generating", tags=("blocked",), rules=("R06",))
    add(1, ROOTS[0], stale, base=4, interpretation="not_called", tags=("race",), rules=("R08",))
    add(1, ROOTS[0], set64, name="set", args={"subtitle_font_size": 64}, event="resend_identical", after=effects("replayed", "same_request", {"subtitle_font_size": 64}, receipt="first_result"), tags=("resend",), rules=("R07",))
    add(1, "字幕サイズを16pxにして", effects("saved", "minimum_valid_value", {"subtitle_font_size": 16}), name="set", args={"subtitle_font_size": 16}, tags=("boundary",), rules=("R03",))

    small = effects("saved", "relative_current_value", {"subtitle_font_size": 46})
    for text in [ROOTS[1], "字幕サイズ、少し下げてください", "字幕を2px小さくして"]:
        add(2, text, small, name="adjust", args={"delta": -2}, rules=("R03",))
    add(2, "字幕を小さくして", effects("needs_input", "missing_amount", ask=["subtitle_amount"], receipt="none"), tags=("omission",), rules=("R03",))
    add(2, ROOTS[1], effects("blocked", "resolved_value_out_of_range", receipt="none"), name="adjust", args={"delta": -2}, settings={"subtitle_font_size": 16}, tags=("boundary", "blocked"), rules=("R03",))
    add(2, "違う、少し小さく", effects("saved", "correction_from_current", {"subtitle_font_size": 54}), name="adjust", args={"delta": -2}, settings={"subtitle_font_size": 56}, prior=[turn("字幕を56pxにして", saved=True)], relation="correction", tags=("correction",), rules=("R03", "R12"))
    add(2, "字幕の文字サイズを4px下げて", effects("saved", "explicit_delta", {"subtitle_font_size": 44}), name="adjust", args={"delta": -4}, rules=("R03",))
    add(2, "字幕は小さくしないで", effects("dismissed", "negated_change", receipt="none"), interpretation="no_operation", tags=("negation",), rules=("R14",))
    for event in ["concurrent_identical", "restart_resend"]:
        add(2, ROOTS[1], small, name="adjust", args={"delta": -2}, event=event, after=effects("replayed", "one_relative_effect", {"subtitle_font_size": 46}, receipt="first_result"), tags=("resend", "race") if event == "concurrent_identical" else ("resend",), rules=("R07",))

    query = effects("queried", "read_only_status")
    for text, state in [(ROOTS[2], "completed"), ("動画はできていますか", "generating"), ("今どうなってる？", "failed"), ("保存してある字幕サイズと動画の反映状況を教えて", "completed"), ("処理の状況を見せて", "pending")]:
        add(3, text, query, name="status.get", status=state, jobs=[job()] if state == "generating" else [], tags=("status",), rules=("R13",), rationale="設定48px・版5と動画の版4を区別して返す。照会で保存や生成をしない。")
    add(3, ROOTS[2], missing_target, target=None, interpretation="not_called", tags=("target", "status"), rules=("R01", "R13"))
    add(3, "プロジェクト202の進行状況を教えて", effects("blocked", "target_conflict", receipt="none"), interpretation="not_called", tags=("target",), rules=("R01",))
    add(3, ROOTS[2], query, name="status.get", event="resend_identical", details={"external_revision": 6, "external_settings": {"subtitle_font_size": 60}}, after=effects("replayed", "original_status_snapshot", receipt="first_result"), tags=("resend", "status"), rules=("R07", "R13"), rationale="同じIDは照会時の版5の結果を返す。最新値を取得する照会は新しいIDを使う。")
    add(3, ROOTS[2], query, name="status.get", event="switch_target", details={"selected_project_id_after": 202, "action": "read_original_request"}, after=effects("queried", "original_target_unchanged", receipt="first_result"), tags=("target",), rules=("R01", "R13"))
    add(3, "動画は生成せず、状態だけ教えて", query, name="status.get", tags=("negation", "status"), rules=("R13",))

    reading = [{"surface": "API", "reading": "エーピーアイ"}]
    compound = {"subtitle_font_size": 64, "pronunciation_overrides": reading}
    saved_pending = effects("saved_awaiting_confirmation", "atomic_save_then_confirm", compound, confirm=True)
    for text in [ROOTS[3], "APIはエーピーアイと読む設定にして、字幕を64pxに。動画も作り直したい"]:
        add(4, text, saved_pending, name="settings", args=compound, generate=True, tags=("compound", "confirmation"), rules=("R04", "R05"))
    missing_read = "字幕を64pxにして、APIの読み方を登録して。作り直して"
    add(4, missing_read, effects("needs_input", "missing_reading", ask=["pronunciation_overrides.API.reading"], receipt="none"), tags=("compound", "omission"), rules=("R04",), limitation="D23で読み方の推測を観測。アプリが質問へ戻してもモデル単体の正解に読み替えない。")
    add(4, "エーピーアイ", saved_pending, name="settings", args=compound, generate=True, prior=[turn(missing_read, question="APIは何と読みますか？")], relation="answer", tags=("compound", "omission"), rules=("R04", "R12", "R15"), limitation="D23では短い回答で字幕値が脱落。部分保存を止めるだけでは、この明確な対話の完了成功に数えない。")
    add(4, "字幕を64pxにしてAPIをエーピーアイと読んで。動画は作らないで", effects("saved", "save_without_generation", compound), name="settings", args=compound, tags=("compound", "negation"), rules=("R04", "R05"))
    add(4, ROOTS[3], saved_pending, name="settings", args=compound, generate=True, event="confirm_twice", after=effects("generation_queued", "one_confirmed_job", compound, jobs=1, job_assertions={"input_revision": 6, "count": 1}), tags=("compound", "confirmation", "resend"), rules=("R05", "R07"))
    add(4, ROOTS[3], saved_pending, name="settings", args=compound, generate=True, event="revision_race", details={"when": "after_save_before_confirm", "external_revision": 7, "external_settings": {"subtitle_font_size": 80}, "then": "confirm_old_generation"}, after=effects("blocked", "stale_generation_confirmation", compound), tags=("compound", "race"), rules=("R05", "R08"))
    add(4, ROOTS[3], blocked, name="settings", args=compound, generate=True, jobs=[job()], status="generating", tags=("compound", "blocked"), rules=("R04", "R06"))
    add(4, "字幕サイズを64pxにして、APIの読みも変えて", effects("needs_input", "missing_reading", ask=["pronunciation_overrides.API.reading"], receipt="none"), tags=("compound", "omission"), rules=("R04",))
    add(4, ROOTS[3], saved_pending, name="settings", args=compound, generate=True, event="same_id_different_body", details={"replacement_text": "字幕を80pxにして"}, after=effects("blocked", "request_id_conflict", compound, confirm=True, receipt="same_id_conflict"), tags=("compound", "resend"), rules=("R07",))

    speed = {"subtitle_font_size": 72, "voicevox_speed_scale": 1.2}
    speed_pending = effects("saved_awaiting_confirmation", "atomic_save_then_confirm", speed, confirm=True)
    for text in [ROOTS[4], "話速1.2倍、字幕72pxに変更してから生成したい"]:
        add(5, text, speed_pending, name="settings", args=speed, generate=True, tags=("compound", "confirmation"), rules=("R04", "R05"), limitation="D23では話速の代わりに指定していない文末間隔が提案された。")
    add(5, "字幕72px、話速1.2倍で保存だけして", effects("saved", "atomic_save", speed), name="settings", args=speed, tags=("compound",), rules=("R04",))
    vague = "字幕を大きくして、速度を1.2倍にして、動画を作って"
    add(5, vague, effects("needs_input", "missing_subtitle_amount", ask=["subtitle_amount"], receipt="none"), tags=("compound", "omission"), rules=("R03", "R04"), limitation="D23の誤保存例。字幕を勝手に2px増やすのは誤り。")
    answered = {"subtitle_font_size": 64, "voicevox_speed_scale": 1.2}
    add(5, "64px", effects("saved_awaiting_confirmation", "complete_pending_request", answered, confirm=True), name="settings", args=answered, generate=True, prior=[turn(vague, question="字幕は何pxにしますか？")], relation="answer", tags=("compound", "omission"), rules=("R12", "R15"), limitation="D23の短答では話速や生成希望が脱落。全内容を保つことが正解で、再質問は安全だが未完了。")
    add(5, "字幕は64px、速度は1.2倍にして、動画も作って", effects("saved_awaiting_confirmation", "complete_pending_request", answered, confirm=True), name="settings", args=answered, generate=True, prior=[turn(vague, question="字幕は何pxにしますか？")], relation="answer", tags=("compound",), rules=("R12",))
    add(5, "字幕を少し大きくして、速度を1.2倍にして", effects("saved", "atomic_relative_settings", {"subtitle_font_size": 50, "voicevox_speed_scale": 1.2}), name="relative_settings", args={"settings": {"voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": 2}, tags=("compound",), rules=("R03", "R04"))
    add(5, "字幕を72pxにして、話速を3倍にして", effects("needs_input", "speed_out_of_range", ask=["voicevox_speed_scale"], receipt="none"), tags=("compound", "boundary"), rules=("R04",), rationale="話速の上限を超えるので全体を保存せず、有効な話速を確認する。")
    add(5, ROOTS[4], blocked, name="settings", args=speed, generate=True, jobs=[job()], status="generating", tags=("compound", "blocked"), rules=("R06",))
    add(5, "違う、字幕を少し小さく。速度はそのまま", effects("saved", "correction_from_current", {"subtitle_font_size": 70}), name="adjust", args={"delta": -2}, settings=speed, prior=[turn("字幕72px、話速1.2倍にして", saved=True)], relation="correction", tags=("correction", "compound"), rules=("R12", "R03"))

    gen = effects("awaiting_confirmation", "generation_permission_required", confirm=True)
    for text in [ROOTS[5], "保存した設定を使って動画を生成してください", "この動画をもう一度作って"]:
        add(6, text, gen, name="generation.start", tags=("confirmation",), rules=("R05",))
    add(6, ROOTS[5], gen, name="generation.start", event="confirm_twice", after=effects("generation_queued", "one_confirmed_job", jobs=1, job_assertions={"input_revision": 5, "count": 1}), tags=("confirmation", "resend"), rules=("R05", "R07"))
    add(6, "今回は作らないで", effects("dismissed", "withdraw_pending_generation", receipt="none"), interpretation="no_operation", prior=[{**turn(ROOTS[5]), "status": "ready"}], relation="dismiss", tags=("negation",), rules=("R12", "R14"))
    add(6, ROOTS[5], blocked, name="generation.start", jobs=[job()], status="generating", tags=("blocked",), rules=("R06",))
    add(6, ROOTS[5], gen, name="generation.start", jobs=[job("unknown")], status="failed", event="confirm_generation", after=effects("blocked", "external_outcome_unknown"), tags=("failure", "confirmation"), rules=("R10",), rationale="外部呼出しの結果未確定なら確認後も新たな生成処理を作成しない。")
    add(6, ROOTS[5], gen, name="generation.start", event="revision_race", details={"when": "after_prepare_before_confirm", "external_revision": 6, "external_settings": {"subtitle_font_size": 80}, "then": "confirm_old_generation"}, after=stale, tags=("race", "confirmation"), rules=("R08",))
    add(6, ROOTS[5], missing_target, target=None, interpretation="not_called", tags=("target",), rules=("R01",))
    add(6, "動画は作り直さないで", effects("dismissed", "negated_generation", receipt="none"), interpretation="no_operation", tags=("negation",), rules=("R14",))

    cancel = effects("cancel_requested", "cooperative_cancel", job_assertions={"job_id": 7, "cancel_requested": True, "status": "running", "no_future_publication": True})
    for text in [ROOTS[6], "ジョブID7の生成を中止してください"]:
        add(7, text, cancel, name="generation.cancel", args={"job_id": 7}, jobs=[job()], status="generating", rules=("R09",))
    add(7, ROOTS[6], effects("cancelled", "pending_job_cancelled", job_assertions={"job_id": 7, "status": "cancelled", "cancel_requested": True}), name="generation.cancel", args={"job_id": 7}, jobs=[job("pending")], rules=("R09",))
    add(7, "生成を止めて", effects("needs_input", "missing_job_reference", ask=["job_id"], receipt="none"), jobs=[job()], status="generating", tags=("omission",), rules=("R09",))
    add(7, ROOTS[6], effects("blocked", "job_not_found"), name="generation.cancel", args={"job_id": 7}, jobs=[job(owner=202)], tags=("target",), rules=("R09",))
    for state in ["completed", "failed"]:
        add(7, ROOTS[6], effects("unchanged", "job_already_terminal"), name="generation.cancel", args={"job_id": 7}, jobs=[job(state)], rules=("R09",), rationale="終了したジョブは取り消さない。公開済みの動画や設定も戻さない。")
    add(7, ROOTS[6], effects("unchanged", "cancel_already_requested"), name="generation.cancel", args={"job_id": 7}, jobs=[job(cancelled=True)], rules=("R09",))
    add(7, ROOTS[6], cancel, name="generation.cancel", args={"job_id": 7}, jobs=[job()], event="resend_identical", after={**cancel, "outcome": "replayed", "receipt_rule": "first_result"}, tags=("resend",), rules=("R07", "R09"))
    add(7, "ジョブ7は止めないで", effects("dismissed", "negated_cancel", receipt="none"), interpretation="no_operation", jobs=[job()], tags=("negation",), rules=("R14",))

    retry = effects("awaiting_confirmation", "retry_permission_required", confirm=True)
    retried = effects("generation_queued", "retry_uses_current_settings", jobs=1, job_assertions={"parent_job_id": 7, "new_job_id_differs_from": 7, "input_revision": 5, "input_settings": {"subtitle_font_size": 72}})
    for state in ["failed", "cancelled"]:
        add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, settings={"subtitle_font_size": 72}, jobs=[job(state)], event="confirm_generation", after=retried, tags=("failure", "confirmation"), rules=("R10", "R05"))
    add(8, "失敗した動画を再試行して", effects("needs_input", "missing_job_reference", ask=["job_id"], receipt="none"), jobs=[job("failed")], tags=("omission",), rules=("R09",))
    for state, reason in [("unknown", "external_outcome_unknown"), ("completed", "job_not_retryable")]:
        add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, jobs=[job(state)], event="confirm_generation", after=effects("blocked", reason), tags=("failure", "confirmation"), rules=("R10",))
    add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, jobs=[job("failed", owner=202)], event="confirm_generation", after=effects("blocked", "job_not_found"), tags=("target", "confirmation"), rules=("R09",))
    add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, jobs=[job("failed")], event="revision_race", details={"when": "after_prepare_before_confirm", "external_revision": 6, "external_settings": {"subtitle_font_size": 80}, "then": "confirm_old_retry"}, after=stale, tags=("race",), rules=("R08",))
    add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, settings={"subtitle_font_size": 72}, jobs=[job("failed")], event="confirm_twice", after=retried, tags=("confirmation", "resend"), rules=("R05", "R07", "R10"))
    add(8, "ジョブ7は再試行しないで", effects("dismissed", "negated_retry", receipt="none"), interpretation="no_operation", jobs=[job("failed")], tags=("negation",), rules=("R14",))
    add(8, ROOTS[7], retry, name="generation.retry", args={"job_id": 7}, jobs=[job("failed")], event="restart_resend", after={**retry, "receipt_rule": "first_result"}, tags=("resend",), rules=("R05", "R07"), rationale="未確認の再試行要求を再送しても、確認待ちを復元するだけでジョブは作らない。")

    restore = effects("saved", "restore_recorded_settings", {"subtitle_font_size": 40})
    for text in [ROOTS[8], "設定の版2を復元して", "revision 2の設定に戻したい"]:
        add(9, text, restore, name="settings.restore", args={"revision": 2}, tags=("history",), rules=("R11",))
    add(9, "以前の設定に戻して", effects("needs_input", "missing_revision", ask=["revision"], receipt="none"), tags=("history", "omission"), rules=("R11",))
    add(9, "設定を第3版に戻して", effects("blocked", "settings_revision_not_found"), name="settings.restore", args={"revision": 3}, tags=("history", "blocked"), rules=("R11",))
    add(9, "設定を第5版に戻して", {**effects("unchanged", "restore_recorded_even_when_equal"), "revision_delta": 1}, name="settings.restore", args={"revision": 5}, tags=("history", "boundary"), rules=("R11", "R02"), rationale="現在と同じ設定でも、復元した事実を版6として記録する。設定値と成功動画は変えず、生成しない。通常の同値保存とは異なる。")
    add(9, ROOTS[8], blocked, name="settings.restore", args={"revision": 2}, jobs=[job()], status="generating", tags=("history", "blocked"), rules=("R06", "R11"))
    add(9, ROOTS[8], stale, base=4, interpretation="not_called", tags=("history", "race"), rules=("R08",))
    add(9, ROOTS[8], restore, name="settings.restore", args={"revision": 2}, event="restart_resend", after=effects("replayed", "one_restore_revision", {"subtitle_font_size": 40}, receipt="first_result"), tags=("history", "resend"), rules=("R07", "R11"))
    add(9, "第2版には戻さないで", effects("dismissed", "negated_restore", receipt="none"), interpretation="no_operation", tags=("negation", "history"), rules=("R14",))

    unsupported = effects("unsupported", "mail_not_supported", receipt="none")
    for text in [ROOTS[9], "完成動画をメールに添付して送信して", "この映像、メールで共有できる？", "動画を相手のメールアドレスに送ってほしい", "メール送信をお願い", "完成したら動画をメールで届けて", "送信先へメールで動画を渡して", "字幕を64pxに変更してから動画をメールで送って"]:
        add(10, text, unsupported, interpretation="unsupported", tags=("unsupported", "compound") if "64" in text else ("unsupported",), rules=("R14",), rationale="メール操作は自然言語ツールの対象外。生成や設定保存に置換せず、複合依頼を部分実行しない。")
    add(10, "動画をメールで送らないで", effects("dismissed", "negated_send", receipt="none"), interpretation="no_operation", tags=("negation", "unsupported"), rules=("R14",))
    add(10, ROOTS[9], unsupported, interpretation="unsupported", status="generating", jobs=[job()], tags=("unsupported", "blocked"), rules=("R14",), rationale="生成中でも非対応操作を対応済みの操作へ置換しない。非対応と実行待ちを区別する。")
    if len(rows) != 100:
        raise ValueError("development target is exactly 100 cases")
    return reviewed_state_details(rows)


def reviewed_state_details(rows: list[Case]) -> list[Case]:
    """Independent specification review: make each synthetic state self-contained."""
    result: list[Case] = []
    for case in rows:
        data = case.model_dump(mode="json")
        number = int(case.case_id[-3:])
        initial = data["initial"]
        expected = data["expected"]
        # Stored readings use the specialist schema's explicit default. Proposal
        # encodings with omitted/null accent both remain acceptable above.
        for effect in (expected["submit"], expected["after_event"]):
            if effect:
                for reading in effect["settings_delta"].get("pronunciation_overrides", []):
                    reading.setdefault("accent", None)
        if number in {23, 25}:
            initial["jobs"] = [{**job("failed" if number == 23 else "pending"),
                                "input_revision": 5, "input_settings": copy.deepcopy(initial["settings"])}]
        if any(item["project_id"] == 101 and item["status"] in {"pending", "running"}
               for item in initial["jobs"]):
            initial["project_status"] = "generating"
        for item in initial["jobs"]:
            if item["project_id"] == initial["project_id"] and item["status"] in {"pending", "running"}:
                item["input_revision"] = initial["revision"]
                item["input_settings"] = copy.deepcopy(initial["settings"])
        if number in {57, 74}:
            initial["jobs"][0]["input_revision"] = initial["revision"]
            initial["jobs"][0]["input_settings"] = copy.deepcopy(initial["settings"])
        if number == 72:
            initial["jobs"][0]["cancel_requested"] = True
        if number == 16:
            initial["prior_turns"][0]["proposal"] = {"kind": "operation", **proposals("set", {"subtitle_font_size": 56})[0]}
        if number == 50:
            initial["prior_turns"][0]["proposal"] = {"kind": "operation", **proposals("settings", {"subtitle_font_size": 72, "voicevox_speed_scale": 1.2})[0]}
        if number == 55:
            initial["prior_turns"][0]["proposal"] = {"kind": "operation", **proposals("generation.start", {})[0]}
        for prior in initial["prior_turns"]:
            if prior["question"] and prior["proposal"] is None:
                prior["proposal"] = {"kind": "clarification", "question": prior["question"], "missing_fields": ["arguments"]}
        if number in {66, 67, 68}:
            item = initial["jobs"][0]
            expected["submit"]["job_assertions"] = {"job_id": item["id"], "status": item["status"],
                                                       "cancel_requested": item["cancel_requested"]}
            if number == 68:
                expected["submit"]["job_assertions"]["no_future_publication"] = True
        if number in {71, 72, 78}:
            expected["after_event"]["job_assertions"]["input_settings"] = copy.deepcopy(initial["settings"])
            expected["after_event"]["job_assertions"]["count"] = 1
        if number == 37:
            data["event"]["details"]["external_settings"] = {"subtitle_font_size": 80,
                "pronunciation_overrides": [{"surface": "API", "reading": "エーピーアイ", "accent": None}]}
            expected["rationale"] = "両設定は版6へ保存済み。その後別の保存が字幕を80pxへ変更し版7となる。旧版6の生成確認は拒否し、後続保存を上書きしない。settings_deltaは当該依頼が版6で行った変更だけを表す。"
        if number in {58, 77}:
            action = "生成" if number == 58 else "再試行"
            expected["rationale"] = f"版5で{action}を確認待ちにした後、別の保存が字幕を80pxへ変更して版6となる。古い確認は拒否し、版6・80pxを保持する。この依頼による設定変更と新しいジョブは0件。"
        target_description = "プロジェクト未選択。参照用の状態はプロジェクト101" if data["request"]["target_project_id"] is None else "選択中はプロジェクト101"
        job_description = "関連ジョブなし" if not initial["jobs"] else "関連ジョブ: " + "、".join(
            f"ID{item['id']}・所有{item['project_id']}・{item['status']}・入力版{item['input_revision']}・取消要求{item['cancel_requested']}"
            for item in initial["jobs"])
        data["situation"] = (f"{target_description}。現在版5、字幕{initial['settings']['subtitle_font_size']}px、"
            f"話速{initial['settings']['voicevox_speed_scale']}倍、状態{initial['project_status']}。"
            f"成功動画は版2と版4で、最新動画は現在設定に未反映。{job_description}。")
        if initial["prior_turns"]:
            data["situation"] += "先行会話はprior_turnsに記録し、今回の回答・訂正・取り下げは別IDで結び付ける。"
        if "external_revision" in data["event"]["details"]:
            data["situation"] += (f"event.detailsの時点で別の保存が版{data['event']['details']['external_revision']}へ更新する。"
                "その変更は当該依頼の増分に含めない。")
        result.append(Case.model_validate(data))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for case in cases:
            stream.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False) + "\n")
    print(f"created {len(cases)} development cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
