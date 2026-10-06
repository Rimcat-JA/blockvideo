"""Synthetic wording coverage for the D17 All Tools interpreter (no execution)."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.service import Interpreter, _SYSTEM
from app.operations.catalog import load_catalog

# Development cases only, not a held-out evaluation or a general accuracy claim.
CASES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("字幕のサイズを52pxに設定して", "set", {"value": 52}),
    ("テロップの文字を64ピクセルにしたい", "set", {"value": 64}),
    ("字幕、今より4px大きくできる？", "adjust", {"delta": 4}),
    ("字幕を6ピクセル小さくお願いします", "adjust", {"delta": -6}),
    ("字幕を少し大きくして", "adjust", {"delta": 2}),
    ("字幕は少し小さくしてほしい", "adjust", {"delta": -2}),
    ("字幕を大きくして", "needs_input", {}),
    ("字幕の文字サイズを変更して", "needs_input", {}),
    ("字幕をもっと読みやすくしたい", "needs_input", {}),
    ("状態を教えて", "status", {}),
    ("今どこまでできてる？", "status", {}),
    ("動画はもう完成してますか", "status", {}),
    ("現在のプロジェクトの進み具合を確認したい", "status", {}),
    ("作り直して", "generate", {"kind": "full"}),
    ("今の設定でもう一度動画を作ってください", "generate", {"kind": "full"}),
    ("保存してある内容で動画を再生成したい", "generate", {"kind": "full"}),
    ("字幕を58pxに変更するだけで、動画は作らないで", "set", {"value": 58}),
    ("字幕を大きくしてから動画も作り直して", "needs_input", {}),
    ("生成をキャンセルして", "needs_input", {}),
    ("ジョブ7の動画生成をキャンセルして", "cancel", {"job_id": 7}),
    ("失敗したジョブ9を今の設定で再試行して", "retry", {"job_id": 9}),
    ("設定をrevision 2に戻して", "restore", {"revision": 2}),
    ("以前の設定に戻して", "needs_input", {}),
    ("読み上げ速度を1.2に変更して", "settings", {"voicevox_speed_scale": 1.2}),
    ("字幕を非表示にして", "settings", {"subtitle_enabled": False}),
    ("友達にメールを送って", "unsupported", {}),
    ("この動画をSNSに投稿して", "unsupported", {}),
)


def matches(actual: dict[str, Any], expected: str, arguments: dict[str, Any]) -> bool:
    if expected in {"needs_input", "unsupported"}:
        return actual["status"] == expected and not actual["executed"]
    proposal = actual.get("proposal") or {}
    operations = {"set": "subtitle-font-size.set", "adjust": "subtitle-font-size.adjust",
                  "status": "status.get", "generate": "generation.start", "cancel": "generation.cancel",
                  "retry": "generation.retry", "restore": "settings.restore", "settings": "settings.update"}
    if actual["status"] != "proposed":
        return False
    # Equivalent absolute settings operations are valid, but no extra fields are.
    if expected == "set" and proposal.get("operation_id") == "project.settings.update":
        return proposal["arguments"] == {"subtitle_font_size": arguments["value"]}
    return (proposal.get("operation_id") == f"project.{operations[expected]}"
            and proposal.get("arguments") == arguments)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "app/operations/definitions.json")
    refs = tuple(CandidateRef(operation_id=d.operation_id, operation_version=d.operation_version)
                 for d in catalog.definitions)
    records = []
    async with LocalChatAdapter(args.base_url, args.model, reasoning_effort="none") as adapter:
        interpreter = Interpreter(catalog, adapter)
        for text, expected, arguments in CASES:
            result = await interpreter.preview(InterpretationInput(text=text, candidates=refs,
                state=MinimalState(selected_project_id=1, revision=3, subtitle_font_size=56, status="pending")))
            actual = result.model_dump(mode="json")
            passed = matches(actual, expected, arguments)
            records.append({"text": text, "expected": expected, "arguments": arguments,
                            "actual": actual, "passed": passed})
            print(f"{len(records):02d} {'PASS' if passed else 'FAIL'} {expected}", flush=True)
    return {"model": args.model, "mode": "all_tools", "candidate_count": len(refs),
            "system_prompt_sha256": hashlib.sha256(_SYSTEM.encode()).hexdigest(),
            "synthetic_only": True, "executes_operations": False, "external_api_cost_yen": 0,
            "passed": sum(record["passed"] for record in records), "total": len(records), "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if evidence["passed"] == evidence["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
