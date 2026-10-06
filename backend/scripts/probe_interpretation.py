"""D16 local inference probe using only fixed synthetic requests; never executes.

Run from backend: python -m scripts.probe_interpretation --model blockvideo-d16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.interpretation.candidates import response_schema, select_candidates
from app.interpretation.contracts import CandidateRef, InterpretationInput, MinimalState
from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.service import Interpreter
from app.operations.catalog import load_catalog

CATALOG_PATH = Path(__file__).resolve().parents[1] / "app/operations/definitions.json"
CASES: dict[str, tuple[str, str, str | None, dict[str, Any] | None]] = {
    "set": ("選択中のプロジェクトの字幕を56pxにしてください。", "operation",
            "project.subtitle-font-size.set", {"value": 56}),
    "adjust": ("字幕を少し大きくして。", "operation", "project.subtitle-font-size.adjust", {"delta": 2}),
    "status": ("選択中の動画の状態を教えて。", "operation", "project.status.get", {}),
    "clarify": ("字幕の文字サイズを変更して。", "clarification", None, None),
    "unsupported": ("このアプリから友達にメールを送って。", "unsupported", None, None),
}


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """No DB/config/.env/project reads; artifact content is explicitly synthetic."""
    catalog = load_catalog(CATALOG_PATH)
    refs = tuple(CandidateRef(operation_id=name) for name in (
        "project.subtitle-font-size.set", "project.subtitle-font-size.adjust", "project.status.get",
    ))
    definitions = select_candidates(catalog, refs)
    schema = response_schema(definitions)
    records: list[dict[str, Any]] = []
    async with LocalChatAdapter(
        args.base_url, args.model, timeout_seconds=args.timeout,
        reasoning_effort=None if args.reasoning_effort == "default" else "none",
    ) as adapter:
        interpreter = Interpreter(catalog, adapter)
        for name in ([args.case] if args.case != "all" else list(CASES)):
            text, kind, operation_id, arguments = CASES[name]
            request = InterpretationInput(
                text=text, candidates=refs,
                state=MinimalState(selected_project_id=101, revision=7, subtitle_font_size=48, status="completed"),
            )
            started = perf_counter()
            outcome = await interpreter.preview(request)
            proposal = outcome.proposal.model_dump() if outcome.proposal else None
            matches = bool(proposal and proposal["kind"] == kind)
            if operation_id:
                matches = matches and proposal.get("operation_id") == operation_id and proposal.get("arguments") == arguments
            record = {
                "case": name, "synthetic_request": request.model_dump(mode="json"),
                "outcome": outcome.model_dump(mode="json"),
                "expected_kind": kind, "matches_expected": matches,
                "elapsed_seconds": round(perf_counter() - started, 3),
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "base_url": adapter.base_url,
        "data_policy": "fixed synthetic requests only; no database or .env reads",
        "external_api_cost_yen": 0, "sdk": "none (httpx)",
        "connection_parameters": {"max_tokens": adapter.max_tokens,
                                  "timeout_seconds": adapter.timeout_seconds,
                                  "reasoning_effort": adapter.reasoning_effort},
        "schema": schema, "records": records,
        "all_match_expected": all(item["matches_expected"] for item in records),
        "executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--reasoning-effort", choices=["none", "default"], default="none",
                        help="Use none for the verified LM Studio/Ternary Bonsai configuration.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        evidence = asyncio.run(run_probe(args))
    except InterpretationError as exc:
        print(exc.as_view().model_dump_json(), flush=True)
        return 1
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if evidence["all_match_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
