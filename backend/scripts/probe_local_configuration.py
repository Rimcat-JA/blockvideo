"""D23 reproducible synthetic-only local inference measurements, no operations."""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from app.interpretation.contracts import CandidateRef, DialogueContextTurn, InterpretationInput, MinimalState
from app.interpretation.errors import InterpretationError
from app.interpretation.local_chat import MAX_HTTP_RESPONSE_BYTES, LocalChatAdapter
from app.interpretation.service import Interpreter, _SYSTEM
from app.operations.catalog import load_catalog
from scripts.probe_interpretation import CASES

CATALOG = Path(__file__).resolve().parents[1] / "app/operations/definitions.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def memory_snapshot() -> dict[str, int] | None:
    """Windows system-wide memory, not a model-only or GPU allocation estimate."""
    if sys.platform != "win32":
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("load_percent", ctypes.c_uint32),
                    *[(name, ctypes.c_uint64) for name in (
                        "total_physical", "available_physical", "commit_limit", "available_commit",
                        "total_virtual", "available_virtual", "available_extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return {name: int(getattr(status, name)) for name in (
        "load_percent", "total_physical", "available_physical", "commit_limit", "available_commit")}


class MeasuredTransport(httpx.AsyncHTTPTransport):
    """Probe-owned bounded evidence capture; never installed in the application."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        record: dict[str, Any] = {"request_bytes": len(request.content),
            "messages_sha256": digest(payload["messages"]),
            "schema_sha256": digest(payload["response_format"]), "memory_before": memory_snapshot()}
        self.calls.append(record)
        started = perf_counter()
        response = await super().handle_async_request(request)
        data = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > MAX_HTTP_RESPONSE_BYTES:
                    raise InterpretationError("response_too_large")
        finally:
            await response.aclose()
            record.update(elapsed_seconds=round(perf_counter() - started, 3), memory_after=memory_snapshot())
        record["http_status"] = response.status_code
        if response.status_code == 200:
            body = json.loads(data)
            record.update(reported_model=body.get("model"), usage=body.get("usage"), choices=body.get("choices"))
        return httpx.Response(response.status_code, headers=response.headers, content=bytes(data), request=request)


def operation(name: str, arguments: dict[str, Any], version: int = 1, generate: bool = False) -> dict[str, Any]:
    return {"kind": "operation", "operation_id": f"project.{name}", "operation_version": version,
            "arguments": arguments, "generate_after_save": generate}


def cases() -> list[tuple[str, str, dict[str, Any], tuple[DialogueContextTurn, ...]]]:
    reading = {"surface": "API", "reading": "エーピーアイ", "accent": None}
    compound = operation("settings.update", {"subtitle_font_size": 64, "pronunciation_overrides": [reading]}, generate=True)
    question = DialogueContextTurn(text="字幕を64pxにして、APIの読み方を登録して。作り直して",
        status="needs_input", proposal={"kind": "clarification", "question": "APIは何と読みますか？", "missing_fields": ["arguments"]},
        question="APIは何と読みますか？")
    saved = DialogueContextTurn(text="字幕を64pxにして、APIをエーピーアイと読んで。作り直して",
                               status="ready", proposal=compound, settings_saved=True)
    return [
        ("absolute_compound", saved.text, compound, ()),
        ("relative_compound", "字幕を少し大きくして、速度を1.2倍に。生成はしないで",
         operation("settings.update", {"settings": {"voicevox_speed_scale": 1.2}, "subtitle_font_size_delta": 2}, 2), ()),
        ("reading_missing", question.text, {"kind": "clarification"}, ()),
        ("short_answer", "エーピーアイ", compound, (question,)),
        ("saved_correction", "違う、少し小さく", operation("subtitle-font-size.adjust", {"delta": -2}), (saved,)),
        ("size_missing", "字幕を大きくして、速度を1.2倍にして、動画を作って", {"kind": "clarification"}, ()),
        ("unsupported_mixed", "字幕を64pxにして、友達にメールを送って", {"kind": "unsupported"}, ()),
        ("restore_missing", "以前の設定に戻して", {"kind": "clarification"}, ()),
        ("cancel_missing", "生成をキャンセルして", {"kind": "clarification"}, ()),
        ("generation", "今の設定で動画を作り直して", operation("generation.start", {"kind": "full"}), ()),
    ]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("Choose a fresh evidence path; previous runs are immutable")
    catalog = load_catalog(CATALOG)
    full = tuple(CandidateRef(operation_id=d.operation_id, operation_version=d.operation_version) for d in catalog.definitions)
    small = tuple(r for r in full if r.operation_id in {
        "project.subtitle-font-size.set", "project.subtitle-font-size.adjust", "project.status.get"})
    basic = [(name, text, {"kind": kind, **({"operation_id": op, "arguments": arguments,
              "generate_after_save": False} if op else {})}, ())
             for name, (text, kind, op, arguments) in CASES.items()]
    queue = [("small", small, case) for case in basic] + [("full", full, case) for case in basic + cases()]
    evidence: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(),
        "synthetic_only": True, "executes_operations": False, "model": args.model, "base_url": args.base_url,
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(_SYSTEM.encode()).hexdigest(),
        "parameters": {"context_observed_separately": True, "temperature": 0, "max_tokens": 768, "reasoning_effort": "none"},
        "memory_scope": "Windows system-wide snapshots before/after each call; not peak, GPU, or model-only memory",
        "records": []}
    transport = MeasuredTransport()
    async with LocalChatAdapter(args.base_url, args.model, reasoning_effort="none", transport=transport) as adapter:
        interpreter = Interpreter(catalog, adapter)
        for group, refs, (name, text, expected, dialogue) in queue:
            request = InterpretationInput(text=text, candidates=refs, dialogue=dialogue,
                state=MinimalState(selected_project_id=101, revision=7, subtitle_font_size=64, status="completed"))
            first_call = len(transport.calls)
            started = perf_counter()
            actual = (await interpreter.preview(request)).model_dump(mode="json")
            proposal = actual.get("proposal") or {}
            passed = all(proposal.get(k) == v for k, v in expected.items())
            evidence["records"].append({"id": f"{group}-{name}", "candidate_count": len(refs),
                "request": request.model_dump(mode="json"), "expected": expected, "actual": actual,
                "passed": passed, "elapsed_seconds": round(perf_counter() - started, 3),
                "calls": transport.calls[first_call:]})
            evidence.update(completed=len(evidence["records"]), total=len(queue),
                            passed=sum(r["passed"] for r in evidence["records"]))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f'{group}-{name}: {"PASS" if passed else "FAIL"} ({actual["attempts"]} calls)', flush=True)
    evidence["latency_seconds"] = {"median": statistics.median(r["elapsed_seconds"] for r in evidence["records"]),
                                    "max": max(r["elapsed_seconds"] for r in evidence["records"])}
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = asyncio.run(run(args))
    return int(evidence["passed"] != evidence["total"])


if __name__ == "__main__":
    raise SystemExit(main())
