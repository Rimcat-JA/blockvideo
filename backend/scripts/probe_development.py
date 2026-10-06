"""Explicit real local inference on human+AI approved development cases only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from app.interpretation.contracts import CandidateRef
from app.interpretation.candidates import candidate_payload, response_schema
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation import service as interpretation_service
from app.operations.catalog import load_catalog
from evaluation.corpus import case_digest, corpus_digest
from evaluation.development_probe import approved_development, interpretation_input, proposal_matches

CATALOG = Path(__file__).resolve().parents[1] / "app/operations/definitions.json"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    cases, gate = approved_development(args.cases, args.human, args.ai)
    catalog = load_catalog(CATALOG)
    refs = tuple(CandidateRef(operation_id=d.operation_id, operation_version=d.operation_version) for d in catalog.definitions)
    if args.prompt_file:
        # Development-only baseline replay, never a product setting or API input.
        interpretation_service._SYSTEM = args.prompt_file.read_text(encoding="utf-8")
    args.output.mkdir(parents=True)
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(), "scope": "approved_development_interpretation_only",
        "approval_gate": gate, "eligible_subset_sha256": corpus_digest(cases),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(interpretation_service._SYSTEM.encode()).hexdigest(),
        "response_schema_ordered_sha256": hashlib.sha256(json.dumps(response_schema(catalog.definitions), ensure_ascii=False).encode()).hexdigest(),
        "candidate_payload_ordered_sha256": hashlib.sha256(json.dumps([candidate_payload(d) for d in catalog.definitions], ensure_ascii=False).encode()).hexdigest(),
        "model": args.model, "base_url": args.base_url,
        "parameters": {"temperature": 0, "max_tokens": 768, "reasoning_effort": "none"},
        "records": [], "score_scope": "proposal kind and accepted operation/arguments; not clarification quality, target resolution or after-event effects",
        "held_out_used": False, "executes_operations": False, "repeats": args.repeats,
    }
    async with LocalChatAdapter(args.base_url, args.model, reasoning_effort="none") as adapter:
        interpreter = interpretation_service.Interpreter(catalog, adapter)
        transport_failures = 0
        for repeat in range(args.repeats):
            for case in cases:
                if case.expected.interpretation == "not_called":
                    report["records"].append({"case_id": case.case_id, "repeat": repeat,
                        "case_sha256": case_digest(case), "skipped": "application_precheck_no_model_expected"})
                    continue
                started = perf_counter()
                actual = await interpreter.preview(interpretation_input(case, refs))
                record = {"case_id": case.case_id, "case_sha256": case_digest(case), "repeat": repeat,
                    "candidate_count": len(refs), "actual": actual.model_dump(mode="json"),
                    "proposal_match": proposal_matches(case, actual.proposal),
                    "elapsed_seconds": round(perf_counter() - started, 3)}
                report["records"].append(record)
                (args.output / "progress.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f'{case.case_id} repeat={repeat + 1} match={record["proposal_match"]} status={actual.status}', flush=True)
                code = actual.failure.reason_code if actual.failure else None
                transport_failures = transport_failures + 1 if code in {"http_error", "timeout"} else 0
                if transport_failures >= 2:
                    report["stopped_reason"] = "repeated_transport_failure"
                    break
                if actual.status == "error" and actual.failure and actual.failure.reason_code in {"connection_failed", "model_mismatch", "configuration_error"}:
                    report["stopped_reason"] = actual.failure.reason_code
                    break
            if report.get("stopped_reason"):
                break
    tested = [r for r in report["records"] if "skipped" not in r]
    latencies = sorted(r["elapsed_seconds"] for r in tested)
    report["summary"] = {"eligible": len(cases), "planned_model_cases": sum(c.expected.interpretation != "not_called" for c in cases) * args.repeats,
        "measured": len(tested), "matched": sum(r["proposal_match"] for r in tested),
        "skipped_prechecks": sum("skipped" in r for r in report["records"]),
        "calls": sum(r["actual"]["attempts"] for r in tested),
        "errors": sum(r["actual"]["status"] == "error" for r in tested),
        "latency_median_seconds": statistics.median(latencies) if latencies else None,
        "latency_max_seconds": max(latencies) if latencies else None}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "human", "ai", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except (ValueError, OSError):
        print("Development probe stopped. No final evaluation score was produced.")
        return 2
    return 2 if report.get("stopped_reason") else 0


if __name__ == "__main__":
    raise SystemExit(main())
