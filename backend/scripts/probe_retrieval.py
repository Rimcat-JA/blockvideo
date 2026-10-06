"""Approved development-only Recall@k and optional real semantic interpretation."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.interpretation import service as interpretation_service
from app.interpretation.contracts import CandidateRef
from app.interpretation.local_chat import LocalChatAdapter
from app.operations.catalog import load_catalog
from app.retrieval.contracts import EmbeddingProfile, OperationRef, SearchScope
from app.retrieval.embeddings import LocalEmbeddingAdapter
from app.retrieval.onnx_embeddings import OnnxEmbeddingAdapter
from app.retrieval.ranking import rank_operations
from app.retrieval.reader import load_index
from app.retrieval.serialization import digest
from app.retrieval.sources import DEFAULT_CATALOG, load_sources
from app.semantic_interpretation.service import SemanticInterpreter, retrieval_query
from evaluation.corpus import case_digest, corpus_digest
from evaluation.development_probe import approved_development, interpretation_input, proposal_matches

KS = (1, 3, 5, 6, 8, 9)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in records if "skipped" not in r]
    positives = [r for r in measured if r["expected_operation"]]
    traces = [r["trace"] for r in measured if "trace" in r]
    latencies = [t["elapsed_ms"] for t in traces]
    return {
        "measured": len(measured), "skipped_prechecks": len(records) - len(measured),
        "operation_cases": len(positives),
        "recall_hits": {str(k): sum(r["hits"][str(k)] for r in positives) for k in KS},
        "recall_denominator": len(positives),
        "proposal_matched": sum(r.get("proposal_match", False) for r in measured) if traces else None,
        "proposal_denominator": len(traces),
        "embedding_calls": sum(t["embedding_calls"] for t in traces) if traces else len(measured),
        "chat_calls": sum(t["chat_calls"] for t in traces),
        "expansions": sum(t["expansion_count"] for t in traces),
        "all_tools_fallbacks": sum(t["all_tools_count"] for t in traces),
        "request_bytes": sum(s["request_bytes"] for t in traces for s in t["stages"]),
        "response_bytes": sum(s["response_bytes"] for t in traces for s in t["stages"]),
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_max_ms": max(latencies) if latencies else None,
        "errors": sum(r.get("actual", {}).get("status") == "error" for r in measured),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    cases, gate = approved_development(args.cases, args.human, args.ai)
    catalog, sources = load_catalog(DEFAULT_CATALOG), load_sources()
    refs = tuple(CandidateRef(operation_id=d.operation_id, operation_version=d.operation_version) for d in catalog.definitions)
    profile = EmbeddingProfile.model_validate_json(args.profile.read_bytes())
    index = load_index(args.index, sources, profile)
    encoder = OnnxEmbeddingAdapter(profile, args.assets) if profile.transport == "local-onnx-e5-v1" else LocalEmbeddingAdapter(profile, args.base_url)
    scope = SearchScope(app_id=sources.app_id,
        capabilities=tuple(sorted({c for d in sources.documents for c in d.required_capabilities})),
        operations=tuple(OperationRef(operation_id=r.operation_id, operation_version=r.operation_version) for r in refs))
    runner = SemanticInterpreter(lambda s: load_index(args.index, s, profile), load_sources, encoder, scope=scope)
    args.output.mkdir(parents=True)
    report: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "approved_development_only", "approval_gate": gate, "eligible_subset_sha256": corpus_digest(cases),
        "profile": profile.model_dump(mode="json"), "index_manifest": index.manifest.model_dump(mode="json"),
        "prompt_sha256": digest(interpretation_service._SYSTEM.encode()), "model": args.model,
        "parameters": {"temperature": 0, "max_tokens": 768, "reasoning_effort": "none"},
        "implementation_sha256": {name: digest((Path(__file__).parents[1] / name).read_bytes()) for name in (
            "app/semantic_interpretation/service.py", "app/retrieval/ranking.py", "app/interpretation/candidates.py")},
        "records": [], "held_out_used": False, "executes_operations": False,
        "metrics": "Recall: explicit acceptable operation versions; proposal match: kind and arguments only. Not end-to-end effects or clarification quality. Byte counts are not tokens.",
    }
    async with encoder, LocalChatAdapter(args.base_url, args.model or "unused", reasoning_effort="none") as chat:
        for case in cases:
            record: dict[str, Any] = {"case_id": case.case_id, "case_sha256": case_digest(case)}
            if case.expected.interpretation == "not_called":
                record["skipped"] = "application_precheck_no_model_expected"
                report["records"].append(record)
                continue
            request = interpretation_input(case, refs)
            if args.model:
                result = await runner.preview(catalog, chat, request)
                ranking = result.trace.ranking
                record.update(actual=result.interpretation.model_dump(mode="json"), trace=result.trace.model_dump(mode="json"),
                    proposal_match=proposal_matches(case, result.interpretation.proposal))
            else:
                vector = await encoder.embed_query(retrieval_query(request))
                ranking = rank_operations(index, vector, scope, load_sources())
            expected = {(p.operation_id, p.operation_version) for p in case.expected.operations}
            record.update(expected_operation=bool(expected), ranking=[r.model_dump(mode="json") for r in ranking],
                hits={str(k): bool(expected.intersection(r.key for r in ranking[:k])) if expected else None for k in KS})
            report["records"].append(record)
            (args.output / "progress.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f'{case.case_id} proposal_match={record.get("proposal_match", "not_measured")}', flush=True)
            if args.model and result.interpretation.failure and result.interpretation.failure.reason_code in {
                "timeout", "http_error", "connection_failed", "model_mismatch", "configuration_error", "retrieval_deadline", "retrieval_integrity_failed"}:
                report["stopped_reason"] = result.interpretation.failure.reason_code
                break
    report["summary"] = summarize(report["records"])
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "human", "ai", "output", "index", "profile"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("storage/embedding-models/multilingual-e5-small"))
    parser.add_argument("--model", help="Omit for embedding recall only; supplying a model makes real local chat calls")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except (ValueError, OSError):
        print("Development retrieval probe stopped; no final evaluation score was produced.")
        return 2
    return 2 if report.get("stopped_reason") else 0


if __name__ == "__main__":
    raise SystemExit(main())
