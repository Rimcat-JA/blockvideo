"""Run the five D29 modes on human/AI-approved development cases only."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import platform
import statistics
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.interpretation.local_chat import LocalChatAdapter, local_base_url
from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.onnx_embeddings import OnnxEmbeddingAdapter
from app.retrieval.reader import load_index
from app.retrieval.serialization import canonical, digest
from app.retrieval.sources import load_sources
from app.semantic_interpretation.service import SemanticInterpreter
from evaluation.comparison import MODES, POLICY
from evaluation.comparison_runner import run_trial
from evaluation.contracts import Case
from evaluation.corpus import corpus_digest
from evaluation.development_probe import approved_development, unreviewed_development


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parents[1]
    # Include the common guards/core/schema as well as the varying selector.
    files = [p for folder in ("app", "evaluation", "scripts") for p in (root / folder).rglob("*.py")]
    files.extend((root / "app/operations").glob("*.json"))
    return {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(files)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MODES:
        rows = [r for r in records if r["mode"] == mode]
        calls = [c for r in rows for c in r["calls"]]
        called = [r for r in rows if r["calls"]]
        timings = [r["response"]["diagnostics"]["interpretation_ms"] for r in called
                   if r["response"]["diagnostics"]["interpretation_ms"] is not None]
        result[mode] = {"trials": len(rows), "submit_effects_match": sum(r["score"]["submit_effects_match"] for r in rows),
            "proposal_match": sum(r["score"]["proposal_match"] for r in rows),
            "model_called_trials": len(called), "model_not_called_trials": len(rows) - len(called),
            "model_proposals_match": sum(r["score"]["proposal_match"] for r in called),
            "chat_calls": len(calls), "request_bytes": sum(c["request_bytes"] for c in calls),
            "response_bytes": sum(c.get("response_bytes", 0) for c in calls),
            "errors": sum(r["response"]["status"] == "error" for r in rows),
            "median_interpretation_ms": statistics.median(timings) if timings else None,
            "interpretation_timing_samples": len(timings),
            "timing_comparable_between_modes": False,
            "timing_limit": "diagnostic only: shared prompt/prefix cache and order effects are uncontrolled",
            "mismatched_cases": [r["case_id"] for r in rows if not r["score"]["submit_effects_match"]],
            "replay_checks_pass": all(r["replay"]["model_calls"] == 0 and r["replay"]["db_unchanged"] for r in rows)}
    return result


def trial_stop_reason(record: dict[str, Any]) -> str | None:
    codes = {c.get("error_code") for c in record["calls"]}
    response = record["response"]
    for failure in (response.get("failure"), (response.get("interpretation") or {}).get("failure")):
        if failure:
            codes.add(failure.get("reason_code"))
    trace = response.get("diagnostics", {}).get("retrieval") or {}
    if codes & {"timeout", "retrieval_deadline"} or trace.get("reason") == "deadline":
        return "interpretation_deadline_exceeded"
    if codes & {"connection_failed", "http_error", "model_mismatch", "configuration_error"}:
        return "local_model_transport_failure"
    return None


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise ValueError("use a new output directory; existing experiments are immutable")
    unreviewed = getattr(args, "unreviewed", False)
    if unreviewed:
        cases, gate = unreviewed_development(args.cases)
    elif args.human is None or args.ai is None:
        raise ValueError("approved runs need --human and --ai; use --unreviewed for an unofficial run")
    else:
        cases, gate = approved_development(args.cases, args.human, args.ai)
    if args.case_id:
        if not set(args.case_id) <= {c.case_id for c in cases}:
            raise ValueError("selected cases must have both current approvals")
        cases = [c for c in cases if c.case_id in args.case_id]
    if not 1 <= args.repeats <= 3 or not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("invalid repeat count or duplicate modes")
    profile = EmbeddingProfile.model_validate_json(args.profile.read_bytes())
    if profile.transport != "local-onnx-e5-v1":
        raise ValueError("D29 comparison uses the fixed D27 local E5 profile")
    sources = load_sources()
    index = load_index(args.index, sources, profile)
    encoder = OnnxEmbeddingAdapter(profile, args.assets)
    semantic = SemanticInterpreter(lambda s: load_index(args.index, s, profile), load_sources, encoder)
    base = local_base_url(args.base_url)
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=10) as http:
        response = await http.get(base + "/models")
        response.raise_for_status()
        if args.model not in {m["id"] for m in response.json()["data"]}:
            raise ValueError("configured model is not advertised by local server")
        instance_details: dict[str, Any] = {"available": False}
        native = await http.get(base.removesuffix("/v1") + "/api/v1/models")
        if native.status_code == 200:
            for model in native.json().get("models", []):
                if model.get("key") == args.model:
                    instance_details = {key: model.get(key) for key in ("key", "quantization", "format", "size_bytes", "loaded_instances")}
    args.output.mkdir(parents=True)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "unreviewed_development_unofficial" if unreviewed else "approved_development_only",
        "held_out_used": False, "approval_gate": gate,
        "selected_case_ids": [c.case_id for c in cases], "selected_subset_sha256": corpus_digest(cases),
        "approval_files_sha256": {name: digest(getattr(args, name).read_bytes()) for name in ("cases", "human", "ai")
                                  if getattr(args, name) is not None},
        "modes": {m: asdict(MODES[m]) for m in args.modes}, "common_policy": POLICY,
        "model": {"id": args.model, "base_url": base, "temperature": 0, "max_tokens": 768, "reasoning_effort": "none",
                  "server_reported_instance": instance_details,
                  "quantization_attestation": "server model identity only; loaded instance evidence recorded separately"},
        "python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ("pydantic", "httpx", "sqlalchemy", "onnxruntime", "tokenizers", "numpy")},
        "embedding_profile": profile.model_dump(mode="json"), "index": index.manifest.model_dump(mode="json"),
        "source_sha256": source_hashes(), "repeats": args.repeats, "order": "case-major, cyclic mode rotation by case and repeat",
        "expected_trials": len(cases) * len(args.modes) * args.repeats,
        "execution": "real shared language service and transactional core; no media dispatcher/providers",
        "fixture_limits": "history media rows are metadata placeholders; prior dialogue is seeded from initial fixture; no real video bytes",
        "scoring": "submit effects and raw proposal agreement separately; events observed where supported, not scored; question wording not scored; not final evaluation",
    }
    write_json(args.output / "manifest.json", manifest)
    return await execute_trials(args, cases, manifest, semantic, encoder,
        LocalChatAdapter(base, args.model, timeout_seconds=180, reasoning_effort="none"))


async def execute_trials(args: argparse.Namespace, cases: list[Case], manifest: dict[str, Any],
                         semantic: Any, encoder: Any, chat: Any) -> dict[str, Any]:
    """Persist a partial report on any failure after the experiment manifest exists."""
    records: list[dict[str, Any]] = []
    stopped = None
    failure_context: dict[str, Any] | None = None
    current_trial = None
    try:
        async with encoder, chat:
            # Asset construction precedes this query timer; it is not cold-load time.
            from time import perf_counter
            started = perf_counter()
            await encoder.embed_query("操作の候補を確認します")
            write_json(args.output / "warmup.json", {"embedding_calls": 1, "milliseconds": round((perf_counter() - started) * 1000), "excluded_from_trial_cost": True})
            for repeat in range(args.repeats):
                for index_case, case in enumerate(cases):
                    offset = (index_case + repeat) % len(args.modes)
                    order = args.modes[offset:] + args.modes[:offset]
                    for mode in order:
                        current_trial = {"case_id": case.case_id, "mode": mode, "repeat": repeat + 1}
                        directory = args.output / f"r{repeat + 1}" / case.case_id / mode.replace("+", "plus")
                        record = await run_trial(case, mode, semantic, chat, directory)
                        record["repeat"] = repeat + 1
                        records.append(record)
                        print(f'{case.case_id} {mode} effects={record["score"]["submit_effects_match"]} calls={len(record["calls"])}', flush=True)
                        write_json(args.output / "progress.json", {"summary_schema_version": 2,
                            "completed_trials": len(records), "expected_trials": manifest["expected_trials"], "summary": summarize(records)})
                        stopped = trial_stop_reason(record)
                        if stopped:
                            failure_context = {"trial": current_trial}
                            break
                    if stopped:
                        break
                if stopped:
                    break
    except Exception as exc:
        stopped = "interpretation_deadline_exceeded" if isinstance(exc, TimeoutError) else "runner_error"
        failure_context = {"trial": current_trial, "exception_type": type(exc).__name__}
    except BaseException as exc:
        stopped = "interrupted"
        failure_context = {"trial": current_trial, "exception_type": type(exc).__name__}
        raise
    finally:
        # Never log exception messages: they may contain paths or provider details.
        same_sources = source_hashes() == manifest["source_sha256"]
        same_initial = all(len({r["initial_state_sha256"] for r in records if r["case_id"] == c.case_id}) <= 1 for c in cases)
        result = {"summary_schema_version": 2, "completed_trials": len(records), "expected_trials": manifest["expected_trials"],
            "complete": len(records) == manifest["expected_trials"] and not stopped and same_sources and same_initial,
            "stopped_reason": stopped, "failure_context": failure_context,
            "sources_unchanged": same_sources, "identical_initial_state_per_case": same_initial,
            "summary": summarize(records), "manifest_sha256": digest(canonical(manifest)), "held_out_used": False,
            "records": [{"case_id": r["case_id"], "mode": r["mode"], "repeat": r["repeat"], "score": r["score"]} for r in records]}
        write_json(args.output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "output", "index", "profile"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("human", "ai"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--unreviewed", action="store_true",
                        help="Unofficial: every development case, approval gate skipped (never an approved score)")
    parser.add_argument("--assets", type=Path, default=Path("storage/embedding-models/multilingual-e5-small"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    try:
        result = asyncio.run(run(args))
    except (ValueError, OSError, httpx.HTTPError):
        print("Comparison stopped before completion; see retained experiment evidence. No final evaluation score.")
        return 2
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
