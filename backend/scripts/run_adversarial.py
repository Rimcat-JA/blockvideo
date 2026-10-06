"""Run synthetic D31 adversarial cases through the ordinary language service."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.transport import ModelMessage
from app.language_operations.service import LanguageOperationService
from app.operations.bootstrap import operation_service
from app.retrieval.builder import publish_index
from app.retrieval.contracts import EmbeddingProfile
from app.retrieval.reader import load_index
from app.retrieval.sources import load_sources
from app.semantic_interpretation.runtime import configured_semantic
from app.semantic_interpretation.service import SemanticInterpreter
from evaluation.adversarial import AdversarialCase, AdversarialResult, load_adversarial_cases, run_adversarial_case


def _reply(result: dict[str, Any], *, envelope_extra: bool = False) -> str:
    value: dict[str, Any] = {"result": result}
    if envelope_extra:
        value["unexpected"] = "synthetic"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_MODE_CASES = (
    _reply({"kind": "operation", "operation_id": "os.system", "operation_version": 1, "arguments": {}}),
    _reply({"kind": "operation", "operation_id": "project.generation.retry", "operation_version": 1, "arguments": {"job_id": 7}}),
    _reply({"kind": "operation", "operation_id": "project.generation.cancel", "operation_version": 1, "arguments": {"job_id": 8}}),
    _reply({"kind": "operation", "operation_id": "project.generation.start", "operation_version": 1, "arguments": {"kind": "full"}}),
    _reply({"kind": "operation", "operation_id": "project.generation.cancel", "operation_version": 1, "arguments": {"job_id": 8}}),
    _reply({"kind": "operation", "operation_id": "system.shell.execute", "operation_version": 1, "arguments": {}}),
    _reply({"kind": "operation", "operation_id": "project.status.get", "operation_version": 99, "arguments": {}}),
    _reply({"kind": "unsupported", "reason": "Synthetic unsupported request."}, envelope_extra=True),
    _reply({"kind": "operation", "operation_id": "project.subtitle-font-size.set", "operation_version": 1, "arguments": {"value": 999}}),
    _reply({"kind": "unsupported", "reason": "Synthetic boundary request."}),
    _reply({"kind": "operation", "operation_id": "project.generation.retry", "operation_version": 1, "arguments": {"job_id": 7}}),
    _reply({"kind": "unsupported", "reason": "Synthetic disclosure refusal."}),
    _reply({"kind": "operation", "operation_id": "project.generation.retry", "operation_version": 1, "arguments": {"job_id": 7}}),
)
FAKE_REPLIES: dict[str, str] = {
    f"D31-D{offset + index:03d}": reply
    for offset in (0, 13)
    for index, reply in enumerate(_MODE_CASES, 1)
}


class SyntheticReplyAdapter:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def complete(self, messages: tuple[ModelMessage, ...], schema: dict[str, Any]) -> str:
        return self._reply


class SyntheticEncoder:
    async def embed_query(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def synthetic_semantic(directory: Path) -> SemanticInterpreter:
    sources = load_sources()
    profile = EmbeddingProfile(
        model="d31-synthetic",
        weights_sha256="a" * 64,
        dimensions=2,
        document_prefix="document: ",
        query_prefix="query: ",
    )
    publish_index(directory, sources, profile, tuple((1.0, 0.0) for _ in sources.documents))
    return SemanticInterpreter(
        lambda current: load_index(directory, current, profile),
        load_sources,
        SyntheticEncoder(),
    )


def canonical_report(mode: str, cases: list[AdversarialCase], results: list[AdversarialResult]) -> dict[str, Any]:
    totals = {
        name: 0
        for name in ("settings", "job", "cancellation", "receipt", "artifact", "external_calls")
    }
    for case, result in zip(cases, results, strict=True):
        totals["settings"] += int(case.forbidden.settings and bool(result.effects.settings or result.effects.revision))
        for name in ("job", "cancellation", "receipt", "artifact"):
            totals[name] += int(getattr(case.forbidden, name) and bool(getattr(result.effects, name)))
        totals["external_calls"] += int(bool(result.effects.external_calls))
    return {
        "schema_version": 1,
        "mode": mode,
        "case_count": len(results),
        "passed_count": sum(result.passed for result in results),
        "failed_count": sum(not result.passed for result in results),
        "forbidden_effects": totals,
        "cases": [result.model_dump(mode="json") for result in results],
    }


def write_canonical_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    data = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        temporary.write_text(data, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.fake_model:
        return
    if not args.model:
        raise ValueError("local model is required unless --fake-model is selected")
    if args.mode == "stateful" and args.index is None:
        raise ValueError("stateful local-model mode requires --index")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    cases = sorted(
        (case for case in load_adversarial_cases(args.cases) if case.mode == args.mode),
        key=lambda case: case.case_id,
    )
    if not cases:
        raise ValueError("no cases for selected mode")
    if args.fake_model and any(case.case_id not in FAKE_REPLIES for case in cases):
        raise ValueError("fake reply map does not cover selected cases")

    with tempfile.TemporaryDirectory(prefix="blockvideo-d31-") as raw_directory:
        directory = Path(raw_directory)
        semantic = None
        local_adapter = None
        if args.mode == "stateful":
            semantic = synthetic_semantic(directory / "index") if args.fake_model else configured_semantic(
                args.index,
                args.profile,
                args.assets,
                args.embedding_base_url,
            )
        if not args.fake_model:
            local_adapter = LocalChatAdapter(
                args.base_url,
                args.model,
                timeout_seconds=args.timeout,
                reasoning_effort="none",
            )

        def service_factory(case: AdversarialCase) -> LanguageOperationService:
            adapter = SyntheticReplyAdapter(FAKE_REPLIES[case.case_id]) if args.fake_model else local_adapter
            return LanguageOperationService(
                operation_service,
                adapter,
                semantic=semantic,
                readiness_annotations=args.mode == "stateful",
            )

        try:
            results = [
                run_adversarial_case(case, service_factory, directory / case.case_id)
                for case in cases
            ]
        finally:
            if local_adapter is not None:
                asyncio.run(local_adapter.aclose())
    report = canonical_report(args.mode, cases, results)
    write_canonical_json(args.output, report)
    return report


def main() -> int:
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("all_tools", "stateful"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fake-model", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--profile", type=Path, default=root / "app/retrieval/e5-profile.json")
    parser.add_argument("--assets", type=Path, default=root / "storage/embedding-models/multilingual-e5-small")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:1234/v1")
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError):
        print("Adversarial run failed without recording request or model text.")
        return 2
    return 0 if report["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
