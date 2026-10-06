"""Synthetic local failures: no server shutdown, fallback, DB, or execution."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.interpretation.connection import inspect_connection
from app.interpretation.contracts import CandidateRef, InterpretationInput
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.service import Interpreter
from app.operations.catalog import load_catalog


async def run(model: str, output: Path) -> None:
    if output.exists():
        raise ValueError("Choose a fresh output path")
    catalog = load_catalog(Path(__file__).resolve().parents[1] / "app/operations/definitions.json")
    records: list[dict[str, Any]] = []
    for name, url, chosen in [("unreachable", "http://127.0.0.1:1235/v1", model),
                              ("model_missing", "http://127.0.0.1:1234/v1", "blockvideo-d23-absent-model")]:
        view = await inspect_connection(url, chosen, check=True)
        async with LocalChatAdapter(url, chosen, reasoning_effort="none", timeout_seconds=10) as adapter:
            result = await Interpreter(catalog, adapter, timeout_seconds=10).preview(InterpretationInput(
                text="動画の状態を教えて", candidates=(CandidateRef(operation_id="project.status.get"),)))
        records.append({"case": name, "connection": view.model_dump(), "interpretation": result.model_dump(),
                        "passed": view.status == name and result.status == "error" and result.attempts == 1})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"synthetic_only": True, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    assert all(record["passed"] for record in records), "Inspect the observed failure evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.output))


if __name__ == "__main__":
    main()
