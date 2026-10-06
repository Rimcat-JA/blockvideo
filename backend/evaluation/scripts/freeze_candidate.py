"""CLI for deterministic D36 candidate freeze evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from evaluation.release_candidate.freeze import freeze_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze the exact detached D35 candidate")
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--candidate-control", required=True, type=Path)
    parser.add_argument("--expected-candidate-control-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest = freeze_candidate(
        candidate_root=arguments.candidate_root,
        candidate_control_path=arguments.candidate_control,
        expected_candidate_control_sha256=arguments.expected_candidate_control_sha256,
        output_root=arguments.output_root,
    )
    result = {
        "candidate_id": manifest.candidate_id,
        "completion_marker": f"{manifest.candidate_id}/.d36-publication-state",
        "freeze_manifest": f"{manifest.candidate_id}/freeze-manifest.json",
        "tool_attestation": f"{manifest.candidate_id}/d36-tool-attestation.json",
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
