"""Fixed D39 final verifier entry point; no caller-selected commands or tool closure."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import NoReturn, Sequence

from evaluation.release_verification import verify_release_candidate


class _RedactedParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("invalid verification arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _RedactedParser(description="Verify one bound immutable release candidate")
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--expected-materialization-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print only fixed public statuses; failed verification never exits zero."""
    try:
        arguments = build_parser().parse_args(argv)
        result = verify_release_candidate(candidate_root=arguments.candidate_root, freeze_manifest_path=arguments.freeze_manifest, runtime_root=arguments.runtime_root, materialization_path=arguments.materialization, expected_materialization_sha256=arguments.expected_materialization_sha256, work_root=arguments.work_root, output_dir=arguments.output, smoke_manifest_path=arguments.smoke_manifest)
        print("verification " + result.status)
        return 0 if result.status == "passed" else 2
    except (OSError, ValueError, subprocess.SubprocessError):
        print("verification rejected")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
