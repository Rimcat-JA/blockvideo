"""D39 materialization and detached marker-bound cleanup entry point."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from evaluation.runtime_materialization import cleanup_candidate_runtime, materialize_candidate_runtime


class _RedactedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError("invalid materialization arguments")


def build_parser() -> argparse.ArgumentParser:
    parser = _RedactedParser(description="Materialize or clean one immutable candidate runtime")
    actions = parser.add_subparsers(dest="action")
    cleanup = actions.add_parser("cleanup")
    cleanup.add_argument("--runtime-root", type=Path, required=True)
    cleanup.add_argument("--work-root", type=Path, required=True)
    cleanup.add_argument("--materialization", type=Path, required=True)
    cleanup.add_argument("--expected-materialization-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Emit fixed statuses, never input paths or failure tracebacks."""
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.action == "cleanup":
            cleanup_candidate_runtime(runtime_root=arguments.runtime_root, work_root=arguments.work_root, materialization_path=arguments.materialization, expected_materialization_sha256=arguments.expected_materialization_sha256)
            print("cleanup completed")
        else:
            if any(getattr(arguments, name) is None for name in ("candidate_root", "freeze_manifest", "work_root", "output")):
                raise ValueError("incomplete materialization arguments")
            result = materialize_candidate_runtime(candidate_root=arguments.candidate_root, freeze_manifest_path=arguments.freeze_manifest, work_root=arguments.work_root, output_path=arguments.output)
            print("materialized " + result.runtime_instance_id)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError):
        print("materialization rejected")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
