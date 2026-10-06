"""Separately attest the fixed D40 decision source closure (this step may use Git)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

from evaluation.release_decision import attest_and_validate_decision_tool

_REFUSED = "decision source attestation refused"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(_REFUSED, file=sys.stderr)
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(add_help=False)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        attest_and_validate_decision_tool(repo_root=args.repo_root, output_path=args.output,
                                          expected_sha256=args.expected_sha256)
    except (OSError, ValueError, subprocess.SubprocessError):
        print(_REFUSED, file=sys.stderr)
        return 2
    print("decision source attestation published")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
