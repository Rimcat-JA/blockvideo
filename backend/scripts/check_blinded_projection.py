"""Check that every held-out case projects into the D37 wire contract.

Output names only case IDs, field locations, key names and identifier-shaped
labels, so an evaluator can share it with the implementation agent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.corpus import load_cases
from evaluation.projection_diagnostics import projection_failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        cases = load_cases(arguments.corpus)
    except (OSError, ValueError):
        print("Corpus could not be loaded without exposing its content.", file=sys.stderr)
        return 2
    failures = [item for item in map(projection_failure, cases) if item is not None]
    report = {"case_count": len(cases), "projected": len(cases) - len(failures), "failures": failures}
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
