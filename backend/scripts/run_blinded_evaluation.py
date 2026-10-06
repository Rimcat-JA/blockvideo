"""Run or resume the externally attested D37 blinded evaluation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from evaluation.blinded_runner import run_blinded_evaluation
from evaluation.blinded_io import read_regular
from evaluation.result_contracts import MAX_RESULT_BUNDLE_BYTES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--output", dest="output_root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--evaluator-name", required=True)
    parser.add_argument("--token-key-file", type=Path, required=True)
    parser.add_argument("--embedding-profile", type=Path)
    parser.add_argument("--embedding-base-url")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        bundle = asyncio.run(run_blinded_evaluation(**vars(arguments)))
    except (OSError, ValueError):
        print(
            "Blinded evaluation failed without exposing private evaluation content.",
            file=sys.stderr,
        )
        return 2
    raw = read_regular(arguments.output_root / "result-bundle.json", maximum=MAX_RESULT_BUNDLE_BYTES)
    result = {
        "candidate_id": bundle.candidate_id,
        "protocol_sha256": bundle.protocol_sha256,
        "result_bundle_sha256": hashlib.sha256(raw).hexdigest(),
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
