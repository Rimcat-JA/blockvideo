"""D24 offline corpus validation, review and approval gate; no inference calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.contracts import Case
from evaluation.corpus import eligibility, load_cases, load_review, pending_ledger, split_summary, summary
from evaluation.review import review_html


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--cases", type=Path, required=True)
    validate.add_argument("--human", type=Path)
    validate.add_argument("--ai", type=Path)
    validate.add_argument("--output", type=Path)
    split = commands.add_parser("split")
    split.add_argument("--development", type=Path, required=True)
    split.add_argument("--held-out", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    review = commands.add_parser("review")
    review.add_argument("--cases", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--pending-output", type=Path, required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("--output", type=Path, required=True)
    eligible = commands.add_parser("eligible")
    eligible.add_argument("--cases", type=Path, required=True)
    eligible.add_argument("--human", type=Path, required=True)
    eligible.add_argument("--ai", type=Path, required=True)
    eligible.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "schema":
            write_new(args.output, Case.model_json_schema())
            report = {"schema_written": True}
        elif args.command == "split":
            report = split_summary(load_cases(args.development), load_cases(args.held_out))
            write_new(args.output, report)
        else:
            cases = load_cases(args.cases)
            report = summary(cases)
            if args.command == "review":
                repo = Path(__file__).resolve().parents[2]
                if any(case.split == "held_out" for case in cases) and any(
                    path.resolve().is_relative_to(repo) for path in (args.output, args.pending_output)
                ):
                    raise ValueError("held-out review artifacts must stay outside the implementation repository")
                if args.output.exists() or args.pending_output.exists():
                    raise ValueError("review output already exists; use a fresh path")
                write_new(args.pending_output, pending_ledger(cases).model_dump(mode="json"))
                write_new(args.output, review_html(cases))
            else:
                human = load_review(args.human, cases, "human") if args.human else None
                ai = load_review(args.ai, cases, "independent_ai") if args.ai else None
                report["approval"] = eligibility(cases, human, ai)
                if args.command == "eligible" and not report["approval"]["eligible_count"]:
                    raise ValueError("no approved cases; final aggregation is prohibited")
                if args.output:
                    write_new(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError):
        # Do not echo invalid input, raw labels, user notes or paths into logs.
        print("Evaluation data check failed. No inference or final score was produced.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
