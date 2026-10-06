"""D40 readiness decision CLI: bounded local reads and output writes only.

No Git, network, subprocess, commit, push, tag, publication, release or deployment.
Exit 0 for Ready / Conditionally ready, 2 for Not ready or a refusal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from evaluation.release_decision import (
    AcceptedNonSafetyLimitation,
    DecisionInputError,
    ReviewEvidence,
    decide_readiness,
    load_d38_accepted_evidence,
    load_d39_verification_evidence,
    load_decision_tool_attestation,
    assert_running_closure,
    load_freeze,
    load_limitations,
    load_review,
    publish_decision,
)

_REFUSED = "readiness decision refused"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(_REFUSED, file=sys.stderr)
        raise SystemExit(2)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = _Parser(add_help=False)
    for name in ("--repo-root", "--decision-tool-attestation", "--freeze", "--output"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--decision-tool-expected-sha256", required=True)
    for name in ("--aggregate", "--import-validation", "--d38-tool-attestation", "--verification",
                 "--verifier-tool-attestation", "--human", "--independent", "--limitations"):
        parser.add_argument(name, type=Path)
    for name in ("--aggregate-expected-sha256", "--import-validation-expected-sha256",
                 "--d38-tool-attestation-expected-sha256", "--verification-expected-sha256",
                 "--verifier-tool-attestation-expected-sha256"):
        parser.add_argument(name)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        # Unattested decision code never evaluates a gate or writes an output, and
        # the attested files must be the modules actually running.
        tool, _ = load_decision_tool_attestation(repo_root=args.repo_root, attestation_path=args.decision_tool_attestation,
                                                 expected_sha256=args.decision_tool_expected_sha256)
        assert_running_closure(args.repo_root)
        freeze, _, d36_tool = load_freeze(args.freeze)
    except (OSError, ValueError):
        print(_REFUSED, file=sys.stderr)
        return 2
    failures: dict[str, str] = {}
    try:
        aggregate, validation, d38_tool, d38_hashes = load_d38_accepted_evidence(
            accepted_result_path=args.aggregate, accepted_result_expected_sha256=args.aggregate_expected_sha256,
            validation_path=args.import_validation, validation_expected_sha256=args.import_validation_expected_sha256,
            d38_tool_attestation_path=args.d38_tool_attestation,
            d38_tool_attestation_expected_sha256=args.d38_tool_attestation_expected_sha256, freeze=freeze)
    except DecisionInputError as error:
        failures.update(error.failures)
        aggregate, validation, d38_tool, d38_hashes = None, None, None, {}
    try:
        verification, verifier_tool, d39_hashes = load_d39_verification_evidence(
            verification_path=args.verification, verification_expected_sha256=args.verification_expected_sha256,
            verifier_tool_attestation_path=args.verifier_tool_attestation,
            verifier_tool_attestation_expected_sha256=args.verifier_tool_attestation_expected_sha256, freeze=freeze)
    except DecisionInputError as error:
        failures.update(error.failures)
        verification, verifier_tool, d39_hashes = None, None, {}
    reviews: dict[str, ReviewEvidence | None] = {}
    for kind, path in (("human_operation", args.human), ("independent_review", args.independent)):
        try:
            reviews[kind] = load_review(path, kind)
        except DecisionInputError as error:
            failures.update(error.failures)
            reviews[kind] = None
    limitations: tuple[AcceptedNonSafetyLimitation, ...] = ()
    if args.limitations is not None:
        try:
            limitations = load_limitations(args.limitations)
        except DecisionInputError as error:
            failures.update(error.failures)
    try:
        decision = decide_readiness(
            freeze=freeze, aggregate=aggregate, import_validation=validation, d38_tool_attestation=d38_tool,
            d38_input_sha256=d38_hashes, verification=verification, verifier_tool_attestation=verifier_tool,
            d39_input_sha256=d39_hashes, human=reviews["human_operation"], independent=reviews["independent_review"],
            limitation_approvals=limitations, decision_tool_attestation=tool, input_failures=failures,
            d36_tool_attestation=d36_tool)
        # The fixed decision sources must be unchanged after evaluation as well.
        after, _ = load_decision_tool_attestation(repo_root=args.repo_root, attestation_path=args.decision_tool_attestation,
                                                  expected_sha256=args.decision_tool_expected_sha256)
        if after != tool:
            raise ValueError("decision source changed during evaluation")
        assert_running_closure(args.repo_root)
        # Every input's directory is protected: the decision never lands inside
        # (and so can never invalidate) the D36, D38 or D39 publications or reviews.
        inputs = (args.freeze, args.aggregate, args.import_validation, args.d38_tool_attestation, args.verification,
                  args.verifier_tool_attestation, args.human, args.independent, args.limitations)
        protected = tuple(path.parent for path in inputs if path is not None)
        publish_decision(decision, args.output, args.repo_root, attestation_path=args.decision_tool_attestation,
                         protected=protected)
    except (OSError, ValueError):
        print(_REFUSED, file=sys.stderr)
        return 2
    print("readiness decision: " + decision.outcome)
    return 0 if decision.outcome in ("Ready", "Conditionally ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
