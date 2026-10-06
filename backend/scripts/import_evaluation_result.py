"""Fixed redacted CLI for synthetic or independently transferred result aggregates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.result_import import import_evaluation_result


class _RedactedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "evaluation result import refused\n")


def main(argv: list[str] | None = None) -> int:
    """Emit only fixed status/verified opaque identities or a redacted exit-2 refusal."""
    parser = _RedactedParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--d36-trial-tool-attestation", type=Path, required=True)
    parser.add_argument("--d37-tool-attestation", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        validation = import_evaluation_result(
            bundle_path=arguments.bundle, expected_sha256=arguments.expected_sha256,
            freeze_manifest_path=arguments.freeze_manifest, protocol_path=arguments.protocol,
            d36_trial_tool_attestation_path=arguments.d36_trial_tool_attestation,
            d37_tool_attestation_path=arguments.d37_tool_attestation,
            output_dir=arguments.output, repo_root=arguments.repo_root,
        )
    except Exception:
        print("evaluation result import refused", file=sys.stderr)
        return 2
    print(f"accepted candidate_id={validation.candidate_id} "
          f"accepted_bundle_sha256={validation.accepted_bundle_sha256} "
          f"d38_import_tool_sha256={validation.d38_import_tool_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
