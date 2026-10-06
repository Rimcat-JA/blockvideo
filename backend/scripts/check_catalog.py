"""Validate every catalog-side file together (definitions, policies, annotations,
prompt rules, search scope). Exit 0 when consistent, 1 otherwise."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from app.operations.catalog_compiler import OPERATIONS_DIR, PROMPT_RULES, compile_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, default=OPERATIONS_DIR)
    parser.add_argument("--prompt-rules", type=Path, default=PROMPT_RULES)
    arguments = parser.parse_args()
    report = compile_catalog(arguments.catalog_dir, arguments.prompt_rules)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=1))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
