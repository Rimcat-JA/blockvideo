"""Check that every catalog-side file agrees before startup, indexing or evaluation.

The catalog is the single source of operation knowledge: definitions (arguments),
policies (safety, confirmation, references, settings layout), annotations (Japanese
phrasings for retrieval and selection), prompt rules (model guidance per operation)
and the search scope (capabilities). Adding an operation means editing these files,
not code; this compiler refuses any disagreement between them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.operations.annotations import parse_annotations
from app.operations.catalog import load_catalog
from app.operations.limits import MAX_CATALOG_OPERATIONS
from app.operations.policies import OperationPolicies
from app.operations.schema_validation import CatalogError

OPERATIONS_DIR = Path(__file__).parent
PROMPT_RULES = OPERATIONS_DIR.parent / "interpretation" / "prompt_rules.json"
# Optional prompt-rule groups that the interpreter can enable (see system_prompt).
KNOWN_FEATURES = ("plan",)


@dataclass
class CatalogReport:
    operations: int = 0
    operation_ids: int = 0
    annotated_operations: int = 0
    annotation_texts: int = 0
    prompt_rules: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def compile_catalog(directory: Path = OPERATIONS_DIR, prompt_rules: Path = PROMPT_RULES) -> CatalogReport:
    """Validate the catalog files in ``directory`` together; never raises on content."""
    report = CatalogReport()
    try:
        catalog = load_catalog(directory / "definitions.json")
    except CatalogError as exc:
        report.errors.append(f"definitions: {exc}")
        return report
    keys = {(item.operation_id, item.operation_version) for item in catalog.definitions}
    ids = {operation_id for operation_id, _ in keys}
    report.operations, report.operation_ids = len(keys), len(ids)
    if len(keys) > MAX_CATALOG_OPERATIONS:
        report.errors.append(f"catalog has {len(keys)} operations; limit is {MAX_CATALOG_OPERATIONS}")

    try:
        policies = OperationPolicies.model_validate(
            json.loads((directory / "operation_policies.json").read_text(encoding="utf-8")))
        policies.require_catalog_coverage(keys)
        for operation_id, policy in policies.operations.items():
            if operation_id not in ids:
                report.errors.append(f"policy for unknown operation: {operation_id}")
            for version in policy.settings_views:
                if (operation_id, version) not in keys:
                    report.errors.append(f"settings view for unknown version: {operation_id}@{version}")
    except (OSError, ValueError, CatalogError) as exc:
        report.errors.append(f"policies: {exc}")

    annotations_path = directory / "operation_annotations.json"
    if annotations_path.is_file():
        try:
            annotations = parse_annotations(annotations_path.read_bytes())
            annotations.require_known_operations(keys)
            for operation_id, version in sorted(keys):
                item = annotations.for_operation(operation_id, version)
                if not item.utterances:
                    report.warnings.append(f"no Japanese utterances for {operation_id}@{version}")
                else:
                    report.annotated_operations += 1
                report.annotation_texts += len(item.searchable())
        except (OSError, CatalogError) as exc:
            report.errors.append(f"annotations: {exc}")
    else:
        report.warnings.append("no operation_annotations.json: retrieval uses English metadata only")

    try:
        rules = json.loads(prompt_rules.read_text(encoding="utf-8"))["rules"]
        report.prompt_rules = len(rules)
        for index, rule in enumerate(rules):
            unknown = set(rule.get("operations", [])) - ids
            if unknown:
                report.errors.append(f"prompt rule {index} names unknown operations: {sorted(unknown)}")
            if set(rule.get("modes", ["normal", "yolo"])) - {"normal", "yolo"}:
                report.errors.append(f"prompt rule {index} has an unknown mode")
            if rule.get("feature") not in (None, *KNOWN_FEATURES):
                report.errors.append(f"prompt rule {index} has an unknown feature")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.errors.append(f"prompt rules: {exc}")

    try:
        scope = json.loads((directory / "search_scope.json").read_text(encoding="utf-8"))
        bound = {(item["operation_id"], item["operation_version"]) for item in scope["bindings"]}
        if bound != keys:
            report.errors.append(f"search scope differs from catalog: missing={sorted(keys - bound)[:5]} "
                                 f"extra={sorted(bound - keys)[:5]}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.errors.append(f"search scope: {exc}")
    return report
