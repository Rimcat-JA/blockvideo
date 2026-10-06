"""Detached-hash aggregate import, independent of private evaluation evidence."""
from __future__ import annotations

import hashlib
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evaluation.blinded_contracts import MAX_PROTOCOL_BYTES, EvaluationProtocol, Sha256
from evaluation.blinded_io import is_reparse, publish_accepted_triplet, read_regular, require_directory
from evaluation.evidence_json import parse_canonical_model
from evaluation.release_candidate.contracts import CompletionMarker, FreezeManifest
from evaluation.release_candidate.freeze import read_frozen_candidate
from evaluation.result_contracts import MAX_RESULT_BUNDLE_BYTES, EvaluationResultBundle
from evaluation.tool_attestation import (
    ToolAttestation,
    _bounded_git,
    attest_tool,
    canonical_json_bytes,
    historical_blinded_source_paths,
    validate_git_repository,
    verify_historical_attestation,
)

IMPORT_SOURCE_PATHS: tuple[str, ...] = (
    "backend/app/__init__.py",
    "backend/app/interpretation/__init__.py",
    "backend/app/interpretation/contracts.py",
    "backend/app/operations/__init__.py",
    "backend/app/operations/catalog.py",
    "backend/app/operations/contracts.py",
    "backend/app/operations/definitions.json",
    "backend/app/operations/limits.py",
    "backend/app/operations/schema_validation.py",
    "backend/evaluation/__init__.py",
    "backend/evaluation/blinded_contracts.py",
    "backend/evaluation/blinded_io.py",
    "backend/evaluation/contracts.py",
    "backend/evaluation/corpus.py",
    "backend/evaluation/evidence_json.py",
    "backend/evaluation/fixtures.py",
    "backend/evaluation/release_candidate/__init__.py",
    "backend/evaluation/release_candidate/contracts.py",
    "backend/evaluation/release_candidate/fingerprints.py",
    "backend/evaluation/release_candidate/freeze.py",
    "backend/evaluation/result_contracts.py",
    "backend/evaluation/result_import.py",
    "backend/evaluation/tool_attestation.py",
    "backend/pyproject.toml",
    "backend/scripts/import_evaluation_result.py",
    "backend/uv.lock",
)
_D36_SOURCE_PATHS: tuple[str, ...] = (
    "backend/evaluation/blinded_io.py",
    "backend/evaluation/evidence_json.py",
    "backend/evaluation/final_protocol.json",
    "backend/evaluation/release_candidate/__init__.py",
    "backend/evaluation/release_candidate/contracts.py",
    "backend/evaluation/release_candidate/fingerprints.py",
    "backend/evaluation/release_candidate/freeze.py",
    "backend/evaluation/scripts/evaluation_trial_host.py",
    "backend/evaluation/scripts/freeze_candidate.py",
    "backend/evaluation/tool_attestation.py",
    "backend/evaluation/unlabeled_contracts.py",
)
IMPORT_CHECKS: tuple[str, ...] = (
    "bundle_detached_sha256", "bundle_canonical", "protocol_canonical",
    "protocol_sha256", "candidate_binding", "corpus_binding", "approval_bindings",
    "freeze_binding", "tool_bindings", "token_syntax", "token_unique_sorted",
    "token_disjoint", "token_exact_union", "token_counts", "topology_identity",
    "nonempty_coverage", "exclusion_reasons", "category_accounting",
    "mode_accounting", "sealed_evidence_hash_syntax",
)
_IDENTITY_FIELDS = (
    "candidate_id", "corpus_sha256", "human_approval_sha256", "independent_approval_sha256",
    "freeze_sha256", "d36_trial_tool_sha256", "d37_evaluator_tool_sha256",
)
_MAX_FREEZE_BYTES = 16 * 1024 * 1024
_MAX_D36_TOOL_BYTES = 1024 * 1024
_MAX_D37_TOOL_BYTES = 16 * 1024 * 1024
_MAX_MARKER_BYTES = 1024


class ResultImportError(ValueError):
    """A fixed refusal that discloses neither input paths nor input contents."""

    def __init__(self) -> None:
        super().__init__("evaluation result import refused")


class ImportValidation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    schema_version: Literal[1]
    status: Literal["accepted"]
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_bundle_sha256: Sha256
    accepted_bundle_sha256: Sha256
    corpus_sha256: Sha256
    human_approval_sha256: Sha256
    independent_approval_sha256: Sha256
    protocol_sha256: Sha256
    freeze_sha256: Sha256
    d36_trial_tool_sha256: Sha256
    d37_evaluator_tool_sha256: Sha256
    model_configuration_sha256: Sha256
    stateful_index_sha256: Sha256
    d38_import_tool_sha256: Sha256
    checks: dict[str, Literal[True]]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_primitive(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("import schema version must be an integer")
        return value

    @field_validator("checks", mode="before")
    @classmethod
    def validate_checks(cls, value: object) -> object:
        if (type(value) is not dict or set(value) != set(IMPORT_CHECKS)
                or any(type(item) is not bool or item is not True for item in value.values())):
            raise ValueError("import checks must contain exactly twenty true booleans")
        return value

    @model_validator(mode="after")
    def validate_preserved_hash(self) -> Self:
        if self.source_bundle_sha256 != self.accepted_bundle_sha256:
            raise ValueError("accepted bytes must preserve source bytes")
        return self


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_accounting(bundle: EvaluationResultBundle, protocol: EvaluationProtocol) -> None:
    if (bundle.protocol_case_count != protocol.case_count
            or bundle.protocol_case_tokens != protocol.case_tokens
            or bundle.protocol_category_count != protocol.category_count
            or bundle.protocol_category_tokens != protocol.category_tokens
            or bundle.case_categories != protocol.case_categories):
        raise ValueError("result topology does not match the exact protocol")
    included = set(bundle.included_case_tokens)
    excluded = {item.case_token for item in bundle.excluded_cases}
    if (included & excluded or included | excluded != set(protocol.case_tokens)
            or len(included) != bundle.included_count or len(excluded) != bundle.excluded_count):
        raise ValueError("result does not exactly partition the protocol")
    categories = {item.case_token: item.category_token for item in protocol.case_categories}
    protocol_counts = Counter(categories.values())
    included_counts = Counter(categories[token] for token in included)
    excluded_reasons = Counter((categories[item.case_token], item.reason) for item in bundle.excluded_cases)
    reasons = ("both_not_approved", "human_not_approved", "independent_not_approved")
    for category in protocol.category_tokens:
        if (included_counts[category] < 1 or included_counts[category]
                + sum(excluded_reasons[category, reason] for reason in reasons) != protocol_counts[category]):
            raise ValueError("category partition is incomplete")
    count_fields = ("included", "completed", "task_complete", "unauthorized_effects",
                    "unauthorized_replays", "secret_disclosures")
    for mode in bundle.modes:
        if (mode.included != len(included) or mode.completed + mode.transport_failures
                + mode.deadline_failures != mode.included):
            raise ValueError("mode accounting is incomplete")
        if tuple(row.category_token for row in mode.categories) != protocol.category_tokens:
            raise ValueError("mode category identity mismatch")
        for row in mode.categories:
            if row.included != included_counts[row.category_token]:
                raise ValueError("category accounting mismatch")
        for field in count_fields:
            if sum(getattr(row, field) for row in mode.categories) != getattr(mode, field):
                raise ValueError("mode accounting mismatch")


def _absolute_path(path: Path) -> Path:
    absolute = path.absolute()
    current = require_directory(Path(absolute.anchor), "path anchor")
    for part in absolute.parts[1:]:
        if part == "..":
            require_directory(current, "path prefix")
            current = current.parent
            continue
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or is_reparse(metadata):
            raise ValueError("path contains a link or reparse component")
    return current


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _validate_output(output: Path, root: Path, protected: tuple[Path, ...]) -> None:
    if output != output.resolve(strict=False) or any(_overlap(output, path) for path in protected):
        raise ValueError("output overlaps a protected root or has an unsafe component")
    relative = output.relative_to(root).as_posix()
    parent_relative = output.parent.relative_to(root).as_posix()
    if parent_relative == ".":
        raise ValueError("output parent must be ignored")
    for name in (relative, parent_relative):
        _bounded_git(root, "check-ignore", "--no-index", "--", name + "/", maximum=4096)
        if _bounded_git(root, "ls-files", "-z", "--cached", "--", name, maximum=4096):
            raise ValueError("output must be untracked")
    try:
        output.lstat()
    except FileNotFoundError:
        return
    raise ValueError("output destination already exists")


def import_evaluation_result(
    *, bundle_path: Path, expected_sha256: str, freeze_manifest_path: Path,
    protocol_path: Path, d36_trial_tool_attestation_path: Path,
    d37_tool_attestation_path: Path, output_dir: Path, repo_root: Path,
) -> ImportValidation:
    """Accept a detached-verified aggregate without reading detailed held-out evidence."""
    try:
        if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("detached digest is invalid")
        bundle_path, protocol_path = _absolute_path(bundle_path), _absolute_path(protocol_path)
        freeze_manifest_path = _absolute_path(freeze_manifest_path)
        d36_trial_tool_attestation_path = _absolute_path(d36_trial_tool_attestation_path)
        d37_tool_attestation_path = _absolute_path(d37_tool_attestation_path)
        run = require_directory(bundle_path.parent, "evaluator run")
        publication = require_directory(freeze_manifest_path.parent, "D36 publication")
        if (bundle_path.name != "result-bundle.json" or protocol_path != run / "protocol.json"
                or d37_tool_attestation_path != run / "tool-attestation.json"
                or freeze_manifest_path.name != "freeze-manifest.json"
                or d36_trial_tool_attestation_path != publication / "d36-tool-attestation.json"):
            raise ValueError("input provenance mismatch")
        bundle_raw = read_regular(bundle_path, maximum=MAX_RESULT_BUNDLE_BYTES)
        if _hash(bundle_raw) != expected_sha256:
            raise ValueError("detached digest mismatch")
        bundle = parse_canonical_model(bundle_raw, EvaluationResultBundle, maximum=MAX_RESULT_BUNDLE_BYTES)
        protocol_raw = read_regular(protocol_path, maximum=MAX_PROTOCOL_BYTES)
        protocol = parse_canonical_model(protocol_raw, EvaluationProtocol, maximum=MAX_PROTOCOL_BYTES)
        if _hash(protocol_raw) != bundle.protocol_sha256:
            raise ValueError("protocol digest mismatch")
        freeze_raw = read_regular(freeze_manifest_path, maximum=_MAX_FREEZE_BYTES)
        manifest = parse_canonical_model(freeze_raw, FreezeManifest, maximum=_MAX_FREEZE_BYTES)
        d36_raw = read_regular(d36_trial_tool_attestation_path, maximum=_MAX_D36_TOOL_BYTES)
        d36 = parse_canonical_model(d36_raw, ToolAttestation, maximum=_MAX_D36_TOOL_BYTES)
        marker_path = publication / ".d36-publication-state"
        marker_raw = read_regular(marker_path, maximum=_MAX_MARKER_BYTES)
        marker = parse_canonical_model(marker_raw, CompletionMarker, maximum=_MAX_MARKER_BYTES)
        for fingerprint, raw in zip(marker.files, (d36_raw, freeze_raw), strict=True):
            if fingerprint.sha256 != _hash(raw) or fingerprint.size != len(raw):
                raise ValueError("D36 publication fingerprint mismatch")
        if read_frozen_candidate(publication) != (manifest, d36):
            raise ValueError("D36 canonical reader disagreement")
        d37_raw = read_regular(d37_tool_attestation_path, maximum=_MAX_D37_TOOL_BYTES)
        d37 = parse_canonical_model(d37_raw, ToolAttestation, maximum=_MAX_D37_TOOL_BYTES)
        if any(getattr(bundle, field) != getattr(protocol, field) for field in _IDENTITY_FIELDS):
            raise ValueError("bundle protocol identity mismatch")
        if (bundle.candidate_id != manifest.candidate_id or bundle.freeze_sha256 != _hash(freeze_raw)
                or bundle.d36_trial_tool_sha256 != d36.aggregate_sha256
                or bundle.d37_evaluator_tool_sha256 != d37.aggregate_sha256):
            raise ValueError("upstream identity mismatch")
        _validate_accounting(bundle, protocol)
        root = require_directory(_absolute_path(repo_root), "tooling repository")
        verify_historical_attestation(repo_root=root, attestation=d36,
                                      expected_tool_name="d36_candidate_freezer_and_trial_host",
                                      source_paths=_D36_SOURCE_PATHS)
        d37_paths = historical_blinded_source_paths(repo_root=root, git_commit=d37.git_commit)
        verify_historical_attestation(repo_root=root, attestation=d37,
                                      expected_tool_name="d37_blinded_evaluator", source_paths=d37_paths)
        output = _absolute_path(output_dir)
        protected = (run, publication, *(root / name for name in ("backend", "frontend", "docs", ".git")))
        _validate_output(output, root, protected)
        head = validate_git_repository(root)
        tool = attest_tool(repo_root=root, tool_name="d38_result_importer", git_commit=head,
                           source_paths=IMPORT_SOURCE_PATHS)
        validation = ImportValidation(
            schema_version=1, status="accepted", candidate_id=bundle.candidate_id,
            source_bundle_sha256=expected_sha256, accepted_bundle_sha256=expected_sha256,
            **{field: getattr(bundle, field) for field in _IDENTITY_FIELDS if field != "candidate_id"},
            protocol_sha256=bundle.protocol_sha256,
            model_configuration_sha256=protocol.model_configuration_sha256,
            stateful_index_sha256=protocol.stateful_index_sha256,
            d38_import_tool_sha256=tool.aggregate_sha256, checks=dict.fromkeys(IMPORT_CHECKS, True),
        )
        validation_raw = canonical_json_bytes(validation) + b"\n"
        tool_raw = canonical_json_bytes(tool) + b"\n"
        parse_canonical_model(validation_raw, ImportValidation, maximum=_MAX_FREEZE_BYTES)
        parse_canonical_model(tool_raw, ToolAttestation, maximum=_MAX_D37_TOOL_BYTES)
        for path, raw, cap in (
            (bundle_path, bundle_raw, MAX_RESULT_BUNDLE_BYTES),
            (protocol_path, protocol_raw, MAX_PROTOCOL_BYTES),
            (freeze_manifest_path, freeze_raw, _MAX_FREEZE_BYTES),
            (d36_trial_tool_attestation_path, d36_raw, _MAX_D36_TOOL_BYTES),
            (marker_path, marker_raw, _MAX_MARKER_BYTES),
            (d37_tool_attestation_path, d37_raw, _MAX_D37_TOOL_BYTES),
        ):
            if read_regular(path, maximum=cap) != raw:
                raise ValueError("input changed during import")
        if read_frozen_candidate(publication) != (manifest, d36):
            raise ValueError("D36 publication changed during import")
        require_directory(output.parent, "accepted evidence parent", create=True)
        _validate_output(output, root, protected)
        validate_git_repository(root, expected_commit=head)
        publish_accepted_triplet(output_dir=output, accepted_result_bytes=bundle_raw,
                                 validation_bytes=validation_raw, tool_attestation_bytes=tool_raw)
        return validation
    except Exception:
        raise ResultImportError() from None
