from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from evaluation.blinded_contracts import MAX_PROTOCOL_CASES
from evaluation.evidence_json import parse_canonical_model
from evaluation.result_contracts import (
    MAX_RESULT_BUNDLE_BYTES,
    CategoryResult,
    EvaluationResultBundle,
    ExcludedCaseToken,
    ModeResult,
)
from evaluation.tool_attestation import (
    FileFingerprint,
    ToolAttestation,
    aggregate_fingerprints,
    attest_tool,
    canonical_json_bytes,
)

FIXTURE = Path(__file__).parent / "fixtures/blinded/synthetic-result-bundle.json"
CHECKS = (
    "bundle_detached_sha256", "bundle_canonical", "protocol_canonical",
    "protocol_sha256", "candidate_binding", "corpus_binding", "approval_bindings",
    "freeze_binding", "tool_bindings", "token_syntax", "token_unique_sorted",
    "token_disjoint", "token_exact_union", "token_counts", "topology_identity",
    "nonempty_coverage", "exclusion_reasons", "category_accounting",
    "mode_accounting", "sealed_evidence_hash_syntax",
)


def _raw(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _bundle() -> dict[str, Any]:
    return json.loads(FIXTURE.read_bytes())


def _parse_bundle(value: dict[str, Any]) -> EvaluationResultBundle:
    return parse_canonical_model(_raw(value), EvaluationResultBundle,
                                 maximum=MAX_RESULT_BUNDLE_BYTES)


def _set(data: Any, path: tuple[str | int, ...], value: object) -> None:
    for part in path[:-1]:
        data = data[part]
    data[path[-1]] = value


def _importer() -> Any:
    assert importlib.util.find_spec("evaluation.result_import") is not None, (
        "D38 import boundary is not implemented"
    )
    return importlib.import_module("evaluation.result_import")


def test_contracts_preserve_failures_safety_and_shared_models() -> None:
    raw = FIXTURE.read_bytes()
    bundle = parse_canonical_model(raw, EvaluationResultBundle,
                                   maximum=MAX_RESULT_BUNDLE_BYTES)
    assert _raw(bundle) == raw
    assert bundle.included_count == 2 and bundle.excluded_count == 3
    assert bundle.protocol_case_count == 5 and bundle.protocol_category_count == 2
    assert tuple(mode.mode for mode in bundle.modes) == ("all_tools", "stateful")
    assert bundle.modes[0].transport_failures == 1
    assert bundle.modes[1].deadline_failures == 1
    assert bundle.modes[0].unauthorized_effects == 1
    assert bundle.modes[1].unauthorized_replays == 1
    assert bundle.modes[1].secret_disclosures == 1
    assert type(bundle.modes[0]) is ModeResult
    assert type(bundle.modes[0].categories[0]) is CategoryResult
    assert type(bundle.excluded_cases[0]) is ExcludedCaseToken
    assert tuple(item.reason for item in bundle.excluded_cases) == (
        "independent_not_approved", "human_not_approved", "both_not_approved",
    )


INVALID_FIELDS = [
    (("schema_version",), True), (("schema_version",), 1.0),
    (("included_count",), True), (("included_count",), 2.0),
    (("included_count",), -1), (("included_count",), MAX_PROTOCOL_CASES + 1),
    (("included_count",), 1), (("excluded_count",), 2),
    (("protocol_case_count",), 4), (("protocol_category_count",), 1),
    (("protocol_case_tokens",), []), (("protocol_category_tokens",), []),
    (("included_case_tokens",), []), (("included_count",), 0),
    (("protocol_case_tokens", 0), "D24-H001"),
    (("protocol_category_tokens", 0), "A" * 64),
    (("included_case_tokens", 0), "6" * 64),
    (("excluded_cases", 0, "reason"), "skipped"),
    (("excluded_cases", 0, "reason"), True),
    (("evaluator_name",), ""), (("evaluator_name",), " padded "),
    (("evaluator_role",), "implementer"),
    (("executed_at",), "2026-09-20T00:00:00+00:00"),
    (("executed_at",), "2026-02-30T00:00:00Z"),
    (("modes", 1, "mode"), "all_tools"),
    (("modes", 0, "included"), 3),
    (("modes", 0, "completed"), 0),
    (("modes", 0, "completed"), 3),
    (("modes", 0, "task_complete"), 2),
    (("modes", 0, "transport_failures"), 0),
    (("modes", 0, "deadline_failures"), 2),
    (("modes", 0, "unauthorized_effects"), 3),
    (("modes", 0, "unauthorized_replays"), 1),
    (("modes", 0, "secret_disclosures"), 1),
    (("modes", 0, "categories", 0, "included"), 0),
    (("modes", 0, "categories", 0, "completed"), 2),
    (("modes", 0, "categories", 0, "task_complete"), 2),
    (("modes", 0, "categories", 0, "unauthorized_effects"), 2),
    (("modes", 0, "categories", 0, "unauthorized_replays"), 2),
    (("modes", 0, "categories", 0, "secret_disclosures"), 2),
    (("modes", 0, "categories", 0, "completed"), True),
    (("modes", 0, "categories", 0, "included"), "1"),
    (("modes", 0, "categories", 1, "category_token"), "a" * 64),
    (("case_categories", 1, "case_token"), "1" * 64),
    (("case_categories", 0, "category_token"), "c" * 64),
]
INVALID_FIELDS += [((field,), "not-a-hash") for field in (
    "freeze_sha256", "corpus_sha256", "human_approval_sha256",
    "independent_approval_sha256", "protocol_sha256", "d36_trial_tool_sha256",
    "d37_evaluator_tool_sha256", "sealed_evidence_sha256",
)]
INVALID_FIELDS += [((field,), "synthetic-poison") for field in (
    "case_id", "text", "category_name", "label", "sealed_evidence_path",
)]


@pytest.mark.parametrize("path,value", INVALID_FIELDS)
def test_contracts_reject_invalid_fields(path: tuple[str | int, ...], value: object) -> None:
    bundle = _bundle()
    _set(bundle, path, value)
    with pytest.raises(ValueError):
        _parse_bundle(bundle)


@pytest.mark.parametrize("field,operation", [
    ("protocol_case_tokens", "duplicate"), ("protocol_case_tokens", "reverse"),
    ("protocol_case_tokens", "missing"), ("protocol_case_tokens", "extra"),
    ("protocol_category_tokens", "duplicate"), ("protocol_category_tokens", "reverse"),
    ("included_case_tokens", "duplicate"), ("included_case_tokens", "reverse"),
    ("excluded_cases", "duplicate"), ("excluded_cases", "reverse"),
    ("case_categories", "duplicate"), ("case_categories", "reverse"),
    ("case_categories", "missing"), ("modes", "reverse"),
])
def test_contracts_reject_nonexact_topology(field: str, operation: str) -> None:
    bundle = _bundle()
    values = bundle[field]
    if operation == "duplicate":
        values.insert(0, copy.deepcopy(values[0]))
    elif operation == "reverse":
        values.reverse()
    elif operation == "missing":
        values.pop()
    else:
        values.append("6" * 64)
    with pytest.raises(ValueError):
        _parse_bundle(bundle)


def test_contracts_reject_partition_overlap_even_with_correct_lengths() -> None:
    bundle = _bundle()
    bundle["excluded_cases"][0]["case_token"] = "1" * 64
    with pytest.raises(ValueError):
        _parse_bundle(bundle)


def test_contracts_reject_zero_category_coverage_with_consistent_totals() -> None:
    bundle = _bundle()
    bundle["included_case_tokens"] = ["1" * 64, "2" * 64]
    bundle["excluded_cases"][0]["case_token"] = "5" * 64
    bundle["excluded_cases"].sort(key=lambda item: item["case_token"])
    for mode in bundle["modes"]:
        mode["categories"][0]["included"] = 2
        mode["categories"][1]["included"] = 0
        mode["categories"][1]["completed"] = 0
        mode["categories"][1]["secret_disclosures"] = 0
        mode["categories"][1]["unauthorized_replays"] = 0
        mode["completed"] = mode["categories"][0]["completed"]
        mode["secret_disclosures"] = 0
        mode["unauthorized_replays"] = 0
        mode["deadline_failures"] = 2 - mode["completed"] - mode["transport_failures"]
    with pytest.raises(ValueError):
        _parse_bundle(bundle)


@pytest.mark.parametrize("transform", [
    lambda raw: raw[:-1], lambda raw: raw + b"\n", lambda raw: raw.replace(b"\n", b"\r\n"),
    lambda raw: raw.replace(b'"schema_version":1', b'"schema_version":1.0'),
    lambda raw: raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
    lambda raw: raw.replace(b'"included_count":2', b'"included_count":NaN'),
    lambda raw: raw.replace(b'"included_count":2', b'"included_count":1e999'),
    lambda raw: raw.replace(b'"included_count":2', b'"included_count": 2'),
])
def test_contracts_reject_noncanonical_bytes(transform: Any) -> None:
    with pytest.raises(ValueError):
        parse_canonical_model(transform(FIXTURE.read_bytes()), EvaluationResultBundle,
                              maximum=MAX_RESULT_BUNDLE_BYTES)


def test_contracts_enforce_preparse_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("oversized bytes reached model construction")

    monkeypatch.setattr(EvaluationResultBundle, "model_validate_json", forbidden)
    raw = FIXTURE.read_bytes()
    with pytest.raises(ValueError, match="size limit"):
        parse_canonical_model(raw, EvaluationResultBundle, maximum=len(raw) - 1)
    with pytest.raises(ValueError, match="string limit"):
        parse_canonical_model(_raw({"poison": "x" * 8192}), EvaluationResultBundle,
                              maximum=MAX_RESULT_BUNDLE_BYTES)


def test_import_validation_requires_exact_twenty_raw_true_checks() -> None:
    importer = _importer()
    validation = {
        "schema_version": 1, "status": "accepted", "candidate_id": "d35-synthetic-candidate",
        **{field: "1" * 64 for field in (
            "source_bundle_sha256", "accepted_bundle_sha256", "corpus_sha256",
            "human_approval_sha256", "independent_approval_sha256", "protocol_sha256",
            "freeze_sha256", "d36_trial_tool_sha256", "d37_evaluator_tool_sha256",
            "model_configuration_sha256", "stateful_index_sha256", "d38_import_tool_sha256",
        )},
        "checks": dict.fromkeys(CHECKS, True),
    }
    model = parse_canonical_model(_raw(validation), importer.ImportValidation,
                                 maximum=16 * 1024 * 1024)
    assert model.checks == dict.fromkeys(CHECKS, True)
    for replacement in (False, 1, "true"):
        invalid = copy.deepcopy(validation)
        invalid["checks"][CHECKS[0]] = replacement
        with pytest.raises(ValueError):
            parse_canonical_model(_raw(invalid), importer.ImportValidation,
                                  maximum=16 * 1024 * 1024)
    for extra in (False, True):
        invalid = copy.deepcopy(validation)
        if extra:
            invalid["checks"]["extra"] = True
        else:
            invalid["checks"].pop(CHECKS[0])
        with pytest.raises(ValueError):
            parse_canonical_model(_raw(invalid), importer.ImportValidation,
                                  maximum=16 * 1024 * 1024)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != "5" * 64


D36_PATHS = (
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
D37_REQUIRED = (
    "backend/app/operations/definitions.json", "backend/pyproject.toml",
    "backend/scripts/run_blinded_evaluation.py", "backend/uv.lock",
)
D38_PATHS = (
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


def _git(root: Path, *args: str, input: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], input=input,
                          capture_output=True, check=True).stdout.strip()


@pytest.fixture
def records(tmp_path: Path) -> dict[str, Any]:
    from evaluation.release_candidate.contracts import FreezeManifest

    repo = tmp_path / "synthetic-tooling"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "d38@example.invalid")
    _git(repo, "config", "user.name", "D38 Synthetic")
    _git(repo, "config", "core.autocrlf", "false")
    historical_paths = sorted(
        (set(D36_PATHS) | set(D38_PATHS) | set(D37_REQUIRED)
         | {"backend/evaluation/blinded_runner.py"})
        - {"backend/evaluation/result_import.py", "backend/scripts/import_evaluation_result.py"}
    )
    for name in historical_paths:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC = 1\n")
    (repo / ".gitignore").write_text("evidence/\n", encoding="ascii")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "synthetic historical tools")
    old = _git(repo, "rev-parse", "HEAD").decode("ascii")
    d36 = attest_tool(repo_root=repo, tool_name="d36_candidate_freezer_and_trial_host",
                      git_commit=old, source_paths=D36_PATHS)
    d37_paths = tuple(sorted({name for name in historical_paths if (
        name.endswith(".py") and name.startswith(("backend/evaluation/", "backend/app/"))
    )} | set(D37_REQUIRED)))
    d37 = attest_tool(repo_root=repo, tool_name="d37_blinded_evaluator",
                      git_commit=old, source_paths=d37_paths)
    (repo / D36_PATHS[0]).write_bytes(b"SYNTHETIC = 2\n")
    for name in ("backend/evaluation/result_import.py", "backend/scripts/import_evaluation_result.py"):
        (repo / name).write_bytes(b"SYNTHETIC = 3\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "synthetic later importer tools")
    files = [FileFingerprint(path="synthetic.py", size=3,
                             sha256=hashlib.sha256(b"x=1").hexdigest())]
    aggregate = aggregate_fingerprints(files)
    candidate_id = f"{aggregate[:16]}-{'a' * 12}"
    freeze = FreezeManifest(
        schema_version=1, candidate_id=candidate_id, git_commit="a" * 40,
        git_tree_clean=True, candidate_control_sha256="b" * 64,
        created_at="2026-09-20T00:00:00Z", runtime={"python": "synthetic"},
        schema_version_number=1, mode_configuration={}, files=files,
        aggregate_sha256=aggregate,
    )
    publication = repo / "evidence/d36" / candidate_id
    publication.mkdir(parents=True)
    freeze_raw, d36_raw = _raw(freeze), _raw(d36)
    freeze_path = publication / "freeze-manifest.json"
    d36_path = publication / "d36-tool-attestation.json"
    freeze_path.write_bytes(freeze_raw)
    d36_path.write_bytes(d36_raw)
    marker = {"schema_version": 1, "files": [
        {"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in ((d36_path.name, d36_raw), (freeze_path.name, freeze_raw))
    ]}
    (publication / ".d36-publication-state").write_bytes(_raw(marker))
    run = repo / "evidence/run"
    run.mkdir()
    (run / "sealed").mkdir()
    (run / "sealed/poison-private.json").write_bytes(b"SYNTHETIC POISON DO NOT OPEN")
    bundle = _bundle()
    bundle.update(candidate_id=candidate_id, freeze_sha256=hashlib.sha256(freeze_raw).hexdigest(),
                  d36_trial_tool_sha256=d36.aggregate_sha256,
                  d37_evaluator_tool_sha256=d37.aggregate_sha256)
    protocol = {
        "schema_version": 1, "candidate_id": candidate_id,
        "modes": ["all_tools", "stateful"], "per_call_deadline_seconds": 180,
        "maximum_model_calls": 4, "isolation": "fresh_case_state_under_source_group",
        **{field: bundle[field] for field in (
            "freeze_sha256", "corpus_sha256", "human_approval_sha256",
            "independent_approval_sha256", "d36_trial_tool_sha256", "d37_evaluator_tool_sha256",
        )},
        "model_configuration_sha256": "8" * 64, "stateful_index_sha256": "9" * 64,
        "case_count": 5, "category_count": 2,
        "case_tokens": bundle["protocol_case_tokens"],
        "category_tokens": bundle["protocol_category_tokens"],
        "case_categories": bundle["case_categories"],
    }
    protocol_path = run / "protocol.json"
    protocol_path.write_bytes(_raw(protocol))
    bundle["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    bundle_path = run / "result-bundle.json"
    bundle_path.write_bytes(_raw(bundle))
    d37_path = run / "tool-attestation.json"
    d37_path.write_bytes(_raw(d37))
    return {
        "repo_root": repo, "old": old, "d36": d36, "d37": d37, "d37_paths": d37_paths,
        "bundle_path": bundle_path, "expected_sha256": hashlib.sha256(_raw(bundle)).hexdigest(),
        "freeze_manifest_path": freeze_path, "protocol_path": protocol_path,
        "d36_trial_tool_attestation_path": d36_path, "d37_tool_attestation_path": d37_path,
        "output_dir": repo / "evidence/accepted",
    }


def _arguments(records: dict[str, Any]) -> dict[str, Any]:
    return {key: records[key] for key in (
        "bundle_path", "expected_sha256", "freeze_manifest_path", "protocol_path",
        "d36_trial_tool_attestation_path", "d37_tool_attestation_path", "output_dir", "repo_root",
    )}


def _rewrite_bundle(records: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    data = json.loads(records["bundle_path"].read_bytes())
    _set(data, path, value)
    raw = _raw(data)
    records["bundle_path"].write_bytes(raw)
    records["expected_sha256"] = hashlib.sha256(raw).hexdigest()


def _refused(records: dict[str, Any]) -> None:
    importer = _importer()
    with pytest.raises(importer.ResultImportError) as error:
        importer.import_evaluation_result(**_arguments(records))
    assert str(error.value) == "evaluation result import refused"
    assert not records["output_dir"].exists()


def test_imports_canonical_synthetic_bundle_with_detached_hash(
    records: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = _importer()
    sealed = records["bundle_path"].parent / "sealed"
    original_open, original_scandir, original_listdir = os.open, os.scandir, os.listdir

    def guard_open(path: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(path, int) and "sealed" in Path(path).parts:
            pytest.fail("importer opened sealed synthetic poison")
        return original_open(path, *args, **kwargs)

    def guard_scan(path: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(path, int) and Path(path) == sealed:
            pytest.fail("importer enumerated sealed synthetic poison")
        return original_scandir(path, *args, **kwargs)

    def guard_list(path: Any) -> Any:
        if not isinstance(path, int) and Path(path) == sealed:
            pytest.fail("importer enumerated sealed synthetic poison")
        return original_listdir(path)

    monkeypatch.setattr(os, "open", guard_open)
    monkeypatch.setattr(os, "scandir", guard_scan)
    monkeypatch.setattr(os, "listdir", guard_list)
    originals = {records[field]: records[field].read_bytes() for field in (
        "bundle_path", "protocol_path", "freeze_manifest_path",
        "d36_trial_tool_attestation_path", "d37_tool_attestation_path",
    )}
    before = originals[records["bundle_path"]]
    validation = importer.import_evaluation_result(**_arguments(records))
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    output = records["output_dir"]
    assert {entry.name for entry in output.iterdir()} == {
        "accepted-result.json", "validation.json", "d38-tool-attestation.json",
    }
    assert (output / "accepted-result.json").read_bytes() == before
    assert validation.source_bundle_sha256 == validation.accepted_bundle_sha256 == (
        hashlib.sha256(before).hexdigest()
    )
    assert validation.model_configuration_sha256 == "8" * 64
    assert validation.stateful_index_sha256 == "9" * 64
    assert validation.checks == dict.fromkeys(CHECKS, True)
    tool = parse_canonical_model((output / "d38-tool-attestation.json").read_bytes(),
                                 ToolAttestation, maximum=16 * 1024 * 1024)
    assert tool.tool_name == "d38_result_importer"
    assert tuple(file.path for file in tool.files) == D38_PATHS
    assert tool.git_commit != records["old"]
    assert tool.aggregate_sha256 == validation.d38_import_tool_sha256
    assert aggregate_fingerprints(tool.files) == tool.aggregate_sha256
    assert json.loads((output / "validation.json").read_bytes())["checks"] == dict.fromkeys(CHECKS, True)
    assert not list(output.parent.glob(".d38-stage-*"))


@pytest.mark.parametrize("path,value", INVALID_FIELDS + [
    ((field,), "f" * 64) for field in (
        "freeze_sha256", "corpus_sha256", "human_approval_sha256", "independent_approval_sha256",
        "protocol_sha256", "d36_trial_tool_sha256", "d37_evaluator_tool_sha256",
    )
] + [(("candidate_id",), "altered-synthetic-candidate")])
def test_import_refuses_invalid_or_unbound_bundle(
    records: dict[str, Any], path: tuple[str | int, ...], value: object,
) -> None:
    _rewrite_bundle(records, path, value)
    _refused(records)


def test_documented_relative_input_repo_and_output_paths_are_accepted(
    records: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = records["repo_root"] / "backend"
    monkeypatch.chdir(cwd)
    arguments = _arguments(records)
    for field, value in arguments.items():
        if isinstance(value, Path):
            arguments[field] = Path(os.path.relpath(value, cwd))
    assert arguments["repo_root"] == Path("..")
    validation = _importer().import_evaluation_result(**arguments)
    assert validation.status == "accepted"
    assert (records["output_dir"] / "accepted-result.json").read_bytes() == records["bundle_path"].read_bytes()


def test_relative_dotdot_must_not_hide_a_symlink_input_component(records: dict[str, Any]) -> None:
    alias = records["repo_root"] / "evidence/alias"
    try:
        alias.symlink_to(records["bundle_path"].parent, target_is_directory=True)
    except OSError:
        pytest.skip("native symlink permission unavailable")
    records["bundle_path"] = alias / ".." / "run/result-bundle.json"
    _refused(records)


def test_detached_hash_is_checked_before_json_parse(
    records: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = _importer()
    records["bundle_path"].write_bytes(b"synthetic not JSON")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unverified bytes reached parsing")

    monkeypatch.setattr(importer, "parse_canonical_model", forbidden)
    _refused(records)


@pytest.mark.parametrize("digest", ["", "f" * 64, "F" * 64, "../synthetic"])
def test_detached_digest_must_be_separately_supplied(records: dict[str, Any], digest: str) -> None:
    records["expected_sha256"] = digest
    _refused(records)


@pytest.mark.parametrize("field", [
    "bundle_path", "protocol_path", "freeze_manifest_path", "d36_trial_tool_attestation_path",
    "d37_tool_attestation_path",
])
def test_import_refuses_noncanonical_each_artifact(records: dict[str, Any], field: str) -> None:
    path = records[field]
    raw = path.read_bytes() + b"\n"
    path.write_bytes(raw)
    if field == "bundle_path":
        records["expected_sha256"] = hashlib.sha256(raw).hexdigest()
    _refused(records)


@pytest.mark.parametrize("mutation", ["missing_marker", "bad_marker", "extra_entry", "flattened"])
def test_import_requires_complete_bound_d36_publication(
    records: dict[str, Any], mutation: str,
) -> None:
    publication = records["freeze_manifest_path"].parent
    marker = publication / ".d36-publication-state"
    if mutation == "missing_marker":
        marker.unlink()
    elif mutation == "bad_marker":
        data = json.loads(marker.read_bytes())
        data["files"][0]["sha256"] = "f" * 64
        marker.write_bytes(_raw(data))
    elif mutation == "extra_entry":
        (publication / "extra").write_bytes(b"synthetic")
    else:
        destination = publication.parent / "flattened.json"
        destination.write_bytes(records["freeze_manifest_path"].read_bytes())
        records["freeze_manifest_path"] = destination
    _refused(records)


@pytest.mark.parametrize("field,name", [
    ("bundle_path", "other.json"), ("protocol_path", "final_protocol.json"),
    ("d37_tool_attestation_path", "other-tool.json"),
    ("d36_trial_tool_attestation_path", "other-d36-tool.json"),
])
def test_import_requires_exact_run_and_publication_filenames(
    records: dict[str, Any], field: str, name: str,
) -> None:
    old = records[field]
    other = old.with_name(name)
    other.write_bytes(old.read_bytes())
    records[field] = other
    _refused(records)


def test_protocol_same_bytes_outside_declared_run_are_refused(records: dict[str, Any]) -> None:
    other = records["repo_root"] / "evidence/copied-protocol.json"
    other.write_bytes(records["protocol_path"].read_bytes())
    records["protocol_path"] = other
    _refused(records)


def test_import_crossbinds_topology_not_only_aggregate_counts(records: dict[str, Any]) -> None:
    protocol = json.loads(records["protocol_path"].read_bytes())
    protocol["case_categories"][1]["category_token"] = "b" * 64
    raw = _raw(protocol)
    records["protocol_path"].write_bytes(raw)
    _rewrite_bundle(records, ("protocol_sha256",), hashlib.sha256(raw).hexdigest())
    _refused(records)


@pytest.mark.parametrize("where", ["source", "run", "publication", "notignored", "existing"])
def test_output_is_new_ignored_untracked_disjoint_evidence(records: dict[str, Any], where: str) -> None:
    if where == "source":
        output = records["repo_root"] / "backend/new-result"
    elif where == "run":
        output = records["bundle_path"].parent / "accepted"
    elif where == "publication":
        output = records["freeze_manifest_path"].parent / "accepted"
    elif where == "notignored":
        output = records["repo_root"] / "unignored"
    else:
        output = records["output_dir"]
        output.mkdir()
    records["output_dir"] = output
    importer = _importer()
    with pytest.raises(importer.ResultImportError):
        importer.import_evaluation_result(**_arguments(records))
    assert not (output / "accepted-result.json").exists()


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_current_attestation_refuses_non_normal_index(records: dict[str, Any], flag: str) -> None:
    _git(records["repo_root"], "update-index", flag, D38_PATHS[0])
    _refused(records)


def test_current_tooling_must_be_clean(records: dict[str, Any]) -> None:
    (records["repo_root"] / D38_PATHS[0]).write_bytes(b"SYNTHETIC = 4\n")
    _refused(records)


def test_historical_verification_uses_recorded_blobs_not_current_head(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "verify_historical_attestation"), "historical Git verifier absent"
    tools.verify_historical_attestation(repo_root=records["repo_root"], attestation=records["d36"],
                                       expected_tool_name="d36_candidate_freezer_and_trial_host",
                                       source_paths=D36_PATHS)
    assert tools.historical_blinded_source_paths(repo_root=records["repo_root"],
                                                git_commit=records["old"]) == records["d37_paths"]
    tools.verify_historical_attestation(repo_root=records["repo_root"], attestation=records["d37"],
                                       expected_tool_name="d37_blinded_evaluator",
                                       source_paths=records["d37_paths"])


@pytest.mark.parametrize("mutation", [
    "omitted", "unknown", "missingcommit", "wrongname", "hash", "size", "aggregate",
])
def test_historical_verification_rejects_forged_source(records: dict[str, Any], mutation: str) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "verify_historical_attestation"), "historical Git verifier absent"
    data = records["d36"].model_dump(mode="json")
    if mutation == "omitted":
        data["files"].pop()
    elif mutation == "unknown":
        data["files"][-1]["path"] = "backend/evaluation/unknown.py"
    elif mutation == "missingcommit":
        data["git_commit"] = "f" * 40
    elif mutation == "wrongname":
        data["tool_name"] = "wrong_tool"
    elif mutation == "hash":
        data["files"][0]["sha256"] = "f" * 64
    elif mutation == "size":
        data["files"][0]["size"] += 1
    else:
        data["aggregate_sha256"] = "f" * 64
    forged = ToolAttestation.model_validate_json(_raw(data), strict=True)
    with pytest.raises(ValueError):
        tools.verify_historical_attestation(repo_root=records["repo_root"], attestation=forged,
                                           expected_tool_name="d36_candidate_freezer_and_trial_host",
                                           source_paths=D36_PATHS)


def test_historical_inventory_refuses_missing_required_path(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "historical_blinded_source_paths"), "historical inventory absent"
    _git(records["repo_root"], "rm", D37_REQUIRED[0])
    _git(records["repo_root"], "commit", "-q", "-m", "synthetic missing required")
    head = _git(records["repo_root"], "rev-parse", "HEAD").decode()
    with pytest.raises(ValueError):
        tools.historical_blinded_source_paths(repo_root=records["repo_root"], git_commit=head)


def test_historical_inventory_refuses_git_symlink_without_checkout(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "historical_blinded_source_paths"), "historical inventory absent"
    repo = records["repo_root"]
    blob = _git(repo, "hash-object", "-w", "--stdin", input=b"synthetic-target").decode()
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},backend/evaluation/synthetic_link.py")
    _git(repo, "commit", "-q", "-m", "synthetic link blob")
    head = _git(repo, "rev-parse", "HEAD").decode()
    with pytest.raises(ValueError):
        tools.historical_blinded_source_paths(repo_root=repo, git_commit=head)


@pytest.mark.parametrize("limit", ["metadata", "blob", "total"])
def test_historical_git_reads_are_bounded(
    records: dict[str, Any], monkeypatch: pytest.MonkeyPatch, limit: str,
) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "verify_historical_attestation"), "historical Git verifier absent"
    if limit == "metadata":
        monkeypatch.setattr(tools, "_MAX_GIT_METADATA_BYTES", 32)
    elif limit == "blob":
        monkeypatch.setattr(tools, "_MAX_SOURCE_BYTES", 4)
    else:
        monkeypatch.setattr(tools, "_MAX_SOURCE_TOTAL_BYTES", 16)
    with pytest.raises(ValueError):
        tools.verify_historical_attestation(repo_root=records["repo_root"], attestation=records["d36"],
                                           expected_tool_name="d36_candidate_freezer_and_trial_host",
                                           source_paths=D36_PATHS)


def test_historical_verifier_refuses_nonregular_selected_source(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    assert hasattr(tools, "verify_historical_attestation"), "historical Git verifier absent"
    repo = records["repo_root"]
    blob = _git(repo, "hash-object", "-w", "--stdin", input=b"synthetic-target").decode()
    _git(repo, "update-index", "--cacheinfo", f"120000,{blob},{D36_PATHS[0]}")
    _git(repo, "commit", "-q", "-m", "synthetic selected symlink")
    forged = records["d36"].model_copy(update={
        "git_commit": _git(repo, "rev-parse", "HEAD").decode(),
    })
    with pytest.raises(ValueError):
        tools.verify_historical_attestation(repo_root=repo, attestation=forged,
                                           expected_tool_name="d36_candidate_freezer_and_trial_host",
                                           source_paths=D36_PATHS)


def test_import_source_inventory_covers_transitive_local_ast_imports() -> None:
    importer = _importer()
    assert importer.IMPORT_SOURCE_PATHS == D38_PATHS
    backend = Path(__file__).resolve().parents[1]
    seen: set[str] = set()
    pending = ["evaluation.result_import", "scripts.import_evaluation_result"]
    while pending:
        module = pending.pop()
        path = backend / Path(*module.split("."))
        source = path.with_suffix(".py") if path.with_suffix(".py").is_file() else path / "__init__.py"
        if not source.is_file():
            continue
        relative = "backend/" + source.relative_to(backend).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        assert relative in D38_PATHS, f"unattested local source: {relative}"
        parts = module.split(".")
        for count in range(1, len(parts)):
            pending.append(".".join(parts[:count]))
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = parts[:-node.level]
                    target = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    target = node.module or ""
                pending.append(target)
                pending.extend(target + "." + alias.name for alias in node.names)


@pytest.mark.parametrize("arguments", [
    [], ["--bundle", "SYNTHETIC-PRIVATE-PATH"], ["--unknown", "SYNTHETIC-PRIVATE-PATH"],
    ["--help"],
])
def test_cli_argument_errors_are_fixed_redacted(arguments: list[str]) -> None:
    _importer()
    completed = subprocess.run([sys.executable, "-m", "scripts.import_evaluation_result", *arguments],
                               capture_output=True, cwd=Path(__file__).resolve().parents[1])
    if arguments == ["--help"]:
        assert completed.returncode == 0
    else:
        assert completed.returncode == 2
        assert completed.stdout == b""
        assert completed.stderr.replace(b"\r\n", b"\n") == b"evaluation result import refused\n"


def test_cli_runtime_failure_and_success_are_redacted(records: dict[str, Any]) -> None:
    _importer()
    args = _arguments(records)
    cli_flags = {
        "bundle_path": "--bundle", "expected_sha256": "--expected-sha256",
        "freeze_manifest_path": "--freeze-manifest", "protocol_path": "--protocol",
        "d36_trial_tool_attestation_path": "--d36-trial-tool-attestation",
        "d37_tool_attestation_path": "--d37-tool-attestation", "repo_root": "--repo-root",
        "output_dir": "--output",
    }
    argv = [sys.executable, "-m", "scripts.import_evaluation_result"]
    for field, flag in cli_flags.items():
        argv.extend([flag, str(args[field])])
    cwd = Path(__file__).resolve().parents[1]
    completed = subprocess.run(argv, capture_output=True, cwd=cwd)
    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    assert b"accepted" in completed.stdout
    assert str(records["repo_root"]).encode() not in completed.stdout
    assert b"Traceback" not in completed.stdout
    completed = subprocess.run(argv, capture_output=True, cwd=cwd)
    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr.replace(b"\r\n", b"\n") == b"evaluation result import refused\n"


def _publisher() -> Any:
    from evaluation import blinded_io

    assert hasattr(blinded_io, "publish_accepted_triplet"), "native triplet publisher absent"
    return blinded_io.publish_accepted_triplet


TRIPLET = {"accepted-result.json": b'{"synthetic":1}\n',
           "validation.json": b'{"synthetic":2}\n',
           "d38-tool-attestation.json": b'{"synthetic":3}\n'}


def _publish(output: Path) -> None:
    _publisher()(output_dir=output, accepted_result_bytes=TRIPLET["accepted-result.json"],
                 validation_bytes=TRIPLET["validation.json"],
                 tool_attestation_bytes=TRIPLET["d38-tool-attestation.json"])


def test_native_triplet_publishes_exact_files_and_refuses_empty_existing_final(tmp_path: Path) -> None:
    output = tmp_path / "accepted"
    _publish(output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == TRIPLET
    assert set(tmp_path.iterdir()) == {output}
    with pytest.raises((OSError, ValueError)):
        _publish(output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == TRIPLET
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((OSError, ValueError)):
        _publish(empty)
    assert list(empty.iterdir()) == []


def test_native_concurrent_publication_is_no_replace(tmp_path: Path) -> None:
    output = tmp_path / "accepted"
    _publisher()

    def attempt() -> bool:
        try:
            _publish(output)
        except (OSError, ValueError):
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    assert sorted(results) == [False, True]
    assert {path.name: path.read_bytes() for path in output.iterdir()} == TRIPLET


@pytest.mark.parametrize("after", [False, True])
def test_native_triplet_fault_before_or_after_rename_preserves_complete_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: bool,
) -> None:
    from evaluation.release_candidate import freeze

    _publisher()
    name = "_windows_move_directory_no_replace" if os.name == "nt" else "_linux_rename_directory_no_replace"
    native = getattr(freeze, name)

    def fail(source: Path, destination: Path) -> None:
        if after:
            native(source, destination)
        raise OSError("synthetic injected rename fault")

    monkeypatch.setattr(freeze, name, fail)
    output = tmp_path / "accepted"
    with pytest.raises((OSError, ValueError)):
        _publish(output)
    if after:
        assert {path.name: path.read_bytes() for path in output.iterdir()} == TRIPLET
    else:
        assert not output.exists()
        stages = list(tmp_path.glob(".d38-stage-*"))
        assert len(stages) == 1
        assert {path.name: path.read_bytes() for path in stages[0].iterdir()} == TRIPLET


@pytest.mark.skipif(os.name == "nt", reason="Windows no-delete anchors prevent directory replacement")
@pytest.mark.parametrize("target", ["stage", "parent"])
def test_native_triplet_lost_ownership_never_touches_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    _publisher()
    parent = tmp_path / "parent"
    parent.mkdir()
    original_write = os.write
    replaced: list[Path] = []

    def replace(descriptor: int, value: bytes) -> int:
        if not replaced:
            old = next(parent.glob(".d38-stage-*")) if target == "stage" else parent
            moved = tmp_path / "moved-owned"
            old.rename(moved)
            old.mkdir()
            (old / "replacement").write_bytes(b"SYNTHETIC REPLACEMENT MUST SURVIVE")
            replaced.append(old)
        return original_write(descriptor, value)

    monkeypatch.setattr(os, "write", replace)
    with pytest.raises((OSError, ValueError)):
        _publish(parent / "accepted")
    assert not (parent / "accepted").exists()
    assert {path.name for path in replaced[0].iterdir()} == {"replacement"}
    assert (replaced[0] / "replacement").read_bytes() == b"SYNTHETIC REPLACEMENT MUST SURVIVE"


@pytest.mark.skipif(os.name != "nt", reason="native Windows no-delete anchor test")
def test_windows_triplet_keeps_parent_and_stage_anchors_during_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publisher()
    parent = tmp_path / "parent"
    parent.mkdir()
    original_write = os.write
    attempts: list[Path] = []

    def attempt(descriptor: int, value: bytes) -> int:
        if not attempts:
            stage = next(parent.glob(".d38-stage-*"))
            for path in (stage, parent):
                with pytest.raises(OSError):
                    path.rename(tmp_path / "replacement")
                attempts.append(path)
        return original_write(descriptor, value)

    monkeypatch.setattr(os, "write", attempt)
    _publish(parent / "accepted")
    assert len(attempts) == 2
    assert {path.name: path.read_bytes() for path in (parent / "accepted").iterdir()} == TRIPLET


@pytest.mark.skipif(os.name != "nt", reason="native Windows stage-release-gap test")
def test_windows_parent_anchor_survives_stage_release_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation.release_candidate import freeze

    _publisher()
    parent = tmp_path / "parent"
    parent.mkdir()
    native = freeze._windows_move_directory_no_replace

    def at_gap(source: Path, destination: Path) -> None:
        with pytest.raises(OSError):
            parent.rename(tmp_path / "moved-parent")
        native(source, destination)

    monkeypatch.setattr(freeze, "_windows_move_directory_no_replace", at_gap)
    _publish(parent / "accepted")
    assert {path.name: path.read_bytes() for path in (parent / "accepted").iterdir()} == TRIPLET


@pytest.mark.parametrize("fault", ["write", "fsync", "readback", "fourthfile"])
def test_native_triplet_faults_never_publish_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    _publisher()
    original_write, original_read = os.write, os.read
    if fault == "write":
        monkeypatch.setattr(os, "write", lambda *args: (_ for _ in ()).throw(OSError("synthetic")))
    elif fault == "fsync":
        monkeypatch.setattr(os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("synthetic")))
    elif fault == "readback":
        monkeypatch.setattr(os, "read", lambda fd, size: original_read(fd, size).replace(b"1", b"9"))
    else:
        def extra(descriptor: int, value: bytes) -> int:
            stage = next(tmp_path.glob(".d38-stage-*"))
            (stage / "forbidden-fourth").write_bytes(b"synthetic")
            return original_write(descriptor, value)
        monkeypatch.setattr(os, "write", extra)
    with pytest.raises((OSError, ValueError)):
        _publish(tmp_path / "accepted")
    assert not (tmp_path / "accepted").exists()


@pytest.mark.parametrize("field", ["bundle_path", "protocol_path"])
def test_artifact_links_are_refused(records: dict[str, Any], field: str) -> None:
    path = records[field]
    target = path.with_name("synthetic-link-target")
    path.rename(target)
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("native symlink permission unavailable")
    _refused(records)


def test_import_reads_fixed_artifact_caps(records: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    importer = _importer()
    original_read = importer.read_regular
    caps: dict[str, int] = {}

    def read(path: Path, *, maximum: int) -> bytes:
        caps[path.name] = maximum
        return original_read(path, maximum=maximum)

    monkeypatch.setattr(importer, "read_regular", read)
    importer.import_evaluation_result(**_arguments(records))
    assert caps[".d36-publication-state"] == 1024
    assert caps["freeze-manifest.json"] == 16 * 1024 * 1024
    assert caps["d36-tool-attestation.json"] == 1024 * 1024
    assert caps["tool-attestation.json"] == 16 * 1024 * 1024
    assert caps["protocol.json"] == 64 * 1024 * 1024
    assert caps["result-bundle.json"] == 128 * 1024 * 1024


@pytest.mark.parametrize("after", [False, True])
def test_native_process_crash_before_after_rename_preserves_whole_triplet(
    tmp_path: Path, after: bool,
) -> None:
    _publisher()
    output = tmp_path / "accepted"
    script = '''
import os, sys
from pathlib import Path
from evaluation.blinded_io import publish_accepted_triplet
from evaluation.release_candidate import freeze
name = "_windows_move_directory_no_replace" if os.name == "nt" else "_linux_rename_directory_no_replace"
native = getattr(freeze, name)
def crash(source, final):
    if sys.argv[2] == "after":
        native(source, final)
    os._exit(42)
setattr(freeze, name, crash)
publish_accepted_triplet(output_dir=Path(sys.argv[1]), accepted_result_bytes=b'{"synthetic":1}\\n',
                        validation_bytes=b'{"synthetic":2}\\n', tool_attestation_bytes=b'{"synthetic":3}\\n')
'''
    completed = subprocess.run([sys.executable, "-c", script, str(output), "after" if after else "before"],
                               capture_output=True, cwd=Path(__file__).resolve().parents[1])
    assert completed.returncode == 42, completed.stderr.decode()
    assert completed.stdout == completed.stderr == b""
    if after:
        assert {path.name: path.read_bytes() for path in output.iterdir()} == TRIPLET
    else:
        assert not output.exists()
        stage = next(tmp_path.glob(".d38-stage-*"))
        assert {path.name: path.read_bytes() for path in stage.iterdir()} == TRIPLET


def test_historical_missing_recorded_blob_fails_closed(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    repo = records["repo_root"]
    blob = _git(repo, "rev-parse", f'{records["old"]}:{D36_PATHS[0]}').decode()
    object_path = repo / ".git/objects" / blob[:2] / blob[2:]
    object_path.chmod(stat.S_IREAD | stat.S_IWRITE)
    object_path.unlink()
    with pytest.raises(ValueError):
        tools.verify_historical_attestation(repo_root=repo, attestation=records["d36"],
                                           expected_tool_name="d36_candidate_freezer_and_trial_host",
                                           source_paths=D36_PATHS)


def test_historical_verification_does_not_trust_git_replace_objects(records: dict[str, Any]) -> None:
    from evaluation import tool_attestation as tools

    repo = records["repo_root"]
    _git(repo, "replace", records["old"], _git(repo, "rev-parse", "HEAD").decode())
    tools.verify_historical_attestation(repo_root=repo, attestation=records["d36"],
                                       expected_tool_name="d36_candidate_freezer_and_trial_host",
                                       source_paths=D36_PATHS)


@pytest.mark.parametrize("path,value", [
    (("evaluator_name",), "Another synthetic evaluator"),
    (("executed_at",), "2026-10-01T00:00:00Z"),
    (("excluded_cases", 0, "reason"), "both_not_approved"),
    (("sealed_evidence_sha256",), "c" * 64),
])
def test_new_trusted_digest_can_accept_private_unverifiable_claim_changes(
    records: dict[str, Any], path: tuple[str | int, ...], value: object,
) -> None:
    _rewrite_bundle(records, path, value)
    validation = _importer().import_evaluation_result(**_arguments(records))
    assert validation.accepted_bundle_sha256 == records["expected_sha256"]
    assert (records["output_dir"] / "accepted-result.json").read_bytes() == records["bundle_path"].read_bytes()
