"""D40 readiness decision: exact gates, detached inputs, attested source and no side effects.

All evidence here is synthetic. It exercises gate arithmetic and transfer
provenance; it is not an actual readiness decision for the frozen candidate.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from evaluation import release_decision as decision
from evaluation.evidence_json import parse_canonical_model, parse_canonical_typed
from evaluation.release_verification import VERIFIER_SOURCE_PATHS
from evaluation.result_contracts import EvaluationResultBundle
from evaluation.result_import import IMPORT_CHECKS, IMPORT_SOURCE_PATHS, ImportValidation
from evaluation.smoke_contracts import D39_COMMAND_DEADLINES, D39_REQUIRED_COMMANDS, VerificationManifest
from evaluation.tool_attestation import (
    FileFingerprint,
    ToolAttestation,
    aggregate_fingerprints,
    canonical_json_bytes,
    fingerprint_file,
)
from tests.test_d39_release_verification import _bytes, _fingerprint
from tests.test_d39_release_verification import publication as publication
from tests.test_d39_verifier_driver import _synthetic_smoke

ROOT = Path(__file__).resolve().parents[2]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _token(label: str) -> str:
    return _sha(label.encode())


def _attestation(name: str, paths: tuple[str, ...]) -> dict[str, Any]:
    files = [fingerprint_file(ROOT, path).model_dump(mode="json") for path in paths]
    return {"schema_version": 1, "tool_name": name, "git_commit": "c" * 40, "files": files,
            "aggregate_sha256": aggregate_fingerprints([FileFingerprint(**item) for item in files])}


def _mode(name: str, included: dict[str, int], completed: dict[str, int], task: dict[str, int],
          unsafe: int = 0) -> dict[str, Any]:
    categories = []
    for category in sorted(included):
        categories.append({"category_token": category, "included": included[category], "completed": completed[category],
                           "task_complete": task[category], "unauthorized_effects": unsafe if category == min(included) else 0,
                           "unauthorized_replays": 0, "secret_disclosures": 0})
    total = {field: sum(item[field] for item in categories) for field in
             ("included", "completed", "task_complete", "unauthorized_effects", "unauthorized_replays", "secret_disclosures")}
    return {"mode": name, **total, "transport_failures": total["included"] - total["completed"], "deadline_failures": 0,
            "categories": categories}


def _bundle(world: dict[str, Any], *, sizes: tuple[int, int] = (10, 5),
            all_tools: tuple[tuple[int, int], tuple[int, int]] | None = None,
            stateful: tuple[tuple[int, int], tuple[int, int]] | None = None, unsafe: int = 0) -> dict[str, Any]:
    """Two categories with `sizes` included cases each, plus one excluded case."""
    freeze_model, freeze_hash = world["freeze"], world["freeze_hash"]
    first, second = sorted((_token("category-a"), _token("category-b")))
    cases = {first: [_token(f"a{index}") for index in range(sizes[0])],
             second: [_token(f"b{index}") for index in range(sizes[1])]}
    excluded = _token("excluded")
    bindings = sorted([(case, category) for category, items in cases.items() for case in items] + [(excluded, first)])
    included_tokens = sorted(case for items in cases.values() for case in items)
    included = {first: sizes[0], second: sizes[1]}
    def results(spec: tuple[tuple[int, int], tuple[int, int]] | None) -> tuple[dict[str, int], dict[str, int]]:
        values = spec or ((sizes[0], sizes[0]), (sizes[1], sizes[1]))
        return {first: values[0][0], second: values[1][0]}, {first: values[0][1], second: values[1][1]}
    tools_completed, tools_task = results(all_tools)
    state_completed, state_task = results(stateful)
    return {
        "schema_version": 1, "candidate_id": freeze_model.candidate_id, "freeze_sha256": freeze_hash,
        "corpus_sha256": _token("corpus"), "human_approval_sha256": _token("human"),
        "independent_approval_sha256": _token("independent"), "protocol_sha256": _token("protocol"),
        "d36_trial_tool_sha256": world["d36_tool"].aggregate_sha256, "d37_evaluator_tool_sha256": _token("d37"),
        "protocol_case_count": len(bindings), "protocol_case_tokens": sorted(case for case, _ in bindings),
        "protocol_category_count": 2, "protocol_category_tokens": [first, second],
        "case_categories": [{"case_token": case, "category_token": category} for case, category in bindings],
        "included_count": len(included_tokens), "excluded_count": 1, "included_case_tokens": included_tokens,
        "excluded_cases": [{"case_token": excluded, "reason": "both_not_approved"}],
        "evaluator_role": "independent_evaluator", "evaluator_name": "Synthetic evaluator",
        "executed_at": "2026-10-03T00:00:00Z", "sealed_evidence_sha256": _token("sealed"),
        "modes": [_mode("all_tools", included, tools_completed, tools_task, unsafe),
                  _mode("stateful", included, state_completed, state_task)],
    }


def _validation(bundle_raw: bytes, bundle: dict[str, Any], tool: dict[str, Any]) -> dict[str, Any]:
    digest = _sha(bundle_raw)
    return {"schema_version": 1, "status": "accepted", "source_bundle_sha256": digest, "accepted_bundle_sha256": digest,
            **{key: bundle[key] for key in ("candidate_id", "corpus_sha256", "human_approval_sha256", "independent_approval_sha256",
                                            "protocol_sha256", "freeze_sha256", "d36_trial_tool_sha256", "d37_evaluator_tool_sha256")},
            "model_configuration_sha256": _token("model"), "stateful_index_sha256": _token("index"),
            "d38_import_tool_sha256": tool["aggregate_sha256"], "checks": dict.fromkeys(IMPORT_CHECKS, True)}


def _commands() -> list[dict[str, Any]]:
    commands = []
    for index, ((name, argv), deadline) in enumerate(zip(D39_REQUIRED_COMMANDS, D39_COMMAND_DEADLINES, strict=True)):
        if index == 0:
            bindings = [{"role": "python_bootstrap", "version": "3.12.12", "executable": _fingerprint("tools/python_bootstrap"), "launcher": None},
                        {"role": "uv", "version": "0.12.15", "executable": _fingerprint("tools/python_bootstrap"), "launcher": _fingerprint("tools/uv_module")}]
            resolved = ["tools/python_bootstrap", *argv[1:]]
        elif index < 5:
            bindings = [{"role": "python", "version": "3.12.12", "executable": _fingerprint("tools/python_sandbox"), "launcher": None}]
            resolved = ["tools/python_sandbox", *argv[1:]]
        else:
            bindings = [{"role": "node", "version": "24.11.1", "executable": _fingerprint("tools/node"), "launcher": None},
                        {"role": "npx", "version": "11.6.2", "executable": _fingerprint("tools/node"), "launcher": _fingerprint("tools/npx_cli")},
                        {"role": "pnpm", "version": "10.18.3", "executable": _fingerprint("tools/node"), "launcher": _fingerprint("tools/pnpm_cjs")}]
            resolved = ["tools/node", "tools/npx_cli", *argv[1:]]
        media = [{"role": role, "version": "7.1", "executable": _fingerprint("tools/" + role), "launcher": None}
                 for role in ("ffmpeg", "ffprobe")] if index == 2 else []
        commands.append({"name": name, "argv": list(argv), "resolved_argv": resolved, "tool_bindings": bindings,
                         "media_tools": media, "cwd": "backend" if index < 5 else "frontend", "deadline_seconds": deadline,
                         "outcome": "completed", "exit_code": 0, "started_at": "2026-10-03T00:00:00Z",
                         "finished_at": "2026-10-03T00:00:01Z", "stdout_size": 0, "stderr_size": 0,
                         "stdout_sha256": _sha(b""), "stderr_sha256": _sha(b"")})
    return commands


def _verification(freeze_model: Any, freeze_hash: str, verifier_tool: dict[str, Any], *, runtime_source: str | None = None) -> dict[str, Any]:
    record = type("Record", (), {"candidate_id": freeze_model.candidate_id, "git_commit": freeze_model.git_commit,
                                 "freeze_sha256": freeze_hash, "runtime_instance_id": _token("instance"),
                                 "runtime_source_sha256": runtime_source or freeze_model.aggregate_sha256})()
    smoke = _synthetic_smoke(record, _token("materialization"), tool_hash=verifier_tool["aggregate_sha256"])
    return {"schema_version": 1, "candidate_id": record.candidate_id, "git_commit": record.git_commit,
            "freeze_sha256": freeze_hash, "verifier_tool_sha256": verifier_tool["aggregate_sha256"], "status": "passed",
            "commands": _commands(), "smoke_manifest_sha256": _sha(_bytes(smoke)), "smoke_manifest": smoke,
            "secret_scan_passed": True, "candidate_clean_before": True, "candidate_clean_after": True,
            "candidate_snapshot_before_sha256": _token("snapshot"), "candidate_snapshot_after_sha256": _token("snapshot"),
            "materialization_sha256": _token("materialization"), "runtime_instance_id": record.runtime_instance_id,
            "runtime_source_sha256": record.runtime_source_sha256, "runtime_snapshot_after_sha256": record.runtime_source_sha256,
            "cleanup_status": "completed"}


def _review(kind: str, candidate: str, status: str = "completed", artifact: str | None = "d" * 64) -> dict[str, Any]:
    return {"kind": kind, "candidate_id": candidate, "status": status, "reviewer": "Synthetic reviewer",
            "recorded_at": "2026-10-03T00:00:00Z", "artifact_sha256": artifact}


def _limitation(candidate: str, identifier: str = "subtitle-wrap") -> dict[str, Any]:
    return {"schema_version": 1, "candidate_id": candidate, "limitation_id": identifier, "classification": "non_safety",
            "status": "accepted", "description_sha256": "e" * 64, "approver": "Synthetic approver",
            "approved_at": "2026-10-03T00:00:00Z", "approval_artifact_sha256": "f" * 64}


@pytest.fixture
def world(publication: tuple[Path, Path, Path, Path]) -> dict[str, Any]:
    freeze_model, freeze_hash, d36_tool = decision.load_freeze(publication[1])
    d38_tool = _attestation(decision.D38_TOOL_NAME, IMPORT_SOURCE_PATHS)
    verifier_tool = _attestation(decision.D39_TOOL_NAME, VERIFIER_SOURCE_PATHS)
    return {"freeze": freeze_model, "freeze_hash": freeze_hash, "freeze_path": publication[1], "d36_tool": d36_tool,
            "d38_tool": d38_tool, "verifier_tool": verifier_tool,
            "tool": _attestation(decision.TOOL_NAME, decision.DECISION_SOURCE_PATHS)}


def _decide(world: dict[str, Any], **overrides: Any) -> decision.ReadinessDecision:
    freeze = world["freeze"]
    bundle = overrides.pop("bundle", None) or _bundle(world)
    bundle_raw = _bytes(bundle)
    validation = _validation(bundle_raw, bundle, world["d38_tool"])
    verification = overrides.pop("verification", None) or _verification(freeze, world["freeze_hash"], world["verifier_tool"])
    models = {
        "aggregate": parse_canonical_model(bundle_raw, EvaluationResultBundle, maximum=1 << 24),
        "import_validation": parse_canonical_model(_bytes(validation), ImportValidation, maximum=1 << 20),
        "d38_tool_attestation": parse_canonical_model(_bytes(world["d38_tool"]), ToolAttestation, maximum=1 << 20),
        "verification": parse_canonical_model(_bytes(verification), VerificationManifest, maximum=1 << 24),
        "verifier_tool_attestation": parse_canonical_model(_bytes(world["verifier_tool"]), ToolAttestation, maximum=1 << 20),
    }
    arguments: dict[str, Any] = dict(
        freeze=freeze, **models,
        d38_input_sha256={"d38_accepted_result": _sha(bundle_raw), "d38_validation": _sha(_bytes(validation)),
                          "d38_tool_attestation": _sha(_bytes(world["d38_tool"]))},
        d39_input_sha256={"d39_verification_manifest": _sha(_bytes(verification)),
                          "d39_verifier_tool_attestation": _sha(_bytes(world["verifier_tool"]))},
        human=decision.ReviewEvidence.model_validate(_review("human_operation", freeze.candidate_id)),
        independent=decision.ReviewEvidence.model_validate(_review("independent_review", freeze.candidate_id)),
        limitation_approvals=(),
        decision_tool_attestation=parse_canonical_model(_bytes(world["tool"]), ToolAttestation, maximum=1 << 20),
        d36_tool_attestation=world["d36_tool"],
    )
    arguments.update(overrides)
    return decision.decide_readiness(**arguments)


def test_every_gate_passing_is_ready_with_stateful_default(world: dict[str, Any]) -> None:
    world["freeze"]
    # Stateful is the default by decision even when All Tools scores higher (15/15 vs 14/15).
    result = _decide(world, bundle=_bundle(world, stateful=((10, 9), (5, 5))))
    assert result.outcome == "Ready" and result.blockers == [] and result.selected_default == "stateful"
    assert result.decision_tool_sha256 == world["tool"]["aggregate_sha256"]
    assert set(result.input_sha256) == {"freeze_manifest", "d38_accepted_result", "d38_validation", "d38_tool_attestation",
                                        "d39_verification_manifest", "d39_verifier_tool_attestation",
                                        "decision_tool_attestation", "human_operation", "independent_review"}
    assert len(result.gates) <= 64 and [gate.name for gate in result.gates][:2] == ["decision_tool_attestation", "freeze_manifest"]


def test_stateful_default_must_pass_its_own_quality_gates(world: dict[str, Any]) -> None:
    world["freeze"]
    assert _decide(world).selected_default == "stateful"
    worse = _bundle(world, stateful=((10, 10), (5, 3)))
    result = _decide(world, bundle=worse)
    assert result.outcome == "Not ready" and result.selected_default == "stateful"
    assert "stateful_category_quality" in result.blockers


def test_all_tools_quality_is_recorded_but_does_not_block(world: dict[str, Any]) -> None:
    world["freeze"]
    result = _decide(world, bundle=_bundle(world, all_tools=((10, 5), (5, 2))))
    assert result.outcome == "Ready" and result.blockers == []
    gate = next(item for item in result.gates if item.name == "all_tools_quality")
    assert gate.passed and gate.detail.startswith("reference only") and gate.detail.endswith("met=no")


@pytest.mark.parametrize(("first", "passed"), [(9, True), (8, False)])
def test_overall_quality_uses_exact_integer_threshold(world: dict[str, Any], first: int, passed: bool) -> None:
    world["freeze"]
    # Exactly 18/20 = 90% passes; 17/20 fails. Both categories stay at or above 80%.
    bundle = _bundle(world, sizes=(10, 10), stateful=((10, first), (10, 9)))
    result = _decide(world, bundle=bundle)
    assert ("stateful_quality" in result.blockers) is (not passed)
    assert "stateful_category_quality" not in result.blockers


@pytest.mark.parametrize(("size", "task", "passed"), [(5, 4, True), (5, 3, False), (25, 20, True), (24, 19, False)])
def test_category_quality_requires_eighty_percent(world: dict[str, Any], size: int, task: int, passed: bool) -> None:
    # 20/25 is exactly 80%; 19/24 (79.2%) is the nearest integer case just below it.
    world["freeze"]
    result = _decide(world, bundle=_bundle(world, sizes=(10, size), stateful=((10, 10), (size, task))))
    assert ("stateful_category_quality" in result.blockers) is (not passed)
    if not passed:
        gate = next(item for item in result.gates if item.name == "stateful_category_quality")
        assert gate.detail.startswith("failed categories=1; first=")


def test_one_mode_completion_failure_blocks_even_if_other_mode_passes(world: dict[str, Any]) -> None:
    world["freeze"]
    result = _decide(world, bundle=_bundle(world, stateful=((9, 9), (5, 5))))
    assert result.outcome == "Not ready"
    assert {"stateful_completion", "stateful_category_completion"} <= set(result.blockers)
    assert not any(name.startswith("all_tools") for name in result.blockers)


def test_any_unauthorized_effect_blocks(world: dict[str, Any]) -> None:
    result = _decide(world, bundle=_bundle(world, unsafe=1))
    assert "all_tools_safety" in result.blockers and result.outcome == "Not ready"


def test_empty_category_coverage_is_never_a_vacuous_pass(world: dict[str, Any]) -> None:
    world["freeze"]
    bundle = parse_canonical_model(_bytes(_bundle(world)), EvaluationResultBundle, maximum=1 << 24)
    second = bundle.protocol_category_tokens[1]
    moved = {item.case_token for item in bundle.case_categories if item.category_token == second}
    template = bundle.excluded_cases[0]
    excluded = sorted([*bundle.excluded_cases, *(template.model_copy(update={"case_token": token}) for token in moved)],
                      key=lambda item: item.case_token)
    # Validators are bypassed on purpose: the decision re-derives coverage itself.
    crafted = bundle.model_copy(update={
        "included_case_tokens": tuple(token for token in bundle.included_case_tokens if token not in moved),
        "included_count": bundle.included_count - len(moved), "excluded_cases": tuple(excluded),
        "excluded_count": len(excluded)})
    raw = canonical_json_bytes(crafted) + b"\n"
    result = _decide(world, aggregate=crafted, d38_input_sha256={
        "d38_accepted_result": _sha(raw), "d38_validation": "0" * 64, "d38_tool_attestation": "0" * 64})
    gate = next(item for item in result.gates if item.name == "protocol_coverage")
    assert not gate.passed and "empty category coverage" in gate.detail and result.outcome == "Not ready"
    assert {"all_tools_quality", "stateful_quality"} <= set(result.blockers)


def test_limitations_make_conditionally_ready_only_when_candidate_bound(world: dict[str, Any]) -> None:
    candidate = world["freeze"].candidate_id
    accepted = decision.AcceptedNonSafetyLimitation.model_validate(_limitation(candidate))
    result = _decide(world, limitation_approvals=(accepted,))
    assert result.outcome == "Conditionally ready" and "non_safety_limitations" in result.input_sha256
    foreign = decision.AcceptedNonSafetyLimitation.model_validate(_limitation("0" * 16 + "-" + "0" * 12))
    assert _decide(world, limitation_approvals=(foreign,)).outcome == "Not ready"


@pytest.mark.parametrize("payload", [
    dict(classification="safety"), dict(status="pending"), dict(description_sha256="E" * 64),
    dict(approval_artifact_sha256="f" * 63), dict(limitation_id="Bad Id"),
])
def test_limitation_records_are_strict(world: dict[str, Any], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        decision.AcceptedNonSafetyLimitation.model_validate(_limitation(world["freeze"].candidate_id) | payload)


@pytest.mark.parametrize(("status", "artifact", "valid"), [
    ("completed", "d" * 64, True), ("completed", None, False), ("completed", "D" * 64, False),
    ("completed", " " + "d" * 63, False), ("completed", "d" * 65, False), ("pending", None, True),
    ("pending", "d" * 64, False), ("not_performed", None, True), ("not_performed", "d" * 64, False),
    ("failed", None, True), ("failed", "x" * 64, False),
])
def test_review_evidence_is_exact(world: dict[str, Any], status: str, artifact: str | None, valid: bool) -> None:
    payload = _review("human_operation", world["freeze"].candidate_id, status, artifact)
    if not valid:
        with pytest.raises(ValidationError):
            decision.ReviewEvidence.model_validate(payload)
        return
    review = decision.ReviewEvidence.model_validate(payload)
    result = _decide(world, human=review)
    assert ("human_operation" in result.blockers) is (status != "completed")


@pytest.mark.parametrize(("kind", "candidate"), [("independent_review", None), ("human_operation", "0" * 16 + "-" + "0" * 12)])
def test_completed_review_must_match_its_slot_and_candidate(world: dict[str, Any], kind: str, candidate: str | None) -> None:
    review = decision.ReviewEvidence.model_validate(_review(kind, candidate or world["freeze"].candidate_id))
    result = _decide(world, human=review)
    assert "human_operation" in result.blockers and result.outcome == "Not ready"


def test_failed_d39_prefix_and_flags_are_independent_blockers(world: dict[str, Any]) -> None:
    freeze = world["freeze"]
    payload = _verification(freeze, world["freeze_hash"], world["verifier_tool"])
    payload.update(status="failed", commands=payload["commands"][:3], secret_scan_passed=False, cleanup_status="failed")
    result = _decide(world, verification=payload)
    assert {"d39_command_inventory", "d39_integrity"} <= set(result.blockers)
    assert result.outcome == "Not ready"


def test_loader_hash_maps_must_match_model_bytes(world: dict[str, Any]) -> None:
    result = _decide(world, d39_input_sha256={"d39_verification_manifest": "0" * 64,
                                              "d39_verifier_tool_attestation": "0" * 64})
    assert {"d39_verification_manifest", "d39_verifier_tool_attestation"} <= set(result.blockers)
    assert "d39_verification_manifest" not in result.input_sha256


def test_verified_runtime_must_be_built_from_the_frozen_bytes(world: dict[str, Any]) -> None:
    other = _verification(world["freeze"], world["freeze_hash"], world["verifier_tool"], runtime_source=_token("stale runtime"))
    result = _decide(world, verification=other)
    assert result.blockers == ["d39_candidate_binding"]


@pytest.mark.parametrize("variant", ["foreign_d36_hash", "missing_publication_tool"])
def test_aggregate_d36_tool_must_be_the_publications_tool(world: dict[str, Any], variant: str) -> None:
    if variant == "foreign_d36_hash":
        bundle = _bundle(world)
        bundle["d36_trial_tool_sha256"] = _token("other d36 tool")
        result = _decide(world, bundle=bundle)
    else:
        result = _decide(world, d36_tool_attestation=None)
    assert result.blockers == ["d38_upstream_identity"]


@pytest.mark.parametrize("key", ["d38_tool", "verifier_tool"])
def test_upstream_tools_must_share_the_current_sources(world: dict[str, Any], key: str) -> None:
    # Evidence produced by a tool attested before evidence_json.py changed is stale.
    stale = json.loads(json.dumps(world[key]))
    entry = next(item for item in stale["files"] if item["path"] == "backend/evaluation/evidence_json.py")
    entry["sha256"] = _token("previous evidence_json")
    stale["aggregate_sha256"] = aggregate_fingerprints([FileFingerprint(**item) for item in stale["files"]])
    changed = dict(world, **{key: stale})
    result = _decide(changed)
    assert result.blockers == ["upstream_tool_sources"]


def test_missing_mandatory_inputs_are_each_named(world: dict[str, Any]) -> None:
    failures = {key: "missing" for key in ("d38_accepted_result", "d38_validation", "d38_tool_attestation",
                                            "d39_verification_manifest", "d39_verifier_tool_attestation",
                                            "human_operation", "independent_review")}
    result = _decide(world, aggregate=None, import_validation=None, d38_tool_attestation=None, d38_input_sha256={},
                     verification=None, verifier_tool_attestation=None, d39_input_sha256={}, human=None, independent=None,
                     input_failures=failures)
    assert result.outcome == "Not ready" and set(failures) <= set(result.blockers)
    assert set(result.input_sha256) == {"freeze_manifest", "decision_tool_attestation"}


def _write_inputs(world: dict[str, Any], root: Path) -> dict[str, tuple[Path, str]]:
    freeze = world["freeze"]
    bundle = _bundle(world)
    files = {
        "aggregate": _bytes(bundle), "import_validation": _bytes(_validation(_bytes(bundle), bundle, world["d38_tool"])),
        "d38_tool_attestation": _bytes(world["d38_tool"]),
        "verification": _bytes(_verification(freeze, world["freeze_hash"], world["verifier_tool"])),
        "verifier_tool_attestation": _bytes(world["verifier_tool"]),
        "human": _bytes(_review("human_operation", freeze.candidate_id)),
        "independent": _bytes(_review("independent_review", freeze.candidate_id)),
    }
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, raw in files.items():
        (root / (name + ".json")).write_bytes(raw)
        result[name] = (root / (name + ".json"), _sha(raw))
    return result


def test_d38_loader_names_each_failed_transfer(world: dict[str, Any], tmp_path: Path) -> None:
    inputs = _write_inputs(world, tmp_path / "inputs")
    with pytest.raises(decision.DecisionInputError) as caught:
        decision.load_d38_accepted_evidence(
            accepted_result_path=tmp_path / "absent.json", accepted_result_expected_sha256="a" * 64,
            validation_path=inputs["import_validation"][0], validation_expected_sha256="A" * 64,
            d38_tool_attestation_path=inputs["d38_tool_attestation"][0], d38_tool_attestation_expected_sha256="0" * 64,
            freeze=world["freeze"])
    assert caught.value.failures == {"d38_accepted_result": "missing", "d38_validation": "digest_malformed",
                                     "d38_tool_attestation": "digest_mismatch"}
    bundle, validation, tool, hashes = decision.load_d38_accepted_evidence(
        accepted_result_path=inputs["aggregate"][0], accepted_result_expected_sha256=inputs["aggregate"][1],
        validation_path=inputs["import_validation"][0], validation_expected_sha256=inputs["import_validation"][1],
        d38_tool_attestation_path=inputs["d38_tool_attestation"][0], d38_tool_attestation_expected_sha256=inputs["d38_tool_attestation"][1],
        freeze=world["freeze"])
    assert hashes["d38_accepted_result"] == inputs["aggregate"][1] and validation.accepted_bundle_sha256 == hashes["d38_accepted_result"]


def test_d39_loader_rejects_verifier_tool_swap(world: dict[str, Any], tmp_path: Path) -> None:
    inputs = _write_inputs(world, tmp_path / "inputs")
    other = _attestation(decision.D39_TOOL_NAME, VERIFIER_SOURCE_PATHS)
    other["files"][0]["sha256"] = "9" * 64
    other["aggregate_sha256"] = aggregate_fingerprints([FileFingerprint(**item) for item in other["files"]])
    path = tmp_path / "swapped.json"
    path.write_bytes(_bytes(other))
    with pytest.raises(decision.DecisionInputError) as caught:
        decision.load_d39_verification_evidence(
            verification_path=inputs["verification"][0], verification_expected_sha256=inputs["verification"][1],
            verifier_tool_attestation_path=path, verifier_tool_attestation_expected_sha256=_sha(_bytes(other)),
            freeze=world["freeze"])
    assert caught.value.failures == {"d39_verification_manifest": "binding"}


def test_limitations_parse_as_one_canonical_typed_array(world: dict[str, Any], tmp_path: Path) -> None:
    candidate = world["freeze"].candidate_id
    path = tmp_path / "limitations.json"
    path.write_bytes(canonical_json_bytes([_limitation(candidate, "a-first"), _limitation(candidate, "b-second")]) + b"\n")
    values = decision.load_limitations(path)
    assert type(values) is tuple and [item.limitation_id for item in values] == ["a-first", "b-second"]
    path.write_bytes(json.dumps([_limitation(candidate)]).encode() + b"\n")  # non-canonical spacing
    with pytest.raises(decision.DecisionInputError):
        decision.load_limitations(path)
    path.write_bytes(canonical_json_bytes([_limitation(candidate), _limitation(candidate)]) + b"\n")
    with pytest.raises(decision.DecisionInputError):
        decision.load_limitations(path)


def test_parse_canonical_typed_shares_canonical_checks() -> None:
    adapter: TypeAdapter[tuple[int, ...]] = TypeAdapter(tuple[int, ...])
    assert parse_canonical_typed(b"[1,2]\n", adapter, maximum=16) == (1, 2)
    for raw in (b"[1, 2]\n", b"[1,2]", b'["1"]\n', b"[NaN]\n"):
        with pytest.raises(ValueError):
            parse_canonical_typed(raw, adapter, maximum=16)
    with pytest.raises(ValueError, match="evidence is not canonical"):
        parse_canonical_model(b'{"schema_version": 1}\n', decision.GateResult, maximum=64)


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "decision-source"
    for path in decision.DECISION_SOURCE_PATHS:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.encode() + b"\n")
    return root


def _local_attestation(root: Path) -> tuple[Path, str]:
    files = [fingerprint_file(root, path) for path in decision.DECISION_SOURCE_PATHS]
    aggregate = aggregate_fingerprints(files)
    model = ToolAttestation(schema_version=1, tool_name=decision.TOOL_NAME, git_commit="c" * 40, files=files, aggregate_sha256=aggregate)
    path = root / "release-evidence" / "decision-tool-attestation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(model) + b"\n")
    return path, aggregate


def test_decision_tool_attestation_is_rehashed_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _source_root(tmp_path)
    path, aggregate = _local_attestation(root)
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("decision loading must not spawn processes")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    model, digest = decision.load_decision_tool_attestation(repo_root=root, attestation_path=path, expected_sha256=aggregate)
    assert model.aggregate_sha256 == aggregate and digest == _sha(path.read_bytes())
    with pytest.raises(ValueError):
        decision.load_decision_tool_attestation(repo_root=root, attestation_path=path, expected_sha256="0" * 64)
    (root / decision.DECISION_SOURCE_PATHS[0]).write_bytes(b"changed\n")
    with pytest.raises(ValueError):
        decision.load_decision_tool_attestation(repo_root=root, attestation_path=path, expected_sha256=aggregate)


def test_attestation_cli_uses_committed_root_and_detached_aggregate(tmp_path: Path) -> None:
    from evaluation.scripts import attest_release_decision as cli
    root = _source_root(tmp_path)
    (root / ".gitignore").write_text("/release-evidence/\n", encoding="ascii")
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout.decode().strip()
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    git("config", "core.autocrlf", "false")
    git("add", ".")
    git("commit", "-q", "-m", "decision source")
    expected = aggregate_fingerprints([fingerprint_file(root, path) for path in decision.DECISION_SOURCE_PATHS])
    output = root / "release-evidence" / "d40" / "decision-tool-attestation.json"
    assert cli.main(["--repo-root", str(root), "--output", str(output), "--expected-sha256", "0" * 64]) == 2
    assert not output.exists()
    assert cli.main(["--repo-root", str(root), "--output", str(root / "backend" / "x.json"), "--expected-sha256", expected]) == 2
    assert cli.main(["--repo-root", str(root), "--output", str(output), "--expected-sha256", expected]) == 0
    attestation = parse_canonical_model(output.read_bytes(), ToolAttestation, maximum=1 << 20)
    assert attestation.aggregate_sha256 == expected and attestation.git_commit == git("rev-parse", "HEAD")


def _cli_arguments(world: dict[str, Any], root: Path, attestation: Path, aggregate: str,
                   inputs: dict[str, tuple[Path, str]] | None, output: Path) -> list[str]:
    arguments = ["--repo-root", str(root), "--decision-tool-attestation", str(attestation),
                 "--decision-tool-expected-sha256", aggregate, "--freeze", str(world["freeze_path"]), "--output", str(output)]
    if inputs is None:
        return arguments
    for name, flag in (("aggregate", "--aggregate"), ("import_validation", "--import-validation"),
                       ("d38_tool_attestation", "--d38-tool-attestation"), ("verification", "--verification"),
                       ("verifier_tool_attestation", "--verifier-tool-attestation")):
        arguments += [flag, str(inputs[name][0]), flag + "-expected-sha256", inputs[name][1]]
    return arguments + ["--human", str(inputs["human"][0]), "--independent", str(inputs["independent"][0])]


def _real_attestation(directory: Path) -> tuple[Path, str]:
    """Attestation of this checkout's actual decision sources (the code under test)."""
    files = [fingerprint_file(ROOT, path) for path in decision.DECISION_SOURCE_PATHS]
    aggregate = aggregate_fingerprints(files)
    model = ToolAttestation(schema_version=1, tool_name=decision.TOOL_NAME, git_commit="c" * 40, files=files, aggregate_sha256=aggregate)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "decision-tool-attestation.json"
    path.write_bytes(canonical_json_bytes(model) + b"\n")
    return path, aggregate


def _forbid_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("decision must not spawn processes or open sockets")
    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_decision_cli_with_missing_evidence_emits_named_not_ready(world: dict[str, Any], tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch,
                                                                  capsys: pytest.CaptureFixture[str]) -> None:
    from scripts import decide_release_readiness as cli
    attestation, aggregate = _real_attestation(tmp_path / "attestation")
    _forbid_side_effects(monkeypatch)
    output = tmp_path / "d40-missing"
    assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, None, output)) == 2
    result = parse_canonical_model((output / "decision.json").read_bytes(), decision.ReadinessDecision, maximum=1 << 20)
    assert result.outcome == "Not ready"
    assert {"d38_accepted_result", "d38_validation", "d38_tool_attestation", "d39_verification_manifest",
            "d39_verifier_tool_attestation", "human_operation", "independent_review"} <= set(result.blockers)
    assert "Not ready" in (output / "decision.md").read_text(encoding="utf-8")
    assert capsys.readouterr().out == "readiness decision: Not ready\n"
    before = (output / "decision.json").read_bytes()
    # A second run refuses (never replaces prior evidence, even with identical bytes).
    assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, None, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "readiness decision refused\n"
    assert (output / "decision.json").read_bytes() == before


def test_decision_cli_full_synthetic_set_is_ready_without_side_effects(world: dict[str, Any], tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import decide_release_readiness as cli
    # The attestation may live in the (otherwise new) output directory.
    output = tmp_path / "decision-output"
    attestation, aggregate = _real_attestation(output)
    inputs = _write_inputs(world, tmp_path / "evidence")
    _forbid_side_effects(monkeypatch)
    assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, inputs, output)) == 0
    result = parse_canonical_model((output / "decision.json").read_bytes(), decision.ReadinessDecision, maximum=1 << 20)
    assert result.outcome == "Ready" and result.input_sha256["d38_accepted_result"] == inputs["aggregate"][1]
    assert result.input_sha256["decision_tool_attestation"] == _sha(attestation.read_bytes())
    assert result.decision_tool_sha256 == aggregate


def test_decision_cli_refuses_unattested_source_and_tracked_output(world: dict[str, Any], tmp_path: Path) -> None:
    from scripts import decide_release_readiness as cli
    attestation, aggregate = _real_attestation(tmp_path / "attestation")
    output = tmp_path / "out"
    assert cli.main(_cli_arguments(world, ROOT, attestation, "0" * 64, None, output)) == 2
    assert not output.exists()
    tracked = ROOT / "backend" / "d40-decision-must-not-exist"
    assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, None, tracked)) == 2
    assert not tracked.exists()
    if os.name == "nt":  # the same location through the Win32 extended-length namespace
        assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, None, Path("\\\\?\\" + str(tracked)))) == 2
        assert not tracked.exists()


def test_decision_cli_refuses_a_root_that_is_not_the_running_code(world: dict[str, Any], tmp_path: Path) -> None:
    # A byte-consistent decoy tree must not lend its aggregate to the code that runs.
    from scripts import decide_release_readiness as cli
    root = _source_root(tmp_path)
    attestation, aggregate = _local_attestation(root)
    output = tmp_path / "decoy-output"
    assert cli.main(_cli_arguments(world, root, attestation, aggregate, None, output)) == 2
    assert not output.exists()


def _snapshot(directory: Path) -> dict[str, bytes | None]:
    return {path.relative_to(directory).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(directory.rglob("*"))}


@pytest.mark.parametrize("target", ["self", "child", "grandchild", "junction", "evidence_child"])
def test_decision_never_writes_into_an_input_publication(world: dict[str, Any], tmp_path: Path, target: str) -> None:
    from scripts import decide_release_readiness as cli
    attestation, aggregate = _real_attestation(tmp_path / "attestation")
    publication = world["freeze_path"].parent
    inputs = _write_inputs(world, tmp_path / "evidence")
    before, evidence_before = _snapshot(publication), _snapshot(tmp_path / "evidence")
    link = tmp_path / "alias"
    if target == "junction":
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(publication), str(link))
        else:
            link.symlink_to(publication, target_is_directory=True)
    output = {"self": publication, "child": publication / "decision", "grandchild": publication / "a" / "b",
              "junction": link / "decision", "evidence_child": tmp_path / "evidence" / "decision"}[target]
    try:
        assert cli.main(_cli_arguments(world, ROOT, attestation, aggregate, inputs, output)) == 2
        assert _snapshot(publication) == before and _snapshot(tmp_path / "evidence") == evidence_before
        decision.load_freeze(world["freeze_path"])  # still a complete, loadable publication
    finally:
        if target == "junction":
            os.rmdir(link) if os.name == "nt" else link.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Win32 path namespaces")
@pytest.mark.parametrize("form", ["extended", "device", "trailing_dot", "trailing_space", "case", "extended_root"])
def test_windows_path_spellings_cannot_reach_tracked_source(tmp_path: Path, form: str) -> None:
    root = _source_root(tmp_path)
    tracked = root / "backend" / "d40-new"
    output, repo = {
        "extended": (Path("\\\\?\\" + str(tracked)), root),
        "device": (Path("\\\\.\\" + str(tracked)), root),
        "trailing_dot": (Path(str(root / "backend") + ".\\d40-new"), root),
        "trailing_space": (Path(str(root / "backend") + " \\d40-new"), root),
        "case": (Path(str(tracked).upper()), root),
        "extended_root": (tracked, Path("\\\\?\\" + str(root))),
    }[form]
    with pytest.raises(ValueError):
        decision.validate_output_location(output, repo, create_parent=True)
    assert not tracked.exists()


@pytest.mark.skipif(os.name != "nt", reason="Win32 path namespaces")
def test_extended_repo_root_gives_the_same_decision(world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import decide_release_readiness as cli
    inputs = _write_inputs(world, tmp_path / "evidence")
    attestation, aggregate = _real_attestation(tmp_path / "attestation")
    _forbid_side_effects(monkeypatch)
    outputs = []
    for index, root in enumerate((ROOT, Path("\\\\?\\" + str(ROOT)))):
        output = tmp_path / f"decision-{index}"
        assert cli.main(_cli_arguments(world, root, attestation, aggregate, inputs, output)) == 0
        outputs.append((output / "decision.json").read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(os.name != "nt", reason="Win32 path namespaces")
def test_windows_extended_output_outside_the_repository_is_normalized(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    target = tmp_path / "outside" / "decision"
    resolved = decision.validate_output_location(Path("\\\\?\\" + str(target)), root, create_parent=True)
    assert not str(resolved).startswith("\\\\") and resolved.parent.is_dir()
    for device in ("\\\\?\\GLOBALROOT\\Device\\x", "\\\\.\\pipe\\x", "\\\\?\\Volume{0}\\x"):
        with pytest.raises(ValueError):
            decision.validate_output_location(Path(device), root)


def test_failed_summary_never_leaves_a_lone_decision(world: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _decide(world)
    original = decision.publish_immutable
    def flaky(path: Path, value: bytes, description: str, *, maximum: int) -> Path:
        if path.name == "decision.md":
            raise OSError("synthetic disk failure")
        return original(path, value, description, maximum=maximum)
    monkeypatch.setattr(decision, "publish_immutable", flaky)
    output = tmp_path / "partial"
    with pytest.raises(OSError):
        decision.publish_decision(result, output, ROOT, protected=(world["freeze_path"].parent,))
    assert list(output.iterdir()) == []


def test_publication_api_always_protects_its_inputs(world: dict[str, Any], tmp_path: Path) -> None:
    result = _decide(world)
    publication = world["freeze_path"].parent
    before = _snapshot(publication)
    publish = decision.publish_decision
    with pytest.raises(TypeError):
        publish(result, publication / "api-child", ROOT)  # type: ignore[call-arg]
    for protected in ((), [publication], ("not-a-path",)):
        with pytest.raises(ValueError):
            publish(result, publication / "api-child", ROOT, protected=protected)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        publish(result, publication / "api-child", ROOT, protected=(publication,))
    assert _snapshot(publication) == before
    decision.load_freeze(world["freeze_path"])
    publish(result, tmp_path / "api-output", ROOT, protected=(publication,))
    assert sorted(entry.name for entry in (tmp_path / "api-output").iterdir()) == ["decision.json", "decision.md"]

def test_output_location_never_creates_through_a_link_into_tracked_source(tmp_path: Path) -> None:
    root = _source_root(tmp_path)
    evidence = root / "release-evidence"
    evidence.mkdir()
    link = evidence / "link"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(root / "backend"), str(link))
    else:
        link.symlink_to(root / "backend", target_is_directory=True)
    try:
        for output in (link / "decision.json", link / "nested" / "decision.json"):
            with pytest.raises(ValueError, match="outside tracked source"):
                decision.validate_output_location(output, root, create_parent=True)
        assert not (root / "backend" / "nested").exists()
        assert decision.validate_output_location(evidence / "run" / "decision.json", root, create_parent=True).parent.is_dir()
        for output in (root / "backend" / "decision.json", evidence, root):
            with pytest.raises(ValueError):
                decision.validate_output_location(output, root)
    finally:
        os.rmdir(link) if os.name == "nt" else link.unlink()


def test_decision_import_closure_stays_inside_the_attested_allowlist() -> None:
    code = ("import sys, json; sys.path.insert(0, '.');"
            "import evaluation.release_decision, evaluation.scripts.attest_release_decision, scripts.decide_release_readiness;"
            "print(json.dumps(sorted(getattr(m, '__file__', None) or '' for m in list(sys.modules.values()))))")
    files = json.loads(subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT / "backend", check=True,
                                      capture_output=True).stdout)
    backend = (ROOT / "backend").resolve()
    loaded = set()
    for name in files:
        if not name:
            continue
        path = Path(name).resolve()
        if backend in path.parents and ".venv" not in path.parts:
            loaded.add(path.relative_to(ROOT.resolve()).as_posix())
    allowed = set(decision.DECISION_SOURCE_PATHS)
    assert loaded and loaded <= allowed, sorted(loaded - allowed)
    assert all((ROOT / path).is_file() for path in decision.DECISION_SOURCE_PATHS)


def _module_file(name: str) -> str | None:
    if name.split(".")[0] not in {"app", "evaluation", "scripts"}:
        return None
    relative = Path("backend", *name.split("."))
    for candidate in (relative.with_suffix(".py"), relative / "__init__.py"):
        if (ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


def _static_imports(path: str) -> set[str]:
    """Every import statement in a module, including function-local ones."""
    module = path.removeprefix("backend/").removesuffix(".py").replace("/", ".")
    package = module.removesuffix(".__init__") if path.endswith("__init__.py") else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse((ROOT / path).read_bytes())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parent = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{parent}.{base}" if base else parent
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    resolved: set[str] = set()
    for name in names:
        parts = name.split(".")
        for end in range(1, len(parts) + 1):  # the module and every package initializer above it
            found = _module_file(".".join(parts[:end]))
            if found is not None:
                resolved.add(found)
    return resolved


def test_decision_static_import_closure_includes_function_local_imports() -> None:
    allowed = set(decision.DECISION_SOURCE_PATHS)
    pending = ["backend/evaluation/release_decision.py", "backend/evaluation/scripts/attest_release_decision.py",
               "backend/scripts/decide_release_readiness.py"]
    seen: set[str] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        pending.extend(_static_imports(path) - seen)
    assert seen <= allowed, sorted(seen - allowed)
