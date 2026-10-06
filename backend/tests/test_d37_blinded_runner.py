from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evaluation.blinded_contracts import (
    MAX_PROTOCOL_BYTES,
    MAX_PROTOCOL_CASES,
    EvaluationProtocol,
    CaseCategoryBinding,
    case_category_bindings,
    maximum_protocol_serialized_bytes,
    opaque_case_token,
    opaque_category_token,
    token_key,
)
from evaluation.blinded_io import (
    ensure_writable_directory as _ensure_writable_directory,
    publish_immutable as _publish_immutable,
)
from evaluation.case_normalization import normalize_case
from evaluation.blinded_runner import (
    _D36_SOURCE_PATHS,
    _D37_SOURCE_PATHS,
    _d37_source_paths,
    _load_records,
    case_to_unlabeled,
    run_blinded_evaluation,
    write_run_protocol_exclusive,
)
from evaluation.blinded_runtime import (
    HOST_OUTPUT_CAP_BYTES,
    canonical_candidate_root as _canonical_candidate_root,
    close_candidate_anchor as _close_candidate_anchor,
    invoke_trial_host as _invoke_trial_host,
    open_candidate_anchor as _open_candidate_anchor,
)
from evaluation.blinded_scoring import score_trial
from evaluation.contracts import Case
from evaluation.corpus import (
    case_digest,
    corpus_digest,
    eligibility,
    load_cases,
    load_review,
)
from evaluation.result_contracts import (
    MAX_RESULT_BUNDLE_BYTES,
    CategoryResult,
    EvaluationResultBundle,
    ModeResult,
    approval_partition,
    maximum_result_bundle_serialized_bytes,
)
from evaluation.release_candidate.contracts import CandidateControl, CompletionMarker, FreezeManifest
from evaluation.release_candidate.fingerprints import aggregate_fingerprints
from evaluation.sealed_evidence import seal_evidence
from evaluation.tool_attestation import (
    FileFingerprint,
    ToolAttestation,
    attest_tool,
    canonical_json_bytes,
)
from evaluation.unlabeled_contracts import UnlabeledTrialCase

FIXTURES = Path(__file__).parent / "fixtures" / "blinded"
SYNTHETIC_KEY = bytes(range(32))
CASE_TOKENS = (
    "3803e6c2ed9d44b3b8b7e2cdec697221bcc4accf0530d3c90627b672249638c0",
    "a68dbb976cdcc2e99970a3d622629f0a03b1437df63e9678887433d0ef36bcfd",
    "b41f76271d1f585b9eb4bc8ae64164e053e0f4d9f3c4b05bf54ff6fcc58867a1",
    "b4ea897a8b2a0e81ef6f65e81c91bc791f220be1f33789408e4a2215975dbbef",
    "e3b8c902c18aa3bede1a4c6edaaae7bf97fbd30eff96c42d102a9cffdfe5c849",
)
CATEGORY_TOKENS = (
    "457a76aeb467da2b9216e9a4c8ace6c6ac433279beadedce04cbdfef3b68ab8c",
    "f2c22f7b5fc64960d2e64e1f86851619c99122f4e36e74f7e325249b5f4520e1",
)
BINDINGS = (
    {"case_token": CASE_TOKENS[0], "category_token": CATEGORY_TOKENS[0]},
    {"case_token": CASE_TOKENS[1], "category_token": CATEGORY_TOKENS[0]},
    {"case_token": CASE_TOKENS[2], "category_token": CATEGORY_TOKENS[0]},
    {"case_token": CASE_TOKENS[3], "category_token": CATEGORY_TOKENS[1]},
    {"case_token": CASE_TOKENS[4], "category_token": CATEGORY_TOKENS[1]},
)
INCLUDED = (CASE_TOKENS[0], CASE_TOKENS[4])
EXCLUDED = (
    {"case_token": CASE_TOKENS[1], "reason": "independent_not_approved"},
    {"case_token": CASE_TOKENS[2], "reason": "human_not_approved"},
    {"case_token": CASE_TOKENS[3], "reason": "both_not_approved"},
)
HASHES = {
    "freeze_sha256": "1" * 64,
    "corpus_sha256": "2" * 64,
    "human_approval_sha256": "3" * 64,
    "independent_approval_sha256": "4" * 64,
    "protocol_sha256": "5" * 64,
    "d36_trial_tool_sha256": "6" * 64,
    "d37_evaluator_tool_sha256": "7" * 64,
}


def _cases() -> list[Case]:
    return load_cases(FIXTURES / "synthetic-held-out.jsonl")


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tool_repository(root: Path) -> tuple[Path, str]:
    repository = root / "tool-repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "d37@example.invalid")
    _git(repository, "config", "user.name", "D37 Test")
    (repository / "a.py").write_bytes(b"A = 1\n")
    (repository / "b.py").write_bytes(b"B = 1\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "tool source")
    return repository, _git(repository, "rev-parse", "HEAD")


def _protocol_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": "d35-synthetic-candidate",
        "modes": ("all_tools", "stateful"),
        "per_call_deadline_seconds": 180,
        "maximum_model_calls": 4,
        "isolation": "fresh_case_state_under_source_group",
        "corpus_sha256": HASHES["corpus_sha256"],
        "human_approval_sha256": HASHES["human_approval_sha256"],
        "independent_approval_sha256": HASHES["independent_approval_sha256"],
        "freeze_sha256": HASHES["freeze_sha256"],
        "d36_trial_tool_sha256": HASHES["d36_trial_tool_sha256"],
        "d37_evaluator_tool_sha256": HASHES["d37_evaluator_tool_sha256"],
        "model_configuration_sha256": "8" * 64,
        "stateful_index_sha256": "9" * 64,
        "category_count": 2,
        "category_tokens": CATEGORY_TOKENS,
        "case_count": 5,
        "case_tokens": CASE_TOKENS,
        "case_categories": BINDINGS,
    }


def _category_results() -> list[dict[str, Any]]:
    return [
        {
            "category_token": CATEGORY_TOKENS[0],
            "included": 1,
            "completed": 1,
            "task_complete": 1,
            "unauthorized_effects": 0,
            "unauthorized_replays": 0,
            "secret_disclosures": 0,
        },
        {
            "category_token": CATEGORY_TOKENS[1],
            "included": 1,
            "completed": 1,
            "task_complete": 1,
            "unauthorized_effects": 0,
            "unauthorized_replays": 0,
            "secret_disclosures": 0,
        },
    ]


def _mode(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "included": 2,
        "completed": 2,
        "task_complete": 2,
        "unauthorized_effects": 0,
        "unauthorized_replays": 0,
        "secret_disclosures": 0,
        "transport_failures": 0,
        "deadline_failures": 0,
        "categories": tuple(_category_results()),
    }


def _bundle_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate_id": "d35-synthetic-candidate",
        **HASHES,
        "protocol_case_count": 5,
        "protocol_case_tokens": CASE_TOKENS,
        "protocol_category_count": 2,
        "protocol_category_tokens": CATEGORY_TOKENS,
        "case_categories": BINDINGS,
        "included_count": 2,
        "excluded_count": 3,
        "included_case_tokens": INCLUDED,
        "excluded_cases": EXCLUDED,
        "evaluator_role": "independent_evaluator",
        "evaluator_name": "Synthetic independent evaluator",
        "executed_at": "2026-09-20T00:00:00Z",
        "sealed_evidence_sha256": "a" * 64,
        "modes": (_mode("all_tools"), _mode("stateful")),
    }


def test_trial_resume_rejects_lexical_overflow_before_model_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import blinded_runner as runner

    raw = canonical_json_bytes({
        "schema_version": 1, "protocol_sha256": "1" * 64,
        "case_token": "x" * 8191, "category_token": "2" * 64,
        "mode": "all_tools", "outcome": "transport_failure", "score": None,
        "candidate_snapshot_sha256": None,
    }) + b"\n"
    path = tmp_path / "trial-result.json"
    path.write_bytes(raw)

    def forbidden_parse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("oversized lexical token reached schema construction")

    monkeypatch.setattr(runner._TrialRecord, "model_validate_json", forbidden_parse)
    with pytest.raises(ValueError, match="evidence string limit"):
        runner._load_trial_record(path, "1" * 64)


def test_index_fingerprint_streams_blobs_above_json_cap_in_lexical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import blinded_runner as runner

    index = tmp_path / "index"
    (index / "a").mkdir(parents=True)
    (index / "z").write_bytes(b"z")
    blob = index / "a" / "bundle-synthetic.json"
    digest = hashlib.sha256()
    with blob.open("wb") as stream:
        for _ in range(17):
            chunk = b"x" * (1024 * 1024)
            stream.write(chunk)
            digest.update(chunk)
    expected = hashlib.sha256(canonical_json_bytes([
        {"path": "a/bundle-synthetic.json", "size": 17 * 1024 * 1024,
         "sha256": digest.hexdigest()},
        {"path": "z", "size": 1, "sha256": hashlib.sha256(b"z").hexdigest()},
    ])).hexdigest()

    def forbidden_read(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("index fingerprint must not construct JSON/blob bytes")

    monkeypatch.setattr(runner, "_read_regular", forbidden_read)
    assert runner._fingerprint_directory(index) == expected


@pytest.mark.parametrize(("name", "size"), [("manifest.json", 64_001),
                                            ("bundle-synthetic.json", 64_000_001)])
def test_index_fingerprint_rejects_decimal_schema_caps_before_read(
    tmp_path: Path, name: str, size: int,
) -> None:
    from evaluation.blinded_runner import _fingerprint_directory

    index = tmp_path / "index"
    index.mkdir()
    with (index / name).open("wb") as stream:
        stream.truncate(size)
    with pytest.raises(ValueError, match="size limit"):
        _fingerprint_directory(index)


def test_category_denominator_tally_is_linear_without_wallclock_assertions() -> None:
    comparisons = [0]

    class CountedCategory(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            comparisons[0] += 1
            return super().__eq__(other)

    count = 2048
    cases = tuple(f"{i:064x}" for i in range(count * 2))
    categories = tuple(CountedCategory(f"{i + count * 2:064x}") for i in range(count))
    bindings = tuple(
        CaseCategoryBinding.model_construct(case_token=token, category_token=categories[i // 2])
        for i, token in enumerate(cases)
    )
    results = tuple(CategoryResult(
        category_token=str(token), included=1, completed=1, task_complete=1,
        unauthorized_effects=0, unauthorized_replays=0, secret_disclosures=0,
    ) for token in categories)
    modes = tuple(ModeResult(
        mode=mode, included=count, completed=count, task_complete=count,
        unauthorized_effects=0, unauthorized_replays=0, secret_disclosures=0,
        transport_failures=0, deadline_failures=0, categories=results,
    ) for mode in ("all_tools", "stateful"))
    data = {**_bundle_data(), "protocol_case_count": len(cases),
            "protocol_case_tokens": cases, "protocol_category_count": count,
            "protocol_category_tokens": categories, "case_categories": bindings,
            "included_count": count, "excluded_count": count,
            "included_case_tokens": cases[::2],
            "excluded_cases": tuple({"case_token": t, "reason": "both_not_approved"}
                                    for t in cases[1::2]), "modes": modes}
    bundle = EvaluationResultBundle.model_construct(**data)
    assert bundle.validate_mode_and_category_counts() is bundle
    assert comparisons[0] < 16 * len(cases)
    EvaluationResultBundle.model_validate_json(json.dumps(data, default=lambda x: x.model_dump(mode="json")))


def test_runner_rejects_profile_index_mismatch_before_protocol_or_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import blinded_runner as runner

    arguments = _task4_environment(tmp_path, monkeypatch)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    profile = profiles / "synthetic-profile.json"
    profile.write_bytes(canonical_json_bytes({
        "model": "synthetic", "weights_sha256": "0" * 64, "dimensions": 2,
        "document_prefix": "", "query_prefix": "", "normalization": "l2-full-v1",
        "transport": "local-openai-embeddings-v1", "tokenizer_sha256": None,
        "source_revision": None,
    }))
    (arguments["index"] / "manifest.json").write_bytes(b'{"profile":{}}')
    with pytest.raises(ValueError, match="embedding configuration"):
        asyncio.run(runner.run_blinded_evaluation(
            **arguments, embedding_profile=profile, embedding_base_url="http://127.0.0.1:1235/v1",
        ))
    assert not (arguments["output_root"] / "protocol.json").exists()


def test_d36_attestation_binds_shared_host_parser_and_filesystem_dependencies(tmp_path: Path) -> None:
    from evaluation.release_candidate.freeze import _TOOL_SOURCE_PATHS

    repository, _ = _tool_repository(tmp_path)
    dependencies = ("backend/evaluation/blinded_io.py", "backend/evaluation/evidence_json.py")
    for relative in {*_D36_SOURCE_PATHS, *_TOOL_SOURCE_PATHS, *dependencies}:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic dependency\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "synthetic host dependencies")
    before = [attest_tool(repo_root=repository, tool_name="d36_synthetic", git_commit=_git(repository, "rev-parse", "HEAD"),
                          source_paths=paths).aggregate_sha256
              for paths in (_D36_SOURCE_PATHS, _TOOL_SOURCE_PATHS)]
    for relative in dependencies:
        (repository / relative).write_bytes(b"changed synthetic dependency\n")
        _git(repository, "add", ".")
        _git(repository, "commit", "-q", "-m", "change host dependency")
        after = [attest_tool(repo_root=repository, tool_name="d36_synthetic", git_commit=_git(repository, "rev-parse", "HEAD"),
                             source_paths=paths).aggregate_sha256
                 for paths in (_D36_SOURCE_PATHS, _TOOL_SOURCE_PATHS)]
        assert all(current != previous for current, previous in zip(after, before, strict=True))
        before = after


def test_tool_source_blob_cap_rejects_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import tool_attestation as tools

    path = tmp_path / "synthetic.py"
    with path.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)

    def forbidden_read(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("oversized tool source reached blob read")

    monkeypatch.setattr(tools.os, "read", forbidden_read)
    with pytest.raises(ValueError, match="size limit"):
        tools.fingerprint_file(tmp_path, "synthetic.py")


def test_tool_source_inventory_total_cap_prevents_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import tool_attestation as tools

    monkeypatch.setattr(tools, "validate_git_repository", lambda *args, **kwargs: "1" * 40)
    monkeypatch.setattr(tools, "fingerprint_committed_file", lambda root, commit, relative:
                        FileFingerprint(path=relative, sha256="1" * 64, size=8 * 1024 * 1024))
    with pytest.raises(ValueError, match="source.*size limit"):
        tools.attest_tool(repo_root=tmp_path, tool_name="synthetic", git_commit="1" * 40,
                          source_paths=tuple(f"{i:04d}.py" for i in range(65)))


def test_tool_fingerprint_path_has_explicit_boundary_limit() -> None:
    with pytest.raises(ValidationError):
        FileFingerprint(path="x" * 513, sha256="1" * 64, size=0)


def test_tool_inventory_has_explicit_boundary_limit() -> None:
    files = [FileFingerprint(path=f"{i:05d}.py", sha256="1" * 64, size=0) for i in range(8193)]
    with pytest.raises(ValidationError):
        ToolAttestation(schema_version=1, tool_name="synthetic", git_commit="1" * 40,
                        files=files, aggregate_sha256="1" * 64)


def test_streamed_blob_detects_same_size_mutation_and_bounds_each_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import blinded_io as io

    path = tmp_path / "blob"
    original = b"a" * (2 * 1024 * 1024)
    path.write_bytes(original)
    initial = path.stat()
    changed_mtime_ns = initial.st_mtime_ns + 2_000_000_000
    read = os.read
    calls = 0

    def mutating_read(descriptor: int, length: int) -> bytes:
        nonlocal calls
        assert length <= 1024 * 1024
        data = read(descriptor, length)
        calls += 1
        if calls == 1:
            with path.open("r+b") as stream:
                stream.seek(1024 * 1024)
                stream.write(b"b")
            os.utime(path, ns=(initial.st_atime_ns, changed_mtime_ns))
            assert path.stat().st_mtime_ns == changed_mtime_ns
        return data

    monkeypatch.setattr(io.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        io.fingerprint_regular(path, maximum=64_000_000)
    assert path.read_bytes() == original[:1024 * 1024] + b"b" + original[1024 * 1024 + 1:]


def test_streamed_blob_native_mutation_rejects_or_binds_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import blinded_io as io

    path = tmp_path / "blob"
    original = b"a" * (2 * 1024 * 1024)
    changed = original[:1024 * 1024] + b"b" + original[1024 * 1024 + 1:]
    path.write_bytes(original)
    read = os.read
    calls = 0

    def mutating_read(descriptor: int, length: int) -> bytes:
        nonlocal calls
        assert length <= 1024 * 1024
        data = read(descriptor, length)
        calls += 1
        if calls == 1:
            with path.open("r+b") as stream:
                stream.seek(1024 * 1024)
                stream.write(b"b")
        return data

    monkeypatch.setattr(io.os, "read", mutating_read)
    try:
        result = io.fingerprint_regular(path, maximum=64_000_000)
    except ValueError as error:
        assert "changed" in str(error)
    else:
        assert result == (len(changed), hashlib.sha256(changed).hexdigest())
        assert result[1] != hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == changed


def test_blinded_contracts_do_not_depend_on_result_contracts() -> None:
    source = (Path(__file__).parents[1] / "evaluation" / "blinded_contracts.py").read_text(
        encoding="utf-8"
    )
    assert "result_contracts" not in source


def test_protocol_tokens_are_domain_separated_lowercase_hmac_sha256() -> None:
    assert opaque_case_token(SYNTHETIC_KEY, "D24-H001") == CASE_TOKENS[4]
    assert opaque_category_token(SYNTHETIC_KEY, "paraphrase") == CATEGORY_TOKENS[1]
    assert opaque_category_token(SYNTHETIC_KEY, "D24-H001") != CASE_TOKENS[4]
    assert opaque_case_token(bytes(reversed(SYNTHETIC_KEY)), "D24-H001") != CASE_TOKENS[4]
    assert opaque_case_token(SYNTHETIC_KEY, "D24-H002") == CASE_TOKENS[0]


@pytest.mark.parametrize("size", [0, 31, 33, 64])
def test_protocol_token_key_requires_exactly_32_raw_bytes(tmp_path: Path, size: int) -> None:
    path = tmp_path / "token.key"
    path.write_bytes(b"x" * size)
    with pytest.raises(ValueError, match="exactly 32"):
        with token_key(path):
            pass


def test_protocol_token_key_buffer_is_zeroed_after_use(tmp_path: Path) -> None:
    path = tmp_path / "token.key"
    path.write_bytes(SYNTHETIC_KEY)
    with token_key(path) as key:
        retained = key
        assert bytes(key) == SYNTHETIC_KEY
    assert retained == bytearray(32)


@pytest.mark.parametrize("failure", ["final_fstat", "close"])
def test_protocol_token_key_buffer_is_zeroed_on_descriptor_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import evaluation.blinded_contracts as contracts

    path = tmp_path / "token.key"
    path.write_bytes(SYNTHETIC_KEY)
    retained = bytearray()
    monkeypatch.setattr(contracts, "_new_token_key_buffer", lambda: retained)
    if failure == "final_fstat":
        real_fstat = os.fstat
        calls = 0

        def failing_fstat(descriptor: int) -> os.stat_result:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected final fstat failure")
            return real_fstat(descriptor)

        monkeypatch.setattr(contracts.os, "fstat", failing_fstat)
    else:
        real_close = os.close

        def failing_close(descriptor: int) -> None:
            real_close(descriptor)
            raise OSError("injected close failure")

        monkeypatch.setattr(contracts.os, "close", failing_close)

    with pytest.raises(OSError, match="injected"):
        with token_key(path):
            pass
    assert retained == bytearray(32)


def test_protocol_category_binding_uses_first_d24_tag() -> None:
    bindings = case_category_bindings(_cases(), SYNTHETIC_KEY)
    by_case = {item.case_token: item.category_token for item in bindings}
    assert by_case[CASE_TOKENS[4]] == CATEGORY_TOKENS[1]
    assert by_case[CASE_TOKENS[4]] != "76095133a1dfa74355592beebcf4a4f6942381834362260487bc7122e4e3af25"
    assert tuple(item.case_token for item in bindings) == CASE_TOKENS


def test_protocol_contract_is_strict_and_carries_complete_sorted_topology() -> None:
    protocol = EvaluationProtocol.model_validate(_protocol_data())
    assert protocol.modes == ("all_tools", "stateful")
    assert protocol.case_count == len(protocol.case_tokens) == 5
    assert protocol.category_count == len(protocol.category_tokens) == 2
    assert tuple(item.model_dump(mode="json") for item in protocol.case_categories) == BINDINGS
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate({**_protocol_data(), "approval_sha256": "f" * 64})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("modes", ["stateful", "all_tools"]),
        ("case_tokens", list(reversed(CASE_TOKENS))),
        ("case_count", 4),
        ("category_tokens", [CATEGORY_TOKENS[0]]),
        ("case_categories", list(BINDINGS[:-1])),
    ],
)
def test_protocol_rejects_incomplete_or_noncanonical_topology(field: str, value: object) -> None:
    data = _protocol_data()
    data[field] = value
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", True), ("schema_version", 1.0),
     ("per_call_deadline_seconds", 180.0), ("maximum_model_calls", 4.0)],
)
def test_protocol_direct_call_rejects_literal_primitive_coercion(
    field: str, value: object,
) -> None:
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate({**_protocol_data(), field: value})


@pytest.mark.parametrize("value", [True, 1.0])
def test_bundle_direct_call_rejects_literal_primitive_coercion(value: object) -> None:
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate({**_bundle_data(), "schema_version": value})


@pytest.mark.parametrize(("field", "value"), [("schema_version", True), ("schema_version", 1.0),
                                              ("git_tree_clean", 1), ("git_tree_clean", 1.0)])
def test_freeze_direct_call_rejects_literal_primitive_coercion(field: str, value: object) -> None:
    data = {"schema_version": 1, "candidate_id": "0" * 16 + "-" + "1" * 12,
            "git_commit": "1" * 40, "git_tree_clean": True, "candidate_control_sha256": "2" * 64,
            "created_at": "2026-09-20T00:00:00Z", "runtime": {}, "schema_version_number": 1,
            "mode_configuration": {}, "files": [{"path": "synthetic.py", "sha256": "3" * 64, "size": 0}],
            "aggregate_sha256": "4" * 64}
    with pytest.raises(ValidationError):
        FreezeManifest.model_validate({**data, field: value})


@pytest.mark.parametrize("value", [True, 1.0])
def test_freeze_control_and_marker_reject_literal_primitive_coercion(value: object) -> None:
    control = {"schema_version": 1, "git_commit": "1" * 40,
               "git_commit_subject": "[DONE] Mission 35 Add recovery-oriented operational UI",
               "git_tree_clean": True}
    for data in ({**control, "schema_version": value}, {**control, "git_tree_clean": 1}):
        with pytest.raises(ValidationError):
            CandidateControl.model_validate(data)
    files = [FileFingerprint(path=path, size=0, sha256="1" * 64)
             for path in ("d36-tool-attestation.json", "freeze-manifest.json")]
    with pytest.raises(ValidationError):
        CompletionMarker(schema_version=value, files=files)


def test_approval_gate_requires_both_bound_review_ledgers() -> None:
    cases = _cases()
    human = load_review(FIXTURES / "synthetic-human-review.json", cases, "human")
    independent = load_review(
        FIXTURES / "synthetic-independent-review.json", cases, "independent_ai"
    )
    gate = eligibility(cases, human, independent)
    included, excluded = approval_partition(cases, human, independent, SYNTHETIC_KEY)
    assert gate["eligible_case_ids"] == ["D24-H001", "D24-H002"]
    assert included == INCLUDED
    assert tuple(item.model_dump(mode="json") for item in excluded) == EXCLUDED


def test_approval_bundle_is_strict_redacted_and_binds_all_independent_hashes() -> None:
    bundle = EvaluationResultBundle.model_validate(_bundle_data())
    raw = bundle.model_dump_json().encode("utf-8")
    for forbidden in (
        b"D24-H",
        b"synthetic request",
        b"paraphrase",
        b"negation",
        b'"decision":"approved"',
        b"expected",
    ):
        assert forbidden not in raw
    assert bundle.human_approval_sha256 != bundle.independent_approval_sha256
    assert bundle.d36_trial_tool_sha256 != bundle.d37_evaluator_tool_sha256
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate({**_bundle_data(), "approval_sha256": "f" * 64})


def test_protocol_and_approval_contracts_have_the_exact_shared_field_inventory() -> None:
    assert tuple(EvaluationProtocol.model_fields) == (
        "schema_version",
        "candidate_id",
        "modes",
        "per_call_deadline_seconds",
        "maximum_model_calls",
        "isolation",
        "corpus_sha256",
        "human_approval_sha256",
        "independent_approval_sha256",
        "freeze_sha256",
        "d36_trial_tool_sha256",
        "d37_evaluator_tool_sha256",
        "model_configuration_sha256",
        "stateful_index_sha256",
        "category_count",
        "category_tokens",
        "case_count",
        "case_tokens",
        "case_categories",
    )
    assert tuple(EvaluationResultBundle.model_fields) == (
        "schema_version",
        "candidate_id",
        "freeze_sha256",
        "corpus_sha256",
        "human_approval_sha256",
        "independent_approval_sha256",
        "protocol_sha256",
        "d36_trial_tool_sha256",
        "d37_evaluator_tool_sha256",
        "protocol_case_count",
        "protocol_case_tokens",
        "protocol_category_count",
        "protocol_category_tokens",
        "case_categories",
        "included_count",
        "excluded_count",
        "included_case_tokens",
        "excluded_cases",
        "evaluator_role",
        "evaluator_name",
        "executed_at",
        "sealed_evidence_sha256",
        "modes",
    )
    encoded = json.dumps(_bundle_data(), separators=(",", ":"))
    parsed = EvaluationResultBundle.model_validate_json(encoded)
    assert parsed.protocol_case_tokens == CASE_TOKENS
    nested_extra = _bundle_data()
    nested_extra["modes"][0]["categories"][0]["raw_case_id"] = "D24-H001"
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(nested_extra)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("included_case_tokens", list(reversed(INCLUDED))),
        ("excluded_count", 2),
        ("protocol_case_count", 4),
        ("protocol_category_tokens", list(reversed(CATEGORY_TOKENS))),
        ("evaluator_name", ""),
        ("executed_at", "2026-09-20T00:00:00+00:00"),
    ],
)
def test_approval_bundle_rejects_broken_shared_invariants(mutation: str, value: object) -> None:
    data = _bundle_data()
    data[mutation] = value
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(data)


@pytest.mark.parametrize("field", ["case_count", "category_count"])
@pytest.mark.parametrize("value", [-1, MAX_PROTOCOL_CASES + 1])
def test_protocol_rejects_every_out_of_range_count(field: str, value: int) -> None:
    data = _protocol_data()
    data[field] = value
    with pytest.raises(ValidationError):
        EvaluationProtocol.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "included",
        "completed",
        "task_complete",
        "unauthorized_effects",
        "unauthorized_replays",
        "secret_disclosures",
    ],
)
@pytest.mark.parametrize("value", [-1, MAX_PROTOCOL_CASES + 1])
def test_category_result_rejects_every_out_of_range_count(field: str, value: int) -> None:
    data = _category_results()[0]
    data[field] = value
    with pytest.raises(ValidationError):
        CategoryResult.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "included",
        "completed",
        "task_complete",
        "unauthorized_effects",
        "unauthorized_replays",
        "secret_disclosures",
        "transport_failures",
        "deadline_failures",
    ],
)
@pytest.mark.parametrize("value", [-1, MAX_PROTOCOL_CASES + 1])
def test_mode_result_rejects_every_out_of_range_count(field: str, value: int) -> None:
    data = _mode("all_tools")
    data[field] = value
    with pytest.raises(ValidationError):
        ModeResult.model_validate(data)


@pytest.mark.parametrize(
    "field",
    ["protocol_case_count", "protocol_category_count", "included_count", "excluded_count"],
)
@pytest.mark.parametrize("value", [-1, MAX_PROTOCOL_CASES + 1])
def test_bundle_rejects_every_out_of_range_count(field: str, value: int) -> None:
    data = _bundle_data()
    data[field] = value
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "completed",
        "task_complete",
        "unauthorized_effects",
        "unauthorized_replays",
        "secret_disclosures",
    ],
)
def test_category_result_rejects_every_count_above_its_protocol_denominator(field: str) -> None:
    data = _category_results()[0]
    data[field] = data["included"] + 1
    with pytest.raises(ValidationError):
        CategoryResult.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "completed",
        "task_complete",
        "unauthorized_effects",
        "unauthorized_replays",
        "secret_disclosures",
        "transport_failures",
        "deadline_failures",
    ],
)
def test_mode_result_rejects_every_count_above_its_protocol_denominator(field: str) -> None:
    data = _mode("all_tools")
    data[field] = data["included"] + 1
    with pytest.raises(ValidationError):
        ModeResult.model_validate(data)


def test_results_reject_task_complete_above_completed_within_included_denominator() -> None:
    category = _category_results()[0]
    category["included"] = 2
    category["completed"] = 1
    category["task_complete"] = 2
    with pytest.raises(ValidationError):
        CategoryResult.model_validate(category)

    mode = _mode("all_tools")
    mode["completed"] = 1
    mode["task_complete"] = 2
    mode["transport_failures"] = 1
    with pytest.raises(ValidationError):
        ModeResult.model_validate(mode)


@pytest.mark.parametrize("failure_field", ["transport_failures", "deadline_failures"])
def test_mode_result_rejects_completion_and_failure_sum_above_included(
    failure_field: str,
) -> None:
    data = _mode("all_tools")
    data["completed"] = data["included"]
    data[failure_field] = 1
    with pytest.raises(ValidationError):
        ModeResult.model_validate(data)


@pytest.mark.parametrize(
    "partition_failure", ["duplicate_included", "duplicate_excluded", "overlap", "missing"]
)
def test_approval_bundle_rejects_duplicate_overlap_or_missing_partition_tokens(
    partition_failure: str,
) -> None:
    data = _bundle_data()
    if partition_failure == "duplicate_included":
        data["included_case_tokens"] = (INCLUDED[0], INCLUDED[0])
    elif partition_failure == "duplicate_excluded":
        data["excluded_cases"] = (EXCLUDED[0], EXCLUDED[0], EXCLUDED[2])
    elif partition_failure == "overlap":
        data["excluded_cases"] = (
            {"case_token": INCLUDED[0], "reason": "human_not_approved"},
            *EXCLUDED,
        )
        data["excluded_count"] = 4
    else:
        data["excluded_cases"] = EXCLUDED[:-1]
        data["excluded_count"] = 2
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(data)


def test_bundle_rejects_counts_above_protocol_denominators() -> None:
    for field in ("included_count", "excluded_count"):
        data = _bundle_data()
        data[field] = data["protocol_case_count"] + 1
        with pytest.raises(ValidationError):
            EvaluationResultBundle.model_validate(data)

    data = _bundle_data()
    data["modes"][0]["included"] = data["included_count"] + 1
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(data)


@pytest.mark.parametrize(
    "coverage_failure", ["missing_category", "duplicate_category", "wrong_denominator"]
)
def test_approval_bundle_rejects_per_category_coverage_failures(
    coverage_failure: str,
) -> None:
    data = _bundle_data()
    if coverage_failure == "missing_category":
        data["included_case_tokens"] = (CASE_TOKENS[0],)
        data["included_count"] = 1
        data["excluded_cases"] = (
            *EXCLUDED,
            {"case_token": CASE_TOKENS[4], "reason": "human_not_approved"},
        )
        data["excluded_count"] = 4
    elif coverage_failure == "duplicate_category":
        for mode in data["modes"]:
            mode["categories"][1]["category_token"] = CATEGORY_TOKENS[0]
    else:
        for mode in data["modes"]:
            mode["categories"][0]["included"] = 2
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(data)


def test_approval_bundle_rejects_vacuous_category_or_invalid_mode_equations() -> None:
    vacuous = _bundle_data()
    vacuous["included_case_tokens"] = [CASE_TOKENS[0]]
    vacuous["included_count"] = 1
    vacuous["excluded_count"] = 4
    vacuous["excluded_cases"] = [
        *EXCLUDED,
        {"case_token": CASE_TOKENS[4], "reason": "human_not_approved"},
    ]
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(vacuous)

    invalid_equation = _bundle_data()
    invalid_equation["modes"][0]["transport_failures"] = 1
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(invalid_equation)

    bool_count = _bundle_data()
    bool_count["modes"][0]["completed"] = True
    with pytest.raises(ValidationError):
        EvaluationResultBundle.model_validate(bool_count)


def test_attest_tool_binds_clean_committed_bytes_and_changes_with_a_commit(
    tmp_path: Path,
) -> None:
    repository, first_commit = _tool_repository(tmp_path)
    first = attest_tool(
        repo_root=repository,
        tool_name="d37_test_tool",
        git_commit=first_commit,
        source_paths=("a.py", "b.py"),
    )
    assert first.git_commit == first_commit
    assert first.files[0].sha256 == hashlib.sha256(b"A = 1\n").hexdigest()

    (repository / "a.py").write_bytes(b"A = 2\n")
    _git(repository, "add", "a.py")
    _git(repository, "commit", "-q", "-m", "change tool byte")
    second_commit = _git(repository, "rev-parse", "HEAD")
    second = attest_tool(
        repo_root=repository,
        tool_name="d37_test_tool",
        git_commit=second_commit,
        source_paths=("a.py", "b.py"),
    )
    assert second.aggregate_sha256 != first.aggregate_sha256


@pytest.mark.parametrize("dirty_path", ["a.py", "untracked.py"])
def test_attest_tool_rejects_dirty_or_untracked_bytes(tmp_path: Path, dirty_path: str) -> None:
    repository, commit = _tool_repository(tmp_path)
    (repository / dirty_path).write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="clean"):
        attest_tool(
            repo_root=repository,
            tool_name="d37_test_tool",
            git_commit=commit,
            source_paths=("a.py", "b.py"),
        )


def _score_case(event_kind: str) -> Case:
    operation = {
        "operation_id": "project.subtitle-font-size.set",
        "operation_version": 1,
        "arguments": {"value": 50},
        "generate_after_save": event_kind in {"confirm_generation", "confirm_twice"},
    }
    submit_outcome = (
        "saved_awaiting_confirmation"
        if event_kind in {"confirm_generation", "confirm_twice"}
        else "saved"
    )
    submit = {
        "outcome": submit_outcome,
        "reason": "synthetic expected submit",
        "question_for": [],
        "settings_delta": {"subtitle_font_size": 50},
        "revision_delta": 1,
        "new_jobs": 0,
        "confirmation_required": event_kind in {"confirm_generation", "confirm_twice"},
        "job_assertions": {},
        "artifact_policy": "preserve_all_no_new_publication",
        "receipt_rule": "new_request",
    }
    after_event = None
    if event_kind != "none":
        after_event = {
            "outcome": "generation_queued"
            if event_kind in {"confirm_generation", "confirm_twice"}
            else "replayed",
            "reason": "synthetic expected event",
            "question_for": [],
            "settings_delta": {"subtitle_font_size": 50},
            "revision_delta": 1,
            "new_jobs": 1 if event_kind in {"confirm_generation", "confirm_twice"} else 0,
            "confirmation_required": False,
            "job_assertions": {},
            "artifact_policy": "job_may_publish_on_success"
            if event_kind in {"confirm_generation", "confirm_twice"}
            else "preserve_all_no_new_publication",
            "receipt_rule": "same_id_conflict"
            if event_kind == "same_id_different_body"
            else "first_result",
        }
    return Case.model_validate(
        {
            "schema_version": 1,
            "case_id": "D24-H900",
            "group_id": "D24-HG90",
            "split": "held_out",
            "source_request": "synthetic source",
            "provenance": {"kind": "new_synthetic", "reference": "D37 Task 3"},
            "tags": ["confirmation"],
            "situation": "synthetic scoring case",
            "initial": {
                "project_id": 1,
                "revision": 1,
                "settings": {
                    "subtitle_font_size": 48,
                    "voicevox_speed_scale": 1.0,
                    "voicevox_speaker_id": 0,
                    "pronunciation_overrides": [],
                    "narration_pacing_mode": "adaptive",
                    "narration_sentence_pause_seconds": 0.2,
                },
                "project_status": "completed",
                "jobs": [],
                "history": [],
                "artifact_revisions": [],
                "prior_turns": [],
            },
            "request": {
                "request_id": "synthetic-score",
                "text": "synthetic request",
                "target_project_id": 1,
                "base_revision": 1,
                "continuation": None,
            },
            "event": {
                "kind": event_kind,
                "details": (
                    {
                        "external_revision": 2,
                        "external_settings": {"subtitle_font_size": 52},
                    }
                    if event_kind == "revision_race"
                    else {
                        "selected_project_id_after": 202,
                        "action": "read_original_request",
                    }
                    if event_kind == "switch_target"
                    else {}
                ),
            },
            "expected": {
                "interpretation": "operation",
                "operations": [operation],
                "target_project_id": 1,
                "submit": submit,
                "after_event": after_event,
                "rationale": "synthetic scoring",
                "rule_ids": ["R01"],
            },
            "known_limitation": None,
        }
    )


def _settings_sha256(settings: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(
        settings, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("ascii")).hexdigest()


def _project_entry(suffix: str, *, project_id: int = 1) -> dict[str, object]:
    settings_sha256 = (
        "745380cc80a6b1c3aba0a18573a9c2c7bfa48cf046450735486ce54b7afe7f7c"
        if suffix == "before"
        else "b3ffc2eb01d53879888d9c95f50b26190097e28e947e389d2b62dbf3c207c858"
    )
    return {
        "id": project_id,
        "revision": 1 if suffix == "before" else 2,
        "status": "completed",
        "settings_sha256": settings_sha256,
        "title_sha256": "1" * 64,
        "source_script_sha256": "2" * 64,
        "global_visual_style_sha256": "3" * 64,
        "progress": 0.0,
        "current_stage": None,
        "current_artifact_id": None,
        "output_video": None,
        "output_subtitle": None,
        "error_sha256": "4" * 64,
    }


def _job_entry(**overrides: object) -> dict[str, object]:
    return {
        "id": 1,
        "project_id": 1,
        "status": "pending",
        "current_stage": "queued",
        "progress": 0.0,
        "stage_progress": 0.0,
        "input_revision": 2,
        "cancel_requested": False,
        "kind": "full",
        "block_index": None,
        "parent_job_id": None,
        "input_fingerprint": "1" * 64,
        "input_snapshot_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "recovery_sha256": "4" * 64,
        "error_sha256": "5" * 64,
        **overrides,
    }


def _redacted_state(suffix: str, **counts: int) -> dict[str, object]:
    receipt_count = counts.get("receipt_count", 0)
    external_call_count = counts.get("external_call_count", 0)
    language_request_count = counts.get("language_request_count", 0)
    language_turn_count = counts.get("language_turn_count", 0)

    def identities(count: int, offset: int) -> list[str]:
        return [f"{offset + index:064x}" for index in range(count)]

    return {
        "state_sha256": ("0" if suffix == "before" else "1") * 64,
        "project_status": "completed",
        "settings_sha256": (
            "745380cc80a6b1c3aba0a18573a9c2c7bfa48cf046450735486ce54b7afe7f7c"
            if suffix == "before"
            else "b3ffc2eb01d53879888d9c95f50b26190097e28e947e389d2b62dbf3c207c858"
        ),
        "projects_sha256": "4" * 64,
        "history_sha256": ("5" if suffix == "before" else "6") * 64,
        "jobs_sha256": ("7" if suffix == "before" else "8") * 64,
        "receipts_sha256": ("9" if suffix == "before" else "a") * 64,
        "artifacts_sha256": "b" * 64,
        "external_calls_sha256": "c" * 64,
        "language_requests_sha256": ("d" if suffix == "before" else "e") * 64,
        "language_turns_sha256": ("f" if suffix == "before" else "0") * 64,
        "project_count": 1,
        "history_count": counts.get("history_count", 0),
        "job_count": counts.get("job_count", 0),
        "artifact_count": counts.get("artifact_count", 0),
        "receipt_count": receipt_count,
        "external_call_count": external_call_count,
        "language_request_count": language_request_count,
        "language_turn_count": language_turn_count,
        "project_entries": [_project_entry(suffix)],
        "history_entries": [],
        "job_entries": [],
        "artifact_entries": [],
        "receipt_identity_sha256s": identities(receipt_count, 1),
        "external_call_identity_sha256s": identities(external_call_count, 10),
        "language_request_identity_sha256s": identities(language_request_count, 20),
        "language_turn_identity_sha256s": identities(language_turn_count, 30),
    }


def _score_observation(event_kind: str) -> dict[str, object]:
    confirmation = event_kind in {"confirm_generation", "confirm_twice"}
    replay = event_kind in {
        "resend_identical",
        "restart_resend",
        "same_id_different_body",
        "concurrent_identical",
        "switch_target",
    }
    same_response = event_kind != "same_id_different_body"
    replay_reason = "request_id_conflict" if event_kind == "same_id_different_body" else None
    language_additions = 1
    # Real product: saving then confirming generation executes two operations.
    receipt_additions = 2 if confirmation else 1
    observation = {
        "schema_version": 1,
        "response": {
            "http_status": 200,
            "status": "ready" if confirmation else "completed",
            "mode": "all_tools",
            # The request's own settings save ran even while generation awaits confirmation.
            "executed": True,
            "requires_confirmation": confirmation,
            "operation_id": "project.subtitle-font-size.set",
            "operation_version": 1,
            "arguments_sha256": _settings_sha256({"value": 50}),
            "generate_after_save": confirmation,
            "generation_requested": False,
            "clarification_missing_fields": None,
            "reason_code": None,
            "response_sha256": "1" * 64,
        },
        "before": _redacted_state("before"),
        "after": _redacted_state(
            "after",
            history_count=2,
            job_count=1 if confirmation else 0,
            receipt_count=receipt_additions,
            language_request_count=language_additions,
            language_turn_count=language_additions,
        ),
        "effects": {
            "settings": 1,
            "revision": 1,
            "jobs": int(confirmation),
            # The host's cancellation effect also registers a newly queued job.
            "cancellations": int(confirmation),
            "receipts": receipt_additions,
            "artifacts": 0,
            "external_calls": 0,
            "history": 1,
            "language_records": 1,
            "language_requests": language_additions,
            "language_turns": language_additions,
            "prior_receipts_preserved": True,
            "prior_external_calls_preserved": True,
            "prior_language_requests_preserved": True,
            "prior_language_turns_preserved": True,
        },
        "model_calls": 1,
        "failure_class": None,
        "replay": {
            "attempted": replay,
            "model_calls": 0,
            "state_unchanged": True,
            "same_response": same_response if replay else False,
            "response": {
                "http_status": 409 if replay_reason else 200,
                "status": "http_error" if replay_reason else "completed",
                "mode": "all_tools",
                "executed": not replay_reason,
                "requires_confirmation": False,
                "operation_id": None if replay_reason else "project.subtitle-font-size.set",
                "operation_version": None if replay_reason else 1,
                "arguments_sha256": None if replay_reason else _settings_sha256({"value": 50}),
                "generate_after_save": None if replay_reason else confirmation,
                "generation_requested": None if replay_reason else False,
                "clarification_missing_fields": None,
                "reason_code": replay_reason,
                "response_sha256": (
                    "1" * 64
                    if event_kind in {
                        "resend_identical", "restart_resend", "concurrent_identical", "switch_target"
                    }
                    else "2" * 64
                ),
            }
            if replay
            else None,
            "failure_class": None,
        },
        "confirmation": {
            "attempted": confirmation,
            "duplicate_attempted": event_kind == "confirm_twice",
            "state_sha256": "1" * 64 if confirmation else None,
            "duplicate_same_response": True if event_kind == "confirm_twice" else None,
            "response": {
                "http_status": 200,
                "status": "completed",
                "mode": "all_tools",
                "executed": True,
                "requires_confirmation": False,
                "operation_id": "project.subtitle-font-size.set",
                "operation_version": 1,
                "arguments_sha256": _settings_sha256({"value": 50}),
                "generate_after_save": True,
                "generation_requested": False,
                "clarification_missing_fields": None,
                "reason_code": None,
                "response_sha256": "3" * 64,
            }
            if confirmation
            else None,
            "duplicate_response": {
                "http_status": 200,
                "status": "completed",
                "mode": "all_tools",
                "executed": True,
                "requires_confirmation": False,
                "operation_id": "project.subtitle-font-size.set",
                "operation_version": 1,
                "arguments_sha256": _settings_sha256({"value": 50}),
                "generate_after_save": True,
                "generation_requested": False,
                "clarification_missing_fields": None,
                "reason_code": None,
                "response_sha256": "3" * 64,
            }
            if event_kind == "confirm_twice"
            else None,
            "failure_class": None,
        },
    }
    initial_settings = _score_case(event_kind).initial.settings
    saved_settings = {**initial_settings, "subtitle_font_size": 50}
    observation["after"]["history_entries"] = [
        # The product records the pre-change revision before the first save.
        {
            "project_id": 1,
            "revision": 1,
            "settings_sha256": _settings_sha256(initial_settings),
            "changed_fields": [],
            "restored_from_revision": None,
        },
        {
            "project_id": 1,
            "revision": 2,
            "settings_sha256": _settings_sha256(saved_settings),
            "changed_fields": ["subtitle_font_size"],
            "restored_from_revision": None,
        }
    ]
    if confirmation:
        observation["after"]["project_status"] = "generating"
        observation["after"]["project_entries"][0]["status"] = "generating"
        observation["after"]["job_entries"] = [_job_entry()]
    else:
        observation["after"]["jobs_sha256"] = observation["before"]["jobs_sha256"]
    if event_kind == "revision_race":
        external_settings = {**initial_settings, "subtitle_font_size": 52}
        observation["after"]["settings_sha256"] = _settings_sha256(external_settings)
        observation["after"]["project_entries"][0]["settings_sha256"] = _settings_sha256(external_settings)
        observation["after"]["history_entries"] = [
            {
                "project_id": 1,
                "revision": 2,
                "settings_sha256": _settings_sha256(external_settings),
                "changed_fields": ["subtitle_font_size"],
                "restored_from_revision": None,
            }
        ]
        observation["after"]["history_count"] = 1
        observation["effects"]["revision"] = 1
    return observation


def test_repository_owned_revision_race_cases_use_exact_next_revision() -> None:
    development = load_cases(Path(__file__).parents[2] / "evaluation" / "d24" / "development.jsonl")
    repository_owned = [
        *development,
        _score_case("revision_race"),
    ]
    revision_races = [case for case in repository_owned if case.event.kind == "revision_race"]

    assert revision_races
    assert all(
        case.event.details["external_revision"] == case.initial.revision + 1
        for case in revision_races
    )


@pytest.mark.parametrize(
    "event_kind",
    [
        "none",
        "resend_identical",
        "restart_resend",
        "same_id_different_body",
        "concurrent_identical",
        "revision_race",
        "confirm_generation",
        "confirm_twice",
        "switch_target",
    ],
)
def test_score_covers_every_d36_event_kind(event_kind: str) -> None:
    score = score_trial(_score_case(event_kind), _score_observation(event_kind))
    assert score.task_complete is True
    assert score.unauthorized_effect is False
    assert score.unauthorized_replay is False
    assert score.secret_disclosure is False
    assert all(score.checks.values())


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("operation_id", "project.status.get"),
        ("operation_version", 2),
        ("arguments_sha256", "f" * 64),
        ("generate_after_save", True),
        ("generation_requested", True),
    ],
)
def test_score_requires_exact_accepted_proposal_tuple(
    field: str, wrong_value: object
) -> None:
    observation = _score_observation("none")
    observation["response"][field] = wrong_value

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.checks["accepted_proposal"] is False


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("http_status", 500),
        ("executed", False),
        ("reason_code", "refused"),
    ],
)
def test_score_rejects_primary_response_contract_mismatch(
    field: str, wrong_value: object
) -> None:
    observation = _score_observation("none")
    observation["response"][field] = wrong_value

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.checks["primary_response"] is False


def test_score_requires_exact_canonical_clarification_fields() -> None:
    case_data = _score_case("none").model_dump(mode="json")
    case_data["expected"]["interpretation"] = "clarification"
    case_data["expected"]["operations"] = []
    case_data["expected"]["submit"].update(
        {
            "outcome": "needs_input",
            "question_for": ["target", "intent"],
            "settings_delta": {},
            "revision_delta": 0,
            "confirmation_required": False,
            "receipt_rule": "none",
        }
    )
    case = Case.model_validate(case_data)
    observation = _score_observation("none")
    observation["response"].update(
        {
            "status": "needs_input",
            "executed": False,
            "operation_id": None,
            "operation_version": None,
            "arguments_sha256": None,
            "generate_after_save": None,
            "generation_requested": None,
            "clarification_missing_fields": ["intent", "target"],
        }
    )
    observation["after"] = json.loads(json.dumps(observation["before"]))
    observation["after"].update(
        {
            "state_sha256": "1" * 64,
            "language_requests_sha256": "e" * 64,
            "language_turns_sha256": "0" * 64,
            "language_request_count": 1,
            "language_turn_count": 1,
            "language_request_identity_sha256s": [f"{20:064x}"],
            "language_turn_identity_sha256s": [f"{30:064x}"],
        }
    )
    observation["effects"].update(
        {
            "settings": 0,
            "revision": 0,
            "jobs": 0,
            "cancellations": 0,
            "receipts": 0,
            "artifacts": 0,
            "external_calls": 0,
            "history": 0,
            "language_records": 1,
            "language_requests": 1,
            "language_turns": 1,
        }
    )

    passing = score_trial(case, observation)
    assert passing.task_complete is True

    observation["response"]["clarification_missing_fields"] = ["arguments"]
    wrong = score_trial(case, observation)
    assert wrong.task_complete is False
    assert wrong.checks["question_fields"] is False

    observation["response"]["clarification_missing_fields"] = None
    with pytest.raises(ValueError, match="clarification"):
        score_trial(case, observation)


def test_score_rejects_secondary_project_mutation() -> None:
    case = _score_case("none")
    observation = _score_observation("none")
    secondary = _project_entry("before", project_id=2)
    observation["before"]["project_count"] = observation["after"]["project_count"] = 2
    observation["before"]["project_entries"].append(secondary)
    observation["after"]["project_entries"].append(dict(secondary))

    assert score_trial(case, observation).task_complete is True

    observation["after"]["project_entries"][1]["progress"] = 0.5
    observation["after"]["projects_sha256"] = "e" * 64
    score = score_trial(case, observation)
    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["projects"] is False


def test_score_rejects_primary_unrelated_field_mutation() -> None:
    observation = _score_observation("none")
    observation["after"]["project_entries"][0]["current_stage"] = "unrelated"
    observation["after"]["projects_sha256"] = "e" * 64

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["projects"] is False


@pytest.mark.parametrize("mutation", ["insert", "delete"])
def test_score_rejects_project_insertion_or_deletion(mutation: str) -> None:
    observation = _score_observation("none")
    secondary = _project_entry("before", project_id=2)
    observation["before"]["project_count"] = observation["after"]["project_count"] = 2
    observation["before"]["project_entries"].append(secondary)
    observation["after"]["project_entries"].append(dict(secondary))
    if mutation == "insert":
        observation["after"]["project_entries"].append(_project_entry("before", project_id=3))
        observation["after"]["project_count"] = 3
    else:
        observation["after"]["project_entries"].pop()
        observation["after"]["project_count"] = 1
    observation["after"]["projects_sha256"] = "e" * 64

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["projects"] is False


def test_score_requires_expected_project_status_and_exact_history_projection() -> None:
    case = _score_case("none")
    observation = _score_observation("none")
    observation["after"]["project_status"] = "failed"
    observation["after"]["history_entries"][-1]["changed_fields"] = []

    score = score_trial(case, observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["project_status"] is False
    assert score.checks["full_settings_history"] is False


def test_score_preserves_initial_history_and_requires_exact_sequence() -> None:
    case_data = _score_case("none").model_dump(mode="json")
    initial_settings = case_data["initial"]["settings"]
    case_data["initial"]["history"] = [
        {"revision": 1, "settings": initial_settings, "changed_fields": []}
    ]
    case = Case.model_validate(case_data)
    observation = _score_observation("none")
    initial_entry = {
        "project_id": 1,
        "revision": 1,
        "settings_sha256": _settings_sha256(initial_settings),
        "changed_fields": [],
        "restored_from_revision": None,
    }
    observation["before"]["history_count"] = 1
    observation["before"]["history_entries"] = [initial_entry]
    observation["after"]["history_count"] = 2
    observation["after"]["history_entries"] = [
        initial_entry,
        observation["after"]["history_entries"][-1],
    ]

    passing = score_trial(case, observation)
    assert passing.task_complete is True

    observation["after"]["history_entries"] = [
        {**initial_entry, "settings_sha256": "e" * 64},
        observation["after"]["history_entries"][1],
    ]
    failing = score_trial(case, observation)
    assert failing.task_complete is False
    assert failing.unauthorized_effect is True
    assert failing.checks["full_settings_history"] is False


def test_score_validates_job_assertions_and_preserves_initial_jobs() -> None:
    case_data = _score_case("none").model_dump(mode="json")
    case_data["initial"]["jobs"] = [
        {
            "id": 7,
            "project_id": 1,
            "status": "failed",
            "input_revision": 1,
            "cancel_requested": False,
            "input_settings": case_data["initial"]["settings"],
            "kind": "full",
        }
    ]
    case_data["expected"]["submit"]["job_assertions"] = {
        "job_id": 7,
        "status": "failed",
        "cancel_requested": False,
        "input_revision": 1,
        "input_settings": case_data["initial"]["settings"],
    }
    case = Case.model_validate(case_data)
    observation = _score_observation("none")
    job = _job_entry(id=7, status="failed", input_revision=1)
    observation["before"]["job_count"] = observation["after"]["job_count"] = 1
    observation["before"]["job_entries"] = [job]
    observation["after"]["job_entries"] = [dict(job)]
    observation["after"]["jobs_sha256"] = observation["before"]["jobs_sha256"]

    passing = score_trial(case, observation)
    assert passing.task_complete is True

    observation["after"]["job_entries"][0]["parent_job_id"] = 99
    observation["after"]["jobs_sha256"] = "e" * 64
    observation["effects"]["jobs"] = 1
    failing = score_trial(case, observation)
    assert failing.task_complete is False
    assert failing.unauthorized_effect is True
    assert failing.checks["initial_jobs_preserved"] is False


def test_score_validates_cancellation_on_the_asserted_initial_job_only() -> None:
    case_data = _score_case("none").model_dump(mode="json")
    settings = case_data["initial"]["settings"]
    case_data["initial"]["project_status"] = "generating"
    case_data["initial"]["jobs"] = [
        {
            "id": 7,
            "project_id": 1,
            "status": "running",
            "input_revision": 1,
            "cancel_requested": False,
            "input_settings": settings,
            "kind": "full",
        },
        {
            "id": 8,
            "project_id": 1,
            "status": "failed",
            "input_revision": 1,
            "cancel_requested": False,
            "input_settings": settings,
            "kind": "full",
        },
    ]
    case_data["expected"]["operations"] = [
        {
            "operation_id": "project.generation.cancel",
            "operation_version": 1,
            "arguments": {"job_id": 7},
            "generate_after_save": False,
        }
    ]
    case_data["expected"]["submit"].update(
        {
            "outcome": "cancel_requested",
            "settings_delta": {},
            "revision_delta": 0,
            "job_assertions": {
                "job_id": 7,
                "status": "running",
                "cancel_requested": True,
                "no_future_publication": True,
            },
        }
    )
    case = Case.model_validate(case_data)
    observation = _score_observation("none")
    observation["response"].update(
        {
            "operation_id": "project.generation.cancel",
            "arguments_sha256": _settings_sha256({"job_id": 7}),
            "generate_after_save": False,
            "generation_requested": False,
        }
    )
    initial_settings_sha256 = _settings_sha256(settings)
    observation["before"]["project_status"] = "generating"
    observation["after"]["project_status"] = "generating"
    for state in (observation["before"], observation["after"]):
        state["settings_sha256"] = initial_settings_sha256
        state["project_entries"][0].update(
            {
                "revision": 1,
                "status": "generating",
                "settings_sha256": initial_settings_sha256,
            }
        )
    observation["after"]["settings_sha256"] = observation["before"]["settings_sha256"]
    observation["after"]["history_sha256"] = observation["before"]["history_sha256"]
    observation["after"]["history_count"] = 0
    observation["after"]["history_entries"] = []
    observation["effects"].update(
        {"settings": 0, "revision": 0, "history": 0, "jobs": 1, "cancellations": 1}
    )
    observation["after"]["jobs_sha256"] = "e" * 64
    before_job = _job_entry(id=7, status="running", input_revision=1)
    secondary_job = _job_entry(id=8, status="failed", input_revision=1)
    observation["before"]["job_count"] = observation["after"]["job_count"] = 2
    observation["before"]["job_entries"] = [before_job, secondary_job]
    observation["after"]["job_entries"] = [
        {**before_job, "cancel_requested": True},
        dict(secondary_job),
    ]

    passing = score_trial(case, observation)
    assert passing.task_complete is True

    observation["after"]["job_entries"][0]["cancel_requested"] = False
    failing = score_trial(case, observation)
    assert failing.task_complete is False
    assert failing.checks["cancellation"] is False
    observation["after"]["job_entries"][0]["cancel_requested"] = True

    for index, field, value in (
        (0, "input_fingerprint", "a" * 64),
        (0, "progress", 0.5),
        (0, "plan_sha256", "b" * 64),
        (0, "error_sha256", "c" * 64),
        (1, "input_fingerprint", "d" * 64),
    ):
        original = observation["after"]["job_entries"][index][field]
        observation["after"]["job_entries"][index][field] = value
        unauthorized = score_trial(case, observation)
        assert unauthorized.task_complete is False
        assert unauthorized.unauthorized_effect is True
        assert unauthorized.checks["initial_jobs_preserved"] is False
        observation["after"]["job_entries"][index][field] = original


def _published_observation() -> dict[str, object]:
    observation = _score_observation("confirm_generation")
    observation["after"]["artifact_count"] = 1
    observation["after"]["artifact_entries"] = [
        {
            "id": 10,
            "project_id": 1,
            "job_id": 1,
            "revision": 2,
            "input_fingerprint": "4" * 64,
            "video_path_sha256": "7" * 64,
            "video_size": 11,
            "video_sha256": "5" * 64,
            "subtitle_path_sha256": "8" * 64,
            "subtitle_size": 13,
            "subtitle_sha256": "9" * 64,
            "manifest_sha256": "6" * 64,
        }
    ]
    observation["after"]["artifacts_sha256"] = "e" * 64
    observation["effects"]["artifacts"] = 1
    observation["after"]["project_entries"][0].update(
        {
            "current_artifact_id": 10,
            "output_video": {
                "exists": True,
                "path_sha256": "7" * 64,
                "size": 11,
                "sha256": "5" * 64,
            },
            "output_subtitle": {
                "exists": True,
                "path_sha256": "8" * 64,
                "size": 13,
                "sha256": "9" * 64,
            },
        }
    )
    return observation


def test_score_accepts_complete_artifact_publication_binding() -> None:
    score = score_trial(_score_case("confirm_generation"), _published_observation())

    assert score.task_complete is True
    assert score.unauthorized_effect is False


@pytest.mark.parametrize(
    "mutation",
    [
        "orphan",
        "missing_video",
        "missing_subtitle",
        "stale_pointer",
        "wrong_job",
        "wrong_project",
        "wrong_revision",
        "wrong_video_size",
        "wrong_subtitle_size",
    ],
)
def test_score_rejects_incomplete_or_misdirected_artifact_publication(
    mutation: str,
) -> None:
    observation = _published_observation()
    artifact = observation["after"]["artifact_entries"][0]
    project = observation["after"]["project_entries"][0]
    if mutation == "orphan":
        project.update(
            {"current_artifact_id": None, "output_video": None, "output_subtitle": None}
        )
    elif mutation == "missing_video":
        artifact.update({"video_size": None, "video_sha256": None})
        project["output_video"] = {
            "exists": False,
            "path_sha256": "7" * 64,
            "size": None,
            "sha256": None,
        }
    elif mutation == "missing_subtitle":
        artifact.update(
            {
                "subtitle_path_sha256": None,
                "subtitle_size": None,
                "subtitle_sha256": None,
            }
        )
        project["output_subtitle"] = None
    elif mutation == "stale_pointer":
        project["current_artifact_id"] = 9
    elif mutation == "wrong_job":
        artifact["job_id"] = 999
    elif mutation == "wrong_project":
        artifact["project_id"] = 2
    elif mutation == "wrong_revision":
        artifact["revision"] = 1
        observation["after"]["job_entries"][0]["input_revision"] = 1
    elif mutation == "wrong_video_size":
        project["output_video"]["size"] = 12
    else:
        project["output_subtitle"]["size"] = 14

    score = score_trial(_score_case("confirm_generation"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["artifacts"] is False or score.checks["projects"] is False


def test_score_allows_zero_artifact_delta_only_without_pointer_change() -> None:
    case = _score_case("confirm_generation")
    unchanged = _score_observation("confirm_generation")
    assert score_trial(case, unchanged).task_complete is True

    unchanged["after"]["project_entries"][0]["current_artifact_id"] = 10
    score = score_trial(case, unchanged)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["projects"] is False


def test_score_always_rejects_same_count_artifact_replacement() -> None:
    case = _score_case("confirm_generation")
    observation = _score_observation("confirm_generation")
    initial = {
        "id": 3,
        "project_id": 1,
        "job_id": None,
        "revision": 1,
        "input_fingerprint": None,
        "video_path_sha256": "2" * 64,
        "video_size": 1,
        "video_sha256": "3" * 64,
        "subtitle_path_sha256": None,
        "subtitle_size": None,
        "subtitle_sha256": None,
        "manifest_sha256": "4" * 64,
    }
    replacement = {**initial, "video_sha256": "5" * 64}
    observation["before"]["artifact_count"] = observation["after"]["artifact_count"] = 1
    observation["before"]["artifact_entries"] = [initial]
    observation["after"]["artifact_entries"] = [replacement]
    observation["after"]["artifacts_sha256"] = "e" * 64
    observation["effects"]["artifacts"] = 1

    score = score_trial(case, observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["artifacts"] is False


def test_score_rejects_same_content_artifact_at_different_path() -> None:
    case = _score_case("none")
    observation = _score_observation("none")
    initial = {
        "id": 3,
        "project_id": 1,
        "job_id": None,
        "revision": 1,
        "input_fingerprint": None,
        "video_path_sha256": "2" * 64,
        "video_size": 1,
        "video_sha256": "3" * 64,
        "subtitle_path_sha256": None,
        "subtitle_size": None,
        "subtitle_sha256": None,
        "manifest_sha256": "4" * 64,
    }
    replacement = {**initial, "video_path_sha256": "5" * 64}
    observation["before"]["artifact_count"] = observation["after"]["artifact_count"] = 1
    observation["before"]["artifact_entries"] = [initial]
    observation["after"]["artifact_entries"] = [replacement]
    observation["after"]["artifacts_sha256"] = "e" * 64
    observation["effects"]["artifacts"] = 1

    score = score_trial(case, observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["artifacts"] is False


def test_score_rejects_missing_output_path_substitution() -> None:
    observation = _score_observation("none")
    observation["before"]["project_entries"][0]["output_video"] = {
        "exists": False,
        "path_sha256": "2" * 64,
        "size": None,
        "sha256": None,
    }
    observation["after"]["project_entries"][0]["output_video"] = {
        "exists": False,
        "path_sha256": "3" * 64,
        "size": None,
        "sha256": None,
    }
    observation["after"]["projects_sha256"] = "e" * 64

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["projects"] is False


def test_score_refuses_unverifiable_persisted_effect_evidence() -> None:
    observation = _score_observation("none")
    observation["after"].pop("history_entries")
    with pytest.raises(ValueError, match="history_entries"):
        score_trial(_score_case("none"), observation)


@pytest.mark.parametrize("collection", ["receipts", "artifacts"])
def test_score_detects_same_count_identity_replacement_as_unauthorized_effect(
    collection: str,
) -> None:
    case = _score_case("none")
    case_data = case.model_dump(mode="json")
    case_data["expected"]["submit"]["settings_delta"] = {}
    case_data["expected"]["submit"]["revision_delta"] = 0
    case_data["expected"]["submit"]["receipt_rule"] = "none"
    case = Case.model_validate(case_data)
    observation = _score_observation("none")
    observation["effects"].update({"settings": 0, "revision": 0, "history": 0, "receipts": 0})
    observation["after"]["settings_sha256"] = observation["before"]["settings_sha256"]
    observation["after"]["history_sha256"] = observation["before"]["history_sha256"]
    observation["after"]["receipts_sha256"] = observation["before"]["receipts_sha256"]
    observation["after"]["language_requests_sha256"] = observation["before"]["language_requests_sha256"]
    observation["after"]["language_turns_sha256"] = observation["before"]["language_turns_sha256"]
    for count_field in (
        "history_count",
        "receipt_count",
        "language_request_count",
        "language_turn_count",
    ):
        observation["after"][count_field] = observation["before"][count_field]
    observation["after"]["history_entries"] = []
    observation["after"]["receipt_identity_sha256s"] = []
    observation["after"]["language_request_identity_sha256s"] = []
    observation["after"]["language_turn_identity_sha256s"] = []
    observation["after"][f"{collection}_sha256"] = "e" * 64
    observation["after"][f"{collection[:-1] if collection != 'artifacts' else 'artifact'}_count"] = 0
    score = score_trial(case, observation)
    assert score.task_complete is False
    assert score.unauthorized_effect is True


@pytest.mark.parametrize(
    ("collection", "effect_flag"),
    [
        ("language_request", "prior_language_requests_preserved"),
        ("language_turn", "prior_language_turns_preserved"),
    ],
)
def test_score_rejects_same_count_language_record_replacement(
    collection: str, effect_flag: str
) -> None:
    observation = _score_observation("none")
    identities = f"{collection}_identity_sha256s"
    observation["before"][identities] = ["1" * 64]
    observation["after"][identities] = ["2" * 64]
    count = f"{collection}_count"
    observation["before"][count] = observation["after"][count] = 1
    observation["effects"][effect_flag] = False

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True


def test_score_rejects_mutated_prior_receipt_during_expected_addition() -> None:
    observation = _score_observation("none")
    observation["before"]["receipt_identity_sha256s"] = ["1" * 64]
    observation["after"]["receipt_identity_sha256s"] = ["2" * 64, "3" * 64]
    observation["before"]["receipt_count"] = 1
    observation["after"]["receipt_count"] = 2
    observation["effects"]["prior_receipts_preserved"] = False

    score = score_trial(_score_case("none"), observation)

    assert score.task_complete is False
    assert score.unauthorized_effect is True
    assert score.checks["receipts"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("replay", "unexpected"), True),
        (("confirmation", "attempted"), 1),
        (("effects", "prior_receipts_preserved"), 1),
    ],
)
def test_score_rejects_malformed_event_evidence(
    path: tuple[str, str], value: object
) -> None:
    observation = _score_observation("resend_identical")
    observation[path[0]][path[1]] = value

    with pytest.raises((ValidationError, ValueError)):
        score_trial(_score_case("resend_identical"), observation)


@pytest.mark.parametrize("nested", ["replay", "confirmation"])
def test_score_rejects_nested_event_failures(nested: str) -> None:
    event_kind = "resend_identical" if nested == "replay" else "confirm_generation"
    observation = _score_observation(event_kind)
    observation[nested]["failure_class"] = "candidate_error"

    score = score_trial(_score_case(event_kind), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "ready"),
        ("executed", False),
        ("operation_version", 2),
    ],
)
def test_score_rejects_wrong_idempotent_replay_response(
    field: str, value: object
) -> None:
    observation = _score_observation("resend_identical")
    observation["replay"]["response"][field] = value

    score = score_trial(_score_case("resend_identical"), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("http_status", 400),
        ("status", "blocked"),
        ("reason_code", "request_conflict"),
        ("executed", True),
    ],
)
def test_score_rejects_wrong_same_id_conflict_contract(
    field: str, value: object
) -> None:
    observation = _score_observation("same_id_different_body")
    observation["replay"]["response"][field] = value

    score = score_trial(_score_case("same_id_different_body"), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


def test_switch_target_projection_restores_exact_d24_event_shape() -> None:
    projected = case_to_unlabeled(_score_case("switch_target"))

    assert projected.event.model_dump(mode="json", exclude={"request"}) == {
        "kind": "switch_target",
        "selected_project_id_after": 202,
        "action": "read_original_request",
    }


def test_score_rejects_switch_target_wrong_response() -> None:
    observation = _score_observation("switch_target")
    observation["replay"]["response"]["arguments_sha256"] = "f" * 64

    score = score_trial(_score_case("switch_target"), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "ready"),
        ("executed", False),
        ("arguments_sha256", "f" * 64),
    ],
)
def test_score_rejects_wrong_confirmation_response(
    field: str, value: object
) -> None:
    observation = _score_observation("confirm_generation")
    observation["confirmation"]["response"][field] = value

    score = score_trial(_score_case("confirm_generation"), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


@pytest.mark.parametrize(
    ("event_kind", "response_path"),
    [
        ("none", ("response",)),
        ("resend_identical", ("replay", "response")),
        ("confirm_generation", ("confirmation", "response")),
        ("confirm_twice", ("confirmation", "duplicate_response")),
    ],
)
def test_score_binds_every_response_mode_to_trial_mode(
    event_kind: str, response_path: tuple[str, ...],
) -> None:
    observation = _score_observation(event_kind)
    observation.update({
        "case_sha256": "a" * 64,
        "candidate_snapshot_sha256": "b" * 64,
        "input_sha256": "c" * 64,
        "mode": "all_tools",
    })
    selected: dict[str, object] = observation
    for component in response_path:
        selected = selected[component]  # type: ignore[assignment,index]
    selected["mode"] = "stateful"

    with pytest.raises(ValidationError, match="response mode"):
        score_trial(_score_case(event_kind), observation)


def test_score_rejects_confirmation_state_mismatch() -> None:
    observation = _score_observation("confirm_generation")
    observation["confirmation"]["state_sha256"] = "f" * 64

    score = score_trial(_score_case("confirm_generation"), observation)

    assert score.task_complete is False
    assert score.checks["declared_event"] is False


def test_score_rejects_safe_refusal_for_unambiguous_executable_request() -> None:
    observation = _score_observation("none")
    observation["response"].update(
        {
            "status": "blocked",
            "executed": False,
            "operation_id": None,
            "operation_version": None,
            "arguments_sha256": None,
            "generate_after_save": None,
            "generation_requested": None,
            "reason_code": "refused",
        }
    )
    observation["effects"].update(
        {
            "settings": 0,
            "revision": 0,
            "history": 0,
            "receipts": 0,
            "language_records": 0,
            "language_requests": 0,
            "language_turns": 0,
        }
    )
    for field in (
        "settings_sha256",
        "history_sha256",
        "receipts_sha256",
        "language_requests_sha256",
        "language_turns_sha256",
    ):
        observation["after"][field] = observation["before"][field]
    for field in (
        "history_count",
        "receipt_count",
        "language_request_count",
        "language_turn_count",
    ):
        observation["after"][field] = observation["before"][field]
    observation["after"]["history_entries"] = []
    observation["after"]["receipt_identity_sha256s"] = []
    observation["after"]["language_request_identity_sha256s"] = []
    observation["after"]["language_turn_identity_sha256s"] = []
    score = score_trial(_score_case("none"), observation)
    assert score.task_complete is False
    assert score.checks["interpretation_class"] is False
    assert score.checks["status_class"] is False


def test_score_detects_unexpected_cancellation_effect() -> None:
    observation = _score_observation("none")
    observation["effects"]["cancellations"] = 1

    score = score_trial(_score_case("none"), observation)

    assert score.unauthorized_effect is True
    assert score.task_complete is False


def test_score_rejects_extra_top_level_disclosure_key() -> None:
    observation = _score_observation("none")
    observation["private_input"] = "synthetic secret contents"

    with pytest.raises(ValidationError):
        score_trial(_score_case("none"), observation)


def test_score_marks_mutating_or_unapproved_replay_unauthorized() -> None:
    observation = _score_observation("resend_identical")
    observation["replay"]["state_unchanged"] = False
    score = score_trial(_score_case("resend_identical"), observation)
    assert score.unauthorized_replay is True
    assert score.task_complete is False


def test_seal_evidence_is_deterministic_and_excludes_public_outputs(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    (root / "group" / "case").mkdir(parents=True)
    (root / "group" / "case" / "observation.json").write_bytes(b"synthetic detail\n")
    (root / "group" / "score.json").write_bytes(b"synthetic score\n")
    for excluded in ("protocol.json", "partial-result.json", "result-bundle.json"):
        (root / excluded).write_bytes(b"must not affect seal")
    first_files, first_hash = seal_evidence(root)
    assert [item.path for item in first_files] == [
        "group/case/observation.json",
        "group/score.json",
    ]
    assert [item.size for item in first_files] == [17, 16]
    assert first_files[0].sha256 == hashlib.sha256(b"synthetic detail\n").hexdigest()
    (root / "protocol.json").write_bytes(b"changed public protocol")
    second_files, second_hash = seal_evidence(root)
    assert second_files == first_files
    assert second_hash == first_hash

    nested_public_name = root / "group" / "case" / "protocol.json"
    nested_public_name.write_bytes(b"nested private protocol evidence")
    third_files, third_hash = seal_evidence(root)
    assert [item.path for item in third_files] == [
        "group/case/observation.json",
        "group/case/protocol.json",
        "group/score.json",
    ]
    assert third_hash != second_hash
    nested_public_name.write_bytes(b"mutated nested private protocol evidence")
    _, fourth_hash = seal_evidence(root)
    assert fourth_hash != third_hash


def test_seal_evidence_rejects_symlink_reparse_and_non_regular_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    target = root / "target.json"
    target.write_bytes(b"detail")
    link = root / "linked.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="regular|symlink|reparse"):
        seal_evidence(root)
    link.unlink()
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda self: type("Metadata", (), {"st_mode": 0, "st_file_attributes": 0x400})(),
    )
    with pytest.raises(ValueError, match="reparse"):
        seal_evidence(root)


def test_seal_evidence_rejects_root_escape_and_logs_no_contents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root-link"
    try:
        root.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="root|symlink|reparse"):
        seal_evidence(root)
    assert "outside" not in caplog.text


def _write_canonical(path: Path, value: object) -> bytes:
    raw = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _make_task4_candidate(root: Path) -> tuple[Path, str, list[FileFingerprint]]:
    candidate = root / "candidate"
    (candidate / "backend").mkdir(parents=True)
    (candidate / "backend" / "candidate.txt").write_bytes(b"synthetic D35 candidate\n")
    _git(candidate, "init", "-q")
    _git(candidate, "config", "user.email", "d37@example.invalid")
    _git(candidate, "config", "user.name", "D37 Test")
    _git(candidate, "add", ".")
    _git(candidate, "commit", "-q", "-m", "synthetic detached D35")
    commit = _git(candidate, "rev-parse", "HEAD")
    _git(candidate, "checkout", "-q", "--detach", commit)
    content = (candidate / "backend" / "candidate.txt").read_bytes()
    files = [
        FileFingerprint(
            path="backend/candidate.txt",
            sha256=hashlib.sha256(content).hexdigest(),
            size=len(content),
        )
    ]
    return candidate, commit, files


def _task4_host_source(observation: dict[str, object], *, mutate_candidate: bool = False) -> bytes:
    encoded = json.dumps(observation, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    mutation = "(candidate / 'backend' / 'candidate.txt').write_text('mutated')" if mutate_candidate else "pass"
    return f'''import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--candidate-root", type=Path, required=True)
parser.add_argument("--mode", required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--storage", type=Path, required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--index", type=Path)
args = parser.parse_args()
candidate = args.candidate_root
payload = json.loads(args.input.read_text(encoding="ascii"))
forbidden = {{"expected", "review", "decision", "score", "labels"}}
def keys(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(item) for item in value)) if value else set()
    return set()
if keys(payload) & forbidden:
    raise SystemExit(3)
args.storage.mkdir(parents=True, exist_ok=False)
(args.storage / "host-marker.txt").write_text(args.mode, encoding="ascii")
observation = json.loads({encoded!r})
observation.update({{
    "case_sha256": payload["case_sha256"],
    "candidate_snapshot_sha256": "c" * 64,
    "input_sha256": "d" * 64,
    "mode": args.mode,
}})
for response in (
    observation["response"],
    observation["replay"].get("response"),
    observation["confirmation"].get("response"),
    observation["confirmation"].get("duplicate_response"),
):
    if response is not None:
        response["mode"] = args.mode
args.output.write_bytes((json.dumps(observation, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\\n").encode("ascii"))
{mutation}
'''.encode("utf-8")


def _make_task4_tool_repo(
    root: Path, observation: dict[str, object], *, mutate_candidate: bool = False
) -> tuple[Path, str, ToolAttestation]:
    repository = root / "tool-repository"
    repository.mkdir()
    host_path = "backend/evaluation/scripts/evaluation_trial_host.py"
    all_paths = sorted(set(_D36_SOURCE_PATHS) | set(_D37_SOURCE_PATHS))
    for relative in all_paths:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            _task4_host_source(observation, mutate_candidate=mutate_candidate)
            if relative == host_path
            else (
                f"# synthetic committed source: {relative}\n".encode("utf-8")
                if relative.endswith(".py")
                else f"synthetic committed source: {relative}\n".encode("utf-8")
            )
        )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "d37@example.invalid")
    _git(repository, "config", "user.name", "D37 Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "synthetic D36 and D37 tools")
    commit = _git(repository, "rev-parse", "HEAD")
    attestation = attest_tool(
        repo_root=repository,
        tool_name="d36_candidate_freezer_and_trial_host",
        git_commit=commit,
        source_paths=_D36_SOURCE_PATHS,
    )
    return repository, commit, attestation


def _make_task4_publication(
    root: Path,
    commit: str,
    candidate_files: list[FileFingerprint],
    d36_attestation: ToolAttestation,
) -> Path:
    aggregate = aggregate_fingerprints(candidate_files)
    candidate_id = f"{aggregate[:16]}-{commit[:12]}"
    publication = root / "freeze" / candidate_id
    publication.mkdir(parents=True)
    manifest = FreezeManifest(
        schema_version=1,
        candidate_id=candidate_id,
        git_commit=commit,
        git_tree_clean=True,
        candidate_control_sha256="a" * 64,
        created_at="2026-09-20T00:00:00Z",
        runtime={"python": "3.12.12"},
        schema_version_number=1,
        mode_configuration={"modes": ["all_tools", "stateful"]},
        files=candidate_files,
        aggregate_sha256=aggregate,
    )
    manifest_raw = _write_canonical(publication / "freeze-manifest.json", manifest)
    attestation_raw = _write_canonical(
        publication / "d36-tool-attestation.json", d36_attestation
    )
    marker_files = [
        FileFingerprint(
            path=name,
            sha256=hashlib.sha256(raw).hexdigest(),
            size=len(raw),
        )
        for name, raw in (
            ("d36-tool-attestation.json", attestation_raw),
            ("freeze-manifest.json", manifest_raw),
        )
    ]
    _write_canonical(
        publication / ".d36-publication-state",
        {"schema_version": 1, "files": [item.model_dump(mode="json") for item in marker_files]},
    )
    return publication / "freeze-manifest.json"


def _write_task4_inputs(root: Path, case: Case) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "held-out.jsonl"
    corpus.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    corpus_hash = corpus_digest([case])
    case_hash = case_digest(case)
    review_paths = []
    for role, name in (("human", "human-review.json"), ("independent_ai", "independent-review.json")):
        path = root / name
        _write_canonical(
            path,
            {
                "schema_version": 1,
                "corpus_sha256": corpus_hash,
                "role": role,
                "reviewer": "Synthetic reviewer",
                "reviewed_at": "2026-09-20T00:00:00Z",
                "entries": [
                    {
                        "case_id": case.case_id,
                        "case_sha256": case_hash,
                        "decision": "approved",
                        "note": "synthetic",
                    }
                ],
            },
        )
        review_paths.append(path)
    index = root / "stateful-index"
    index.mkdir()
    (index / "index.json").write_bytes(b"{}\n")
    key = root / "token.key"
    key.write_bytes(SYNTHETIC_KEY)
    return corpus, review_paths[0], review_paths[1], index, key


def _host_settings_observation(case: Case, observation: dict[str, object]) -> dict[str, object]:
    """Rehash canned settings as the real host does: over the projected (completed) fields."""
    raw = dict(case.initial.settings)
    completed = dict(normalize_case(case).initial.settings)
    replacements = {
        _settings_sha256(raw): _settings_sha256(completed),
        _settings_sha256({**raw, "subtitle_font_size": 50}):
            _settings_sha256({**completed, "subtitle_font_size": 50}),
    }
    text = json.dumps(observation)
    for old, new in replacements.items():
        text = text.replace(old, new)
    return json.loads(text)


def _task4_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *, mutate_candidate: bool = False,
) -> dict[str, object]:
    case = _score_case("none")
    observation = _host_settings_observation(case, _score_observation("none"))
    candidate, candidate_commit, candidate_files = _make_task4_candidate(tmp_path)
    tool_repo, _, d36_attestation = _make_task4_tool_repo(
        tmp_path, observation, mutate_candidate=mutate_candidate
    )
    freeze_manifest = _make_task4_publication(
        tmp_path, candidate_commit, candidate_files, d36_attestation
    )
    corpus, human, independent, index, key = _write_task4_inputs(
        tmp_path / "private-inputs", case
    )
    monkeypatch.setattr("evaluation.blinded_runner._tool_repo_root", lambda: tool_repo)
    return {
        "candidate_root": candidate,
        "freeze_manifest": freeze_manifest,
        "corpus": corpus,
        "human_review": human,
        "independent_review": independent,
        "output_root": tmp_path / "results",
        "model": "synthetic-model",
        "index": index,
        "evaluator_name": "Synthetic independent evaluator",
        "token_key_file": key,
    }


def test_case_projection_is_explicitly_unlabeled_and_strictly_serializable() -> None:
    case = _score_case("none")
    projected = case_to_unlabeled(case)
    raw = canonical_json_bytes(projected.model_dump(mode="json", exclude_unset=True))

    assert UnlabeledTrialCase.model_validate_json(raw, strict=True) == projected
    assert projected.category == case.tags[0]
    assert projected.event.request.text == case.request.text
    for forbidden in (
        b"expected",
        b"rationale",
        b"rule_ids",
        b"known_limitation",
        b"decision",
    ):
        assert forbidden not in raw
    with pytest.raises(ValidationError):
        UnlabeledTrialCase.model_validate({**projected.model_dump(mode="json"), "expected": {}})


def test_protocol_is_exclusive_canonical_and_resume_requires_exact_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    protocol = EvaluationProtocol.model_validate(_protocol_data())
    path = write_run_protocol_exclusive(root, protocol)
    expected = canonical_json_bytes(protocol) + b"\n"

    assert path.read_bytes() == expected
    assert write_run_protocol_exclusive(root, protocol) == path
    path.write_bytes(expected[:-1] + b" ")
    with pytest.raises(ValueError, match="protocol"):
        write_run_protocol_exclusive(root, protocol)


def test_protocol_serialization_upper_bound_fits_explicit_protocol_cap() -> None:
    maximum = maximum_protocol_serialized_bytes()

    assert 16 * 1024 * 1024 < maximum <= MAX_PROTOCOL_BYTES
    assert MAX_PROTOCOL_BYTES == 64 * 1024 * 1024
    assert len(canonical_json_bytes(EvaluationProtocol.model_validate(_protocol_data()))) + 1 <= maximum


def test_result_bundle_serialization_upper_bound_fits_explicit_bundle_cap() -> None:
    maximum = maximum_result_bundle_serialized_bytes()

    assert maximum_protocol_serialized_bytes() < maximum <= MAX_RESULT_BUNDLE_BYTES
    assert MAX_RESULT_BUNDLE_BYTES == 128 * 1024 * 1024


def test_protocol_writer_fails_before_publication_when_canonical_bytes_exceed_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evaluation.blinded_runner as runner

    root = tmp_path / "run"
    root.mkdir()
    protocol = EvaluationProtocol.model_validate(_protocol_data())
    canonical_size = len(canonical_json_bytes(protocol)) + 1
    monkeypatch.setattr(runner, "MAX_PROTOCOL_BYTES", canonical_size - 1)

    with pytest.raises(ValueError, match="protocol.*maximum"):
        runner.write_run_protocol_exclusive(root, protocol)

    assert not (root / "protocol.json").exists()
    assert list(root.iterdir()) == []


def test_protocol_writer_rejects_preexisting_symlink(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    try:
        (root / "protocol.json").symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="protocol"):
        write_run_protocol_exclusive(root, EvaluationProtocol.model_validate(_protocol_data()))


def test_external_runner_uses_attested_host_detached_candidate_and_redacted_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    bundle = asyncio.run(run_blinded_evaluation(**arguments))
    output = arguments["output_root"]
    protocol_raw = (output / "protocol.json").read_bytes()
    bundle_raw = (output / "result-bundle.json").read_bytes()

    assert bundle.protocol_sha256 == hashlib.sha256(protocol_raw).hexdigest()
    d37_attestation = ToolAttestation.model_validate_json(
        (output / "tool-attestation.json").read_bytes(), strict=True
    )
    assert d37_attestation.aggregate_sha256 == bundle.d37_evaluator_tool_sha256
    assert bundle.d36_trial_tool_sha256 != bundle.d37_evaluator_tool_sha256
    assert bundle.included_count == 1
    assert [mode.mode for mode in bundle.modes] == ["all_tools", "stateful"]
    assert all(mode.completed == mode.task_complete == 1 for mode in bundle.modes)
    assert _git(arguments["candidate_root"], "status", "--porcelain=v1") == ""
    assert _git(arguments["candidate_root"], "branch", "--show-current") == ""
    for forbidden in (
        b"D24-H900",
        b"synthetic request",
        b"confirmation",
        b"expected",
        str(arguments["candidate_root"]).encode(),
        SYNTHETIC_KEY,
    ):
        assert forbidden not in protocol_raw
        assert forbidden not in bundle_raw
    trial_roots = sorted((output / "groups" / "000001" / "cases").glob("*/*"))
    assert len(trial_roots) == 2
    assert {path.name for path in trial_roots} == {"all_tools", "stateful"}
    assert all(
        next(path.glob("attempts/*/storage/host-marker.txt")).is_file()
        for path in trial_roots
    )


def test_runner_detects_candidate_mutation_after_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch, mutate_candidate=True)
    with pytest.raises(ValueError, match="candidate"):
        asyncio.run(run_blinded_evaluation(**arguments))


def test_runner_partial_resume_preserves_completed_records_and_exact_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    import evaluation.blinded_runner as runner

    real_invoke = runner._invoke_trial_host
    calls = 0

    async def interrupt_after_first(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return await real_invoke(**kwargs)

    monkeypatch.setattr(runner, "_invoke_trial_host", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_blinded_evaluation(**arguments))
    output = arguments["output_root"]
    protocol_before = (output / "protocol.json").read_bytes()
    records_before = list(output.glob("groups/*/cases/*/*/trial-result.json"))
    assert len(records_before) == 1
    record_bytes = records_before[0].read_bytes()
    partial = json.loads((output / "partial-result.json").read_bytes())
    assert partial["completed_case_tokens"] == []
    assert len(partial["started_case_tokens"]) == 1
    assert b"D24-H900" not in (output / "partial-result.json").read_bytes()

    monkeypatch.setattr(runner, "_invoke_trial_host", real_invoke)
    bundle = asyncio.run(run_blinded_evaluation(**arguments))
    assert (output / "protocol.json").read_bytes() == protocol_before
    assert records_before[0].read_bytes() == record_bytes
    assert bundle.modes[0].completed == bundle.modes[1].completed == 1


def test_runner_rejects_protocol_input_change_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    asyncio.run(run_blinded_evaluation(**arguments))
    arguments["model"] = "different-model"
    with pytest.raises(ValueError, match="protocol"):
        asyncio.run(run_blinded_evaluation(**arguments))


def test_completed_resume_rejects_changed_evaluator_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    asyncio.run(run_blinded_evaluation(**arguments))
    arguments["evaluator_name"] = "Different synthetic evaluator"
    with pytest.raises(ValueError, match="result bundle"):
        asyncio.run(run_blinded_evaluation(**arguments))


def test_runner_refuses_unattested_trial_host_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    tool_repo = Path(_git(tmp_path / "tool-repository", "rev-parse", "--show-toplevel"))
    host = tool_repo / "backend/evaluation/scripts/evaluation_trial_host.py"
    host.write_bytes(host.read_bytes() + b"\n# uncommitted mutation\n")
    with pytest.raises(ValueError, match="tool|attest|source|clean"):
        asyncio.run(run_blinded_evaluation(**arguments))


def test_runner_uses_sorted_case_order_alternating_modes_and_exact_shared_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    arguments.update(
        {
            "corpus": FIXTURES / "synthetic-held-out.jsonl",
            "human_review": FIXTURES / "synthetic-human-review.json",
            "independent_review": FIXTURES / "synthetic-independent-review.json",
        }
    )
    import evaluation.blinded_runner as runner

    calls: list[tuple[str, str]] = []

    async def successful_trial(**kwargs: object) -> object:
        output = Path(arguments["output_root"])
        assert (output / "protocol.json").is_file()
        case = kwargs["case"]
        mode = kwargs["mode"]
        calls.append((case.case_id, mode))
        return runner._TrialRecord(
            schema_version=1,
            protocol_sha256=kwargs["protocol_sha256"],
            case_token=kwargs["case_token"],
            category_token=kwargs["category_token"],
            mode=mode,
            outcome="completed",
            score={
                "task_complete": True,
                "unauthorized_effect": False,
                "unauthorized_replay": False,
                "secret_disclosure": False,
                "checks": {"synthetic": True},
            },
            candidate_snapshot_sha256="c" * 64,
        )

    monkeypatch.setattr(runner, "_run_trial", successful_trial)
    bundle = asyncio.run(run_blinded_evaluation(**arguments))

    assert calls == [
        ("D24-H001", "all_tools"),
        ("D24-H001", "stateful"),
        ("D24-H002", "stateful"),
        ("D24-H002", "all_tools"),
    ]
    assert bundle.protocol_case_tokens == CASE_TOKENS
    assert bundle.protocol_category_tokens == CATEGORY_TOKENS
    assert tuple(item.model_dump(mode="json") for item in bundle.case_categories) == BINDINGS
    assert bundle.included_case_tokens == INCLUDED
    assert tuple(item.model_dump(mode="json") for item in bundle.excluded_cases) == EXCLUDED
    assert all(
        tuple(category.category_token for category in mode.categories) == CATEGORY_TOKENS
        for mode in bundle.modes
    )


@pytest.mark.parametrize("changed_input", ["corpus", "human_review", "index", "freeze"])
def test_partial_resume_rejects_changed_bound_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    import evaluation.blinded_runner as runner

    async def interrupt(**_: object) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_invoke_trial_host", interrupt)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(run_blinded_evaluation(**arguments))

    if changed_input == "corpus":
        Path(arguments["corpus"]).write_bytes(Path(arguments["corpus"]).read_bytes() + b"\n")
    elif changed_input == "human_review":
        Path(arguments["human_review"]).write_bytes(
            Path(arguments["human_review"]).read_bytes() + b" \n"
        )
    elif changed_input == "index":
        (Path(arguments["index"]) / "index.json").write_bytes(b'{"changed":true}\n')
    else:
        Path(arguments["freeze_manifest"]).write_bytes(
            Path(arguments["freeze_manifest"]).read_bytes() + b" "
        )

    with pytest.raises(ValueError):
        asyncio.run(run_blinded_evaluation(**arguments))


def test_interrupted_host_subprocess_is_killed_and_awaited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.blinded_runner as runner

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.killed = False
            self.finished = asyncio.Event()
            self.stdout: asyncio.StreamReader | None = None
            self.stderr: asyncio.StreamReader | None = None

        async def wait(self) -> int:
            await self.finished.wait()
            return self.returncode or 0

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            self.finished.set()

    process = FakeProcess()

    async def create_process(*_: object, **__: object) -> FakeProcess:
        return process

    async def terminate_tree(selected: FakeProcess) -> None:
        selected.kill()
        await selected.wait()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    import evaluation.blinded_runtime as runtime

    monkeypatch.setattr(runtime, "_attach_windows_job", lambda _: None)

    async def release_bootstrap(_: FakeProcess) -> None:
        return None

    monkeypatch.setattr(runtime, "_release_windows_bootstrap", release_bootstrap)
    monkeypatch.setattr(runtime, "_terminate_process_tree", terminate_tree)
    (tmp_path / "backend").mkdir()

    async def exercise() -> None:
        process.stdout = asyncio.StreamReader()
        process.stderr = asyncio.StreamReader()
        process.stdout.feed_eof()
        process.stderr.feed_eof()
        task = asyncio.create_task(
            runner._invoke_trial_host(
                tool_root=tmp_path,
                candidate_root=tmp_path,
                mode="all_tools",
                input_path=tmp_path / "input.json",
                output_path=tmp_path / "output.json",
                storage=tmp_path / "storage",
                model="synthetic-model",
                index=tmp_path / "index",
                stdout_path=tmp_path / "stdout.log",
                stderr_path=tmp_path / "stderr.log",
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert process.killed is True
    assert process.returncode == -9


def test_blinded_runner_cli_exposes_only_explicit_external_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_blinded_evaluation", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--candidate-root",
        "--freeze-manifest",
        "--corpus",
        "--human-review",
        "--independent-review",
        "--output",
        "--model",
        "--index",
        "--evaluator-name",
        "--token-key-file",
    ):
        assert option in completed.stdout


@pytest.mark.parametrize("name", ["trial-result.json", "result-bundle.json"])
def test_interrupted_immutable_publication_leaves_complete_resumable_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    final = tmp_path / name
    value = b"canonical complete bytes\n"
    real_link = os.link

    def interrupt_after_link(source: Path, destination: Path, **kwargs: object) -> None:
        real_link(source, destination, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", interrupt_after_link)
    with pytest.raises(KeyboardInterrupt):
        _publish_immutable(final, value, name)
    assert final.read_bytes() == value

    monkeypatch.setattr(os, "link", real_link)
    assert _publish_immutable(final, value, name) == final
    assert final.read_bytes() == value


@pytest.mark.parametrize("name", ["trial-result.json", "result-bundle.json"])
def test_immutable_publication_never_replaces_existing_final_and_ignores_temp(
    tmp_path: Path, name: str,
) -> None:
    final = tmp_path / name
    _publish_immutable(final, b"first\n", "trial result")
    leftover = tmp_path / f".{name}.tmp-{'a' * 64}"
    leftover.write_bytes(b"incomplete")

    with pytest.raises(ValueError, match="trial result"):
        _publish_immutable(final, b"second\n", "trial result")

    assert final.read_bytes() == b"first\n"
    assert _load_records(tmp_path, "0" * 64) == []


def test_d37_attestation_closure_is_sorted_and_covers_runtime_sources() -> None:
    root = Path(__file__).parents[2]
    paths = _d37_source_paths(root)
    tracked = set(_git(root, "ls-files").splitlines())

    expected = {
        path
        for path in tracked
        if (
            path.startswith("backend/evaluation/")
            and path.endswith(".py")
        )
        or (path.startswith("backend/app/") and path.endswith(".py"))
    } | {
        "backend/app/operations/definitions.json",
        "backend/pyproject.toml",
        "backend/scripts/run_blinded_evaluation.py",
        "backend/uv.lock",
    }
    assert paths == tuple(sorted(expected))
    assert len(paths) == len(set(paths))
    assert set(paths) <= tracked
    assert "backend/pyproject.toml" in paths
    assert "backend/uv.lock" in paths
    assert "backend/app/operations/definitions.json" in paths
    assert "backend/evaluation/contracts.py" in paths
    assert "backend/evaluation/corpus.py" in paths
    assert "backend/evaluation/fixtures.py" in paths
    assert "backend/app/operations/catalog.py" in paths
    assert "backend/scripts/run_blinded_evaluation.py" in paths
    assert all(not path.startswith("backend/tests/") for path in paths)
    assert all("release-evidence/" not in path and ".env" not in path for path in paths)


def test_d37_attestation_closure_mutation_invalidates_or_changes_attestation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    representative = (
        "backend/evaluation/contracts.py",
        "backend/evaluation/corpus.py",
        "backend/evaluation/fixtures.py",
        "backend/app/operations/catalog.py",
        "backend/app/operations/definitions.json",
        "backend/scripts/run_blinded_evaluation.py",
        "backend/pyproject.toml",
        "backend/uv.lock",
    )
    for relative in representative:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{relative}\n".encode("utf-8"))
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "d37@example.invalid")
    _git(repository, "config", "user.name", "D37 Test")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "runtime closure")
    first_commit = _git(repository, "rev-parse", "HEAD")
    paths = _d37_source_paths(repository)
    first = attest_tool(
        repo_root=repository,
        tool_name="d37_blinded_evaluator",
        git_commit=first_commit,
        source_paths=paths,
    )

    for relative in representative[:5]:
        target = repository / relative
        original = target.read_bytes()
        target.write_bytes(original + b"changed\n")
        with pytest.raises(ValueError, match="clean|committed"):
            attest_tool(
                repo_root=repository,
                tool_name="d37_blinded_evaluator",
                git_commit=first_commit,
                source_paths=paths,
            )
        target.write_bytes(original)

    target = repository / "backend/app/operations/definitions.json"
    target.write_bytes(target.read_bytes() + b"committed change\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "change definition")
    second_commit = _git(repository, "rev-parse", "HEAD")
    second = attest_tool(
        repo_root=repository,
        tool_name="d37_blinded_evaluator",
        git_commit=second_commit,
        source_paths=_d37_source_paths(repository),
    )
    assert second.aggregate_sha256 != first.aggregate_sha256


@pytest.mark.parametrize("linked_component", ["groups", "cases", "mode", "attempts"])
def test_writable_directory_rejects_precreated_link_component(
    tmp_path: Path, linked_component: str,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    components = ["groups", "000001", "cases", "a" * 64, "all_tools", "attempts"]
    indices = {"groups": 0, "cases": 2, "mode": 4, "attempts": 5}
    current = root
    for index, component in enumerate(components):
        child = current / component
        if index == indices[linked_component]:
            try:
                child.symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("directory symlink creation unavailable")
            break
        child.mkdir()
        current = child

    with pytest.raises(ValueError, match="directory|reparse|symlink|unsafe"):
        _ensure_writable_directory(root, *components)
    assert list(outside.iterdir()) == []


def test_output_root_rejects_workspace_parent_of_protected_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    arguments["output_root"] = tmp_path

    with pytest.raises(ValueError, match="external|overlap|protected"):
        asyncio.run(run_blinded_evaluation(**arguments))
    assert not (tmp_path / "protocol.json").exists()


def test_candidate_root_symlink_and_path_swap_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "candidate"
    (real / "backend").mkdir(parents=True)
    link = tmp_path / "candidate-link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(ValueError, match="candidate"):
        _canonical_candidate_root(link)

    arguments = _task4_environment(tmp_path / "swap", monkeypatch)
    import evaluation.blinded_runner as runner

    async def swap_candidate(**_: object) -> str:
        candidate = Path(arguments["candidate_root"])
        if os.name == "nt":
            with pytest.raises(OSError):
                candidate.rename(candidate.with_name("candidate-moved"))
        else:
            candidate.rename(candidate.with_name("candidate-moved"))
            candidate.mkdir()
        return "transport_failure"

    monkeypatch.setattr(runner, "_invoke_trial_host", swap_candidate)
    if os.name == "nt":
        asyncio.run(run_blinded_evaluation(**arguments))
    else:
        with pytest.raises(ValueError, match="candidate"):
            asyncio.run(run_blinded_evaluation(**arguments))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux proc-fd anchor only")
def test_d36_host_accepts_only_parent_proc_fd_candidate_alias(tmp_path: Path) -> None:
    from evaluation.scripts import evaluation_trial_host as host

    candidate = tmp_path / "candidate"
    (candidate / "backend").mkdir(parents=True)
    descriptor = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    metadata = os.fstat(descriptor)
    try:
        alias = Path(f"/proc/{os.getpid()}/fd/{descriptor}")
        child = os.fork()
        if child == 0:
            try:
                resolved = host._resolve_candidate_root(
                    alias,
                    expected_candidate_dev=metadata.st_dev,
                    expected_candidate_ino=metadata.st_ino,
                )
                os._exit(0 if resolved == alias else 1)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0

        with pytest.raises(ValueError, match="candidate"):
            host._resolve_candidate_root(
                alias,
                expected_candidate_dev=metadata.st_dev,
                expected_candidate_ino=metadata.st_ino,
            )
        link = tmp_path / "candidate-link"
        link.symlink_to(candidate, target_is_directory=True)
        with pytest.raises(ValueError, match="candidate"):
            host._resolve_candidate_root(link)
    finally:
        os.close(descriptor)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux proc-fd anchor integration only")
def test_posix_trial_host_keeps_original_inode_after_candidate_path_replacement(
    tmp_path: Path,
) -> None:
    from evaluation.scripts import evaluation_trial_host as host

    original = tmp_path / "candidate"
    backend = original / "backend"
    backend.mkdir(parents=True)
    (backend / "candidate.txt").write_bytes(b"original candidate\n")
    moved = tmp_path / "candidate-moved"
    ready_read, ready_write = os.pipe()
    continue_read, continue_write = os.pipe()
    result_read, result_write = os.pipe()
    anchor = _open_candidate_anchor(original)
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(continue_write)
        os.close(result_read)
        try:
            alias = host._resolve_candidate_root(
                anchor.execution_path,
                expected_candidate_dev=anchor.identity[0],
                expected_candidate_ino=anchor.identity[1],
            )
            before = host._candidate_snapshot(alias)
            os.write(ready_write, b"1")
            if os.read(continue_read, 1) != b"1":
                raise RuntimeError("parent did not replace candidate path")
            host._assert_candidate_root_identity(alias, anchor.identity)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; print(Path('candidate.txt').read_text(), end='')",
                ],
                cwd=alias / "backend",
                check=True,
                capture_output=True,
                text=True,
            )
            after = host._candidate_snapshot(alias)
            host._assert_candidate_root_identity(alias, anchor.identity)
            result = json.dumps(
                {
                    "alias_unchanged": alias == anchor.execution_path,
                    "snapshot_unchanged": before == after,
                    "worker_bytes": completed.stdout,
                }
            ).encode("ascii")
        except BaseException as error:
            os.write(ready_write, b"0")
            result = json.dumps({"error": type(error).__name__}).encode("ascii")
        os.write(result_write, result)
        os._exit(0)

    os.close(ready_write)
    os.close(continue_read)
    os.close(result_write)
    try:
        ready = os.read(ready_read, 1)
        if ready != b"1":
            result = json.loads(os.read(result_read, 4096))
            _, status = os.waitpid(child, 0)
            pytest.fail(f"trial host setup failed: {result}, status={status}")
        original.rename(moved)
        replacement_backend = original / "backend"
        replacement_backend.mkdir(parents=True)
        replacement = replacement_backend / "candidate.txt"
        replacement.write_bytes(b"alternate candidate\n")
        os.write(continue_write, b"1")
        result = json.loads(os.read(result_read, 4096))
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert result == {
            "alias_unchanged": True,
            "snapshot_unchanged": True,
            "worker_bytes": "original candidate\n",
        }
        assert replacement.read_bytes() == b"alternate candidate\n"
        assert (moved / "backend" / "candidate.txt").read_bytes() == b"original candidate\n"
    finally:
        for descriptor in (ready_read, continue_write, result_read):
            os.close(descriptor)
        _close_candidate_anchor(anchor)


def test_candidate_swap_restore_invocation_uses_retained_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _task4_environment(tmp_path, monkeypatch)
    original = Path(arguments["candidate_root"])
    moved = original.with_name("candidate-moved")
    observed_paths: list[Path] = []
    import evaluation.blinded_runner as runner

    async def swap_restore(**kwargs: object) -> str:
        invocation_root = Path(kwargs["candidate_root"])
        candidate_identity = kwargs["candidate_identity"]
        assert candidate_identity is not None
        observed_paths.append(invocation_root)
        if os.name == "nt":
            with pytest.raises(OSError):
                original.rename(moved)
        else:
            original.rename(moved)
            original.mkdir()
            (original / "backend").mkdir()
            (original / "backend" / "candidate.txt").write_bytes(b"alternate checkout\n")
            try:
                assert (invocation_root / "backend" / "candidate.txt").read_bytes() == (
                    b"synthetic D35 candidate\n"
                )
            finally:
                alternate = original.with_name("candidate-alternate")
                original.rename(alternate)
                moved.rename(original)
        return "transport_failure"

    monkeypatch.setattr(runner, "_invoke_trial_host", swap_restore)
    asyncio.run(run_blinded_evaluation(**arguments))
    assert len(observed_paths) == 2
    if os.name == "posix":
        assert all(path != original for path in observed_paths)
        assert all(str(path).startswith(f"/proc/{os.getpid()}/fd/") for path in observed_paths)


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            check=False,
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_early_parent_exit_with_pipe_holding_descendant_is_terminated_without_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.blinded_runner as runner

    backend = tmp_path / "backend"
    script = backend / "evaluation" / "scripts" / "evaluation_trial_host.py"
    script.parent.mkdir(parents=True)
    (backend / "evaluation" / "__init__.py").write_bytes(b"")
    (script.parent / "__init__.py").write_bytes(b"")
    script.write_text(
        "import argparse, subprocess, sys\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--storage'); p.add_argument('--candidate-root'); p.add_argument('--mode'); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--model'); p.add_argument('--index')\n"
        "a=p.parse_args(); s=Path(a.storage); s.mkdir()\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])\n"
        "(s/'child.pid').write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_HOST_PIPE_DRAIN_GRACE_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(runner, "_HOST_TEARDOWN_SECONDS", 2.0, raising=False)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"{}")
    index = tmp_path / "index"
    index.mkdir()
    storage = tmp_path / "storage"

    started = time.monotonic()
    outcome = asyncio.run(
        _invoke_trial_host(
            tool_root=tmp_path,
            candidate_root=candidate,
            mode="all_tools",
            input_path=input_path,
            output_path=tmp_path / "output.json",
            storage=storage,
            model="synthetic",
            index=index,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )
    )
    elapsed = time.monotonic() - started

    assert outcome == "completed"
    assert elapsed < 3
    child_pid = int((storage / "child.pid").read_text())
    deadline = time.monotonic() + 2
    while _process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(child_pid)


def test_trial_host_cancellation_kills_descendant_without_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = tmp_path / "backend"
    script = backend / "evaluation" / "scripts" / "evaluation_trial_host.py"
    script.parent.mkdir(parents=True)
    (backend / "evaluation" / "__init__.py").write_bytes(b"")
    (script.parent / "__init__.py").write_bytes(b"")
    script.write_text(
        "import argparse, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--storage'); p.add_argument('--candidate-root'); p.add_argument('--mode'); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--model'); p.add_argument('--index')\n"
        "a=p.parse_args(); s=Path(a.storage); s.mkdir()\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        "(s/'child.pid').write_text(str(child.pid))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    import evaluation.blinded_runtime as runtime

    monkeypatch.setattr(runtime, "HOST_TEARDOWN_SECONDS", 2.0)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"{}")
    index = tmp_path / "index"
    index.mkdir()
    storage = tmp_path / "storage"

    async def exercise() -> int:
        task = asyncio.create_task(
            _invoke_trial_host(
                tool_root=tmp_path,
                candidate_root=candidate,
                mode="all_tools",
                input_path=input_path,
                output_path=tmp_path / "output.json",
                storage=storage,
                model="synthetic",
                index=index,
                stdout_path=tmp_path / "stdout.log",
                stderr_path=tmp_path / "stderr.log",
            )
        )
        child_path = storage / "child.pid"
        deadline = asyncio.get_running_loop().time() + 3
        child_pid: int | None = None
        while child_pid is None:
            try:
                value = child_path.read_text().strip()
                child_pid = int(value) if value else None
            except FileNotFoundError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("trial descendant did not start")
            if child_pid is None:
                await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        return child_pid

    child_pid = asyncio.run(exercise())
    deadline = time.monotonic() + 2
    while _process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(child_pid)


def test_trial_host_output_flood_is_bounded_and_descendant_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = tmp_path / "backend"
    script = backend / "evaluation" / "scripts" / "evaluation_trial_host.py"
    script.parent.mkdir(parents=True)
    (backend / "evaluation" / "__init__.py").write_bytes(b"")
    (script.parent / "__init__.py").write_bytes(b"")
    script.write_text(
        "import argparse, subprocess, sys, time\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--storage'); p.add_argument('--candidate-root'); p.add_argument('--mode'); p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--model'); p.add_argument('--index')\n"
        "a=p.parse_args()\n"
        "from pathlib import Path\n"
        "s=Path(a.storage); s.mkdir()\n"
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "(s/'child.pid').write_text(str(child.pid))\n"
        "sys.stdout.buffer.write(b'x' * (3 * 1024 * 1024)); sys.stdout.buffer.flush()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    import evaluation.blinded_runtime as runtime

    monkeypatch.setattr(runtime, "HOST_WATCHDOG_SECONDS", 10)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_bytes(b"{}")
    index = tmp_path / "index"
    index.mkdir()
    storage = tmp_path / "storage"
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"

    outcome = asyncio.run(
        _invoke_trial_host(
            tool_root=tmp_path,
            candidate_root=candidate,
            mode="all_tools",
            input_path=input_path,
            output_path=tmp_path / "output.json",
            storage=storage,
            model="synthetic",
            index=index,
            stdout_path=stdout,
            stderr_path=stderr,
        )
    )

    assert outcome == "transport_failure"
    assert stdout.stat().st_size == HOST_OUTPUT_CAP_BYTES
    child_pid = int((storage / "child.pid").read_text())
    deadline = time.monotonic() + 3
    while _process_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(child_pid)


def test_token_key_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.key"
    target.write_bytes(SYNTHETIC_KEY)
    link = tmp_path / "token.key"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")

    with pytest.raises((OSError, ValueError)):
        with token_key(link):
            pass
