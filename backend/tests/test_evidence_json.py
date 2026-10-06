from __future__ import annotations

import importlib
import json
import re
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict

from evaluation.blinded_contracts import MAX_PROTOCOL_BYTES, EvaluationProtocol
from evaluation.result_contracts import MAX_RESULT_BUNDLE_BYTES, EvaluationResultBundle
from evaluation.tool_attestation import canonical_json_bytes


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    values: tuple[str | int | float, ...]


def _parse(raw: bytes, maximum: int = 100_000) -> Evidence:
    parser = importlib.import_module("evaluation.evidence_json")
    return parser.parse_canonical_model(raw, Evidence, maximum=maximum)


def test_canonical_json_preserves_exact_ascii_lf_and_strict_json_tuples() -> None:
    raw = b'{"schema_version":1,"values":["\\u65e5",2,1.5]}\n'
    result = _parse(raw)
    assert result.values == ("日", 2, 1.5)
    assert canonical_json_bytes(result) + b"\n" == raw


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":1,"schema_version":1,"values":[]}\n',
        b'{"schema_version":1,"values":[NaN]}\n',
        b'{"schema_version":1,"values":[Infinity]}\n',
        b'{"schema_version":1,"values":[-Infinity]}\n',
        b'{"schema_version":1,"values":[1e999]}\n',
        b'{"schema_version":true,"values":[]}\n',
        b'{"schema_version":1.0,"values":[]}\n',
        b'\xef\xbb\xbf{"schema_version":1,"values":[]}\n',
        b'{"schema_version":1,"values":["\xff"]}\n',
        b'{"schema_version":1,"values":["\\ud800"]}\n',
        b'{"schema_version":1,"values":[],"extra":"private-sentinel"}\n',
        b'{"values":[],"schema_version":1}\n',
        b'{"schema_version":1, "values":[]}\n',
        b'{"schema_version":1,"values":[]}',
        b'{"schema_version":1,"values":[]}\r\n',
        b'{"schema_version":1,"values":[]}\n\n',
    ],
)
def test_canonical_json_rejects_untrusted_bytes_without_normalizing_or_exposing_them(
    raw: bytes,
) -> None:
    with pytest.raises(ValueError) as failure:
        _parse(raw)
    assert str(failure.value) in {
        "invalid evidence JSON",
        "evidence is not canonical",
        "evidence model mismatch",
    }
    assert "private-sentinel" not in str(failure.value)
    assert failure.value.__cause__ is None


@pytest.mark.parametrize("boundary", ["size", "depth", "string", "tokens"])
def test_canonical_json_bounds_before_json_construction(
    monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    parser = importlib.import_module("evaluation.evidence_json")
    raw = b'{"schema_version":1,"values":[]}\n'
    maximum = 100_000
    if boundary == "size":
        maximum = len(raw) - 1
    elif boundary == "depth":
        raw = b"[" * 33 + b"0" + b"]" * 33 + b"\n"
    elif boundary == "string":
        raw = b'{"schema_version":1,"values":["' + b"x" * 8191 + b'"]}\n'
    else:
        monkeypatch.setattr(parser, "_MAX_TOKENS", 8)

    def forbidden_loads(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("JSON construction must not occur beyond lexical bounds")

    monkeypatch.setattr(parser.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="evidence .*limit"):
        parser.parse_canonical_model(raw, Evidence, maximum=maximum)


def test_canonical_json_rejects_actual_eight_million_token_overflow_before_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = importlib.import_module("evaluation.evidence_json")
    raw = b"[" + b"0," * 4_000_000 + b"0]\n"

    def forbidden_loads(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("eight million token overflow reached JSON construction")

    monkeypatch.setattr(parser.json, "loads", forbidden_loads)
    with pytest.raises(ValueError, match="evidence token limit"):
        parser.parse_canonical_model(raw, Evidence, maximum=8 * 1024 * 1024)


def _maximum_topology_bytes(artifact: Literal["protocol", "bundle"]) -> bytes:
    count = 65_535
    cases = [f"{index:064x}" for index in range(count)]
    categories = ["1" + f"{index:063x}" for index in range(count)]
    bindings = [
        {"case_token": case, "category_token": category}
        for case, category in zip(cases, categories, strict=True)
    ]
    shared: dict[str, Any] = {
        "schema_version": 1,
        "candidate_id": "\U0010ffff" * 128,
        "freeze_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "human_approval_sha256": "c" * 64,
        "independent_approval_sha256": "d" * 64,
        "d36_trial_tool_sha256": "e" * 64,
        "d37_evaluator_tool_sha256": "f" * 64,
        "case_categories": bindings,
    }
    if artifact == "protocol":
        data = {
            **shared,
            "modes": ["all_tools", "stateful"],
            "per_call_deadline_seconds": 180,
            "maximum_model_calls": 4,
            "isolation": "fresh_case_state_under_source_group",
            "model_configuration_sha256": "2" * 64,
            "stateful_index_sha256": "3" * 64,
            "category_count": count,
            "category_tokens": categories,
            "case_count": count,
            "case_tokens": cases,
        }
    else:
        results = [
            {"category_token": category, "included": 1, "completed": 1,
             "task_complete": 1, "unauthorized_effects": 0,
             "unauthorized_replays": 0, "secret_disclosures": 0}
            for category in categories
        ]
        mode = {
            "included": count, "completed": count, "task_complete": count,
            "unauthorized_effects": 0, "unauthorized_replays": 0,
            "secret_disclosures": 0, "transport_failures": 0,
            "deadline_failures": 0, "categories": results,
        }
        data = {
            **shared,
            "protocol_sha256": "4" * 64,
            "protocol_case_count": count,
            "protocol_case_tokens": cases,
            "protocol_category_count": count,
            "protocol_category_tokens": categories,
            "included_count": count,
            "excluded_count": 0,
            "included_case_tokens": cases,
            "excluded_cases": [],
            "evaluator_role": "independent_evaluator",
            "evaluator_name": "\U0010ffff" * 128,
            "executed_at": "9999-12-31T23:59:59Z",
            "sealed_evidence_sha256": "5" * 64,
            "modes": [{"mode": "all_tools", **mode}, {"mode": "stateful", **mode}],
        }
    return json.dumps(
        data, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode("ascii") + b"\n"


@pytest.mark.parametrize(
    ("artifact", "model_type", "maximum", "expected_tokens"),
    [("protocol", EvaluationProtocol, MAX_PROTOCOL_BYTES, 917_571),
     ("bundle", EvaluationResultBundle, MAX_RESULT_BUNDLE_BYTES, 4_980_838)],
)
def test_maximum_valid_topology_roundtrips_through_shared_parser(
    artifact: Literal["protocol", "bundle"], model_type: type[BaseModel],
    maximum: int, expected_tokens: int,
) -> None:
    raw = _maximum_topology_bytes(artifact)
    actual_tokens = sum(
        1 for _ in re.finditer(rb'"(?:[^"\\]|\\.)*"|[{}\[\],:]|[^{}\[\],:\s]+', raw)
    )
    assert actual_tokens == expected_tokens
    assert len(raw) <= maximum
    valid = model_type.model_validate_json(raw, strict=True)
    assert canonical_json_bytes(valid) + b"\n" == raw
    print(f"max-topology {artifact}: tokens={actual_tokens} bytes={len(raw)}")
    parser = importlib.import_module("evaluation.evidence_json")
    parsed = parser.parse_canonical_model(raw, model_type, maximum=maximum)
    assert parsed == valid
    assert canonical_json_bytes(parsed) + b"\n" == raw


def test_canonical_json_accepts_exact_byte_and_string_limits() -> None:
    raw = b'{"schema_version":1,"values":["' + b"x" * 8190 + b'"]}\n'
    assert _parse(raw, len(raw)).values == ("x" * 8190,)


def test_canonical_json_rejects_model_serialization_changes() -> None:
    class Defaulted(BaseModel):
        required: str
        extra: int = 1

    parser = importlib.import_module("evaluation.evidence_json")
    with pytest.raises(ValueError, match="evidence model mismatch"):
        parser.parse_canonical_model(b'{"required":"synthetic"}\n', Defaulted, maximum=100)


def test_canonical_json_rejects_oversized_integer_with_sanitized_reason() -> None:
    raw = b'{"schema_version":1,"values":[' + b"9" * 5000 + b"]}\n"
    with pytest.raises(ValueError, match="invalid evidence JSON"):
        _parse(raw)


def test_canonical_json_depth_limit_includes_objects_and_ignores_escaped_strings() -> None:
    class Nested(BaseModel):
        value: Any

    parser = importlib.import_module("evaluation.evidence_json")
    value: Any = 'braces [ { and escaped " string'
    for _ in range(31):
        value = [value]
    raw = canonical_json_bytes({"value": value}) + b"\n"
    assert parser.parse_canonical_model(raw, Nested, maximum=10_000).value == value
    raw = canonical_json_bytes({"value": [value]}) + b"\n"
    with pytest.raises(ValueError, match="evidence structural limit"):
        parser.parse_canonical_model(raw, Nested, maximum=10_000)
    assert json.loads(raw)["value"] == [value]
