"""Fail-closed recursive subset of JSON Schema for typed operation arguments."""
from __future__ import annotations

import math
from typing import Any


class CatalogError(ValueError):
    """Malformed executable metadata or argument schema/value."""


def _kind(rule: dict[str, Any]) -> tuple[str, bool]:
    value = rule.get("type")
    if isinstance(value, list):
        if len(value) != 2 or not all(isinstance(item, str) for item in value) or value.count("null") != 1:
            raise CatalogError("only a single type plus null is supported")
        value = next(item for item in value if item != "null")
        return value, True
    if not isinstance(value, str):
        raise CatalogError("schema type must be a string or single type plus null")
    return value, False


def validate_schema(rule: dict[str, Any], depth: int = 0) -> None:
    if not isinstance(rule, dict) or depth > 8:
        raise CatalogError("invalid or overly nested input schema")
    kind, nullable = _kind(rule)
    keys = {"type", "enum"}
    if kind in {"integer", "number"}:
        keys |= {"minimum", "maximum"}
        limits = [rule[name] for name in ("minimum", "maximum") if name in rule]
        types = (int,) if kind == "integer" else (int, float)
        if any(isinstance(value, bool) or not isinstance(value, types) or not math.isfinite(value) for value in limits):
            raise CatalogError("numeric constraints must be finite numbers of the declared type")
        if rule.get("minimum", -math.inf) > rule.get("maximum", math.inf):
            raise CatalogError("minimum exceeds maximum")
    elif kind in {"string", "array"}:
        lower, upper = ("minLength", "maxLength") if kind == "string" else ("minItems", "maxItems")
        keys |= {lower, upper}
        for name in (lower, upper):
            if name in rule and (type(rule[name]) is not int or rule[name] < 0):
                raise CatalogError("length constraints must be non-negative integers")
        if rule.get(lower, 0) > rule.get(upper, math.inf):
            raise CatalogError("minimum length exceeds maximum length")
        if kind == "array":
            keys.add("items")
            validate_schema(rule.get("items"), depth + 1)
    elif kind == "object":
        keys |= {"properties", "required", "additionalProperties"}
        properties, required = rule.get("properties"), rule.get("required")
        if rule.get("additionalProperties") is not False or not isinstance(properties, dict) or not isinstance(required, list):
            raise CatalogError("object schema requires properties, required and additionalProperties false")
        if not all(isinstance(name, str) for name in required) or len(set(required)) != len(required) or not set(required) <= set(properties):
            raise CatalogError("invalid required fields")
        for name, child in properties.items():
            if not isinstance(name, str):
                raise CatalogError("property name must be string")
            validate_schema(child, depth + 1)
    elif kind != "boolean":
        raise CatalogError(f"unsupported schema type: {kind}")
    unknown = set(rule) - keys
    if unknown:
        raise CatalogError(f"unsupported schema keys: {sorted(unknown)}")
    if "enum" in rule:
        if not isinstance(rule["enum"], list) or not rule["enum"]:
            raise CatalogError("enum must be a nonempty list")
        for value in rule["enum"]:
            validate_value({key: val for key, val in rule.items() if key != "enum"}, value)
    if depth == 0 and (kind != "object" or nullable):
        raise CatalogError("input schema root type must be object")


def validate_value(rule: dict[str, Any], value: Any, name: str = "arguments") -> None:
    kind, nullable = _kind(rule)
    if value is None and nullable:
        return
    valid = {"integer": type(value) is int,
             "number": type(value) in {int, float} and math.isfinite(value),
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "object": isinstance(value, dict), "array": isinstance(value, list)}
    if not valid.get(kind, False):
        raise CatalogError(f"{name} must be {kind}")
    if "enum" in rule and value not in rule["enum"]:
        raise CatalogError(f"{name} is not an allowed value")
    if kind in {"integer", "number"}:
        if value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
            raise CatalogError(f"{name} is outside allowed bounds")
    if kind in {"string", "array"}:
        lower, upper = ("minLength", "maxLength") if kind == "string" else ("minItems", "maxItems")
        if not rule.get(lower, 0) <= len(value) <= rule.get(upper, math.inf):
            raise CatalogError(f"{name} has invalid length")
        if kind == "array":
            for index, child in enumerate(value):
                validate_value(rule["items"], child, f"{name}[{index}]")
    if kind == "object":
        missing = set(rule["required"]) - set(value)
        extra = set(value) - set(rule["properties"])
        if missing or extra:
            raise CatalogError(f"{name}: missing required arguments {sorted(missing)}; unexpected arguments {sorted(extra)}")
        for key, child in value.items():
            validate_value(rule["properties"][key], child, f"{name}.{key}")
