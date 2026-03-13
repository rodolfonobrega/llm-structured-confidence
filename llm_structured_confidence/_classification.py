"""Helpers for schema-driven classification detection and enum-value resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ._converter import NormalizedToken
from ._types import TopAlternative


def detect_classification_fields(response_schema: type | dict[str, Any] | None) -> list[str]:
    """Inspect a response schema for classification fields."""
    normalized = normalize_response_schema(response_schema)
    if not normalized:
        return []
    return list(_json_schema_values_by_field(normalized).keys())


def classification_values_by_field(
    response_schema: type | dict[str, Any] | None,
) -> dict[str, list[Any]]:
    """Return allowed classification values for each detectable schema field."""
    normalized = normalize_response_schema(response_schema)
    if not normalized:
        return {}
    return _json_schema_values_by_field(normalized)


def normalize_response_schema(
    response_schema: type | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize Pydantic or wrapped schemas into a plain JSON Schema object."""
    if response_schema is None:
        return None

    if hasattr(response_schema, "model_json_schema"):
        schema = response_schema.model_json_schema()
        return _resolve_local_refs(schema) if isinstance(schema, dict) else None

    if isinstance(response_schema, dict):
        unwrapped = _unwrap_json_schema(response_schema)
        return _resolve_local_refs(unwrapped) if isinstance(unwrapped, dict) else None

    return None


def extract_top_alternatives(
    indices: list[int],
    normalized_tokens: list[NormalizedToken],
    *,
    allowed_values: list[Any] | None = None,
) -> list[TopAlternative]:
    """Build top alternatives from the first meaningful token in a span."""
    token_alts = _first_top_logprobs(indices, normalized_tokens)
    if not token_alts:
        return []

    return [
        TopAlternative(
            token=token,
            logprob=logprob,
            resolved_value=resolve_token_prefix(token, allowed_values),
        )
        for token, logprob in token_alts
    ]


def resolve_token_prefix(
    token: str,
    allowed_values: list[Any] | None,
) -> Any | None:
    """Resolve a token prefix to a full enum/literal value when unique."""
    if not token or not allowed_values:
        return None

    matches = [
        value
        for value in allowed_values
        if isinstance(value, str) and value.startswith(token)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _first_top_logprobs(
    indices: list[int],
    normalized_tokens: list[NormalizedToken],
) -> list[tuple[str, float]]:
    for idx in indices:
        nt = normalized_tokens[idx]
        if nt.logprob != 0.0 and nt.top_logprobs:
            return nt.top_logprobs
    if indices:
        nt = normalized_tokens[indices[0]]
        if nt.top_logprobs:
            return nt.top_logprobs
    return []


def _json_schema_values_by_field(response_schema: dict[str, Any]) -> dict[str, list[Any]]:
    properties = response_schema.get("properties")
    if not isinstance(properties, dict):
        return {}

    values: dict[str, list[Any]] = {}
    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        choices = _json_schema_choices(property_schema)
        if choices:
            values[name] = choices
    return values


def _unwrap_json_schema(response_schema: dict[str, Any]) -> dict[str, Any] | None:
    if "properties" in response_schema:
        return response_schema

    json_schema = response_schema.get("json_schema")
    if isinstance(json_schema, dict):
        inner_schema = json_schema.get("schema")
        if isinstance(inner_schema, dict):
            return inner_schema
        return json_schema if "properties" in json_schema else None

    inner_schema = response_schema.get("schema")
    if isinstance(inner_schema, dict):
        return inner_schema

    return None


def _json_schema_choices(property_schema: dict[str, Any]) -> list[Any]:
    enum_values = property_schema.get("enum")
    if isinstance(enum_values, list):
        return enum_values

    if "const" in property_schema:
        return [property_schema["const"]]

    if property_schema.get("type") == "array":
        items = property_schema.get("items")
        if isinstance(items, dict):
            return _json_schema_choices(items)

    return []


def _resolve_local_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local JSON Schema refs so enum detection has a single shape."""
    root = deepcopy(schema)
    return _resolve_node(root, root)


def _resolve_node(node: Any, root: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        if "$ref" in node and isinstance(node["$ref"], str):
            resolved = _resolve_ref(node["$ref"], root)
            merged = {
                **(_resolve_node(resolved, root) if isinstance(resolved, dict) else {}),
                **{k: v for k, v in node.items() if k != "$ref"},
            }
            return _resolve_node(merged, root)
        return {key: _resolve_node(value, root) for key, value in node.items()}

    if isinstance(node, list):
        return [_resolve_node(value, root) for value in node]

    return node


def _resolve_ref(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        return {}

    current: Any = root
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            return {}
        current = current[key]
    return current
