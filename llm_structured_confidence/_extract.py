"""Shared extraction helpers for top-level fields and nested field paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._classification import (
    classification_values_by_field,
    classification_values_by_path,
    detect_classification_fields,
    detect_classification_paths,
    extract_top_alternatives,
)
from ._converter import NormalizedToken, normalize_response
from ._parser import (
    _ValueSpan,
    build_token_char_ranges,
    get_overlapping_indices,
    parse_json_spans,
    tokens_for_span,
)
from ._types import FieldLogprob, PathFieldLogprob, TopAlternative


@dataclass(frozen=True)
class _SpanMatch:
    selector: str
    path: str
    span: _ValueSpan


def extract_field_value_map(
    response: Any,
    *,
    field: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, FieldLogprob]:
    """Extract logprobs for top-level fields only, preserving the legacy API."""
    normalized = normalize_response(response)

    target_fields = _resolve_target_fields(field, response_schema, normalized.content)
    allowed_values_by_field = classification_values_by_field(response_schema)

    parsed = parse_json_spans(normalized.content)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected JSON object at top level, got {type(parsed).__name__}"
        )

    token_ranges = build_token_char_ranges(normalized.tokens)
    result: dict[str, FieldLogprob] = {}

    for fname in target_fields:
        if fname not in parsed:
            continue
        field_value = parsed[fname]

        if isinstance(field_value, _ValueSpan):
            fl = _build_field_logprob(
                field_value,
                normalized.tokens,
                token_ranges,
                allowed_values=allowed_values_by_field.get(fname),
            )
            result[str(field_value.value)] = fl
            continue

        if isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, _ValueSpan):
                    fl = _build_field_logprob(
                        item,
                        normalized.tokens,
                        token_ranges,
                        allowed_values=allowed_values_by_field.get(fname),
                    )
                    result[str(item.value)] = fl

    return result


def extract_path_entries(
    response: Any,
    *,
    field: str | None = None,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> list[PathFieldLogprob]:
    """Extract path-aware logprobs for atomic values, preserving item order."""
    normalized = normalize_response(response)
    parsed = parse_json_spans(normalized.content)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected JSON object at top level, got {type(parsed).__name__}"
        )

    token_ranges = build_token_char_ranges(normalized.tokens)
    allowed_values_by_path = classification_values_by_path(response_schema)
    allowed_values_by_field = classification_values_by_field(response_schema)

    if field_path is not None:
        matches = _match_field_path(parsed, field_path)
    elif field is not None:
        matches = _match_top_level_field(parsed, field)
    elif response_schema is not None:
        detected_paths = detect_classification_paths(response_schema)
        if detected_paths:
            matches = []
            for detected_path in detected_paths:
                matches.extend(_match_field_path(parsed, detected_path))
        else:
            matches = _match_top_level_all(parsed)
    else:
        matches = _match_top_level_all(parsed)

    results: list[PathFieldLogprob] = []
    for match in matches:
        if field_path is not None or "[" in match.selector or "." in match.selector:
            allowed_values = allowed_values_by_path.get(match.selector)
        else:
            allowed_values = allowed_values_by_field.get(match.selector)

        field_logprob = _build_field_logprob(
            match.span,
            normalized.tokens,
            token_ranges,
            allowed_values=allowed_values,
        )
        results.append(
            PathFieldLogprob(
                path=match.path,
                value=match.span.value,
                field_logprob=field_logprob,
            )
        )

    return results


def _build_field_logprob(
    span: _ValueSpan,
    normalized_tokens: list[NormalizedToken],
    token_ranges: list[tuple[int, int]],
    *,
    allowed_values: list[Any] | None = None,
) -> FieldLogprob:
    token_infos = tokens_for_span(
        span.char_start,
        span.char_end,
        normalized_tokens,
        token_ranges,
    )
    top_alts = _extract_top_logprobs(
        span.char_start,
        span.char_end,
        normalized_tokens,
        token_ranges,
        allowed_values=allowed_values,
    )
    return FieldLogprob.compute(span.value, token_infos, top_alts)


def _extract_top_logprobs(
    char_start: int,
    char_end: int,
    normalized_tokens: list[NormalizedToken],
    token_ranges: list[tuple[int, int]],
    *,
    allowed_values: list[Any] | None = None,
) -> list[TopAlternative]:
    indices = get_overlapping_indices(char_start, char_end, token_ranges)
    return extract_top_alternatives(
        indices,
        normalized_tokens,
        allowed_values=allowed_values,
    )


def _resolve_target_fields(
    field: str | None,
    response_schema: type | dict[str, Any] | None,
    content: str,
) -> list[str]:
    if field is not None:
        return [field]

    if response_schema is not None:
        detected = detect_classification_fields(response_schema)
        if detected:
            return detected

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return list(parsed.keys())
    return []


def _match_top_level_all(parsed: dict[str, Any]) -> list[_SpanMatch]:
    matches: list[_SpanMatch] = []
    for field_name, field_value in parsed.items():
        matches.extend(_match_top_level_value(field_name, field_value))
    return matches


def _match_top_level_field(parsed: dict[str, Any], field: str) -> list[_SpanMatch]:
    if field not in parsed:
        return []
    return _match_top_level_value(field, parsed[field])


def _match_top_level_value(field: str, value: Any) -> list[_SpanMatch]:
    if isinstance(value, _ValueSpan):
        return [_SpanMatch(selector=field, path=field, span=value)]

    if isinstance(value, list):
        return [
            _SpanMatch(selector=field, path=f"{field}[{idx}]", span=item)
            for idx, item in enumerate(value)
            if isinstance(item, _ValueSpan)
        ]

    return []


def _match_field_path(parsed: dict[str, Any], field_path: str) -> list[_SpanMatch]:
    segments = _parse_field_path(field_path)
    states: list[tuple[str, Any]] = [("", parsed)]

    for key, is_array in segments:
        next_states: list[tuple[str, Any]] = []
        for current_path, node in states:
            location = current_path or "<root>"
            if isinstance(node, list):
                raise ValueError(
                    f"field_path {field_path!r} resolved to an array at {location!r}; "
                    "use [] to select array items"
                )
            if not isinstance(node, dict):
                raise ValueError(
                    f"field_path {field_path!r} expected an object at {location}"
                )
            if key not in node:
                continue

            child = node[key]
            child_path = f"{current_path}.{key}" if current_path else key

            if is_array:
                if not isinstance(child, list):
                    raise ValueError(
                        f"field_path {field_path!r} expected an array at {child_path!r}"
                    )
                for idx, item in enumerate(child):
                    next_states.append((f"{child_path}[{idx}]", item))
                continue

            next_states.append((child_path, child))

        states = next_states

    matches: list[_SpanMatch] = []
    for resolved_path, node in states:
        if isinstance(node, _ValueSpan):
            matches.append(
                _SpanMatch(selector=field_path, path=resolved_path, span=node)
            )
            continue

        if isinstance(node, list):
            raise ValueError(
                f"field_path {field_path!r} resolved to an array at {resolved_path!r}; "
                "use [] to select array items"
            )

        if isinstance(node, dict):
            raise ValueError(
                f"field_path {field_path!r} resolved to an object at {resolved_path!r}; "
                "select an atomic child field"
            )

    return matches


def _parse_field_path(field_path: str) -> list[tuple[str, bool]]:
    if not field_path:
        raise ValueError("field_path must not be empty")

    segments: list[tuple[str, bool]] = []
    for raw_segment in field_path.split("."):
        if not raw_segment:
            raise ValueError(f"Invalid field_path {field_path!r}")

        is_array = raw_segment.endswith("[]")
        key = raw_segment[:-2] if is_array else raw_segment
        if not key or "[" in key or "]" in key:
            raise ValueError(f"Invalid field_path segment {raw_segment!r}")
        segments.append((key, is_array))

    return segments
