"""Shared extraction helpers built around path-aware results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._classification import (
    classification_values_by_path,
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


def extract_logprob_entries(
    response: Any,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> list[PathFieldLogprob]:
    """Extract path-aware logprobs for atomic values, preserving item order."""
    normalized = normalize_response(response)
    parsed = parse_json_spans(normalized.content)
    if not isinstance(parsed, (dict, list)):
        raise ValueError(
            f"Expected JSON object or array at top level, got {type(parsed).__name__}"
        )

    token_ranges = build_token_char_ranges(normalized.tokens)
    allowed_values_by_path = classification_values_by_path(response_schema)

    if field_path is not None:
        matches = _match_field_path(parsed, field_path)
    elif response_schema is not None:
        detected_paths = detect_classification_paths(response_schema)
        if detected_paths:
            matches = []
            for detected_path in detected_paths:
                matches.extend(_match_field_path(parsed, detected_path))
        else:
            matches = _match_all_atomic_values(parsed)
    else:
        matches = _match_all_atomic_values(parsed)

    return [
        PathFieldLogprob(
            path=match.path,
            value=match.span.value,
            field_logprob=_build_field_logprob(
                match.span,
                normalized.tokens,
                token_ranges,
                allowed_values=allowed_values_by_path.get(match.selector),
            ),
        )
        for match in matches
    ]


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


def _match_all_atomic_values(node: Any, *, current_path: str = "") -> list[_SpanMatch]:
    if isinstance(node, _ValueSpan):
        return [
            _SpanMatch(
                selector=current_path,
                path=current_path,
                span=node,
            )
        ]

    if isinstance(node, dict):
        matches: list[_SpanMatch] = []
        for key, value in node.items():
            child_path = f"{current_path}.{key}" if current_path else key
            matches.extend(_match_all_atomic_values(value, current_path=child_path))
        return matches

    if isinstance(node, list):
        matches: list[_SpanMatch] = []
        for idx, item in enumerate(node):
            child_path = f"{current_path}[{idx}]"
            matches.extend(_match_all_atomic_values(item, current_path=child_path))
        return matches

    return []


def _match_field_path(parsed: dict[str, Any] | list[Any], field_path: str) -> list[_SpanMatch]:
    segments = _parse_field_path(field_path)
    states: list[tuple[str, Any]] = [("", parsed)]

    for key, is_array in segments:
        next_states: list[tuple[str, Any]] = []
        for current_path, node in states:
            location = current_path or "<root>"

            if key == "" and is_array:
                if not isinstance(node, list):
                    raise ValueError(
                        f"field_path {field_path!r} expected a top-level array"
                    )
                for idx, item in enumerate(node):
                    next_states.append((f"[{idx}]", item))
                continue

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
    raw_segments = field_path.split(".")

    for i, raw_segment in enumerate(raw_segments):
        if not raw_segment:
            raise ValueError(f"Invalid field_path {field_path!r}")

        is_array = raw_segment.endswith("[]")
        key = raw_segment[:-2] if is_array else raw_segment

        if i == 0 and key == "" and is_array:
            segments.append(("", True))
            continue

        if not key or "[" in key or "]" in key:
            raise ValueError(f"Invalid field_path segment {raw_segment!r}")
        segments.append((key, is_array))

    return segments
