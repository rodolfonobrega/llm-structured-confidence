"""llm-confidence — extract per-field confidence from LLM JSON responses.

Usage::

    from llm_confidence import extract_field_logprobs

    result = extract_field_logprobs(response, field="category")
    print(result["health and wellness"].joint_probability)   # 0.845
    print(result["health and wellness"].mean_nonzero_probability)  # 0.845
"""

from __future__ import annotations

import enum
import json
import typing
from typing import Any

from ._converter import NormalizedResponse, NormalizedToken, normalize_response
from ._parser import (
    _ValueSpan,
    build_token_char_ranges,
    get_overlapping_indices,
    parse_json_spans,
    tokens_for_span,
)
from ._types import FieldLogprob, TokenInfo, TopAlternative

__version__ = "0.1.0"

__all__ = [
    "extract_field_logprobs",
    "FieldLogprob",
    "TokenInfo",
    "TopAlternative",
]


def extract_field_logprobs(
    response: Any,
    *,
    field: str | None = None,
    model: type | None = None,
) -> dict[str, FieldLogprob]:
    """Extract logprob metrics for values in a structured JSON response.

    Parameters
    ----------
    response
        A ``litellm.ModelResponse``, ``openai.ChatCompletion``, or
        ``google.genai`` ``GenerateContentResponse`` **with logprobs enabled**.
    field
        JSON field name to analyse (e.g. ``"category"``).  Takes precedence
        over *model*.
    model
        Optional Pydantic model class.  When *field* is ``None``, the model
        is inspected to auto-detect ``Enum`` / ``list[Enum]`` /
        ``Literal[...]`` fields.

    Returns
    -------
    dict[str, FieldLogprob]
        Mapping from **value** (as string) to its ``FieldLogprob`` metrics.
        For a scalar field this is a single-entry dict; for an array field
        there is one entry per element.
    """
    normalized = normalize_response(response)

    target_fields = _resolve_target_fields(field, model, normalized.content)

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
            fl = _build_field_logprob(field_value, normalized.tokens, token_ranges)
            result[str(field_value.value)] = fl

        elif isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, _ValueSpan):
                    fl = _build_field_logprob(item, normalized.tokens, token_ranges)
                    result[str(item.value)] = fl

    return result


# ── internals ─────────────────────────────────────────────────────────

def _build_field_logprob(
    span: _ValueSpan,
    normalized_tokens: list[NormalizedToken],
    token_ranges: list[tuple[int, int]],
) -> FieldLogprob:
    """Compute all metrics for one atomic value span."""
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
    )

    return FieldLogprob.compute(span.value, token_infos, top_alts)


def _extract_top_logprobs(
    char_start: int,
    char_end: int,
    normalized_tokens: list[NormalizedToken],
    token_ranges: list[tuple[int, int]],
) -> list[TopAlternative]:
    """Get ``top_logprobs`` from the first non-zero token in the span.

    This is the most informative position for classification: the token
    where the model actually chose between alternatives.  With ENUM
    constraints the remaining tokens are typically deterministic.
    """
    indices = get_overlapping_indices(char_start, char_end, token_ranges)
    for idx in indices:
        nt = normalized_tokens[idx]
        if nt.logprob != 0.0 and nt.top_logprobs:
            return [
                TopAlternative(token=tok, logprob=lp)
                for tok, lp in nt.top_logprobs
            ]
    if indices:
        nt = normalized_tokens[indices[0]]
        if nt.top_logprobs:
            return [
                TopAlternative(token=tok, logprob=lp)
                for tok, lp in nt.top_logprobs
            ]
    return []


def _resolve_target_fields(
    field: str | None,
    model: type | None,
    content: str,
) -> list[str]:
    """Decide which JSON field(s) to analyse."""
    if field is not None:
        return [field]

    if model is not None:
        detected = _detect_classification_fields(model)
        if detected:
            return detected

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return list(parsed.keys())
    return []


def _detect_classification_fields(model: type) -> list[str]:
    """Inspect a Pydantic model for Enum / list[Enum] / Literal fields."""
    if not hasattr(model, "model_fields"):
        return []

    fields: list[str] = []
    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if annotation is None:
            continue

        if _is_enum_type(annotation):
            fields.append(name)
            continue

        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)

        if origin is list and args and _is_enum_type(args[0]):
            fields.append(name)
            continue

        if origin is typing.Literal:
            fields.append(name)
            continue

        if origin is list and args:
            inner_origin = typing.get_origin(args[0])
            if inner_origin is typing.Literal:
                fields.append(name)
                continue

    return fields


def _is_enum_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, enum.Enum)
