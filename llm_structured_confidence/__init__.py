"""llm-structured-confidence — extract per-field confidence from LLM JSON responses.

Usage::

    from llm_structured_confidence import extract_field_logprobs

    result = extract_field_logprobs(response, field="category")
    print(result["health and wellness"].joint_probability)   # 0.845
    print(result["health and wellness"].mean_nonzero_probability)  # 0.845
"""

from __future__ import annotations

import json
from typing import Any

from ._classification import (
    classification_values_by_field,
    detect_classification_fields,
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
from ._pandas import add_confidence_columns, extract_confidence
from ._types import FieldLogprob, TokenInfo, TopAlternative

__version__ = "0.1.1"

__all__ = [
    "extract_field_logprobs",
    "extract_confidence",
    "add_confidence_columns",
    "FieldLogprob",
    "TokenInfo",
    "TopAlternative",
]


def extract_field_logprobs(
    response: Any,
    *,
    field: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, FieldLogprob]:
    """Extract logprob metrics for values in a structured JSON response.

    Parameters
    ----------
    response
        A ``litellm.ModelResponse``, ``openai.ChatCompletion``, or
        ``google.genai`` ``GenerateContentResponse`` **with logprobs enabled**.
        Also accepts raw dicts from the OpenAI / Vertex AI batch APIs.
    field
        JSON field name to analyse (e.g. ``"category"``).  Takes precedence
        over *response_schema*.
    response_schema
        Optional Pydantic model class or JSON Schema dict. When *field* is
        ``None``, the schema is inspected to auto-detect enum-valued fields.

    Returns
    -------
    dict[str, FieldLogprob]
        Mapping from **value** (as string) to its ``FieldLogprob`` metrics.
        For a scalar field this is a single-entry dict; for an array field
        there is one entry per element.
    """
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

        elif isinstance(field_value, list):
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


# ── internals ─────────────────────────────────────────────────────────

def _build_field_logprob(
    span: _ValueSpan,
    normalized_tokens: list[NormalizedToken],
    token_ranges: list[tuple[int, int]],
    *,
    allowed_values: list[Any] | None = None,
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
    """Get ``top_logprobs`` from the first non-zero token in the span.

    This is the most informative position for classification: the token
    where the model actually chose between alternatives.  With ENUM
    constraints the remaining tokens are typically deterministic.
    """
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
    """Decide which JSON field(s) to analyse."""
    if field is not None:
        return [field]

    if response_schema is not None:
        detected = _detect_classification_fields(response_schema)
        if detected:
            return detected

    parsed = json.loads(content)
    if isinstance(parsed, dict):
        return list(parsed.keys())
    return []


def _detect_classification_fields(
    response_schema: type | dict[str, Any],
) -> list[str]:
    return detect_classification_fields(response_schema)
