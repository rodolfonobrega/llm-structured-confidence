"""llm-structured-confidence — extract per-field confidence from LLM JSON responses.

Usage::

    from llm_structured_confidence import extract_field_logprobs

    result = extract_field_logprobs(response, field="category")
    print(result["health and wellness"].joint_probability)   # 0.845
    print(result["health and wellness"].mean_nonzero_probability)  # 0.845
"""

from __future__ import annotations

from typing import Any

from ._extract import extract_field_value_map, extract_path_entries
from ._pandas import add_confidence_columns, extract_confidence
from ._types import FieldLogprob, PathFieldLogprob, TokenInfo, TopAlternative

__version__ = "0.3.0"

__all__ = [
    "extract_field_logprobs",
    "extract_path_logprobs",
    "extract_confidence",
    "add_confidence_columns",
    "FieldLogprob",
    "PathFieldLogprob",
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
    return extract_field_value_map(
        response,
        field=field,
        response_schema=response_schema,
    )


def extract_path_logprobs(
    response: Any,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> list[PathFieldLogprob]:
    """Extract path-aware logprob metrics for atomic JSON values.

    Parameters
    ----------
    response
        A provider SDK object or raw dict batch response with logprobs.
    field_path
        Explicit path to an atomic field, using ``.`` for object traversal
        and ``[]`` for arrays, e.g. ``"classifications[].name"``.
    response_schema
        Optional Pydantic model class or JSON Schema dict. When *field_path*
        is ``None``, the schema is inspected recursively to auto-detect
        enum-valued paths.

    Returns
    -------
    list[PathFieldLogprob]
        Ordered path-aware results, one item per resolved atomic value.
    """
    return extract_path_entries(
        response,
        field_path=field_path,
        response_schema=response_schema,
    )
