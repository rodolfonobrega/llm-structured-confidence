"""llm-structured-confidence — extract path-aware confidence from LLM JSON responses."""

from __future__ import annotations

from typing import Any

from ._extract import extract_logprob_entries
from ._pandas import add_confidence_columns, extract_confidence
from ._types import FieldLogprob, PathFieldLogprob, TokenInfo, TopAlternative

__version__ = "0.4.5"

__all__ = [
    "extract_logprobs",
    "extract_confidence",
    "add_confidence_columns",
    "FieldLogprob",
    "PathFieldLogprob",
    "TokenInfo",
    "TopAlternative",
]


def extract_logprobs(
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
        When omitted, all atomic values are returned, or enum-valued paths
        detected from ``response_schema`` when available.
    response_schema
        Optional Pydantic model class or JSON Schema dict. When provided,
        it is used to auto-detect enum-valued paths and to resolve
        ``TopAlternative.resolved_value`` from token prefixes.

    Returns
    -------
    list[PathFieldLogprob]
        Ordered path-aware results, one item per resolved atomic value.
    """
    return extract_logprob_entries(
        response,
        field_path=field_path,
        response_schema=response_schema,
    )
