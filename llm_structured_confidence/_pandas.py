"""Pandas integration — add confidence columns from batch API responses.

Usage::

    import pandas as pd
    from llm_structured_confidence import add_confidence_columns

    df = pd.read_json("vertex_batch_output.jsonl", lines=True)
    df = add_confidence_columns(df, response_column="response", field="category")

    # Now df has: confidence_value, confidence_prob, confidence_joint_prob,
    #             confidence_top_alt, confidence_top_alt_prob
"""

from __future__ import annotations

from typing import Any

from ._converter import normalize_response
from ._parser import _ValueSpan, build_token_char_ranges, parse_json_spans, tokens_for_span
from ._types import FieldLogprob, TopAlternative


def _extract_first_field_logprob(
    response: Any,
    field: str | None,
) -> FieldLogprob | None:
    """Core extraction: normalize, parse, compute metrics for the first value."""
    normalized = normalize_response(response)
    parsed = parse_json_spans(normalized.content)
    if not isinstance(parsed, dict):
        return None

    if field is not None:
        target_fields = [field]
    else:
        target_fields = list(parsed.keys())

    token_ranges = build_token_char_ranges(normalized.tokens)

    for fname in target_fields:
        if fname not in parsed:
            continue
        field_value = parsed[fname]

        if isinstance(field_value, _ValueSpan):
            return _compute(field_value, normalized.tokens, token_ranges)

        if isinstance(field_value, list):
            for item in field_value:
                if isinstance(item, _ValueSpan):
                    return _compute(item, normalized.tokens, token_ranges)

    return None


def _compute(
    span: _ValueSpan,
    normalized_tokens: list,
    token_ranges: list[tuple[int, int]],
) -> FieldLogprob:
    """Compute FieldLogprob for one value span."""
    from ._parser import get_overlapping_indices

    token_infos = tokens_for_span(
        span.char_start, span.char_end, normalized_tokens, token_ranges,
    )

    indices = get_overlapping_indices(span.char_start, span.char_end, token_ranges)
    top_alts: list[TopAlternative] = []
    for idx in indices:
        nt = normalized_tokens[idx]
        if nt.logprob != 0.0 and nt.top_logprobs:
            top_alts = [TopAlternative(token=t, logprob=lp) for t, lp in nt.top_logprobs]
            break
    if not top_alts and indices:
        nt = normalized_tokens[indices[0]]
        if nt.top_logprobs:
            top_alts = [TopAlternative(token=t, logprob=lp) for t, lp in nt.top_logprobs]

    return FieldLogprob.compute(span.value, token_infos, top_alts)


def extract_confidence(
    response: Any,
    *,
    field: str | None = None,
    model: type | None = None,
) -> dict[str, Any]:
    """Extract confidence metrics from a single response into a flat dict.

    Parameters
    ----------
    response
        SDK object or raw dict (OpenAI / Vertex AI batch).
    field
        JSON field name (e.g. ``"category"``).
    model
        Optional Pydantic model for auto-detection.  Currently only
        ``field`` is used; pass the field name directly for clarity.

    Returns
    -------
    dict
        Flat dict with keys: ``value``, ``joint_probability``,
        ``mean_probability``, ``mean_nonzero_probability``,
        ``top_alternative``, ``top_alternative_probability``,
        ``top_logprobs`` (list of ``(token, probability)`` tuples).
        Returns dict with ``None`` values on error.
    """
    empty: dict[str, Any] = {
        "value": None,
        "joint_probability": None,
        "mean_probability": None,
        "mean_nonzero_probability": None,
        "top_alternative": None,
        "top_alternative_probability": None,
        "top_logprobs": None,
        "error": None,
    }
    try:
        fl = _extract_first_field_logprob(response, field)
    except Exception as e:
        return {**empty, "error": f"{type(e).__name__}: {e}"}

    if fl is None:
        return {**empty, "error": "no values found"}

    top_alt = fl.top_logprobs[1] if len(fl.top_logprobs) > 1 else None

    return {
        "value": str(fl.value),
        "joint_probability": fl.joint_probability,
        "mean_probability": fl.mean_probability,
        "mean_nonzero_probability": fl.mean_nonzero_probability,
        "top_alternative": top_alt.token if top_alt else None,
        "top_alternative_probability": top_alt.probability if top_alt else None,
        "top_logprobs": [
            (a.token, a.probability) for a in fl.top_logprobs
        ] if fl.top_logprobs else None,
        "error": None,
    }


def add_confidence_columns(
    df: Any,
    *,
    response_column: str = "response",
    field: str | None = None,
    model: type | None = None,
    prefix: str = "confidence",
) -> Any:
    """Add confidence metric columns to a DataFrame of batch API responses.

    Parameters
    ----------
    df
        pandas DataFrame with a column containing response dicts.
    response_column
        Name of the column with the response dicts. For Vertex AI batch
        output this is ``"response"``; for OpenAI batch output you may
        need to extract ``row["response"]["body"]`` first.
    field
        JSON field name to analyse (e.g. ``"category"``).
    model
        Optional Pydantic model for auto-detection.
    prefix
        Prefix for the new columns (default ``"confidence"``).

    Returns
    -------
    DataFrame
        The original DataFrame with new columns appended:
        ``{prefix}_value``, ``{prefix}_prob``, ``{prefix}_joint_prob``,
        ``{prefix}_top_alt``, ``{prefix}_top_alt_prob``.

    Example
    -------
    ::

        # Vertex AI batch output
        df = pd.read_json("vertex_batch_output.jsonl", lines=True)
        df = add_confidence_columns(df, response_column="response", field="category")

        # OpenAI batch output — extract body first
        df = pd.read_json("openai_batch_output.jsonl", lines=True)
        df["body"] = df["response"].apply(lambda r: r["body"])
        df = add_confidence_columns(df, response_column="body", field="category")
    """
    import pandas as pd

    records = df[response_column].apply(
        lambda resp: extract_confidence(resp, field=field, model=model)
    )
    metrics_df = pd.DataFrame(records.tolist())

    df = df.copy()
    df[f"{prefix}_value"] = metrics_df["value"]
    df[f"{prefix}_prob"] = metrics_df["mean_nonzero_probability"]
    df[f"{prefix}_joint_prob"] = metrics_df["joint_probability"]
    df[f"{prefix}_top_alt"] = metrics_df["top_alternative"]
    df[f"{prefix}_top_alt_prob"] = metrics_df["top_alternative_probability"]
    df[f"{prefix}_error"] = metrics_df["error"]

    return df
