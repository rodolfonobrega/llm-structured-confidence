"""Pandas integration — add confidence columns from batch API responses.

Usage::

    import pandas as pd
    from llm_structured_confidence import add_confidence_columns

    df = pd.read_json("vertex_batch_output.jsonl", lines=True)
    df = add_confidence_columns(df, response_column="response", field="category")

    # Now df has: confidence_value, confidence_path, confidence_prob,
    #             confidence_joint_prob, confidence_top_alt,
    #             confidence_top_alt_prob
"""

from __future__ import annotations

from typing import Any

from ._extract import extract_path_entries
from ._types import PathFieldLogprob


def _extract_first_field_logprob(
    response: Any,
    field: str | None,
    field_path: str | None,
    response_schema: type | dict[str, Any] | None,
) -> PathFieldLogprob | None:
    """Core extraction: compute metrics for the first matching atomic value."""
    entries = extract_path_entries(
        response,
        field=field,
        field_path=field_path,
        response_schema=response_schema,
    )
    return entries[0] if entries else None


def extract_confidence(
    response: Any,
    *,
    field: str | None = None,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract confidence metrics from a single response into a flat dict.

    Parameters
    ----------
    response
        SDK object or raw dict (OpenAI / Vertex AI batch).
    field
        JSON field name (e.g. ``"category"``).
    field_path
        Optional nested path to an atomic field, e.g.
        ``"classifications[].name"``. Takes precedence over *field*.
    response_schema
        Optional Pydantic model or JSON Schema used to auto-detect fields
        and resolve enum/literal alternatives from token prefixes.

    Returns
    -------
    dict
        Flat dict with keys: ``value``, ``joint_probability``,
        ``mean_probability``, ``mean_nonzero_probability``, ``path``,
        ``top_alternative``, ``top_alternative_resolved``,
        ``top_alternative_probability``,
        ``top_logprobs`` (list of ``(token, probability)`` tuples).
        Returns dict with ``None`` values on error.
    """
    empty: dict[str, Any] = {
        "value": None,
        "joint_probability": None,
        "mean_probability": None,
        "mean_nonzero_probability": None,
        "path": None,
        "top_alternative": None,
        "top_alternative_resolved": None,
        "top_alternative_probability": None,
        "top_logprobs": None,
        "error": None,
    }
    try:
        entry = _extract_first_field_logprob(response, field, field_path, response_schema)
    except Exception as e:
        return {**empty, "error": f"{type(e).__name__}: {e}"}

    if entry is None:
        return {**empty, "error": "no values found"}

    fl = entry.field_logprob
    top_alt = fl.top_logprobs[1] if len(fl.top_logprobs) > 1 else None

    return {
        "value": str(fl.value),
        "joint_probability": fl.joint_probability,
        "mean_probability": fl.mean_probability,
        "mean_nonzero_probability": fl.mean_nonzero_probability,
        "path": entry.path,
        "top_alternative": top_alt.token if top_alt else None,
        "top_alternative_resolved": top_alt.resolved_value if top_alt else None,
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
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
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
    field_path
        Optional nested path to an atomic field, e.g.
        ``"classifications[].name"``. Takes precedence over *field*.
    response_schema
        Optional Pydantic model or JSON Schema for auto-detection and
        enum/literal alternative resolution.
    prefix
        Prefix for the new columns (default ``"confidence"``).

    Returns
    -------
    DataFrame
        The original DataFrame with new columns appended:
        ``{prefix}_value``, ``{prefix}_path``, ``{prefix}_prob``, ``{prefix}_joint_prob``,
        ``{prefix}_top_alt``, ``{prefix}_top_alt_resolved``,
        ``{prefix}_top_alt_prob``.

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
        lambda resp: extract_confidence(
            resp,
            field=field,
            field_path=field_path,
            response_schema=response_schema,
        )
    )
    metrics_df = pd.DataFrame(records.tolist())

    df = df.copy()
    df[f"{prefix}_value"] = metrics_df["value"]
    df[f"{prefix}_path"] = metrics_df["path"]
    df[f"{prefix}_prob"] = metrics_df["mean_nonzero_probability"]
    df[f"{prefix}_joint_prob"] = metrics_df["joint_probability"]
    df[f"{prefix}_top_alt"] = metrics_df["top_alternative"]
    df[f"{prefix}_top_alt_resolved"] = metrics_df["top_alternative_resolved"]
    df[f"{prefix}_top_alt_prob"] = metrics_df["top_alternative_probability"]
    df[f"{prefix}_error"] = metrics_df["error"]

    return df
