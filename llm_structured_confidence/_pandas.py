"""Pandas integration — add confidence columns from batch API responses."""

from __future__ import annotations

from typing import Any

from ._extract import extract_logprob_entries
from ._types import PathFieldLogprob


def _extract_first_logprob(
    response: Any,
    field_path: str | None,
    response_schema: type | dict[str, Any] | None,
) -> PathFieldLogprob | None:
    """Compute metrics for the first matching atomic value."""
    entries = extract_logprob_entries(
        response,
        field_path=field_path,
        response_schema=response_schema,
    )
    return entries[0] if entries else None


def extract_confidence(
    response: Any,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract confidence metrics from a single response into a flat dict."""
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
        entry = _extract_first_logprob(response, field_path, response_schema)
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
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
    prefix: str = "confidence",
) -> Any:
    """Add confidence metric columns to a DataFrame of batch API responses."""
    import pandas as pd

    records = df[response_column].apply(
        lambda resp: extract_confidence(
            resp,
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
