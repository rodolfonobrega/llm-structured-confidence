"""Normalize LLM responses from different providers into a common format.

Supported inputs
----------------
* **litellm** ``ModelResponse``  /  **OpenAI** ``ChatCompletion``
  → reads ``choices[0].message.content`` and ``choices[0].logprobs.content``
* **google-genai** ``GenerateContentResponse``  (Vertex AI / Google AI Studio)
  → reads ``candidates[0].content.parts[0].text`` and
  ``candidates[0].logprobs_result`` using the same mapping logic as
  litellm's ``VertexGeminiConfig._transform_logprobs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NormalizedToken:
    """Provider-agnostic token with logprob and top alternatives."""

    token: str
    logprob: float
    top_logprobs: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class NormalizedResponse:
    """Provider-agnostic response: the JSON content string plus its tokens."""

    content: str
    tokens: list[NormalizedToken]


def normalize_response(response: Any) -> NormalizedResponse:
    """Detect the response type and convert to ``NormalizedResponse``.

    Raises ``TypeError`` for unsupported types and ``ValueError`` when
    logprobs are missing from an otherwise valid response.
    """
    if hasattr(response, "choices"):
        return _from_openai(response)

    if hasattr(response, "candidates"):
        return _from_genai(response)

    raise TypeError(
        f"Unsupported response type: {type(response).__name__}. "
        "Expected litellm ModelResponse, openai ChatCompletion, "
        "or google-genai GenerateContentResponse."
    )


# ── litellm / OpenAI ─────────────────────────────────────────────────

def _from_openai(response: Any) -> NormalizedResponse:
    choice = response.choices[0]
    content: str = choice.message.content or ""

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        raise ValueError(
            "Response has no logprobs. "
            "Pass logprobs=True when calling the API."
        )

    lp_content = getattr(logprobs, "content", None)
    if lp_content is None:
        raise ValueError(
            "Response logprobs.content is None. "
            "Pass logprobs=True when calling the API."
        )

    tokens: list[NormalizedToken] = []
    for t in lp_content:
        top: list[tuple[str, float]] = []
        for alt in getattr(t, "top_logprobs", None) or []:
            top.append((alt.token, alt.logprob))
        tokens.append(NormalizedToken(token=t.token, logprob=t.logprob, top_logprobs=top))

    return NormalizedResponse(content=content, tokens=tokens)


# ── google-genai (Vertex AI / Google AI Studio) ──────────────────────

def _from_genai(response: Any) -> NormalizedResponse:
    """Convert using the same field mapping as litellm ``_transform_logprobs``."""
    candidate = response.candidates[0]

    content = ""
    if getattr(candidate, "content", None) and candidate.content.parts:
        content = candidate.content.parts[0].text or ""

    lr = getattr(candidate, "logprobs_result", None)
    if lr is None:
        raise ValueError(
            "Response has no logprobs_result. "
            "Pass response_logprobs=True in GenerateContentConfig."
        )

    chosen = getattr(lr, "chosen_candidates", None) or []
    top_candidates_list = getattr(lr, "top_candidates", None) or []

    tokens: list[NormalizedToken] = []
    for idx, ch in enumerate(chosen):
        top: list[tuple[str, float]] = []
        if idx < len(top_candidates_list):
            for alt in getattr(top_candidates_list[idx], "candidates", None) or []:
                top.append((alt.token, alt.log_probability))
        tokens.append(
            NormalizedToken(
                token=ch.token,
                logprob=ch.log_probability,
                top_logprobs=top,
            )
        )

    return NormalizedResponse(content=content, tokens=tokens)
