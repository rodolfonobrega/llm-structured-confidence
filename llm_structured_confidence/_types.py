"""Data types for llm-structured-confidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenInfo:
    """A single token contributing to a JSON value, with its logprob and position."""

    token: str
    logprob: float
    char_start: int
    char_end: int

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


@dataclass(frozen=True)
class TopAlternative:
    """An alternative token the model considered at a given position."""

    token: str
    logprob: float
    resolved_value: Any | None = None

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)


@dataclass(frozen=True)
class FieldLogprob:
    """Logprob metrics for a single value in a structured JSON response.

    Three confidence metrics are provided:
      - joint: product of all token probabilities (strictest)
      - mean: geometric mean of all token probabilities
      - mean_nonzero: geometric mean of only non-deterministic tokens
        (most informative for ENUM classification, where only the first
        discriminating token carries real uncertainty).
        Falls back to 1.0 (100%) when all tokens are deterministic
        (logprob=0), meaning the model is fully confident.
        Only ``None`` when no tokens exist at all.
    """

    value: Any
    tokens: list[TokenInfo]

    joint_logprob: float
    joint_probability: float

    mean_logprob: float
    mean_probability: float

    mean_nonzero_logprob: float | None
    mean_nonzero_probability: float | None

    top_logprobs: list[TopAlternative]

    @staticmethod
    def compute(
        value: Any,
        tokens: list[TokenInfo],
        top_logprobs: list[TopAlternative] | None = None,
    ) -> FieldLogprob:
        """Build a FieldLogprob by computing all 3 metrics from token list."""
        if top_logprobs is None:
            top_logprobs = []

        if not tokens:
            return FieldLogprob(
                value=value,
                tokens=tokens,
                joint_logprob=0.0,
                joint_probability=1.0,
                mean_logprob=0.0,
                mean_probability=1.0,
                mean_nonzero_logprob=None,
                mean_nonzero_probability=None,
                top_logprobs=top_logprobs,
            )

        joint = sum(t.logprob for t in tokens)
        mean = joint / len(tokens)

        nonzero = [t.logprob for t in tokens if t.logprob != 0.0]
        if nonzero:
            mean_nz = sum(nonzero) / len(nonzero)
            mean_nz_prob = math.exp(mean_nz)
        else:
            mean_nz = 0.0
            mean_nz_prob = 1.0

        return FieldLogprob(
            value=value,
            tokens=tokens,
            joint_logprob=joint,
            joint_probability=math.exp(joint),
            mean_logprob=mean,
            mean_probability=math.exp(mean),
            mean_nonzero_logprob=mean_nz,
            mean_nonzero_probability=mean_nz_prob,
            top_logprobs=top_logprobs,
        )


@dataclass(frozen=True)
class PathFieldLogprob:
    """A resolved JSON path together with the confidence for its atomic value."""

    path: str
    value: Any
    field_logprob: FieldLogprob
