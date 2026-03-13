"""Shared fixtures for llm-structured-confidence tests."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from llm_structured_confidence._converter import NormalizedResponse, NormalizedToken


# ── helper to build fake tokens ──────────────────────────────────────

def make_tokens(raw: list[tuple[str, float]] | list[tuple[str, float, list[tuple[str, float]]]]) -> list[NormalizedToken]:
    """Build NormalizedToken list from ``[(text, logprob), ...]`` or
    ``[(text, logprob, [(alt_text, alt_lp), ...]), ...]``."""
    result: list[NormalizedToken] = []
    for item in raw:
        if len(item) == 2:
            text, lp = item
            top: list[tuple[str, float]] = []
        else:
            text, lp, top = item[0], item[1], item[2]
        result.append(NormalizedToken(token=text, logprob=lp, top_logprobs=top))
    return result


def make_normalized(content: str, raw_tokens: list) -> NormalizedResponse:
    return NormalizedResponse(content=content, tokens=make_tokens(raw_tokens))


# ── fake litellm/OpenAI response objects ─────────────────────────────

def make_openai_response(content: str, raw_tokens: list):
    """Build a fake object mimicking litellm.ModelResponse / openai ChatCompletion."""
    token_objects = []
    for item in raw_tokens:
        if len(item) == 2:
            text, lp = item
            top_lp = []
        else:
            text, lp, top_raw = item[0], item[1], item[2]
            top_lp = [SimpleNamespace(token=t, logprob=l) for t, l in top_raw]
        token_objects.append(SimpleNamespace(token=text, logprob=lp, top_logprobs=top_lp))

    logprobs = SimpleNamespace(content=token_objects)
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, logprobs=logprobs)
    return SimpleNamespace(choices=[choice])


def make_genai_response(content: str, raw_tokens: list):
    """Build a fake object mimicking google-genai GenerateContentResponse."""
    chosen = []
    top_candidates_list = []
    for item in raw_tokens:
        if len(item) == 2:
            text, lp = item
            top_raw = []
        else:
            text, lp, top_raw = item[0], item[1], item[2]
        chosen.append(SimpleNamespace(token=text, log_probability=lp))
        alts = [SimpleNamespace(token=t, log_probability=l) for t, l in top_raw]
        top_candidates_list.append(SimpleNamespace(candidates=alts))

    logprobs_result = SimpleNamespace(
        chosen_candidates=chosen,
        top_candidates=top_candidates_list,
    )
    part = SimpleNamespace(text=content)
    cand_content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=cand_content, logprobs_result=logprobs_result)
    return SimpleNamespace(candidates=[candidate])


def make_vertex_batch_dict(content: str, raw_tokens: list) -> dict:
    """Build a raw Vertex AI batch response dict with camelCase keys."""
    chosen_candidates = []
    top_candidates = []
    for item in raw_tokens:
        if len(item) == 2:
            text, lp = item
            top_raw = []
        else:
            text, lp, top_raw = item[0], item[1], item[2]
        chosen_candidates.append({"token": text, "logProbability": lp, "tokenId": 0})
        alts = [{"token": t, "logProbability": l, "tokenId": 0} for t, l in top_raw]
        top_candidates.append({"candidates": alts})

    return {
        "candidates": [{
            "content": {"parts": [{"text": content}], "role": "model"},
            "finishReason": "STOP",
            "logprobsResult": {
                "chosenCandidates": chosen_candidates,
                "topCandidates": top_candidates,
            },
        }],
    }


# ── Pydantic models for tests ────────────────────────────────────────

class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"
    technology = "technology"
    entertainment = "entertainment"


class SingleCategoryModel(BaseModel):
    category: CategoryEnum


class MultipleCategoriesModel(BaseModel):
    categories: list[CategoryEnum]


class LiteralModel(BaseModel):
    mood: Literal["happy", "sad", "neutral"]


class MixedModel(BaseModel):
    category: CategoryEnum
    confidence: float
    note: str


class NamedClassificationEnum(str, Enum):
    positive = "Positive"
    negative = "Negative"
    neutral = "Neutral"


class ClassificationItemModel(BaseModel):
    id: int
    name: NamedClassificationEnum
    color: str


class NestedClassificationModel(BaseModel):
    classifications: list[ClassificationItemModel]


# ── standard token fixtures ──────────────────────────────────────────

# Gemini 2.5-style: tokens nicely separated, space before opening quote
GEMINI25_SCALAR_CONTENT = '{"category": "health and wellness"}'
GEMINI25_SCALAR_TOKENS = [
    ('{"', -0.115685),
    ("category", 0.0),
    ('":', -0.000074),
    (' "', -0.000000),
    ("health", -0.012331, [("health", -0.012331), ("sport", -4.425068), ("tech", -10.174607)]),
    (" and", 0.0),
    (" wellness", 0.0),
    ('"}', -0.000021),
]

# Gemini 3-style: no space, colon+quotes merged into one token
GEMINI3_SCALAR_CONTENT = '{"category":"health and wellness"}'
GEMINI3_SCALAR_TOKENS = [
    ('{"', -0.006128),
    ("category", 0.0),
    ('":"', -0.200847),
    ("health", -0.168335, [("health", -0.168335), ("sport", -1.866217), ("tech", -9.090357)]),
    (" and", 0.0),
    (" wellness", -0.000000),
    ('"}', -0.000000),
]

# Array of categories
ARRAY_CONTENT = '{"categories":["health and wellness","sports"]}'
ARRAY_TOKENS = [
    ('{"', -0.005),
    ("categories", 0.0),
    ('":["', -0.100),
    ("health", -0.168, [("health", -0.168), ("sport", -1.866)]),
    (" and", 0.0),
    (" wellness", 0.0),
    ('","', -0.001),
    ("sport", -0.050, [("sport", -0.050), ("health", -2.300)]),
    ("s", 0.0),
    ('"]}', -0.000),
]

# Multi-field JSON
MULTI_FIELD_CONTENT = '{"category":"technology","amount":150}'
MULTI_FIELD_TOKENS = [
    ('{"', -0.01),
    ("category", 0.0),
    ('":"', -0.20),
    ("tech", -0.088, [("tech", -0.088), ("sport", -2.899)]),
    ("nology", 0.0),
    ('","', -0.001),
    ("amount", 0.0),
    ('":', -0.001),
    ("150", -0.500),
    ("}", -0.000),
]

NESTED_CLASSIFICATION_CONTENT = (
    '{"classifications":['
    '{"id":0,"name":"Positive","color":"#00FF00"},'
    '{"id":1,"name":"Negative","color":"#FF0000"},'
    '{"id":2,"name":"Positive","color":"#0000AA"}'
    ']}'
)

NESTED_CLASSIFICATION_TOKENS = [
    ('{"', -0.01),
    ("classifications", 0.0),
    ('":[{"', -0.02),
    ("id", 0.0),
    ('":', -0.01),
    ("0", -0.001),
    (',"', -0.01),
    ("name", 0.0),
    ('":"', -0.05),
    ("Pos", -0.04, [("Pos", -0.04), ("Neg", -1.5), ("Neu", -2.0)]),
    ("itive", 0.0),
    ('","', -0.01),
    ("color", 0.0),
    ('":"', -0.01),
    ("#00FF00", -0.001),
    ('"},{"', -0.01),
    ("id", 0.0),
    ('":', -0.01),
    ("1", -0.001),
    (',"', -0.01),
    ("name", 0.0),
    ('":"', -0.05),
    ("Neg", -0.06, [("Neg", -0.06), ("Pos", -1.2), ("Neu", -1.8)]),
    ("ative", 0.0),
    ('","', -0.01),
    ("color", 0.0),
    ('":"', -0.01),
    ("#FF0000", -0.001),
    ('"},{"', -0.01),
    ("id", 0.0),
    ('":', -0.01),
    ("2", -0.001),
    (',"', -0.01),
    ("name", 0.0),
    ('":"', -0.05),
    ("Pos", -0.03, [("Pos", -0.03), ("Neg", -1.4), ("Neu", -2.1)]),
    ("itive", 0.0),
    ('","', -0.01),
    ("color", 0.0),
    ('":"', -0.01),
    ("#0000AA", -0.001),
    ('"}]}', 0.0),
]


@pytest.fixture
def gemini25_scalar():
    return make_openai_response(GEMINI25_SCALAR_CONTENT, GEMINI25_SCALAR_TOKENS)


@pytest.fixture
def gemini3_scalar():
    return make_openai_response(GEMINI3_SCALAR_CONTENT, GEMINI3_SCALAR_TOKENS)


@pytest.fixture
def array_response():
    return make_openai_response(ARRAY_CONTENT, ARRAY_TOKENS)


@pytest.fixture
def multi_field_response():
    return make_openai_response(MULTI_FIELD_CONTENT, MULTI_FIELD_TOKENS)


@pytest.fixture
def genai_scalar():
    return make_genai_response(GEMINI3_SCALAR_CONTENT, GEMINI3_SCALAR_TOKENS)


@pytest.fixture
def vertex_batch_scalar():
    return make_vertex_batch_dict(GEMINI3_SCALAR_CONTENT, GEMINI3_SCALAR_TOKENS)


@pytest.fixture
def vertex_batch_array():
    return make_vertex_batch_dict(ARRAY_CONTENT, ARRAY_TOKENS)


@pytest.fixture
def nested_classification_response():
    return make_openai_response(
        NESTED_CLASSIFICATION_CONTENT,
        NESTED_CLASSIFICATION_TOKENS,
    )


@pytest.fixture
def nested_vertex_batch_response():
    return make_vertex_batch_dict(
        NESTED_CLASSIFICATION_CONTENT,
        NESTED_CLASSIFICATION_TOKENS,
    )
