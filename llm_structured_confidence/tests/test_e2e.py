"""End-to-end tests that call real LLM APIs.

Requires OPENAI_API_KEY (and optionally OPENAI_API_BASE) in the environment.

Run with:
    pytest llm_structured_confidence/tests/test_e2e.py -v -s
"""

from __future__ import annotations

import os
from enum import Enum

import pytest
from pydantic import BaseModel

import litellm

from llm_structured_confidence import extract_logprobs

MODEL = os.environ.get("LLM_CONFIDENCE_TEST_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = (
    "You are an assistant that classifies text into topics. "
    "Given a short description, classify it into exactly one category. "
    "Return JSON with a single key 'category'."
)


class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"
    technology = "technology"
    entertainment = "entertainment"
    science = "science"


class SingleCategory(BaseModel):
    category: CategoryEnum


class MultiCategory(BaseModel):
    categories: list[CategoryEnum]


CATEGORY_NAMES = [e.value for e in CategoryEnum]

SCALAR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORY_NAMES}
            },
            "required": ["category"],
            "additionalProperties": False,
        },
    },
}

ARRAY_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "multi_topic_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": CATEGORY_NAMES},
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        },
    },
}


def _call(text: str, *, response_format=None, system=None):
    return litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format=response_format or SCALAR_RESPONSE_FORMAT,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
        num_retries=2,
    )


# ═══════════════════════════════════════════════════════════════════════
# Scalar field
# ═══════════════════════════════════════════════════════════════════════

def test_e2e_scalar():
    resp = _call("Morning yoga and meditation session")
    result = extract_logprobs(resp, field_path="category")

    assert len(result) == 1
    entry = result[0]
    assert entry.value in CATEGORY_NAMES
    fl = entry.field_logprob

    assert 0.0 < fl.joint_probability <= 1.0
    assert fl.joint_logprob <= 0.0
    assert fl.mean_probability <= 1.0
    assert len(fl.tokens) >= 1
    assert all(t.token for t in fl.tokens)

    token_texts = [t.token for t in fl.tokens]
    assert '":"' not in token_texts
    assert '"}'  not in token_texts

    if fl.mean_nonzero_logprob is not None:
        assert fl.mean_nonzero_probability <= 1.0

    print(f"\n  [{MODEL}] → {entry.value} "
          f"(joint={fl.joint_probability:.2%}, "
          f"mean_nz={fl.mean_nonzero_probability or 'N/A'})")
    print(f"  tokens: {[t.token for t in fl.tokens]}")
    if fl.top_logprobs:
        print(f"  top_logprobs: {[(a.token, f'{a.probability:.2%}') for a in fl.top_logprobs]}")


# ═══════════════════════════════════════════════════════════════════════
# Response schema auto-detection
# ═══════════════════════════════════════════════════════════════════════

def test_e2e_pydantic_detection():
    resp = _call("Yoga and meditation retreat in the mountains")
    result = extract_logprobs(resp, response_schema=SingleCategory)
    assert len(result) == 1
    entry = result[0]
    assert entry.value in CATEGORY_NAMES
    print(f"\n  [pydantic auto-detect] → {entry.value} "
          f"(joint={entry.field_logprob.joint_probability:.2%})")


# ═══════════════════════════════════════════════════════════════════════
# Array field
# ═══════════════════════════════════════════════════════════════════════

def test_e2e_array():
    prompt = (
        "Classify each item:\n"
        "1. Morning yoga and meditation\n"
        "2. Yoga and meditation retreat\n"
        "3. New smartphone release\n"
        "Return JSON with key 'categories' as an array of categories, one per item, in order."
    )
    resp = _call(
        prompt,
        response_format=ARRAY_RESPONSE_FORMAT,
        system=SYSTEM_PROMPT + " Return a JSON array of categories.",
    )
    result = extract_logprobs(resp, field_path="categories[]")

    assert len(result) >= 2
    for entry in result:
        fl = entry.field_logprob
        assert entry.value in CATEGORY_NAMES
        assert 0.0 < fl.joint_probability <= 1.0
        assert len(fl.tokens) >= 1
        print(f"\n  {entry.value}: joint={fl.joint_probability:.2%}, "
              f"mean_nz={fl.mean_nonzero_probability or 'N/A'}")
