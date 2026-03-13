# llm-structured-confidence

Extract per-field confidence scores from LLM structured JSON outputs using token-level log-probabilities.

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python->=3.10-3776ab?style=flat-square)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/llm-structured-confidence?style=flat-square&cacheSeconds=60)](https://pypi.org/project/llm-structured-confidence/)

[The Problem](#the-problem) • [Installation](#installation) • [Quick Start](#quick-start) • [Features](#features) • [API Reference](#api-reference) • [Supported Providers](#supported-providers) • [Usage Guide](docs/USAGE.md)

---

Designed for [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — available in [OpenAI](https://platform.openai.com/docs/guides/structured-outputs), [Gemini](https://ai.google.dev/gemini-api/docs/structured-output), and other providers. Works with any JSON schema, but **ideal for ENUM-based classification** where the model picks from a fixed set of values.

We recommend [litellm](https://docs.litellm.ai/docs/completion/json_mode#pass-in-json_schema) as a unified interface for calling any provider with structured output and logprobs.

For a compact end-to-end guide to the full public API, see [docs/USAGE.md](docs/USAGE.md).

## The Problem

When an LLM returns structured JSON with logprobs, tokens don't align with field values. A token like `":"` can merge a colon, a quote, and part of the value — all with one logprob.

```
Token          Logprob     What it contains
──────────────────────────────────────────────
'{"'           -0.006      { and opening "
'category'      0.000      the key
'":"'          -0.200      closing ", colon, opening "  ← structural, NOT the value
'health'       -0.168      ← actual value content
' and'          0.000      ← actual value content
' wellness'     0.000      ← actual value content
'"}'            0.000      closing " and }
```

Naively summing all overlapping tokens gives 69% instead of the correct 84.5%.

**This library parses the JSON precisely, strips structural tokens, and computes confidence using only the tokens that carry actual value content.**

## Installation

```bash
pip install llm-structured-confidence
```

For DataFrame helpers:

```bash
pip install "llm-structured-confidence[pandas]"
```

Import path:

```python
from llm_structured_confidence import extract_field_logprobs
```

Or from source:

```bash
git clone https://github.com/rodolfonobrega/llm-structured-confidence.git
cd llm-structured-confidence
pip install -e ".[dev]"
```

## Quick Start

```python
import litellm
from llm_structured_confidence import extract_field_logprobs

response = litellm.completion(
    model="gpt-4.1-mini",  # or any provider: "vertex_ai/gemini-2.5-flash", etc.
    messages=[
        {"role": "system", "content": "Classify this text."},
        {"role": "user", "content": "Morning yoga and meditation session"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["sports", "health and wellness", "technology"],
                    }
                },
                "required": ["category"],
                "additionalProperties": False,
            },
        },
    },
    logprobs=True,
    top_logprobs=5,
)

result = extract_field_logprobs(response, field="category")

for value, fl in result.items():
    print(f"{value}: {fl.mean_nonzero_probability:.2%}")  # health and wellness: 84.51%
```

## How to Use It

You do **not** need to read the source code to use this library. There are three main ways to call it:

### 1. Explicit field name

Use this when you already know the JSON key you want to analyze.

```python
result = extract_field_logprobs(response, field="category")
value, fl = next(iter(result.items()))
```

### 2. `response_schema` with Pydantic

Use this when your structured output already comes from a Pydantic model.

```python
result = extract_field_logprobs(response, response_schema=Classification)
```

### 3. `response_schema` with JSON Schema

Use this when your structured output is defined with a raw schema dict.

```python
result = extract_field_logprobs(response, response_schema=schema)
```

`response_schema=` accepts both:
- a Pydantic model class
- a JSON Schema dict

Internally, both are normalized to JSON Schema before field detection and enum resolution.

## Features

### Three confidence metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| `joint_probability` | `exp(sum(logprobs))` | Strictest — literal sequence probability |
| `mean_probability` | `exp(mean(logprobs))` | General-purpose — fair across token counts |
| `mean_nonzero_probability` | `exp(mean(logprobs ≠ 0))` | **ENUM classification** — ignores deterministic tokens |

> [!TIP]
> With ENUMs, only the first token carries real uncertainty — the rest are forced by the constraint. `mean_nonzero_probability` filters those out, giving you the model's true confidence regardless of category name length.

### Scalar fields

```python
result = extract_field_logprobs(response, field="category")
fl = result["health and wellness"]
```

### Array fields (batch classification)

```python
# {"categories": ["health and wellness", "sports", "technology"]}
result = extract_field_logprobs(response, field="categories")
for value, fl in result.items():
    print(f"{value}: {fl.joint_probability:.2%}")
```

### Batch API raw dicts

Raw OpenAI / Vertex AI batch payloads are supported directly.

```python
from llm_structured_confidence import extract_field_logprobs

# OpenAI batch output line -> use response["body"]
scores = extract_field_logprobs(batch_row["response"]["body"], field="category")

# Vertex AI batch output line -> response dict itself
scores = extract_field_logprobs(batch_row["response"], field="category")
```

### Pydantic auto-detection

Pass `response_schema=` with either the Pydantic model or the JSON Schema you used for structured output. The library finds enum-valued fields automatically.

Internally, both inputs are normalized to JSON Schema before field detection and enum resolution.

```python
from enum import Enum
from pydantic import BaseModel

class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"

class Classification(BaseModel):
    category: CategoryEnum

result = extract_field_logprobs(response, response_schema=Classification)
```

### JSON Schema auto-detection

```python
schema = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["sports", "health and wellness", "technology"],
        }
    },
    "required": ["category"],
    "additionalProperties": False,
}

result = extract_field_logprobs(response, response_schema=schema)
```

### google-genai native support

Pass a `google.genai.GenerateContentResponse` directly — converted internally using the same logic as litellm's Vertex AI adapter.

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="my-project", location="global")
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[...],
    config=types.GenerateContentConfig(response_logprobs=True, logprobs=5),
)

result = extract_field_logprobs(response, field="category")  # same interface
```

### Pandas integration

For batch output files loaded into a DataFrame, use `add_confidence_columns`.

```python
import pandas as pd
from llm_structured_confidence import add_confidence_columns

# Vertex AI batch output
df = pd.read_json("vertex_batch_output.jsonl", lines=True)
df = add_confidence_columns(df, response_column="response", field="category")

# OpenAI batch output
df = pd.read_json("openai_batch_output.jsonl", lines=True)
df["body"] = df["response"].apply(lambda r: r["body"])
df = add_confidence_columns(df, response_column="body", field="category")
```

When `response_schema=` is provided, `add_confidence_columns()` also adds `{prefix}_top_alt_resolved` with the full enum/literal value for the best alternative when the first token uniquely identifies it.

The same resolution is available in `extract_confidence(...)` via the `top_alternative_resolved` key.

### Token inspection

```python
for token in fl.tokens:
    print(f"  {token.token!r:20s}  logprob={token.logprob:.4f}  prob={token.probability:.2%}")
# 'health'             logprob=-0.1683  prob=84.51%
# ' and'               logprob= 0.0000  prob=100.00%
# ' wellness'          logprob= 0.0000  prob=100.00%
```

Structural tokens (`":"`, `"}"`, `{"`) are **never** included.

### Top alternatives

```python
for alt in fl.top_logprobs:
    print(f"  {alt.token!r:20s}  prob={alt.probability:.2%}")
# 'health'             prob=84.51%
# 'tech'               prob=15.47%    ← "technology"
# 'sport'              prob=0.01%     ← "sports"
```

If you also pass `response_schema=...`, each alternative preserves the raw token and exposes the resolved full value when the token is enough to identify a unique enum/literal choice.

```python
result = extract_field_logprobs(response, response_schema=Classification)
fl = result["health and wellness"]

for alt in fl.top_logprobs:
    print(alt.token, "->", alt.resolved_value)
# health -> health and wellness
# tech -> technology
# sport -> sports
```

If a token prefix is ambiguous across allowed values, `resolved_value` stays `None`.

## Understanding the Metrics

<details>
<summary><strong>Why log-probabilities? (numerical stability)</strong></summary>

Multiplying many small probabilities causes underflow. Logarithms convert multiplication to addition:

```
log(A × B) = log(A) + log(B)
```

So instead of `P("health") × P(" and") × P(" wellness") = 0.845`, we compute `sum(logprobs) = -0.168` and convert back: `exp(-0.168) = 0.845`.

</details>

<details>
<summary><strong>Why mean_nonzero matters for ENUMs</strong></summary>

With the enum `["health and wellness", "sports", "technology"]`, once the model generates `"health"`, the remaining tokens are forced (logprob = 0). The regular mean gets diluted by those zeros:

```
"health and wellness" (3 tokens): mean = (-0.168 + 0 + 0) / 3 → 94.6%  ← inflated
"technology" (2 tokens):          mean = (-0.088 + 0) / 2     → 95.7%  ← inflated differently
```

Longer names get more dilution. `mean_nonzero` fixes this by averaging only tokens where the model had a choice:

```
"health and wellness": mean_nonzero = -0.168 / 1 → 84.5%  ← real confidence
"technology":          mean_nonzero = -0.088 / 1 → 91.6%  ← real confidence
```

</details>

## API Reference

### `extract_field_logprobs(response, *, field=None, response_schema=None)`

| Parameter  | Type | Description |
|------------|------|-------------|
| `response` | `Any` | `litellm.ModelResponse`, `openai.ChatCompletion`, or `google.genai.GenerateContentResponse` with logprobs |
| `field` | `str \| None` | JSON field name (e.g. `"category"`). Takes precedence over `response_schema`. |
| `response_schema` | `type \| dict[str, Any] \| None` | Pydantic model or JSON Schema — auto-detects enum-valued fields |

**Returns** `dict[str, FieldLogprob]` — maps each value (as string) to its metrics.

**Precedence**: `field` > `response_schema` > all top-level fields.

### `extract_confidence(response, *, field=None, response_schema=None)`

Returns a flat `dict[str, Any]` for the first extracted value. This is the easiest option when you want to store one result per response in a database, log, or DataFrame-ready record.

Main keys:
- `value`
- `joint_probability`
- `mean_probability`
- `mean_nonzero_probability`
- `top_alternative`
- `top_alternative_resolved`
- `top_alternative_probability`
- `top_logprobs`
- `error`

### `add_confidence_columns(df, *, response_column="response", field=None, response_schema=None, prefix="confidence")`

Adds confidence columns to a pandas DataFrame of batch results.

Main added columns:
- `{prefix}_value`
- `{prefix}_prob`
- `{prefix}_joint_prob`
- `{prefix}_top_alt`
- `{prefix}_top_alt_resolved`
- `{prefix}_top_alt_prob`
- `{prefix}_error`

### `FieldLogprob`

| Attribute | Type | Description |
|-----------|------|-------------|
| `value` | `Any` | The parsed value |
| `tokens` | `list[TokenInfo]` | Tokens included in the calculation |
| `joint_logprob` | `float` | Sum of all token logprobs |
| `joint_probability` | `float` | `exp(joint_logprob)` |
| `mean_logprob` | `float` | Mean of all token logprobs |
| `mean_probability` | `float` | `exp(mean_logprob)` |
| `mean_nonzero_logprob` | `float \| None` | Mean of logprobs where logprob ≠ 0 (or 0.0 if all zero) |
| `mean_nonzero_probability` | `float \| None` | `exp(mean_nonzero_logprob)` (or 1.0 if all zero) |
| `top_logprobs` | `list[TopAlternative]` | Alternatives from the first uncertain token |

### `TokenInfo`

| Attribute | Type | Description |
|-----------|------|-------------|
| `token` | `str` | Token text |
| `logprob` | `float` | Log-probability |
| `probability` | `float` | `exp(logprob)` — property |
| `char_start` / `char_end` | `int` | Position in the JSON string |

### `TopAlternative`

| Attribute | Type | Description |
|-----------|------|-------------|
| `token` | `str` | Alternative token text |
| `logprob` | `float` | Its log-probability |
| `resolved_value` | `Any \| None` | Full enum/literal value when `response_schema=` provides choices and the token prefix matches exactly one |
| `probability` | `float` | `exp(logprob)` — property |

## How It Works

1. **Normalize** — detect litellm/OpenAI or google-genai format, convert to common `(content, tokens)` representation
2. **Parse** — feed JSON to a [Lark](https://github.com/lark-parser/lark) LALR parser with position tracking
3. **Strip quotes** — shrink string ranges by 1 on each side to exclude `"`
4. **Overlap** — include only tokens whose character span overlaps the value range
5. **Metrics** — compute the three logprob metrics from included tokens

```
JSON:   {"category":"health and wellness"}
                    ^^^^^^^^^^^^^^^^^^^^
                    value range [13, 31) ← quotes stripped

Tokens:  '{"'  'category'  '":"'  'health'  ' and'  ' wellness'  '"}'
          ↑                  ↑                                      ↑
       excluded           excluded                               excluded

Included: 'health' + ' and' + ' wellness'  ✓
```

## Supported Providers

| Provider | Response type | Logprobs | Structured output docs |
|----------|--------------|----------|----------------------|
| **litellm** (recommended) | `ModelResponse` | `logprobs=True, top_logprobs=5` | [JSON mode](https://docs.litellm.ai/docs/completion/json_mode) |
| **OpenAI** | `ChatCompletion` | `logprobs=True, top_logprobs=5` | [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) |
| **OpenAI batch** | raw `dict` body with `choices` | from batch output file | [Batch API](https://platform.openai.com/docs/guides/batch) |
| **google-genai** | `GenerateContentResponse` | `response_logprobs=True, logprobs=5` | [Structured output](https://ai.google.dev/gemini-api/docs/structured-output) |
| **Vertex AI batch** | raw `dict` response with `candidates` | from batch output file | [Batch predictions](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage) |

> [!TIP]
> For classification tasks, consider disabling thinking/reasoning to get cleaner logprobs (no reasoning tokens mixed in). This applies to any model that supports it, but depends on your use case — reasoning may improve accuracy for complex classifications.
> - litellm: `reasoning_effort="none"`
> - google-genai: `thinking_config=types.ThinkingConfig(thinking_budget=0)`

## Lower-level API

For custom workflows, internal modules are available:

```python
from llm_structured_confidence._parser import parse_json_spans, build_token_char_ranges, tokens_for_span
from llm_structured_confidence._converter import normalize_response

parsed = parse_json_spans('{"category": "sports", "count": 2}')
# parsed["category"] → _ValueSpan(value="sports", char_start=15, char_end=21)

norm = normalize_response(response)
# norm.content → JSON string, norm.tokens → list of NormalizedToken
```

> [!NOTE]
> These are underscore-prefixed internal APIs that may change in minor releases. Prefer `extract_field_logprobs` when possible.

## Using with AI Agents

The [`AGENTS.md`](AGENTS.md) file contains a compact API reference designed for LLM-based coding agents (Cursor, Copilot, etc.).

## Running Tests

```bash
# Unit tests (55 tests, no API calls)
pytest llm_structured_confidence/tests/test_unit.py -v

# E2E tests (6 tests, calls Vertex AI)
pytest llm_structured_confidence/tests/test_e2e.py -v -s

# All tests
pytest -v
```

## Publishing

Release automation for PyPI via GitHub Actions is documented in [RELEASING.md](RELEASING.md).

For the common release flow, you can also use:

```bash
./scripts/release.sh X.Y.Z
```

The script refuses to run outside `main`, with a dirty Git tree, without `gh` auth, or if the release tag already exists.
