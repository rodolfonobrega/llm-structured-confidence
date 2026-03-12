# llm-structured-confidence

Extract per-field confidence scores from LLM structured JSON outputs using token-level log-probabilities.

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python->=3.10-3776ab?style=flat-square)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/llm-structured-confidence?style=flat-square)](https://pypi.org/project/llm-structured-confidence/)

[The Problem](#the-problem) • [Installation](#installation) • [Quick Start](#quick-start) • [Features](#features) • [API Reference](#api-reference) • [Supported Providers](#supported-providers)

---

Designed for [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — available in [OpenAI](https://platform.openai.com/docs/guides/structured-outputs), [Gemini](https://ai.google.dev/gemini-api/docs/structured-output), and other providers. Works with any JSON schema, but **ideal for ENUM-based classification** where the model picks from a fixed set of values.

We recommend [litellm](https://docs.litellm.ai/docs/completion/json_mode#pass-in-json_schema) as a unified interface for calling any provider with structured output and logprobs.

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

### Pydantic auto-detection

Pass the Pydantic model you used for structured output — the library finds `Enum`, `list[Enum]`, and `Literal` fields automatically.

```python
from enum import Enum
from pydantic import BaseModel

class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"

class Classification(BaseModel):
    category: CategoryEnum

result = extract_field_logprobs(response, model=Classification)
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

### `extract_field_logprobs(response, *, field=None, model=None)`

| Parameter  | Type | Description |
|------------|------|-------------|
| `response` | `Any` | `litellm.ModelResponse`, `openai.ChatCompletion`, or `google.genai.GenerateContentResponse` with logprobs |
| `field` | `str \| None` | JSON field name (e.g. `"category"`). Takes precedence over `model`. |
| `model` | `type \| None` | Pydantic model — auto-detects `Enum`/`list[Enum]`/`Literal` fields |

**Returns** `dict[str, FieldLogprob]` — maps each value (as string) to its metrics.

**Precedence**: `field` > `model` > all top-level fields.

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
| **google-genai** | `GenerateContentResponse` | `response_logprobs=True, logprobs=5` | [Structured output](https://ai.google.dev/gemini-api/docs/structured-output) |

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
./scripts/release.sh 0.1.1
```

The script refuses to run outside `main`, with a dirty Git tree, without `gh` auth, or if the release tag already exists.
