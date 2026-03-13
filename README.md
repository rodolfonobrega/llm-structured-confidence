# llm-structured-confidence

Extract per-field confidence scores from LLM structured JSON outputs using token-level log-probabilities.

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python->=3.10-3776ab?style=flat-square)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/llm-structured-confidence?style=flat-square&cacheSeconds=60)](https://pypi.org/project/llm-structured-confidence/)

[The Problem](#the-problem) • [Installation](#installation) • [Quick Start](#quick-start) • [Highlights](#highlights) • [Documentation](#documentation) • [Supported Providers](#supported-providers)

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
from llm_structured_confidence import extract_field_logprobs, extract_path_logprobs
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

## Highlights

### Three confidence metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| `joint_probability` | `exp(sum(logprobs))` | Strictest — literal sequence probability |
| `mean_probability` | `exp(mean(logprobs))` | General-purpose — fair across token counts |
| `mean_nonzero_probability` | `exp(mean(logprobs ≠ 0))` | **ENUM classification** — ignores deterministic tokens |

> [!TIP]
> With ENUMs, only the first token carries real uncertainty — the rest are forced by the constraint. `mean_nonzero_probability` filters those out, giving you the model's true confidence regardless of category name length.

### Works with explicit fields, Pydantic, and JSON Schema

```python
result = extract_field_logprobs(response, field="category")
result = extract_field_logprobs(response, response_schema=Classification)
result = extract_field_logprobs(response, response_schema=schema)
```

### Arrays and batch classification

```python
# {"categories": ["health and wellness", "sports", "technology"]}
result = extract_field_logprobs(response, field="categories")
```

Simple arrays of atomic values are also supported directly:

```python
# {"classifications": ["Positive", "Negative", "Neutral"]}
result = extract_field_logprobs(response, field="classifications")

for value, fl in result.items():
    print(value, fl.mean_nonzero_probability)
```

If you need item positions, use the path-aware API:

```python
results = extract_path_logprobs(response, field_path="classifications[]")
print(results[0].path)   # classifications[0]
print(results[0].value)  # Positive
```

### Nested arrays of objects

Use the path-aware API when values live inside arrays or nested objects.

```python
results = extract_path_logprobs(response, field_path="classifications[].name")

for entry in results:
    print(entry.path, entry.value, entry.field_logprob.mean_nonzero_probability)
    # classifications[0].name Positive 0.96
```

### Raw batch payloads are supported

Raw OpenAI / Vertex AI batch payloads are supported directly.

```python
# OpenAI batch output line -> use response["body"]
scores = extract_field_logprobs(batch_row["response"]["body"], field="category")

# Vertex AI batch output line -> response dict itself
scores = extract_field_logprobs(batch_row["response"], field="category")
```

### Pandas helpers

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

### Resolved top alternatives

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

## Documentation

Detailed docs live here:

- [docs/USAGE.md](docs/USAGE.md): full public API, all methods, return types, and examples
- [examples/examples.ipynb](examples/examples.ipynb): notebook walkthrough
- [AGENTS.md](AGENTS.md): compact reference for coding agents

The public API covered in the guide:

- `extract_field_logprobs(...)`
- `extract_path_logprobs(...)`
- `extract_confidence(...)`
- `add_confidence_columns(...)`
- `FieldLogprob`
- `PathFieldLogprob`
- `TokenInfo`
- `TopAlternative`

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

## Running Tests

```bash
# Unit tests
pytest llm_structured_confidence/tests/test_unit.py -v

# E2E tests
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
