# llm-structured-confidence

Extract path-aware confidence scores from LLM structured JSON outputs using token-level log-probabilities.

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python->=3.10-3776ab?style=flat-square)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/llm-structured-confidence?style=flat-square&cacheSeconds=60)](https://pypi.org/project/llm-structured-confidence/)

[Installation](#installation) • [Supported JSON Formats](#supported-json-formats) • [Quick Start](#quick-start) • [Path Syntax](#path-syntax) • [Documentation](#documentation) • [Supported Providers](#supported-providers)

---

Designed for structured JSON outputs from OpenAI, Gemini, and compatible providers. The library aligns token logprobs to exact JSON value spans, strips structural tokens, and computes confidence only from the tokens that belong to the value itself.

## Installation

```bash
pip install llm-structured-confidence
```

For DataFrame helpers:

```bash
pip install "llm-structured-confidence[pandas]"
```

## Supported JSON Formats

The library works with any structured JSON output. Here are the most common patterns and how to extract confidence from each.

### Single scalar field

```json
{"category": "health and wellness"}
```

```python
entries = extract_logprobs(response, field_path="category")
# entries[0].path  -> "category"
# entries[0].value -> "health and wellness"
```

### Nested scalar field

```json
{"classification": {"name": "Positive", "score": 0.95}}
```

```python
entries = extract_logprobs(response, field_path="classification.name")
# entries[0].path  -> "classification.name"
# entries[0].value -> "Positive"
```

### Array of strings

```json
{"categories": ["health and wellness", "sports", "technology"]}
```

```python
entries = extract_logprobs(response, field_path="categories[]")
# entries[0].path -> "categories[0]", entries[0].value -> "health and wellness"
# entries[1].path -> "categories[1]", entries[1].value -> "sports"
# entries[2].path -> "categories[2]", entries[2].value -> "technology"
```

### Array of objects (pick one field)

```json
{"results": [{"id": 1, "category": "deposits"}, {"id": 2, "category": "shopping"}]}
```

```python
entries = extract_logprobs(response, field_path="results[].category")
# entries[0].path -> "results[0].category", entries[0].value -> "deposits"
# entries[1].path -> "results[1].category", entries[1].value -> "shopping"
```

### Top-level array of objects (Vertex AI only)

Vertex AI supports `"type": "ARRAY"` at the schema root, producing a bare JSON array.
OpenAI requires a top-level object, so these formats are Vertex-only.

```json
[{"id": 1, "category": "deposits"}, {"id": 2, "category": "shopping"}]
```

```python
entries = extract_logprobs(response, field_path="[].category")
# entries[0].path -> "[0].category", entries[0].value -> "deposits"
# entries[1].path -> "[1].category", entries[1].value -> "shopping"
```

### Top-level array of strings (Vertex AI only)

The most compact multi-classification format — a flat list of category strings with no wrapper object.
The `enum` constraint in the schema is enforced by Vertex AI.

```json
["bars and restaurants", "transportation", "digital services"]
```

```python
entries = extract_logprobs(response, field_path="[]")
# entries[0].path -> "[0]", entries[0].value -> "bars and restaurants"
# entries[1].path -> "[1]", entries[1].value -> "transportation"
# entries[2].path -> "[2]", entries[2].value -> "digital services"
```

### Deeply nested arrays

```json
{"groups": [{"items": [{"label": "A"}, {"label": "B"}]}]}
```

```python
entries = extract_logprobs(response, field_path="groups[].items[].label")
# entries[0].path -> "groups[0].items[0].label", entries[0].value -> "A"
# entries[1].path -> "groups[0].items[1].label", entries[1].value -> "B"
```

### Provider compatibility

| JSON Format | OpenAI | Vertex AI | `field_path` |
|---|:---:|:---:|---|
| `{"category": "..."}` | Yes | Yes | `category` |
| `{"classification": {"name": "..."}}` | Yes | Yes | `classification.name` |
| `{"categories": ["a", "b"]}` | Yes | Yes | `categories[]` |
| `{"results": [{"id": 1, "category": "..."}]}` | Yes | Yes | `results[].category` |
| `["cat_a", "cat_b", "cat_c"]` | No | Yes | `[]` |
| `[{"id": 1, "category": "..."}]` | No | Yes | `[].category` |
| `{"groups": [{"items": [{"label": "..."}]}]}` | Yes | Yes | `groups[].items[].label` |

> **Note**: OpenAI's `json_schema` response format requires `"type": "object"` at the root. To get array-like output on OpenAI, wrap it in an object (e.g. `{"results": [...]}`). Top-level arrays (both string and object variants) are exclusive to Vertex AI.

## Quick Start

```python
import litellm
from llm_structured_confidence import extract_logprobs

response = litellm.completion(
    model="gpt-4.1-mini",
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

entries = extract_logprobs(response, field_path="category")
entry = entries[0]

print(entry.path)                                    # category
print(entry.value)                                   # health and wellness
print(entry.field_logprob.mean_nonzero_probability)  # 0.8451
```

### Schema auto-detection

If you pass `response_schema=`, the library auto-detects enum-valued paths recursively.

```python
entries = extract_logprobs(response, response_schema=ClassificationModel)
```

That also enables `TopAlternative.resolved_value`, so token prefixes like `"Pos"` can resolve back to `"Positive"` when the match is unique.

## Path Syntax

| Pattern | Meaning | Example output |
|---|---|---|
| `category` | Scalar field at root | `{"category": "..."}` |
| `classification.name` | Nested scalar field | `{"classification": {"name": "..."}}` |
| `categories[]` | Each element in an array | `{"categories": ["a", "b"]}` |
| `results[].category` | Field inside each array element | `{"results": [{"category": "..."}]}` |
| `[]` | Each element of a top-level string array | `["cat_a", "cat_b"]` |
| `[].category` | Field inside each element of a top-level array | `[{"category": "..."}]` |
| `groups[].items[].label` | Deeply nested: array inside array | `{"groups": [{"items": [{"label": "..."}]}]}` |

## Flat/DataFrame Helpers

```python
from llm_structured_confidence import extract_confidence, add_confidence_columns

metrics = extract_confidence(response, field_path="category")

df = add_confidence_columns(
    df,
    response_column="response",
    field_path="classifications[].name",
)
```

Helpers always return the first matching value and also expose its resolved `path`.

## Documentation

- [docs/USAGE.md](docs/USAGE.md): full public API and examples
- [examples/examples.ipynb](examples/examples.ipynb): notebook walkthrough
- [AGENTS.md](AGENTS.md): compact reference for coding agents

## Supported Providers

- `litellm.ModelResponse`
- `openai.ChatCompletion`
- `google.genai.GenerateContentResponse`
- raw OpenAI batch response dict with `choices`
- raw Vertex AI batch response dict with `candidates`

## Lower-level API

Internal modules remain available for custom workflows:

```python
from llm_structured_confidence._parser import parse_json_spans, build_token_char_ranges, tokens_for_span
from llm_structured_confidence._converter import normalize_response
```

These are underscore-prefixed internals and may change in minor releases.
