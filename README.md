# llm-structured-confidence

Extract path-aware confidence scores from LLM structured JSON outputs using token-level log-probabilities.

[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python->=3.10-3776ab?style=flat-square)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/llm-structured-confidence?style=flat-square&cacheSeconds=60)](https://pypi.org/project/llm-structured-confidence/)

[Installation](#installation) • [Quick Start](#quick-start) • [Path Syntax](#path-syntax) • [Documentation](#documentation) • [Supported Providers](#supported-providers)

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

### Arrays

```python
# {"categories": ["health and wellness", "sports", "technology"]}
entries = extract_logprobs(response, field_path="categories[]")

for entry in entries:
    print(entry.path, entry.value)
```

### Nested arrays of objects

```python
# {"classifications": [{"id": 0, "name": "Positive", "color": "#00FF00"}]}
entries = extract_logprobs(response, field_path="classifications[].name")

for entry in entries:
    print(entry.path, entry.value, entry.field_logprob.mean_nonzero_probability)
```

### Schema auto-detection

If you pass `response_schema=`, the library auto-detects enum-valued paths recursively.

```python
entries = extract_logprobs(response, response_schema=ClassificationModel)
```

That also enables `TopAlternative.resolved_value`, so token prefixes like `"Pos"` can resolve back to `"Positive"` when the match is unique.

## Path Syntax

- `category`
- `categories[]`
- `classification.name`
- `classifications[].name`
- `groups[].items[].label`

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
