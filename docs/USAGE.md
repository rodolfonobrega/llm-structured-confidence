# Usage Guide

Quick reference for the public API of `llm-structured-confidence`.

This guide covers:
- `extract_logprobs(...)`
- `extract_confidence(...)`
- `add_confidence_columns(...)`
- `FieldLogprob`
- `PathFieldLogprob`
- `TokenInfo`
- `TopAlternative`

## Install

```bash
pip install llm-structured-confidence
```

For pandas helpers:

```bash
pip install "llm-structured-confidence[pandas]"
```

## Import

```python
from llm_structured_confidence import (
    extract_logprobs,
    extract_confidence,
    add_confidence_columns,
    FieldLogprob,
    PathFieldLogprob,
    TokenInfo,
    TopAlternative,
)
```

## Supported Response Inputs

You can pass any of these as `response`:

- `litellm.ModelResponse`
- `openai.ChatCompletion`
- `google.genai.GenerateContentResponse`
- raw OpenAI batch response body dict with `"choices"`
- raw Vertex AI batch response dict with `"candidates"`

The LLM call must include token logprobs.

## Main Method: `extract_logprobs(...)`

```python
def extract_logprobs(
    response,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> list[PathFieldLogprob]
```

Behavior:

- `field_path=` explicitly selects an atomic JSON value
- `response_schema=` auto-detects enum-valued paths recursively
- if neither is provided, all atomic values in the JSON are returned
- return order follows the JSON order

## Path Syntax

Use:

- `.` for object traversal
- `[]` for arrays

Examples:

- `category`
- `categories[]`
- `classification.name`
- `classifications[].name`
- `groups[].items[].label`

## Examples

### Scalar field

```python
entries = extract_logprobs(response, field_path="category")
entry = entries[0]

print(entry.path)                                    # category
print(entry.value)                                   # health and wellness
print(entry.field_logprob.mean_nonzero_probability)  # 0.845
```

### Simple array of strings

```python
# {"classifications": ["Positive", "Negative", "Neutral"]}
entries = extract_logprobs(response, field_path="classifications[]")

for entry in entries:
    print(entry.path, entry.value)
```

### Nested array of objects

```python
# {
#   "classifications": [
#     {"id": 0, "name": "Positive", "color": "#00FF00"},
#     {"id": 1, "name": "Negative", "color": "#FF0000"},
#   ]
# }
entries = extract_logprobs(response, field_path="classifications[].name")

for entry in entries:
    print(entry.path, entry.value, entry.field_logprob.mean_nonzero_probability)
```

### Pydantic schema auto-detection

```python
from enum import Enum
from pydantic import BaseModel


class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"
    technology = "technology"


class Classification(BaseModel):
    category: CategoryEnum


entries = extract_logprobs(response, response_schema=Classification)
print(entries[0].path)   # category
print(entries[0].value)  # health and wellness
```

### Nested schema auto-detection

```python
class Label(str, Enum):
    positive = "Positive"
    negative = "Negative"
    neutral = "Neutral"


class ClassificationItem(BaseModel):
    id: int
    name: Label
    color: str


class Output(BaseModel):
    classifications: list[ClassificationItem]


entries = extract_logprobs(response, response_schema=Output)
print(entries[0].path)   # classifications[0].name
print(entries[0].value)  # Positive
```

## Return Type: `PathFieldLogprob`

Each result contains:

- `path`: resolved path, e.g. `classifications[0].name`
- `value`: parsed Python value
- `field_logprob`: the usual `FieldLogprob`

Example:

```python
entry = entries[0]
fl = entry.field_logprob

print(entry.path)
print(entry.value)
print(fl.top_logprobs[0].resolved_value)
```

## Return Type: `FieldLogprob`

Attributes:

- `value`
- `tokens`
- `joint_logprob`
- `joint_probability`
- `mean_logprob`
- `mean_probability`
- `mean_nonzero_logprob`
- `mean_nonzero_probability`
- `top_logprobs`

## Inspecting Tokens: `TokenInfo`

Each included token has:

- `token`
- `logprob`
- `probability`
- `char_start`
- `char_end`

## Top Alternatives: `TopAlternative`

Each top alternative has:

- `token`
- `logprob`
- `probability`
- `resolved_value`

`resolved_value` is filled only when:

- `response_schema=` is provided
- the token prefix matches exactly one enum/literal option

Example:

```python
entries = extract_logprobs(response, response_schema=Classification)
fl = entries[0].field_logprob

for alt in fl.top_logprobs:
    print(alt.token, alt.probability, alt.resolved_value)
```

If the prefix is ambiguous, `resolved_value` stays `None`.

## Flat Helper: `extract_confidence(...)`

```python
def extract_confidence(
    response,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, Any]
```

Use this when you want a flat dict instead of nested dataclasses.

Example:

```python
metrics = extract_confidence(
    response,
    field_path="classifications[].name",
    response_schema=Output,
)

print(metrics)
```

Returned keys:

- `path`
- `value`
- `joint_probability`
- `mean_probability`
- `mean_nonzero_probability`
- `top_alternative`
- `top_alternative_resolved`
- `top_alternative_probability`
- `top_logprobs`
- `error`

Notes:

- the helper always returns the first matching value
- on failure, it returns `error` instead of raising

## DataFrame Helper: `add_confidence_columns(...)`

```python
def add_confidence_columns(
    df,
    *,
    response_column: str = "response",
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
    prefix: str = "confidence",
)
```

Use this for batch API outputs loaded into pandas.

Example:

```python
import pandas as pd

df = pd.read_json("vertex_batch_output.jsonl", lines=True)
df = add_confidence_columns(
    df,
    response_column="response",
    field_path="classifications[].name",
)
```

Added columns:

- `{prefix}_value`
- `{prefix}_path`
- `{prefix}_prob`
- `{prefix}_joint_prob`
- `{prefix}_top_alt`
- `{prefix}_top_alt_resolved`
- `{prefix}_top_alt_prob`
- `{prefix}_error`

## Which Probability Should You Use?

- `mean_nonzero_probability`: best for enum classification
- `mean_probability`: fair comparison across token counts
- `joint_probability`: strictest sequence probability

Why `mean_nonzero_probability` matters:

- with enums, usually only the first discriminating token is uncertain
- the remaining tokens are often deterministic (`logprob = 0`)
- averaging only non-zero logprobs avoids inflated confidence for longer labels

## Batch Dict Examples

### OpenAI batch

```python
entries = extract_logprobs(batch_row["response"]["body"], field_path="category")
```

### Vertex AI batch

```python
entries = extract_logprobs(batch_row["response"], field_path="category")
```

## Lower-level Internal Modules

These are available, but are internal and may change in minor releases:

```python
from llm_structured_confidence._parser import (
    parse_json_spans,
    build_token_char_ranges,
    tokens_for_span,
    get_overlapping_indices,
)
from llm_structured_confidence._converter import normalize_response
```

Example: nested arrays of objects are preserved in the parsed tree, with `_ValueSpan`
only at atomic leaves.

```python
parsed = parse_json_spans(
    '{"classifications":[{"id":0,"name":"Positive","color":"#00FF00"}]}'
)

item = parsed["classifications"][0]
print(type(item).__name__)          # dict
print(item["id"].value)             # 0
print(item["name"].value)           # "Positive"
print(item["color"].value)          # "#00FF00"
print(item["name"].char_start)      # character offset of Positive in the JSON string
```

## Minimal Checklist

To get reliable results:

- enable logprobs in the LLM call
- use structured output / JSON schema
- pass `field_path=` when you know the target path
- pass `response_schema=` when you want enum auto-detection and `resolved_value`
