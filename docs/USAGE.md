# Usage Guide

Quick reference for the public API of `llm-structured-confidence`.

This guide covers:
- `extract_field_logprobs(...)`
- `extract_path_logprobs(...)`
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
    extract_field_logprobs,
    extract_path_logprobs,
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

## Main Method: `extract_field_logprobs(...)`

```python
def extract_field_logprobs(
    response,
    *,
    field: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, FieldLogprob]
```

Behavior:

- `field=` explicitly selects the JSON key to analyze
- `response_schema=` auto-detects enum-valued fields from either:
  - a Pydantic model
  - a JSON Schema dict
- precedence is: `field` > `response_schema` > all top-level fields

### Example: explicit field

```python
result = extract_field_logprobs(response, field="category")

value, fl = next(iter(result.items()))
print(value)
print(fl.mean_nonzero_probability)
```

### Example: Pydantic schema

```python
from enum import Enum
from pydantic import BaseModel


class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"
    technology = "technology"


class Classification(BaseModel):
    category: CategoryEnum


result = extract_field_logprobs(response, response_schema=Classification)
```

### Example: JSON Schema

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

### Example: array field

```python
# {"categories": ["health and wellness", "sports", "technology"]}
result = extract_field_logprobs(response, field="categories")

for value, fl in result.items():
    print(value, fl.mean_nonzero_probability)
```

### Example: simple array of strings

```python
# {"classifications": ["Positive", "Negative", "Neutral"]}
result = extract_field_logprobs(response, field="classifications")

for value, fl in result.items():
    print(value, fl.mean_nonzero_probability)
```

This works because the array elements are atomic values. If you need to preserve
positions, use:

```python
results = extract_path_logprobs(response, field_path="classifications[]")

print(results[0].path)   # classifications[0]
print(results[0].value)  # Positive
```

## Nested Paths: `extract_path_logprobs(...)`

Use this when the values you want are inside nested objects or arrays of objects.

```python
def extract_path_logprobs(
    response,
    *,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
) -> list[PathFieldLogprob]
```

Behavior:

- `field_path=` selects an atomic JSON value using:
  - `.` for object traversal
  - `[]` for arrays
- examples:
  - `classification.name`
  - `classifications[].name`
  - `groups[].items[].label`
- if `field_path` is omitted and `response_schema=` is provided, enum-valued paths are auto-detected recursively
- return order matches the JSON order, so repeated values do not collide

### Example: explicit nested path

```python
# {
#   "classifications": [
#     {"id": 0, "name": "Positive", "color": "#00FF00"},
#     {"id": 1, "name": "Negative", "color": "#FF0000"},
#   ]
# }
results = extract_path_logprobs(response, field_path="classifications[].name")

for entry in results:
    print(entry.path, entry.value, entry.field_logprob.mean_nonzero_probability)
```

### Example: nested auto-detection with Pydantic

```python
from enum import Enum
from pydantic import BaseModel


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


results = extract_path_logprobs(response, response_schema=Output)
print(results[0].path)   # classifications[0].name
print(results[0].value)  # Positive
```

## Return Type: `PathFieldLogprob`

Each path-aware result contains:

- `path`: resolved path, e.g. `classifications[0].name`
- `value`: parsed Python value
- `field_logprob`: the usual `FieldLogprob`

Example:

```python
entry = results[0]

print(entry.path)
print(entry.value)
print(entry.field_logprob.top_logprobs[0].resolved_value)
```

## Return Type: `FieldLogprob`

Each extracted value maps to one `FieldLogprob`.

Attributes:

- `value`: parsed Python value
- `tokens`: list of `TokenInfo`
- `joint_logprob`
- `joint_probability`
- `mean_logprob`
- `mean_probability`
- `mean_nonzero_logprob`
- `mean_nonzero_probability`
- `top_logprobs`: list of `TopAlternative`

Example:

```python
fl = result["health and wellness"]

print(fl.value)
print(fl.joint_probability)
print(fl.mean_probability)
print(fl.mean_nonzero_probability)
```

## Inspecting Tokens: `TokenInfo`

Each included token has:

- `token`
- `logprob`
- `probability`
- `char_start`
- `char_end`

Example:

```python
for token in fl.tokens:
    print(token.token, token.logprob, token.probability)
```

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
result = extract_field_logprobs(response, response_schema=Classification)
fl = result["health and wellness"]

for alt in fl.top_logprobs:
    print(alt.token, alt.probability, alt.resolved_value)
```

Example output:

```python
health 0.8451 health and wellness
tech 0.1547 technology
sport 0.0001 sports
```

If the prefix is ambiguous, `resolved_value` stays `None`.

## Flat Helper: `extract_confidence(...)`

```python
def extract_confidence(
    response,
    *,
    field: str | None = None,
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

- `field_path` takes precedence over `field`
- `top_alternative` is the best alternative after the chosen token/value
- on failure, the helper returns `error` instead of raising

## DataFrame Helper: `add_confidence_columns(...)`

```python
def add_confidence_columns(
    df,
    *,
    response_column: str = "response",
    field: str | None = None,
    field_path: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
    prefix: str = "confidence",
)
```

Use this for batch API outputs loaded into pandas.

If `field_path=` is used, the helper takes the first matching value and also adds `{prefix}_path`.

Example: Vertex AI batch output

```python
import pandas as pd

df = pd.read_json("vertex_batch_output.jsonl", lines=True)
df = add_confidence_columns(
    df,
    response_column="response",
    field="category",
    response_schema=Classification,
)
```

Example: OpenAI batch output

```python
df = pd.read_json("openai_batch_output.jsonl", lines=True)
df["body"] = df["response"].apply(lambda r: r["body"])
df = add_confidence_columns(
    df,
    response_column="body",
    field="category",
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
result = extract_field_logprobs(batch_row["response"]["body"], field="category")
```

### Vertex AI batch

```python
result = extract_field_logprobs(batch_row["response"], field="category")
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

Use them only if you need custom parsing/token workflows.

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
- pass `field=` when you know the target key
- pass `response_schema=` when you want enum auto-detection and `resolved_value`
