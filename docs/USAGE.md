# Usage Guide

Quick reference for the public API of `llm-structured-confidence`.

This guide covers:
- `extract_field_logprobs(...)`
- `extract_confidence(...)`
- `add_confidence_columns(...)`
- `FieldLogprob`
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
    extract_confidence,
    add_confidence_columns,
    FieldLogprob,
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
    response_schema: type | dict[str, Any] | None = None,
) -> dict[str, Any]
```

Use this when you want a flat dict instead of nested dataclasses.

Example:

```python
metrics = extract_confidence(
    response,
    field="category",
    response_schema=Classification,
)

print(metrics)
```

Returned keys:

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

- `top_alternative` is the best alternative after the chosen token/value
- on failure, the helper returns `error` instead of raising

## DataFrame Helper: `add_confidence_columns(...)`

```python
def add_confidence_columns(
    df,
    *,
    response_column: str = "response",
    field: str | None = None,
    response_schema: type | dict[str, Any] | None = None,
    prefix: str = "confidence",
)
```

Use this for batch API outputs loaded into pandas.

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

## Minimal Checklist

To get reliable results:

- enable logprobs in the LLM call
- use structured output / JSON schema
- pass `field=` when you know the target key
- pass `response_schema=` when you want enum auto-detection and `resolved_value`
