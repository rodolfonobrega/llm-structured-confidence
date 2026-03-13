# llm-structured-confidence

Extract path-aware confidence from LLM structured JSON responses with logprobs.

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

## Main Function

```python
def extract_logprobs(
    response: Any,           # SDK objects OR raw dicts from batch APIs
    *,
    field_path: str | None,  # Atomic path, e.g. "category", "categories[]", "classifications[].name"
    response_schema: type | dict[str, Any] | None,  # Pydantic model or JSON Schema
) -> list[PathFieldLogprob]
```

Supported response types:
- `litellm.ModelResponse` / `openai.ChatCompletion`
- `google.genai.GenerateContentResponse`
- raw `dict` with `"choices"` key
- raw `dict` with `"candidates"` key

Behavior:
- `field_path=` explicitly selects an atomic value
- `response_schema=` auto-detects enum-valued paths recursively
- if neither is provided, all atomic values are returned
- results preserve JSON order and repeated values

Internally, `response_schema` is normalized to plain JSON Schema before path detection and enum resolution.

## Path Syntax

```python
"category"
"categories[]"
"classification.name"
"classifications[].name"
"groups[].items[].label"
```

## PathFieldLogprob attributes

```python
path: str               # resolved path, e.g. "classifications[0].name"
value: Any              # parsed value
field_logprob: FieldLogprob
```

## FieldLogprob attributes

```python
value: Any
tokens: list[TokenInfo]
joint_logprob: float
joint_probability: float
mean_logprob: float
mean_probability: float
mean_nonzero_logprob: float | None
mean_nonzero_probability: float | None
top_logprobs: list[TopAlternative]
```

## TokenInfo attributes

```python
token: str
logprob: float
probability: float
char_start: int
char_end: int
```

## TopAlternative attributes

```python
token: str
logprob: float
resolved_value: Any | None
probability: float
```

`resolved_value` is only populated when `response_schema=` provides the allowed Enum/Literal values and a token prefix matches exactly one value. Ambiguous prefixes stay `None`.

## Which metric to use

- `mean_nonzero_probability` — use this for ENUM classification
- `joint_probability` — strictest, penalizes longer values
- `mean_probability` — geometric mean, fair across token counts

## Examples

### Scalar field

```python
entries = extract_logprobs(resp, field_path="category")
entry = entries[0]

print(entry.path)                                    # "category"
print(entry.value)                                   # "health and wellness"
print(entry.field_logprob.mean_nonzero_probability)  # 0.845
```

### Array field

```python
entries = extract_logprobs(resp, field_path="categories[]")

for entry in entries:
    print(entry.path, entry.value)
```

### Nested path

```python
entries = extract_logprobs(resp, field_path="classifications[].name")

for entry in entries:
    print(entry.path)                                   # "classifications[0].name"
    print(entry.value)                                  # "Positive"
    print(entry.field_logprob.mean_nonzero_probability) # 0.961
```

### Response schema auto-detection

```python
entries = extract_logprobs(resp, response_schema=SingleCategory)  # category
entries = extract_logprobs(resp, response_schema=MultiCategory)   # categories[]
entries = extract_logprobs(resp, response_schema=schema)          # inferred paths from JSON Schema
```

### Resolved alternatives

```python
entries = extract_logprobs(resp, response_schema=SingleCategory)
fl = entries[0].field_logprob

print(fl.top_logprobs[0].token)           # "health"
print(fl.top_logprobs[0].resolved_value)  # "health and wellness"
```

### google-genai SDK

```python
result = extract_logprobs(resp, field_path="category")
```

### Vertex AI batch API (raw dicts)

```python
result = extract_logprobs(row["response"], field_path="category")
```

## Pandas integration

```python
from llm_structured_confidence import add_confidence_columns, extract_confidence

metrics = extract_confidence(row["response"], field_path="category")

df = add_confidence_columns(
    df,
    response_column="response",
    field_path="classifications[].name",
)
```

Added DataFrame columns:

```python
confidence_value
confidence_path
confidence_prob
confidence_joint_prob
confidence_top_alt
confidence_top_alt_resolved
confidence_top_alt_prob
confidence_error
```

Helpers always use the first matching value.

## Lower-level API

```python
from llm_structured_confidence._parser import parse_json_spans, build_token_char_ranges, tokens_for_span, get_overlapping_indices
from llm_structured_confidence._converter import normalize_response

parsed = parse_json_spans('{"category": "bars"}')

parsed = parse_json_spans(
    '{"classifications":[{"id":0,"name":"Positive","color":"#00FF00"}]}'
)
item = parsed["classifications"][0]
print(item["id"].value)    # 0
print(item["name"].value)  # "Positive"

ranges = build_token_char_ranges(tokens)
indices = get_overlapping_indices(char_start, char_end, ranges)
token_infos = tokens_for_span(char_start, char_end, normalized_tokens, ranges)
norm = normalize_response(response)
```

APIs in `_parser` and `_converter` are internal; may change in minor releases.

## Requirements for the LLM call

- `logprobs=True` and `top_logprobs=N` (litellm/OpenAI)
- `response_logprobs=True` and `logprobs=N` (google-genai)
- structured output (JSON schema or Pydantic model) for reliable parsing

## File structure

```python
llm_structured_confidence/
  __init__.py         # extract_logprobs()
  _extract.py         # path-aware extraction helpers
  _classification.py  # schema path detection + token-prefix resolution
  _types.py           # FieldLogprob, PathFieldLogprob, TokenInfo, TopAlternative
  _parser.py          # JSON span parser + overlap logic
  _converter.py       # provider normalization
  _pandas.py          # add_confidence_columns(), extract_confidence()
examples/
  examples.ipynb
```
