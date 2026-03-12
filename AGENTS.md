# llm-structured-confidence

Extract per-field confidence from LLM structured JSON responses with logprobs.

## Import

```python
from llm_structured_confidence import extract_field_logprobs, FieldLogprob, TokenInfo, TopAlternative
```

## Function

```python
def extract_field_logprobs(
    response: Any,           # SDK objects OR raw dicts from batch APIs (see below)
    *,
    field: str | None,       # JSON field name, e.g. "category"
    model: type | None,      # Pydantic model — auto-detects Enum/list[Enum]/Literal fields
) -> dict[str, FieldLogprob]
```

Supported response types:
- `litellm.ModelResponse` / `openai.ChatCompletion` (SDK objects with `.choices`)
- `google.genai.GenerateContentResponse` (SDK object with `.candidates`)
- Raw `dict` with `"choices"` key (OpenAI batch API response body)
- Raw `dict` with `"candidates"` key (Vertex AI batch API response, camelCase)

Returns dict keyed by **value as string**. Scalar field → 1 entry. Array field → 1 entry per element.

Priority: `field` > `model` > all fields.

## FieldLogprob attributes

```
value: Any                          # parsed value
tokens: list[TokenInfo]             # tokens used in calculation
joint_logprob: float                # sum(logprobs)
joint_probability: float            # exp(joint_logprob) — product of all token probs
mean_logprob: float                 # mean(logprobs)
mean_probability: float             # exp(mean_logprob) — geometric mean
mean_nonzero_logprob: float | None  # mean of logprobs where logprob != 0
mean_nonzero_probability: float | None  # exp(mean_nonzero_logprob) — best for ENUM
top_logprobs: list[TopAlternative]  # alternatives from first uncertain token
```

## TokenInfo attributes

```
token: str
logprob: float
probability: float  # property, exp(logprob)
char_start: int
char_end: int
```

## TopAlternative attributes

```
token: str
logprob: float
probability: float  # property, exp(logprob)
```

## Which metric to use

- `mean_nonzero_probability` — **use this for ENUM classification**. Only averages tokens where the model had a real choice. With ENUMs, only the first token carries uncertainty; the rest are deterministic (logprob=0). This avoids inflating confidence for longer category names.
- `joint_probability` — strictest, penalizes longer values.
- `mean_probability` — geometric mean, fair across different token counts.

## Examples

### Scalar field

```python
resp = litellm.completion(model="gpt-4.1-mini", messages=[...],
    response_format=json_schema, logprobs=True, top_logprobs=5)

result = extract_field_logprobs(resp, field="category")

# Iterate — you don't need to know the value beforehand:
for value, fl in result.items():
    print(value)                        # "health and wellness"
    print(fl.mean_nonzero_probability)  # 0.845
    print(fl.top_logprobs[0].token)     # "health"
    print(fl.top_logprobs[1].token)     # "tech" (technology prefix)

# Or unpack the single entry directly:
value, fl = next(iter(result.items()))

# Or access by value if you already know it:
fl = result["health and wellness"]
```

### Array field

```python
# Response: {"categories": ["health and wellness", "sports", "technology"]}
result = extract_field_logprobs(resp, field="categories")

for value, fl in result.items():
    print(f"{value}: {fl.mean_nonzero_probability}")
# health and wellness: 0.845
# sports: 0.951
# technology: 0.916
```

### Pydantic auto-detection

```python
class CategoryEnum(str, Enum):
    health_and_wellness = "health and wellness"
    sports = "sports"

class SingleCategory(BaseModel):
    category: CategoryEnum

class MultiCategory(BaseModel):
    categories: list[CategoryEnum]

result = extract_field_logprobs(resp, model=SingleCategory)   # detects "category"
result = extract_field_logprobs(resp, model=MultiCategory)    # detects "categories"
```

### google-genai SDK

```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="my-project", location="global")
resp = client.models.generate_content(model="gemini-2.5-flash", contents=[...],
    config=types.GenerateContentConfig(response_logprobs=True, logprobs=5,
        thinking_config=types.ThinkingConfig(thinking_budget=0)))

result = extract_field_logprobs(resp, field="category")  # same interface
```

### Vertex AI batch API (raw dicts)

```python
import json

# Each line in the batch output JSONL has: {"id": ..., "response": {...}, ...}
with open("vertex_batch_output.jsonl") as f:
    for line in f:
        row = json.loads(line)
        # Pass the "response" dict directly — no wrapper needed
        result = extract_field_logprobs(row["response"], field="category")
        for value, fl in result.items():
            print(f"{row['id']}: {value} ({fl.mean_nonzero_probability:.2%})")
```

Batch JSONL request must include `"responseLogprobs": true` and `"logprobs": N` in
`generationConfig`. The response dict uses camelCase keys (`logprobsResult`,
`chosenCandidates`, `logProbability`) — the library handles this automatically.

## Pandas integration

For batch API outputs loaded into DataFrames:

```python
import pandas as pd
from llm_structured_confidence import add_confidence_columns

# Vertex AI batch output
df = pd.read_json("vertex_batch_output.jsonl", lines=True)
df = add_confidence_columns(df, response_column="response", field="category")
# Adds: confidence_value, confidence_prob, confidence_joint_prob,
#        confidence_top_alt, confidence_top_alt_prob, confidence_error

# OpenAI batch output — extract body first
df = pd.read_json("openai_batch_output.jsonl", lines=True)
df["body"] = df["response"].apply(lambda r: r["body"])
df = add_confidence_columns(df, response_column="body", field="category")

# Custom prefix
df = add_confidence_columns(df, response_column="response", field="category", prefix="conf")
# Adds: conf_value, conf_prob, conf_joint_prob, conf_top_alt, conf_top_alt_prob, conf_error
```

For single-response extraction into a flat dict:

```python
from llm_structured_confidence import extract_confidence

row = {"response": {...}}  # raw batch response dict
metrics = extract_confidence(row["response"], field="category")
# metrics = {
#     "value": "technology",
#     "joint_probability": 0.9999,
#     "mean_probability": 0.9999,
#     "mean_nonzero_probability": 0.9999,
#     "top_alternative": "science",
#     "top_alternative_probability": 0.0001,
#     "top_logprobs": [("technology", 0.9999), ("science", 0.0001), ...],
#     "error": None,
# }
```

## Lower-level API

For custom workflows (parse JSON without response, build own metrics):

```python
from llm_structured_confidence._parser import parse_json_spans, build_token_char_ranges, tokens_for_span, get_overlapping_indices
from llm_structured_confidence._converter import normalize_response

# JSON → dict with _ValueSpan(value, char_start, char_end) for each atomic value
parsed = parse_json_spans('{"category": "bars"}')  # parsed["category"].char_start, .char_end

# Token char ranges from concatenation
ranges = build_token_char_ranges(tokens)

# Tokens overlapping a span
indices = get_overlapping_indices(char_start, char_end, ranges)
token_infos = tokens_for_span(char_start, char_end, normalized_tokens, ranges)

# Normalize any provider response → (content, tokens)
norm = normalize_response(response)  # norm.content, norm.tokens
```

APIs in `_parser` and `_converter` are internal; may change in minor releases.

## Requirements for the LLM call

- `logprobs=True` and `top_logprobs=N` (litellm/OpenAI)
- `response_logprobs=True` and `logprobs=N` (google-genai)
- For models with reasoning/thinking: consider disabling it for cleaner logprobs (`reasoning_effort="none"` for litellm, `thinking_budget=0` for google-genai)
- Structured output (JSON schema or Pydantic model) for reliable parsing

## File structure

```
llm_structured_confidence/
  __init__.py    # extract_field_logprobs(), Pydantic detection
  _types.py      # FieldLogprob, TokenInfo, TopAlternative
  _parser.py     # Lark JSON parser, char-range overlap logic
  _converter.py  # normalizes litellm/OpenAI/google-genai/raw dicts → internal format
  _pandas.py     # add_confidence_columns(), extract_confidence()
examples/
  examples.ipynb # usage notebook with all features
```
