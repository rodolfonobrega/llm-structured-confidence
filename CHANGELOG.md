# Changelog

All notable changes to this project are recorded in this file.

## [0.4.1]
- Fixed: JSON Schema `type` field is now normalized to lowercase for case-insensitive matching. Schemas with uppercase types like `"type": "OBJECT"` now work correctly.

## [0.4.0]
- **Breaking**: `extract_field_logprobs`, `extract_path_logprobs`, and `field=`/`field_path=` arguments everywhere were removed in favor of a single `extract_logprobs(..., field_path=...)` API. Helpers now accept only `field_path`.
- **Breaking**: The pandas helper columns now always include `{prefix}_path`. `extract_confidence` always returns the resolved `path`.
- Internal schema detection and extraction paths now only work with fully resolved JSON paths; helper utilities that exposed top-level-only dictionaries were removed.
- Updated documentation, README, AGENTS.md, and the example notebook to describe the new API surface.
- Version bump to `0.4.0` ahead of the release.

## [0.3.0]
- Added `response_schema=` (Pydantic or JSON Schema) to auto-detect enum-valued paths and populate `TopAlternative.resolved_value` from token prefixes.
- Introduced `extract_path_logprobs(...)`, `PathFieldLogprob`, and schema-driven detection so arrays/objects can return positional results without collisions.
- Pandas/table helpers gained `path` columns and surfaced resolved paths when `response_schema=` is provided.
- Documentation, AGENTS, and examples were rewritten around the richer structured-output experience.

## [0.2.0]
- Normalized Pydantic/JSON Schema models via `_classification` helpers, adding `classification_values_by_field` and ensuring enum/literal detection worked across JSON/YAML wrappers.
- Shared parser/converter helpers (`parse_json_spans`, token overlap logic, `normalize_response`) became the core of `extract_field_logprobs`, `extract_confidence`, and `add_confidence_columns`.
- Added richer README/AGENTS/docs/examples describing enums, metrics, and provider-specific expectations.

## [0.1.1]
- Added logging and helper coverage for OpenAI/Vertex batch responses, plus pandas integration for `extract_confidence` and column scaffolding.

## [0.1.0]
- Initial release of `llm-structured-confidence`, shipping the first extractor and token-based metrics for structured LLM outputs.
