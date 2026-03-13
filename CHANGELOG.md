# Changelog

All notable changes to this project are recorded in this file.

## [0.4.0]
- **Breaking**: `extract_field_logprobs`, `extract_path_logprobs`, and `field=`/`field_path=` arguments everywhere were removed in favor of a single `extract_logprobs(..., field_path=...)` API. Helpers now accept only `field_path`.
- **Breaking**: The pandas helper columns now always include `{prefix}_path`. `extract_confidence` always returns the resolved `path`.
- Internal schema detection and extraction paths now only work with fully resolved JSON paths; helper utilities that exposed top-level-only dictionaries were removed.
- Updated documentation, README, AGENTS.md, and the example notebook to describe the new API surface.
- Version bump to `0.4.0` ahead of the release.
