# Releasing

This repository is configured to publish the distribution `llm-structured-confidence` to PyPI using GitHub Actions and PyPI Trusted Publishing.

The import path remains `llm_structured_confidence`:

```python
from llm_structured_confidence import extract_field_logprobs
```

## Recommended naming

- Distribution on PyPI: `llm-structured-confidence`
- Import path: `llm_structured_confidence`
- GitHub repository: `llm-structured-confidence`

This keeps the public package name specific and available, while preserving a clean Python import path.

## One-time setup

1. Rename the GitHub repository if you want it to match the new public name.
2. If you rename the repository, update the URLs in `pyproject.toml`.
3. Create the PyPI project `llm-structured-confidence`.
4. In PyPI, add a Trusted Publisher for this GitHub repository:
   - Owner: `rodolfonobrega`
   - Repository: your final repository name
   - Workflow: `release.yml`
   - Environment name: `pypi`
5. In GitHub, create an environment named `pypi`.
6. Optionally protect the `pypi` environment with required reviewers.

## Release flow

1. Update the version in `pyproject.toml`.
2. Update `llm_structured_confidence/__init__.py` to the same version.
3. Commit and push to `main`.
4. Create a Git tag in the format `vX.Y.Z`.
5. Create and publish a GitHub Release from that tag.
6. The workflow `.github/workflows/release.yml` will validate the version, run unit tests, build the distributions, run `twine check`, and publish to PyPI.

## Release script

For the standard flow, you can use:

```bash
./scripts/release.sh 0.1.1
```

The script updates `pyproject.toml` and `llm_structured_confidence/__init__.py`, creates a release commit, pushes `main`, creates tag `vX.Y.Z`, pushes the tag, and publishes the GitHub Release.

## Local preflight

```bash
python3 -m pip install -e ".[dev]"
pytest llm_structured_confidence/tests/test_unit.py -v
python3 -m build
python3 -m twine check dist/*
```
