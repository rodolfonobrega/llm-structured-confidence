#!/usr/bin/env bash
set -euo pipefail

repo="rodolfonobrega/llm-structured-confidence"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 X.Y.Z" >&2
  exit 1
fi

version="$1"

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version must match X.Y.Z" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

branch="$(git branch --show-current)"
if [[ "$branch" != "main" ]]; then
  echo "Releases must be created from the main branch. Current branch: $branch" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree has uncommitted or untracked changes." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run 'gh auth login' first." >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
  echo "Tag v$version already exists locally." >&2
  exit 1
fi

if git ls-remote --tags --exit-code origin "refs/tags/v$version" >/dev/null 2>&1; then
  echo "Tag v$version already exists on origin." >&2
  exit 1
fi

if gh release view "v$version" --repo "$repo" >/dev/null 2>&1; then
  echo "GitHub release v$version already exists." >&2
  exit 1
fi

python3 - "$version" <<'PY'
import pathlib
import re
import sys

version = sys.argv[1]

files = {
    pathlib.Path("pyproject.toml"): (
        r'^version = "[^"]+"$',
        f'version = "{version}"',
        "pyproject.toml",
    ),
    pathlib.Path("llm_structured_confidence/__init__.py"): (
        r'^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
        "llm_structured_confidence/__init__.py",
    ),
}

for path, (pattern, replacement, label) in files.items():
    text = path.read_text(encoding="utf-8")
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Could not update version in {label}")
    path.write_text(text, encoding="utf-8")
PY

git add pyproject.toml llm_structured_confidence/__init__.py
git commit -m "Release v$version"
git push origin main
git tag -a "v$version" -m "Release v$version"
git push origin "v$version"
gh release create "v$version" \
  --repo "$repo" \
  --title "v$version" \
  --notes "Release v$version."
