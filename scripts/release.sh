#!/usr/bin/env bash
set -euo pipefail

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

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Working tree has uncommitted tracked changes." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI 'gh' is required." >&2
  exit 1
fi

python3 - "$version" <<'PY'
import pathlib
import re
import sys

version = sys.argv[1]

pyproject = pathlib.Path("pyproject.toml")
text = pyproject.read_text()
text, count = re.subn(
    r'^version = "[^"]+"$',
    f'version = "{version}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise SystemExit("Could not update version in pyproject.toml")
pyproject.write_text(text)

init_file = pathlib.Path("llm_structured_confidence/__init__.py")
text = init_file.read_text()
text, count = re.subn(
    r'^__version__ = "[^"]+"$',
    f'__version__ = "{version}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    raise SystemExit("Could not update __version__ in llm_structured_confidence/__init__.py")
init_file.write_text(text)
PY

git add pyproject.toml llm_structured_confidence/__init__.py
git commit -m "Release v$version"
git push origin main
git tag -a "v$version" -m "Release v$version"
git push origin "v$version"
gh release create "v$version" \
  --repo rodolfonobrega/llm-structured-confidence \
  --title "v$version" \
  --notes "Release v$version."
