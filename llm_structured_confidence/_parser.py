"""JSON parser that extracts character-level spans for every atomic value.

Uses a Lark LALR grammar with ``propagate_positions=True`` so that each
terminal in the parse tree carries its exact start/end character offsets
inside the JSON string.  The ``_SpanExtractor`` Transformer walks the tree
bottom-up and returns a nested Python structure where every leaf is a
``_ValueSpan`` (value + char range) and containers are plain dicts/lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lark import Lark, Token, Transformer_NonRecursive, Tree, v_args
from lark.tree import Meta

from ._types import TokenInfo

_JSON_GRAMMAR = r"""
    start: value
    ?value: object | array | string | SIGNED_NUMBER -> number | true | false | null
    true:  "true"
    false: "false"
    null:  "null"
    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : key ":" value
    key    : ESCAPED_STRING
    string : ESCAPED_STRING
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

_parser = Lark(
    _JSON_GRAMMAR,
    parser="lalr",
    propagate_positions=True,
    maybe_placeholders=False,
)


@dataclass
class _ValueSpan:
    """A parsed atomic value together with the character range of its
    *content* (quotes excluded for strings)."""

    value: Any
    char_start: int
    char_end: int


@v_args(meta=True)
class _SpanExtractor(Transformer_NonRecursive):

    def string(self, meta: Meta, children: list[Token]) -> _ValueSpan:
        return _ValueSpan(
            value=str(children[0])[1:-1],
            char_start=meta.start_pos + 1,
            char_end=meta.end_pos - 1,
        )

    def number(self, meta: Meta, children: list[Token]) -> _ValueSpan:
        raw = str(children[0])
        value: int | float = float(raw) if "." in raw else int(raw)
        return _ValueSpan(value=value, char_start=meta.start_pos, char_end=meta.end_pos)

    def true(self, meta: Meta, children: list[Token]) -> _ValueSpan:
        return _ValueSpan(value=True, char_start=meta.start_pos, char_end=meta.end_pos)

    def false(self, meta: Meta, children: list[Token]) -> _ValueSpan:
        return _ValueSpan(value=False, char_start=meta.start_pos, char_end=meta.end_pos)

    def null(self, meta: Meta, children: list[Token]) -> _ValueSpan:
        return _ValueSpan(value=None, char_start=meta.start_pos, char_end=meta.end_pos)

    def array(self, meta: Meta, children: list[Any]) -> list[Any]:
        return children

    def object(self, meta: Meta, children: list[tuple[str, Any]]) -> dict[str, Any]:
        return {k: v for k, v in children}

    def pair(self, meta: Meta, children: list[Any]) -> tuple[str, Any]:
        key = children[0]
        value = children[1]
        if isinstance(value, Tree) and not value.children:
            value = _ValueSpan(value=None, char_start=meta.start_pos, char_end=meta.end_pos)
        return (key, value)

    def key(self, meta: Meta, children: list[Token]) -> str:
        return str(children[0])[1:-1]

    def start(self, meta: Meta, children: list[Any]) -> Any:
        return children[0]


def parse_json_spans(json_string: str) -> dict[str, Any] | list[Any] | _ValueSpan:
    """Parse *json_string* and return a tree of ``_ValueSpan`` leaves."""
    tree = _parser.parse(json_string)
    return _SpanExtractor().transform(tree)


# ── token / char-range helpers ────────────────────────────────────────

def build_token_char_ranges(
    tokens: list[Any],
) -> list[tuple[int, int]]:
    """Return ``(char_start, char_end)`` for every token based on concatenation."""
    ranges: list[tuple[int, int]] = []
    offset = 0
    for t in tokens:
        text = t.token if hasattr(t, "token") else t["token"]
        length = len(text)
        ranges.append((offset, offset + length))
        offset += length
    return ranges


def get_overlapping_indices(
    char_start: int,
    char_end: int,
    token_char_ranges: list[tuple[int, int]],
) -> list[int]:
    """Return indices of tokens whose character span overlaps ``[char_start, char_end)``."""
    result: list[int] = []
    for idx, (t_start, t_end) in enumerate(token_char_ranges):
        if max(char_start, t_start) < min(char_end, t_end):
            result.append(idx)
    return result


def tokens_for_span(
    char_start: int,
    char_end: int,
    normalized_tokens: list[Any],
    token_char_ranges: list[tuple[int, int]],
) -> list[TokenInfo]:
    """Build ``TokenInfo`` objects for tokens overlapping a character range."""
    indices = get_overlapping_indices(char_start, char_end, token_char_ranges)
    result: list[TokenInfo] = []
    for idx in indices:
        nt = normalized_tokens[idx]
        t_start, t_end = token_char_ranges[idx]
        result.append(
            TokenInfo(
                token=nt.token,
                logprob=nt.logprob,
                char_start=t_start,
                char_end=t_end,
            )
        )
    return result
