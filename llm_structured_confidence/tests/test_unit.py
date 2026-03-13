"""Comprehensive unit tests for llm-structured-confidence.

All tests use mock token data — no API calls needed.
"""

from __future__ import annotations

import math
from typing import Literal

import pytest
from pydantic import BaseModel

from llm_structured_confidence import (
    extract_logprobs,
    FieldLogprob,
    PathFieldLogprob,
    TokenInfo,
    TopAlternative,
)
from llm_structured_confidence._classification import (
    classification_values_by_path,
    detect_classification_paths,
    normalize_response_schema,
)
from llm_structured_confidence._converter import normalize_response, NormalizedToken, NormalizedResponse
from llm_structured_confidence._parser import (
    parse_json_spans,
    build_token_char_ranges,
    get_overlapping_indices,
    tokens_for_span,
    _ValueSpan,
)
from llm_structured_confidence._types import FieldLogprob as FL

from .conftest import (
    make_openai_response,
    make_genai_response,
    make_vertex_batch_dict,
    make_tokens,
    make_normalized,
    SingleCategoryModel,
    MultipleCategoriesModel,
    LiteralModel,
    MixedModel,
    NestedClassificationModel,
    CategoryEnum,
    GEMINI25_SCALAR_CONTENT,
    GEMINI25_SCALAR_TOKENS,
    GEMINI3_SCALAR_CONTENT,
    GEMINI3_SCALAR_TOKENS,
    ARRAY_CONTENT,
    ARRAY_TOKENS,
    MULTI_FIELD_CONTENT,
    MULTI_FIELD_TOKENS,
)

CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [
                "health and wellness",
                "sports",
                "technology",
                "entertainment",
            ],
        }
    },
    "required": ["category"],
    "additionalProperties": False,
}

ARRAY_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "health and wellness",
                    "sports",
                    "technology",
                    "entertainment",
                ],
            },
        }
    },
    "required": ["categories"],
    "additionalProperties": False,
}

STRUCTURED_OUTPUT_WRAPPER = {
    "type": "json_schema",
    "json_schema": {
        "name": "classification",
        "strict": True,
        "schema": CLASSIFICATION_JSON_SCHEMA,
    },
}

NESTED_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {
                        "type": "string",
                        "enum": ["Positive", "Negative", "Neutral"],
                    },
                    "color": {"type": "string"},
                },
                "required": ["id", "name", "color"],
            },
        }
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


def _entries_by_path(entries: list[PathFieldLogprob]) -> dict[str, PathFieldLogprob]:
    return {entry.path: entry for entry in entries}


def _logprobs_by_value(entries: list[PathFieldLogprob]) -> dict[str, FieldLogprob]:
    return {str(entry.value): entry.field_logprob for entry in entries}


def _single_entry(entries: list[PathFieldLogprob]) -> PathFieldLogprob:
    assert len(entries) == 1
    return entries[0]


# ═══════════════════════════════════════════════════════════════════════
# 1. FieldLogprob.compute — metric calculations
# ═══════════════════════════════════════════════════════════════════════

class TestResponseSchemaNormalization:

    def test_normalize_pydantic_model_to_json_schema(self):
        schema = normalize_response_schema(SingleCategoryModel)
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert schema["properties"]["category"]["enum"] == [
            "health and wellness",
            "sports",
            "technology",
            "entertainment",
        ]

    def test_normalize_json_schema_wrapper(self):
        schema = normalize_response_schema(STRUCTURED_OUTPUT_WRAPPER)
        assert schema == CLASSIFICATION_JSON_SCHEMA

    def test_detect_paths_cover_top_level_pydantic_fields(self):
        assert detect_classification_paths(SingleCategoryModel) == ["category"]
        assert classification_values_by_path(SingleCategoryModel)["category"] == [
            "health and wellness",
            "sports",
            "technology",
            "entertainment",
        ]

    def test_detect_paths_cover_top_level_json_schema_fields(self):
        assert detect_classification_paths(ARRAY_CLASSIFICATION_JSON_SCHEMA) == ["categories[]"]
        assert classification_values_by_path(ARRAY_CLASSIFICATION_JSON_SCHEMA)["categories[]"] == [
            "health and wellness",
            "sports",
            "technology",
            "entertainment",
        ]

    def test_detect_paths_recurse_through_pydantic(self):
        assert detect_classification_paths(NestedClassificationModel) == [
            "classifications[].name"
        ]
        assert classification_values_by_path(NestedClassificationModel)[
            "classifications[].name"
        ] == ["Positive", "Negative", "Neutral"]

    def test_detect_paths_recurse_through_json_schema(self):
        assert detect_classification_paths(NESTED_CLASSIFICATION_JSON_SCHEMA) == [
            "classifications[].name"
        ]
        assert classification_values_by_path(NESTED_CLASSIFICATION_JSON_SCHEMA)[
            "classifications[].name"
        ] == ["Positive", "Negative", "Neutral"]


class TestFieldLogprobCompute:

    def test_empty_tokens(self):
        fl = FL.compute("test", [])
        assert fl.value == "test"
        assert fl.joint_logprob == 0.0
        assert fl.joint_probability == 1.0
        assert fl.mean_logprob == 0.0
        assert fl.mean_probability == 1.0
        assert fl.mean_nonzero_logprob is None
        assert fl.mean_nonzero_probability is None

    def test_single_nonzero_token(self):
        tokens = [TokenInfo(token="bars", logprob=-0.168, char_start=0, char_end=4)]
        fl = FL.compute("bars", tokens)
        assert fl.joint_logprob == pytest.approx(-0.168)
        assert fl.joint_probability == pytest.approx(math.exp(-0.168))
        assert fl.mean_logprob == pytest.approx(-0.168)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.168)
        assert fl.mean_nonzero_probability == pytest.approx(math.exp(-0.168))

    def test_mixed_zero_and_nonzero(self):
        tokens = [
            TokenInfo(token="health", logprob=-0.168, char_start=0, char_end=6),
            TokenInfo(token=" and", logprob=0.0, char_start=6, char_end=10),
            TokenInfo(token=" wellness", logprob=0.0, char_start=10, char_end=18),
        ]
        fl = FL.compute("health and wellness", tokens)

        assert fl.joint_logprob == pytest.approx(-0.168)
        assert fl.mean_logprob == pytest.approx(-0.168 / 3)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.168)
        assert fl.mean_nonzero_probability == pytest.approx(math.exp(-0.168))

    def test_all_zero_tokens(self):
        tokens = [
            TokenInfo(token="a", logprob=0.0, char_start=0, char_end=1),
            TokenInfo(token="b", logprob=0.0, char_start=1, char_end=2),
        ]
        fl = FL.compute("ab", tokens)
        assert fl.joint_logprob == 0.0
        assert fl.joint_probability == 1.0
        assert fl.mean_nonzero_logprob == 0.0
        assert fl.mean_nonzero_probability == 1.0

    def test_multiple_nonzero_tokens(self):
        tokens = [
            TokenInfo(token="tech", logprob=-0.088, char_start=0, char_end=4),
            TokenInfo(token="nology", logprob=-0.012, char_start=4, char_end=10),
        ]
        fl = FL.compute("technology", tokens)
        expected_joint = -0.088 + -0.012
        assert fl.joint_logprob == pytest.approx(expected_joint)
        assert fl.mean_logprob == pytest.approx(expected_joint / 2)
        assert fl.mean_nonzero_logprob == pytest.approx(expected_joint / 2)

    def test_top_logprobs_stored(self):
        tokens = [TokenInfo(token="x", logprob=-0.1, char_start=0, char_end=1)]
        alts = [TopAlternative(token="x", logprob=-0.1), TopAlternative(token="y", logprob=-2.3)]
        fl = FL.compute("x", tokens, alts)
        assert len(fl.top_logprobs) == 2
        assert fl.top_logprobs[0].token == "x"
        assert fl.top_logprobs[0].resolved_value is None
        assert fl.top_logprobs[1].probability == pytest.approx(math.exp(-2.3))


# ═══════════════════════════════════════════════════════════════════════
# 2. Parser — JSON span extraction
# ═══════════════════════════════════════════════════════════════════════

class TestParser:

    def test_simple_string_excludes_quotes(self):
        parsed = parse_json_spans('{"k": "hello"}')
        assert isinstance(parsed, dict)
        span = parsed["k"]
        assert isinstance(span, _ValueSpan)
        assert span.value == "hello"
        json_str = '{"k": "hello"}'
        assert json_str[span.char_start:span.char_end] == "hello"

    def test_number_value(self):
        parsed = parse_json_spans('{"n": 42}')
        span = parsed["n"]
        assert span.value == 42
        assert '{"n": 42}'[span.char_start:span.char_end] == "42"

    def test_float_value(self):
        parsed = parse_json_spans('{"f": 3.14}')
        span = parsed["f"]
        assert span.value == pytest.approx(3.14)

    def test_bool_values(self):
        parsed = parse_json_spans('{"t": true, "f": false}')
        assert parsed["t"].value is True
        assert parsed["f"].value is False

    def test_null_value(self):
        parsed = parse_json_spans('{"x": null}')
        assert parsed["x"].value is None

    def test_array(self):
        parsed = parse_json_spans('{"items": ["a", "b", "c"]}')
        items = parsed["items"]
        assert isinstance(items, list)
        assert len(items) == 3
        assert items[0].value == "a"
        assert items[2].value == "c"

    def test_nested_object(self):
        parsed = parse_json_spans('{"outer": {"inner": "val"}}')
        inner = parsed["outer"]
        assert isinstance(inner, dict)
        assert inner["inner"].value == "val"

    def test_no_space_before_value(self):
        """Gemini 3 style: no space after colon."""
        parsed = parse_json_spans('{"k":"hello"}')
        span = parsed["k"]
        assert span.value == "hello"
        assert '{"k":"hello"}'[span.char_start:span.char_end] == "hello"

    def test_empty_string(self):
        parsed = parse_json_spans('{"k": ""}')
        span = parsed["k"]
        assert span.value == ""
        assert span.char_start == span.char_end


class TestTokenCharRanges:

    def test_build_ranges(self):
        tokens = make_tokens([("ab", 0.0), ("cde", -0.1), ("f", -0.2)])
        ranges = build_token_char_ranges(tokens)
        assert ranges == [(0, 2), (2, 5), (5, 6)]

    def test_overlapping_indices_exact(self):
        ranges = [(0, 2), (2, 5), (5, 8), (8, 10)]
        assert get_overlapping_indices(2, 8, ranges) == [1, 2]
        assert get_overlapping_indices(0, 2, ranges) == [0]

    def test_overlapping_indices_partial(self):
        ranges = [(0, 3), (3, 6), (6, 9)]
        assert get_overlapping_indices(1, 7, ranges) == [0, 1, 2]

    def test_no_overlap(self):
        ranges = [(0, 3), (3, 6)]
        assert get_overlapping_indices(6, 9, ranges) == []

    def test_tokens_for_span(self):
        tokens = make_tokens([("abc", -0.1), ("def", -0.2), ("ghi", -0.3)])
        ranges = build_token_char_ranges(tokens)
        infos = tokens_for_span(3, 6, tokens, ranges)
        assert len(infos) == 1
        assert infos[0].token == "def"
        assert infos[0].logprob == -0.2


# ═══════════════════════════════════════════════════════════════════════
# 3. Converter — response normalization
# ═══════════════════════════════════════════════════════════════════════

class TestConverter:

    def test_openai_format(self, gemini25_scalar):
        nr = normalize_response(gemini25_scalar)
        assert isinstance(nr, NormalizedResponse)
        assert nr.content == GEMINI25_SCALAR_CONTENT
        assert len(nr.tokens) == len(GEMINI25_SCALAR_TOKENS)
        assert nr.tokens[0].token == '{"'
        assert nr.tokens[4].token == "health"
        assert nr.tokens[4].logprob == pytest.approx(-0.012331)

    def test_genai_format(self, genai_scalar):
        nr = normalize_response(genai_scalar)
        assert nr.content == GEMINI3_SCALAR_CONTENT
        assert len(nr.tokens) == len(GEMINI3_SCALAR_TOKENS)
        assert nr.tokens[3].token == "health"
        assert nr.tokens[3].logprob == pytest.approx(-0.168335)

    def test_genai_top_logprobs(self, genai_scalar):
        nr = normalize_response(genai_scalar)
        assert len(nr.tokens[3].top_logprobs) == 3
        assert nr.tokens[3].top_logprobs[0] == ("health", -0.168335)

    def test_openai_missing_logprobs(self):
        from types import SimpleNamespace
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"), logprobs=None)]
        )
        with pytest.raises(ValueError, match="no logprobs"):
            normalize_response(resp)

    def test_genai_missing_logprobs(self):
        from types import SimpleNamespace
        part = SimpleNamespace(text="{}")
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            logprobs_result=None,
        )
        resp = SimpleNamespace(candidates=[candidate])
        with pytest.raises(ValueError, match="no logprobs_result"):
            normalize_response(resp)

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            normalize_response(42)

    def test_unsupported_dict(self):
        with pytest.raises(TypeError, match="choices.*candidates"):
            normalize_response({"not": "a response"})

    def test_vertex_batch_dict(self, vertex_batch_scalar):
        nr = normalize_response(vertex_batch_scalar)
        assert nr.content == GEMINI3_SCALAR_CONTENT
        assert len(nr.tokens) == len(GEMINI3_SCALAR_TOKENS)
        assert nr.tokens[3].token == "health"
        assert nr.tokens[3].logprob == pytest.approx(-0.168335)

    def test_vertex_batch_dict_top_logprobs(self, vertex_batch_scalar):
        nr = normalize_response(vertex_batch_scalar)
        assert len(nr.tokens[3].top_logprobs) == 3
        assert nr.tokens[3].top_logprobs[0] == ("health", -0.168335)

    def test_vertex_batch_dict_missing_logprobs(self):
        response = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
        with pytest.raises(ValueError, match="no logprobsResult"):
            normalize_response(response)

    def test_vertex_batch_dict_matches_genai_sdk(self, genai_scalar, vertex_batch_scalar):
        nr_sdk = normalize_response(genai_scalar)
        nr_dict = normalize_response(vertex_batch_scalar)
        assert nr_sdk.content == nr_dict.content
        assert len(nr_sdk.tokens) == len(nr_dict.tokens)
        for sdk_tok, dict_tok in zip(nr_sdk.tokens, nr_dict.tokens):
            assert sdk_tok.token == dict_tok.token
            assert sdk_tok.logprob == pytest.approx(dict_tok.logprob)
            assert sdk_tok.top_logprobs == dict_tok.top_logprobs


# ═══════════════════════════════════════════════════════════════════════
# 3b. Batch dict — extract_logprobs end-to-end with Vertex dicts
# ═══════════════════════════════════════════════════════════════════════

class TestVertexBatchDict:

    def test_scalar_extraction(self, vertex_batch_scalar):
        entry = _single_entry(extract_logprobs(vertex_batch_scalar, field_path="category"))
        fl = entry.field_logprob
        assert entry.path == "category"
        assert entry.value == "health and wellness"
        assert fl.value == "health and wellness"
        assert len(fl.tokens) == 3

    def test_scalar_matches_sdk(self, genai_scalar, vertex_batch_scalar):
        fl_sdk = _single_entry(extract_logprobs(genai_scalar, field_path="category")).field_logprob
        fl_dict = _single_entry(extract_logprobs(vertex_batch_scalar, field_path="category")).field_logprob
        assert fl_sdk.joint_probability == pytest.approx(fl_dict.joint_probability)
        assert fl_sdk.mean_probability == pytest.approx(fl_dict.mean_probability)
        assert fl_sdk.mean_nonzero_probability == pytest.approx(fl_dict.mean_nonzero_probability)

    def test_array_extraction(self, vertex_batch_array):
        result = extract_logprobs(vertex_batch_array, field_path="categories[]")
        assert [entry.value for entry in result] == ["health and wellness", "sports"]

    def test_pydantic_model_detection(self, vertex_batch_scalar):
        result = extract_logprobs(vertex_batch_scalar, response_schema=SingleCategoryModel)
        assert [entry.value for entry in result] == ["health and wellness"]


# ═══════════════════════════════════════════════════════════════════════
# 4. Integration — extract_logprobs end-to-end with mocks
# ═══════════════════════════════════════════════════════════════════════

class TestExtractScalar:

    def test_gemini25_category(self, gemini25_scalar):
        entry = _single_entry(extract_logprobs(gemini25_scalar, field_path="category"))
        fl = entry.field_logprob
        assert entry.value == "health and wellness"
        assert fl.value == "health and wellness"
        assert len(fl.tokens) == 3
        assert fl.tokens[0].token == "health"
        assert fl.tokens[1].token == " and"
        assert fl.tokens[2].token == " wellness"

    def test_gemini25_excludes_quotes_and_colon(self, gemini25_scalar):
        """The tokens '":' and ' "' must NOT appear in the result."""
        fl = _single_entry(extract_logprobs(gemini25_scalar, field_path="category")).field_logprob
        token_texts = [t.token for t in fl.tokens]
        assert '":'  not in token_texts
        assert ' "'  not in token_texts
        assert '{"'  not in token_texts
        assert '"}' not in token_texts

    def test_gemini3_excludes_merged_colon_token(self, gemini3_scalar):
        """Gemini 3 merges '":"' into one token — it must be excluded."""
        fl = _single_entry(extract_logprobs(gemini3_scalar, field_path="category")).field_logprob
        token_texts = [t.token for t in fl.tokens]
        assert '":"' not in token_texts
        assert fl.tokens[0].token == "health"

    def test_gemini3_metrics(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, field_path="category")).field_logprob
        assert fl.joint_logprob == pytest.approx(-0.168335)
        assert fl.joint_probability == pytest.approx(math.exp(-0.168335))
        assert fl.mean_logprob == pytest.approx(-0.168335 / 3)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.168335)

    def test_gemini25_metrics(self, gemini25_scalar):
        fl = _single_entry(extract_logprobs(gemini25_scalar, field_path="category")).field_logprob
        assert fl.joint_logprob == pytest.approx(-0.012331)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.012331)

    def test_top_logprobs_from_first_significant(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, field_path="category")).field_logprob
        assert len(fl.top_logprobs) == 3
        assert fl.top_logprobs[0].token == "health"
        assert fl.top_logprobs[1].token == "sport"


class TestResolvedTopAlternatives:

    def test_pydantic_schema_resolves_unique_prefixes(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, response_schema=SingleCategoryModel)).field_logprob
        assert fl.top_logprobs[0].token == "health"
        assert fl.top_logprobs[0].resolved_value == "health and wellness"
        assert fl.top_logprobs[1].token == "sport"
        assert fl.top_logprobs[1].resolved_value == "sports"
        assert fl.top_logprobs[2].token == "tech"
        assert fl.top_logprobs[2].resolved_value == "technology"

    def test_without_schema_keeps_top_alternatives_unresolved(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, field_path="category")).field_logprob
        assert [alt.resolved_value for alt in fl.top_logprobs] == [None, None, None]

    def test_array_resolves_each_element(self, array_response):
        result = _logprobs_by_value(extract_logprobs(array_response, response_schema=MultipleCategoriesModel))
        bars = result["health and wellness"]
        sports = result["sports"]
        assert bars.top_logprobs[0].resolved_value == "health and wellness"
        assert bars.top_logprobs[1].resolved_value == "sports"
        assert sports.top_logprobs[0].resolved_value == "sports"
        assert sports.top_logprobs[1].resolved_value == "health and wellness"

    def test_literal_model_resolves_unique_prefixes(self):
        content = '{"mood":"happy"}'
        tokens = [
            ('{"', -0.01),
            ("mood", 0.0),
            ('":"', -0.10),
            ("hap", -0.05, [("hap", -0.05), ("sad", -2.1), ("neu", -2.3)]),
            ("py", 0.0),
            ('"}', 0.0),
        ]
        resp = make_openai_response(content, tokens)
        fl = _single_entry(extract_logprobs(resp, response_schema=LiteralModel)).field_logprob
        assert fl.top_logprobs[0].resolved_value == "happy"
        assert fl.top_logprobs[1].resolved_value == "sad"
        assert fl.top_logprobs[2].resolved_value == "neutral"

    def test_json_schema_resolves_unique_prefixes(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, response_schema=CLASSIFICATION_JSON_SCHEMA)).field_logprob
        assert fl.top_logprobs[0].resolved_value == "health and wellness"
        assert fl.top_logprobs[1].resolved_value == "sports"
        assert fl.top_logprobs[2].resolved_value == "technology"

    def test_structured_output_wrapper_resolves_unique_prefixes(self, gemini3_scalar):
        fl = _single_entry(extract_logprobs(gemini3_scalar, response_schema=STRUCTURED_OUTPUT_WRAPPER)).field_logprob
        assert fl.top_logprobs[0].resolved_value == "health and wellness"

    def test_ambiguous_prefix_stays_unresolved(self):
        class AmbiguousCategoryModel(BaseModel):
            category: Literal["Bar and Restaurants", "Bars and Nightlife"]

        content = '{"category":"Bar and Restaurants"}'
        tokens = [
            ('{"', -0.01),
            ("category", 0.0),
            ('":"', -0.10),
            ("Bar", -0.05, [("Bar", -0.05), ("Foo", -2.0)]),
            (" and", 0.0),
            (" Restaurants", 0.0),
            ('"}', 0.0),
        ]
        resp = make_openai_response(content, tokens)
        fl = _single_entry(extract_logprobs(resp, response_schema=AmbiguousCategoryModel)).field_logprob
        assert fl.top_logprobs[0].resolved_value is None


class TestExtractArray:

    def test_array_returns_entries_in_order(self, array_response):
        result = extract_logprobs(array_response, field_path="categories[]")
        assert [entry.path for entry in result] == ["categories[0]", "categories[1]"]
        assert [entry.value for entry in result] == ["health and wellness", "sports"]

    def test_array_each_element_has_metrics(self, array_response):
        result = _entries_by_path(extract_logprobs(array_response, field_path="categories[]"))
        bars = result["categories[0]"].field_logprob
        groc = result["categories[1]"].field_logprob
        assert bars.joint_logprob == pytest.approx(-0.168)
        assert groc.joint_logprob == pytest.approx(-0.050)

    def test_array_element_tokens(self, array_response):
        bars = _entries_by_path(extract_logprobs(array_response, field_path="categories[]"))["categories[0]"].field_logprob
        assert bars.tokens[0].token == "health"
        assert bars.tokens[1].token == " and"
        assert bars.tokens[2].token == " wellness"

    def test_array_excludes_delimiters(self, array_response):
        result = extract_logprobs(array_response, field_path="categories[]")
        for entry in result:
            fl = entry.field_logprob
            token_texts = [t.token for t in fl.tokens]
            assert '":["' not in token_texts
            assert '","' not in token_texts
            assert '"]}' not in token_texts

    def test_array_top_logprobs_per_element(self, array_response):
        result = _entries_by_path(extract_logprobs(array_response, field_path="categories[]"))
        bars = result["categories[0]"].field_logprob
        groc = result["categories[1]"].field_logprob
        assert bars.top_logprobs[0].token == "health"
        assert groc.top_logprobs[0].token == "sport"


class TestExtractPathLogprobs:

    def test_nested_object_array_path_returns_all_items(self, nested_classification_response):
        result = extract_logprobs(
            nested_classification_response,
            field_path="classifications[].name",
        )
        assert [entry.path for entry in result] == [
            "classifications[0].name",
            "classifications[1].name",
            "classifications[2].name",
        ]
        assert [entry.value for entry in result] == ["Positive", "Negative", "Positive"]

    def test_nested_path_preserves_repeated_values(self, nested_classification_response):
        result = extract_logprobs(
            nested_classification_response,
            field_path="classifications[].name",
        )
        positives = [entry for entry in result if entry.value == "Positive"]
        assert len(positives) == 2
        assert positives[0].path == "classifications[0].name"
        assert positives[1].path == "classifications[2].name"

    def test_nested_path_resolves_top_alternatives_with_pydantic_schema(
        self,
        nested_classification_response,
    ):
        result = extract_logprobs(
            nested_classification_response,
            field_path="classifications[].name",
            response_schema=NestedClassificationModel,
        )
        assert result[0].field_logprob.top_logprobs[0].resolved_value == "Positive"
        assert result[0].field_logprob.top_logprobs[1].resolved_value == "Negative"
        assert result[1].field_logprob.top_logprobs[0].resolved_value == "Negative"

    def test_response_schema_auto_detects_nested_paths(self, nested_classification_response):
        result = extract_logprobs(
            nested_classification_response,
            response_schema=NestedClassificationModel,
        )
        assert [entry.path for entry in result] == [
            "classifications[0].name",
            "classifications[1].name",
            "classifications[2].name",
        ]

    def test_json_schema_auto_detects_nested_paths(self, nested_classification_response):
        result = extract_logprobs(
            nested_classification_response,
            response_schema=NESTED_CLASSIFICATION_JSON_SCHEMA,
        )
        assert [entry.value for entry in result] == ["Positive", "Negative", "Positive"]

    def test_array_without_brackets_raises_clear_error(self, nested_classification_response):
        with pytest.raises(ValueError, match="resolved to an array"):
            extract_logprobs(
                nested_classification_response,
                field_path="classifications.name",
            )

    def test_object_terminal_raises_clear_error(self, nested_classification_response):
        with pytest.raises(ValueError, match="resolved to an object"):
            extract_logprobs(
                nested_classification_response,
                field_path="classifications[]",
            )

    def test_missing_nested_path_returns_empty_list(self, nested_classification_response):
        result = extract_logprobs(
            nested_classification_response,
            field_path="classifications[].missing",
        )
        assert result == []

    def test_simple_array_with_repeated_values_preserves_alignment(self):
        content = '{"classifications":["Positive","Negative","Positive"]}'
        tokens = [
            ('{"', -0.01),
            ("classifications", 0.0),
            ('":["', -0.03),
            ("Pos", -0.04, [("Pos", -0.04), ("Neg", -1.5), ("Neu", -2.0)]),
            ("itive", 0.0),
            ('","', -0.01),
            ("Neg", -0.06, [("Neg", -0.06), ("Pos", -1.2), ("Neu", -1.8)]),
            ("ative", 0.0),
            ('","', -0.01),
            ("Pos", -0.03, [("Pos", -0.03), ("Neg", -1.4), ("Neu", -2.1)]),
            ("itive", 0.0),
            ('"]}', 0.0),
        ]
        resp = make_openai_response(content, tokens)

        result = extract_logprobs(resp, field_path="classifications[]")

        assert [entry.path for entry in result] == [
            "classifications[0]",
            "classifications[1]",
            "classifications[2]",
        ]
        assert [entry.value for entry in result] == ["Positive", "Negative", "Positive"]
        assert result[0].field_logprob.tokens[0].token == "Pos"
        assert result[1].field_logprob.tokens[0].token == "Neg"
        assert result[2].field_logprob.tokens[0].token == "Pos"
        assert result[0].field_logprob.top_logprobs[0].token == "Pos"
        assert result[1].field_logprob.top_logprobs[0].token == "Neg"
        assert result[2].field_logprob.top_logprobs[0].token == "Pos"


class TestFieldSelection:

    def test_explicit_field_path(self, multi_field_response):
        result = extract_logprobs(multi_field_response, field_path="category")
        assert [entry.path for entry in result] == ["category"]
        assert [entry.value for entry in result] == ["technology"]

    def test_no_field_path_returns_all_atomic_values(self, multi_field_response):
        result = extract_logprobs(multi_field_response)
        assert [entry.path for entry in result] == ["category", "amount"]
        assert [str(entry.value) for entry in result] == ["technology", "150"]

    def test_nonexistent_field_path_returns_empty(self, multi_field_response):
        result = extract_logprobs(multi_field_response, field_path="nonexistent")
        assert result == []


class TestPydanticDetection:

    def test_single_enum(self, gemini3_scalar):
        result = extract_logprobs(gemini3_scalar, response_schema=SingleCategoryModel)
        assert [entry.value for entry in result] == ["health and wellness"]

    def test_list_enum(self, array_response):
        result = extract_logprobs(array_response, response_schema=MultipleCategoriesModel)
        assert [entry.value for entry in result] == ["health and wellness", "sports"]

    def test_literal_field(self):
        content = '{"mood":"happy"}'
        tokens = [('{"', -0.01), ("mood", 0.0), ('":"', -0.10), ("happy", -0.05), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_logprobs(resp, response_schema=LiteralModel)
        assert [entry.value for entry in result] == ["happy"]

    def test_mixed_schema_only_enum(self):
        content = '{"category":"technology","confidence":0.9,"note":"test"}'
        tokens = [
            ('{"', -0.01),
            ("category", 0.0),
            ('":"', -0.10),
            ("tech", -0.05),
            ("nology", 0.0),
            ('","', -0.001),
            ("confidence", 0.0),
            ('":', -0.001),
            ("0", -0.01),
            (".", 0.0),
            ("9", -0.02),
            (',"', -0.001),
            ("note", 0.0),
            ('":"', -0.001),
            ("test", -0.01),
            ('"}', 0.0),
        ]
        resp = make_openai_response(content, tokens)
        result = extract_logprobs(resp, response_schema=MixedModel)
        assert [entry.path for entry in result] == ["category"]
        assert [entry.value for entry in result] == ["technology"]

    def test_json_schema_detects_enum_field(self, gemini3_scalar):
        result = extract_logprobs(gemini3_scalar, response_schema=CLASSIFICATION_JSON_SCHEMA)
        assert [entry.value for entry in result] == ["health and wellness"]

    def test_json_schema_detects_array_enum_field(self, array_response):
        result = extract_logprobs(array_response, response_schema=ARRAY_CLASSIFICATION_JSON_SCHEMA)
        assert [entry.value for entry in result] == ["health and wellness", "sports"]

    def test_field_path_overrides_response_schema(self, multi_field_response):
        result = extract_logprobs(
            multi_field_response,
            field_path="amount",
            response_schema=SingleCategoryModel,
        )
        assert [str(entry.value) for entry in result] == ["150"]


class TestGoogleGenaiInput:

    def test_genai_scalar(self, genai_scalar):
        entry = _single_entry(extract_logprobs(genai_scalar, field_path="category"))
        fl = entry.field_logprob
        assert entry.value == "health and wellness"
        assert fl.tokens[0].token == "health"
        assert fl.joint_logprob == pytest.approx(-0.168335)

    def test_genai_metrics_match_openai(self, gemini3_scalar, genai_scalar):
        """Same token data via different providers must yield same metrics."""
        fl_oai = _single_entry(extract_logprobs(gemini3_scalar, field_path="category")).field_logprob
        fl_gen = _single_entry(extract_logprobs(genai_scalar, field_path="category")).field_logprob

        assert fl_oai.joint_logprob == pytest.approx(fl_gen.joint_logprob)
        assert fl_oai.mean_logprob == pytest.approx(fl_gen.mean_logprob)
        assert fl_oai.mean_nonzero_logprob == pytest.approx(fl_gen.mean_nonzero_logprob)

    def test_genai_array(self):
        from .conftest import make_genai_response, ARRAY_CONTENT, ARRAY_TOKENS
        resp = make_genai_response(ARRAY_CONTENT, ARRAY_TOKENS)
        result = extract_logprobs(resp, field_path="categories[]")
        assert [entry.value for entry in result] == ["health and wellness", "sports"]


class TestEdgeCases:

    def test_single_token_value(self):
        content = '{"k":"v"}'
        tokens = [('{"', -0.01), ("k", 0.0), ('":"', -0.10), ("v", -0.05), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        entry = _single_entry(extract_logprobs(resp, field_path="k"))
        fl = entry.field_logprob
        assert entry.value == "v"
        assert len(fl.tokens) == 1
        assert fl.tokens[0].token == "v"

    def test_empty_string_value(self):
        content = '{"k":""}'
        tokens = [('{"', -0.01), ("k", 0.0), ('":"', -0.10), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        entry = _single_entry(extract_logprobs(resp, field_path="k"))
        fl = entry.field_logprob
        assert entry.value == ""
        assert len(fl.tokens) == 0
        assert fl.joint_probability == 1.0

    def test_boolean_value(self):
        content = '{"flag":true}'
        tokens = [('{"', -0.01), ("flag", 0.0), ('":', -0.01), ("true", -0.02), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        fl = _single_entry(extract_logprobs(resp, field_path="flag")).field_logprob
        assert fl.value is True
        assert fl.joint_logprob == pytest.approx(-0.02)

    def test_number_value(self):
        content = '{"count":42}'
        tokens = [('{"', -0.01), ("count", 0.0), ('":', -0.01), ("42", -0.05), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        entry = _single_entry(extract_logprobs(resp, field_path="count"))
        assert entry.value == 42

    def test_null_value(self):
        content = '{"x":null}'
        tokens = [('{"', -0.01), ("x", 0.0), ('":', -0.01), ("null", -0.03), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        entry = _single_entry(extract_logprobs(resp, field_path="x"))
        assert entry.value is None

    def test_probability_property(self):
        ti = TokenInfo(token="test", logprob=-0.5, char_start=0, char_end=4)
        assert ti.probability == pytest.approx(math.exp(-0.5))

    def test_top_alternative_probability(self):
        alt = TopAlternative(token="x", logprob=-1.0)
        assert alt.probability == pytest.approx(math.exp(-1.0))

    def test_path_field_logprob_keeps_path_and_payload(self):
        fl = FL.compute("Positive", [TokenInfo(token="Pos", logprob=-0.1, char_start=0, char_end=3)])
        entry = PathFieldLogprob(
            path="classifications[0].name",
            value="Positive",
            field_logprob=fl,
        )
        assert entry.path == "classifications[0].name"
        assert entry.value == "Positive"
        assert entry.field_logprob.mean_nonzero_probability == pytest.approx(math.exp(-0.1))


# ═══════════════════════════════════════════════════════════════════════
# 8. Pandas integration — extract_confidence, add_confidence_columns
# ═══════════════════════════════════════════════════════════════════════

class TestExtractConfidence:

    def test_returns_flat_dict(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field_path="category")
        assert result["value"] == "health and wellness"
        assert result["path"] == "category"
        assert result["error"] is None
        assert isinstance(result["mean_nonzero_probability"], float)
        assert isinstance(result["joint_probability"], float)

    def test_top_alternative(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field_path="category")
        assert result["top_alternative"] == "sport"
        assert result["top_alternative_resolved"] is None
        assert isinstance(result["top_alternative_probability"], float)

    def test_top_alternative_resolved_with_model(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(
            vertex_batch_scalar,
            field_path="category",
            response_schema=SingleCategoryModel,
        )
        assert result["top_alternative"] == "sport"
        assert result["top_alternative_resolved"] == "sports"

    def test_response_schema_auto_detects_field(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, response_schema=SingleCategoryModel)
        assert result["value"] == "health and wellness"
        assert result["top_alternative_resolved"] == "sports"

    def test_json_schema_resolves_top_alternative(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(
            vertex_batch_scalar,
            field_path="category",
            response_schema=CLASSIFICATION_JSON_SCHEMA,
        )
        assert result["top_alternative_resolved"] == "sports"

    def test_error_on_bad_input(self):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence({"garbage": True}, field_path="category")
        assert result["value"] is None
        assert result["error"] is not None

    def test_error_on_missing_field_path(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field_path="nonexistent")
        assert result["value"] is None
        assert result["error"] == "no values found"

    def test_field_path_returns_first_nested_match(self, nested_vertex_batch_response):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(
            nested_vertex_batch_response,
            field_path="classifications[].name",
            response_schema=NestedClassificationModel,
        )
        assert result["value"] == "Positive"
        assert result["path"] == "classifications[0].name"
        assert result["top_alternative_resolved"] == "Negative"


class TestAddConfidenceColumns:

    def test_vertex_batch(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([
            {"id": "req-1", "response": vertex_batch_scalar},
            {"id": "req-2", "response": vertex_batch_scalar},
        ])
        result = add_confidence_columns(df, response_column="response", field_path="category")

        assert "confidence_value" in result.columns
        assert "confidence_path" in result.columns
        assert "confidence_prob" in result.columns
        assert "confidence_joint_prob" in result.columns
        assert "confidence_top_alt" in result.columns
        assert "confidence_top_alt_resolved" in result.columns
        assert "confidence_top_alt_prob" in result.columns
        assert "confidence_error" in result.columns
        assert list(result["confidence_value"]) == ["health and wellness", "health and wellness"]
        assert list(result["confidence_path"]) == ["category", "category"]
        assert result["confidence_top_alt_resolved"].isna().all()
        assert all(result["confidence_error"].isna())

    def test_custom_prefix(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": vertex_batch_scalar}])
        result = add_confidence_columns(df, response_column="response", field_path="category", prefix="conf")
        assert "conf_value" in result.columns
        assert "conf_path" in result.columns
        assert "conf_prob" in result.columns

    def test_response_schema_populates_resolved_top_alt_column(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": vertex_batch_scalar}])
        result = add_confidence_columns(
            df,
            response_column="response",
            field_path="category",
            response_schema=SingleCategoryModel,
        )
        assert list(result["confidence_top_alt_resolved"]) == ["sports"]

    def test_does_not_mutate_original(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": vertex_batch_scalar}])
        original_cols = list(df.columns)
        add_confidence_columns(df, response_column="response", field_path="category")
        assert list(df.columns) == original_cols

    def test_field_path_adds_resolved_path_column(self, nested_vertex_batch_response):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": nested_vertex_batch_response}])
        result = add_confidence_columns(
            df,
            response_column="response",
            field_path="classifications[].name",
            response_schema=NestedClassificationModel,
        )
        assert list(result["confidence_value"]) == ["Positive"]
        assert list(result["confidence_path"]) == ["classifications[0].name"]
        assert list(result["confidence_top_alt_resolved"]) == ["Negative"]
