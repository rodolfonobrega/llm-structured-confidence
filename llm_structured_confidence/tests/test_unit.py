"""Comprehensive unit tests for llm-structured-confidence.

All tests use mock token data — no API calls needed.
"""

from __future__ import annotations

import math

import pytest

from llm_structured_confidence import extract_field_logprobs, FieldLogprob, TokenInfo, TopAlternative
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


# ═══════════════════════════════════════════════════════════════════════
# 1. FieldLogprob.compute — metric calculations
# ═══════════════════════════════════════════════════════════════════════

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
# 3b. Batch dict — extract_field_logprobs end-to-end with Vertex dicts
# ═══════════════════════════════════════════════════════════════════════

class TestVertexBatchDict:

    def test_scalar_extraction(self, vertex_batch_scalar):
        result = extract_field_logprobs(vertex_batch_scalar, field="category")
        assert "health and wellness" in result
        fl = result["health and wellness"]
        assert fl.value == "health and wellness"
        assert len(fl.tokens) == 3

    def test_scalar_matches_sdk(self, genai_scalar, vertex_batch_scalar):
        result_sdk = extract_field_logprobs(genai_scalar, field="category")
        result_dict = extract_field_logprobs(vertex_batch_scalar, field="category")
        fl_sdk = result_sdk["health and wellness"]
        fl_dict = result_dict["health and wellness"]
        assert fl_sdk.joint_probability == pytest.approx(fl_dict.joint_probability)
        assert fl_sdk.mean_probability == pytest.approx(fl_dict.mean_probability)
        assert fl_sdk.mean_nonzero_probability == pytest.approx(fl_dict.mean_nonzero_probability)

    def test_array_extraction(self, vertex_batch_array):
        result = extract_field_logprobs(vertex_batch_array, field="categories")
        assert "health and wellness" in result
        assert "sports" in result

    def test_pydantic_model_detection(self, vertex_batch_scalar):
        result = extract_field_logprobs(vertex_batch_scalar, model=SingleCategoryModel)
        assert "health and wellness" in result


# ═══════════════════════════════════════════════════════════════════════
# 4. Integration — extract_field_logprobs end-to-end with mocks
# ═══════════════════════════════════════════════════════════════════════

class TestExtractScalar:

    def test_gemini25_category(self, gemini25_scalar):
        result = extract_field_logprobs(gemini25_scalar, field="category")
        assert "health and wellness" in result
        fl = result["health and wellness"]
        assert fl.value == "health and wellness"
        assert len(fl.tokens) == 3
        assert fl.tokens[0].token == "health"
        assert fl.tokens[1].token == " and"
        assert fl.tokens[2].token == " wellness"

    def test_gemini25_excludes_quotes_and_colon(self, gemini25_scalar):
        """The tokens '":' and ' "' must NOT appear in the result."""
        result = extract_field_logprobs(gemini25_scalar, field="category")
        fl = result["health and wellness"]
        token_texts = [t.token for t in fl.tokens]
        assert '":'  not in token_texts
        assert ' "'  not in token_texts
        assert '{"'  not in token_texts
        assert '"}' not in token_texts

    def test_gemini3_excludes_merged_colon_token(self, gemini3_scalar):
        """Gemini 3 merges '":"' into one token — it must be excluded."""
        result = extract_field_logprobs(gemini3_scalar, field="category")
        fl = result["health and wellness"]
        token_texts = [t.token for t in fl.tokens]
        assert '":"' not in token_texts
        assert fl.tokens[0].token == "health"

    def test_gemini3_metrics(self, gemini3_scalar):
        result = extract_field_logprobs(gemini3_scalar, field="category")
        fl = result["health and wellness"]
        assert fl.joint_logprob == pytest.approx(-0.168335)
        assert fl.joint_probability == pytest.approx(math.exp(-0.168335))
        assert fl.mean_logprob == pytest.approx(-0.168335 / 3)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.168335)

    def test_gemini25_metrics(self, gemini25_scalar):
        result = extract_field_logprobs(gemini25_scalar, field="category")
        fl = result["health and wellness"]
        assert fl.joint_logprob == pytest.approx(-0.012331)
        assert fl.mean_nonzero_logprob == pytest.approx(-0.012331)

    def test_top_logprobs_from_first_significant(self, gemini3_scalar):
        result = extract_field_logprobs(gemini3_scalar, field="category")
        fl = result["health and wellness"]
        assert len(fl.top_logprobs) == 3
        assert fl.top_logprobs[0].token == "health"
        assert fl.top_logprobs[1].token == "sport"


class TestExtractArray:

    def test_array_returns_dict_by_value(self, array_response):
        result = extract_field_logprobs(array_response, field="categories")
        assert "health and wellness" in result
        assert "sports" in result

    def test_array_each_element_has_metrics(self, array_response):
        result = extract_field_logprobs(array_response, field="categories")
        bars = result["health and wellness"]
        groc = result["sports"]
        assert bars.joint_logprob == pytest.approx(-0.168)
        assert groc.joint_logprob == pytest.approx(-0.050)

    def test_array_element_tokens(self, array_response):
        result = extract_field_logprobs(array_response, field="categories")
        bars = result["health and wellness"]
        assert bars.tokens[0].token == "health"
        assert bars.tokens[1].token == " and"
        assert bars.tokens[2].token == " wellness"

    def test_array_excludes_delimiters(self, array_response):
        result = extract_field_logprobs(array_response, field="categories")
        for fl in result.values():
            token_texts = [t.token for t in fl.tokens]
            assert '":["' not in token_texts
            assert '","' not in token_texts
            assert '"]}' not in token_texts

    def test_array_top_logprobs_per_element(self, array_response):
        result = extract_field_logprobs(array_response, field="categories")
        bars = result["health and wellness"]
        groc = result["sports"]
        assert bars.top_logprobs[0].token == "health"
        assert groc.top_logprobs[0].token == "sport"


class TestFieldSelection:

    def test_explicit_field(self, multi_field_response):
        result = extract_field_logprobs(multi_field_response, field="category")
        assert "technology" in result
        assert len(result) == 1

    def test_no_field_returns_all(self, multi_field_response):
        result = extract_field_logprobs(multi_field_response)
        assert "technology" in result
        assert "150" in result
        assert len(result) == 2

    def test_nonexistent_field_returns_empty(self, multi_field_response):
        result = extract_field_logprobs(multi_field_response, field="nonexistent")
        assert result == {}


class TestPydanticDetection:

    def test_single_enum(self, gemini3_scalar):
        result = extract_field_logprobs(gemini3_scalar, model=SingleCategoryModel)
        assert "health and wellness" in result

    def test_list_enum(self, array_response):
        result = extract_field_logprobs(array_response, model=MultipleCategoriesModel)
        assert "health and wellness" in result
        assert "sports" in result

    def test_literal_field(self):
        content = '{"mood":"happy"}'
        tokens = [('{"', -0.01), ("mood", 0.0), ('":"', -0.10), ("happy", -0.05), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, model=LiteralModel)
        assert "happy" in result

    def test_mixed_model_only_enum(self):
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
        result = extract_field_logprobs(resp, model=MixedModel)
        assert "technology" in result
        assert len(result) == 1

    def test_field_overrides_model(self, multi_field_response):
        result = extract_field_logprobs(multi_field_response, field="amount", model=SingleCategoryModel)
        assert "150" in result
        assert "technology" not in result


class TestGoogleGenaiInput:

    def test_genai_scalar(self, genai_scalar):
        result = extract_field_logprobs(genai_scalar, field="category")
        assert "health and wellness" in result
        fl = result["health and wellness"]
        assert fl.tokens[0].token == "health"
        assert fl.joint_logprob == pytest.approx(-0.168335)

    def test_genai_metrics_match_openai(self, gemini3_scalar, genai_scalar):
        """Same token data via different providers must yield same metrics."""
        r_oai = extract_field_logprobs(gemini3_scalar, field="category")
        r_gen = extract_field_logprobs(genai_scalar, field="category")

        fl_oai = r_oai["health and wellness"]
        fl_gen = r_gen["health and wellness"]

        assert fl_oai.joint_logprob == pytest.approx(fl_gen.joint_logprob)
        assert fl_oai.mean_logprob == pytest.approx(fl_gen.mean_logprob)
        assert fl_oai.mean_nonzero_logprob == pytest.approx(fl_gen.mean_nonzero_logprob)

    def test_genai_array(self):
        from .conftest import make_genai_response, ARRAY_CONTENT, ARRAY_TOKENS
        resp = make_genai_response(ARRAY_CONTENT, ARRAY_TOKENS)
        result = extract_field_logprobs(resp, field="categories")
        assert "health and wellness" in result
        assert "sports" in result


class TestEdgeCases:

    def test_single_token_value(self):
        content = '{"k":"v"}'
        tokens = [('{"', -0.01), ("k", 0.0), ('":"', -0.10), ("v", -0.05), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, field="k")
        assert "v" in result
        fl = result["v"]
        assert len(fl.tokens) == 1
        assert fl.tokens[0].token == "v"

    def test_empty_string_value(self):
        content = '{"k":""}'
        tokens = [('{"', -0.01), ("k", 0.0), ('":"', -0.10), ('"}', 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, field="k")
        assert "" in result
        fl = result[""]
        assert len(fl.tokens) == 0
        assert fl.joint_probability == 1.0

    def test_boolean_value(self):
        content = '{"flag":true}'
        tokens = [('{"', -0.01), ("flag", 0.0), ('":', -0.01), ("true", -0.02), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, field="flag")
        assert "True" in result
        assert result["True"].value is True
        assert result["True"].joint_logprob == pytest.approx(-0.02)

    def test_number_value(self):
        content = '{"count":42}'
        tokens = [('{"', -0.01), ("count", 0.0), ('":', -0.01), ("42", -0.05), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, field="count")
        assert "42" in result
        assert result["42"].value == 42

    def test_null_value(self):
        content = '{"x":null}'
        tokens = [('{"', -0.01), ("x", 0.0), ('":', -0.01), ("null", -0.03), ("}", 0.0)]
        resp = make_openai_response(content, tokens)
        result = extract_field_logprobs(resp, field="x")
        assert "None" in result
        assert result["None"].value is None

    def test_probability_property(self):
        ti = TokenInfo(token="test", logprob=-0.5, char_start=0, char_end=4)
        assert ti.probability == pytest.approx(math.exp(-0.5))

    def test_top_alternative_probability(self):
        alt = TopAlternative(token="x", logprob=-1.0)
        assert alt.probability == pytest.approx(math.exp(-1.0))


# ═══════════════════════════════════════════════════════════════════════
# 8. Pandas integration — extract_confidence, add_confidence_columns
# ═══════════════════════════════════════════════════════════════════════

class TestExtractConfidence:

    def test_returns_flat_dict(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field="category")
        assert result["value"] == "health and wellness"
        assert result["error"] is None
        assert isinstance(result["mean_nonzero_probability"], float)
        assert isinstance(result["joint_probability"], float)

    def test_top_alternative(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field="category")
        assert result["top_alternative"] == "sport"
        assert isinstance(result["top_alternative_probability"], float)

    def test_error_on_bad_input(self):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence({"garbage": True}, field="category")
        assert result["value"] is None
        assert result["error"] is not None

    def test_error_on_missing_field(self, vertex_batch_scalar):
        from llm_structured_confidence import extract_confidence
        result = extract_confidence(vertex_batch_scalar, field="nonexistent")
        assert result["value"] is None
        assert result["error"] == "no values found"


class TestAddConfidenceColumns:

    def test_vertex_batch(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([
            {"id": "req-1", "response": vertex_batch_scalar},
            {"id": "req-2", "response": vertex_batch_scalar},
        ])
        result = add_confidence_columns(df, response_column="response", field="category")

        assert "confidence_value" in result.columns
        assert "confidence_prob" in result.columns
        assert "confidence_joint_prob" in result.columns
        assert "confidence_top_alt" in result.columns
        assert "confidence_top_alt_prob" in result.columns
        assert "confidence_error" in result.columns
        assert list(result["confidence_value"]) == ["health and wellness", "health and wellness"]
        assert all(result["confidence_error"].isna())

    def test_custom_prefix(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": vertex_batch_scalar}])
        result = add_confidence_columns(df, response_column="response", field="category", prefix="conf")
        assert "conf_value" in result.columns
        assert "conf_prob" in result.columns

    def test_does_not_mutate_original(self, vertex_batch_scalar):
        import pandas as pd
        from llm_structured_confidence import add_confidence_columns

        df = pd.DataFrame([{"response": vertex_batch_scalar}])
        original_cols = list(df.columns)
        add_confidence_columns(df, response_column="response", field="category")
        assert list(df.columns) == original_cols
