"""Tests for django_milvus utils."""

import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.settings")
django.setup()

import pytest
from django_milvus.utils import build_filter_expr, _format_value


class TestFormatValue:
    def test_string(self):
        assert _format_value("hello") == '"hello"'

    def test_string_with_quotes(self):
        assert _format_value('say "hi"') == '"say \\"hi\\""'

    def test_int(self):
        assert _format_value(42) == "42"

    def test_float(self):
        assert _format_value(3.14) == "3.14"

    def test_bool_true(self):
        assert _format_value(True) == "true"

    def test_bool_false(self):
        assert _format_value(False) == "false"

    def test_none(self):
        assert _format_value(None) == "null"


class TestBuildFilterExpr:
    def test_empty(self):
        assert build_filter_expr({}) == ""

    def test_equality(self):
        result = build_filter_expr({"name": "test"})
        assert result == 'name == "test"'

    def test_gt(self):
        result = build_filter_expr({"score__gt": 0.5})
        assert result == "score > 0.5"

    def test_gte(self):
        result = build_filter_expr({"score__gte": 0.5})
        assert result == "score >= 0.5"

    def test_lt(self):
        result = build_filter_expr({"age__lt": 30})
        assert result == "age < 30"

    def test_lte(self):
        result = build_filter_expr({"age__lte": 30})
        assert result == "age <= 30"

    def test_ne(self):
        result = build_filter_expr({"status__ne": "deleted"})
        assert result == 'status != "deleted"'

    def test_in(self):
        result = build_filter_expr({"category__in": ["a", "b", "c"]})
        assert result == 'category in ["a", "b", "c"]'

    def test_nin(self):
        result = build_filter_expr({"category__nin": [1, 2]})
        assert result == "category not in [1, 2]"

    def test_like(self):
        result = build_filter_expr({"title__like": "hello%"})
        assert result == 'title like "hello%"'

    def test_exists_true(self):
        result = build_filter_expr({"field__exists": True})
        assert result == "exists field"

    def test_exists_false(self):
        result = build_filter_expr({"field__exists": False})
        assert result == "not exists field"

    def test_multiple_filters(self):
        result = build_filter_expr({
            "category": "test",
            "score__gt": 0.5,
        })
        assert 'category == "test"' in result
        assert "score > 0.5" in result
        assert " and " in result

    def test_json_contains(self):
        result = build_filter_expr({"tags__json_contains": "python"})
        assert result == 'json_contains(tags, "python")'

    def test_array_contains(self):
        result = build_filter_expr({"items__array_contains": 42})
        assert result == "array_contains(items, 42)"

    def test_unknown_op_treated_as_field(self):
        result = build_filter_expr({"some__nested__field": "value"})
        assert result == 'some__nested__field == "value"'

    def test_in_requires_list(self):
        with pytest.raises(ValueError):
            build_filter_expr({"field__in": "not a list"})
