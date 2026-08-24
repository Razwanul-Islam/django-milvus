"""Tests for MILVUS_CACHE parsing and validation."""

import pytest
from django.test import override_settings

from django_milvus.cache.config import (
    DEFAULTS,
    deep_merge,
    is_configured,
    load_config,
    parse_size,
    validate_all,
)
from django_milvus.exceptions import CacheConfigurationError

from .conftest import cache_settings


class TestParseSize:
    @pytest.mark.parametrize("value,expected", [
        ("1024", 1024),
        ("1KB", 1024),
        ("1 KB", 1024),
        ("256MB", 268435456),
        ("1.5GB", 1610612736),
        ("2GiB", 2147483648),
        ("512B", 512),
        (4096, 4096),
        (None, None),
    ])
    def test_parses(self, value, expected):
        assert parse_size(value) == expected

    def test_is_case_insensitive(self):
        assert parse_size("256mb") == parse_size("256MB")

    @pytest.mark.parametrize("value", ["banana", "10 QB", "", True])
    def test_rejects_nonsense(self, value):
        with pytest.raises(CacheConfigurationError):
            parse_size(value, name="TEST")

    def test_error_names_the_setting(self):
        with pytest.raises(CacheConfigurationError) as info:
            parse_size("banana", name="L1.MAX_MEMORY")
        assert "L1.MAX_MEMORY" in str(info.value)

    def test_rejects_negative(self):
        with pytest.raises(CacheConfigurationError):
            parse_size(-5)


class TestDeepMerge:
    def test_overrides_leaves_only(self):
        merged = deep_merge(
            {"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 99}}
        )
        assert merged == {"a": 1, "b": {"c": 99, "d": 3}}

    def test_does_not_mutate_the_base(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_empty_override_returns_a_copy(self):
        base = {"a": 1}
        assert deep_merge(base, None) == base
        assert deep_merge(base, None) is not base


class TestLoadConfig:
    def test_unconfigured_project(self):
        with override_settings(MILVUS_CACHE=None):
            assert is_configured() is False

    def test_defaults_apply_to_omitted_keys(self):
        with override_settings(MILVUS_CACHE={"default": {}}):
            config = load_config("default")
        assert config.ttl == DEFAULTS["TTL"]
        assert config.l1.algorithm == "w-tinylfu"
        assert config.l1.max_memory == 268435456
        assert config.l2 is None

    def test_user_values_win(self):
        with override_settings(MILVUS_CACHE=cache_settings(TTL=42)):
            config = load_config("default")
        assert config.ttl == 42
        # ...without wiping the sibling defaults
        assert config.negative_ttl == DEFAULTS["NEGATIVE_TTL"]

    def test_unknown_alias_is_loud(self):
        with override_settings(MILVUS_CACHE=cache_settings()):
            with pytest.raises(CacheConfigurationError) as info:
                load_config("typo")
        assert "typo" in str(info.value)
        assert "default" in str(info.value)

    def test_l2_gets_its_own_defaults(self):
        with override_settings(MILVUS_CACHE=cache_settings(
            L2={"BACKEND": "django_milvus.cache.backends.djangocache."
                           "DjangoCacheBackend"}
        )):
            config = load_config("default")
        assert config.l2 is not None
        assert config.l2.serializer == "pickle"
        assert config.l2.breaker_failures == 5

    def test_validate_all_covers_every_alias(self):
        settings_dict = cache_settings()
        settings_dict["second"] = {"TTL": 60}
        with override_settings(MILVUS_CACHE=settings_dict):
            configs = validate_all()
        assert set(configs) == {"default", "second"}


class TestValidation:
    """Bad configuration must fail at start-up, not at the first query."""

    def _load(self, **overrides):
        with override_settings(MILVUS_CACHE=cache_settings(**overrides)):
            return load_config("default")

    def test_rejects_unknown_algorithm(self):
        with pytest.raises(CacheConfigurationError) as info:
            self._load(L1={"MAX_ENTRIES": 100, "ALGORITHM": "psychic"})
        assert "psychic" in str(info.value)

    def test_rejects_an_unbounded_cache(self):
        with pytest.raises(CacheConfigurationError) as info:
            self._load(L1={"MAX_MEMORY": None, "MAX_ENTRIES": None})
        assert "unbounded" in str(info.value).lower()

    def test_rejects_inverted_watermarks(self):
        with pytest.raises(CacheConfigurationError) as info:
            self._load(L1={"MAX_ENTRIES": 100,
                           "WATERMARK": {"high": 0.5, "low": 0.9}})
        assert "low" in str(info.value)

    def test_rejects_out_of_range_ratios(self):
        with pytest.raises(CacheConfigurationError):
            self._load(L1={"MAX_ENTRIES": 100,
                           "WINDOW": {"admission_ratio": 5.0}})

    def test_rejects_bad_similarity_threshold(self):
        with pytest.raises(CacheConfigurationError) as info:
            self._load(SEMANTIC={"metric": "COSINE", "threshold": 4.0})
        assert "threshold" in str(info.value)

    def test_allows_large_l2_thresholds(self):
        # An L2 threshold is a squared distance, not a similarity, so it
        # is legitimately unbounded.
        config = self._load(SEMANTIC={"metric": "L2", "threshold": 12.0})
        assert config.semantic.threshold == 12.0

    def test_rejects_unknown_metric(self):
        with pytest.raises(CacheConfigurationError):
            self._load(SEMANTIC={"metric": "MANHATTAN"})

    def test_rejects_unknown_serializer(self):
        with pytest.raises(CacheConfigurationError):
            self._load(L2={"BACKEND": "x.Y", "SERIALIZER": "yaml"})

    def test_rejects_negative_max_entries(self):
        with pytest.raises(CacheConfigurationError):
            self._load(L1={"MAX_ENTRIES": -5})

    def test_rejects_non_dict_alias(self):
        with override_settings(MILVUS_CACHE={"default": "yes please"}):
            with pytest.raises(CacheConfigurationError):
                load_config("default")

    def test_rejects_non_dict_setting(self):
        with override_settings(MILVUS_CACHE=["default"]):
            with pytest.raises(CacheConfigurationError):
                load_config("default")


class TestSystemCheck:
    """Misconfiguration surfaces through `manage.py check`."""

    def test_clean_config_reports_nothing(self):
        from django_milvus.apps import check_cache_settings
        with override_settings(MILVUS_CACHE=cache_settings()):
            assert check_cache_settings(None) == []

    def test_broken_config_is_reported(self):
        from django_milvus.apps import check_cache_settings
        with override_settings(MILVUS_CACHE=cache_settings(
            L1={"ALGORITHM": "psychic", "MAX_ENTRIES": 10}
        )):
            errors = check_cache_settings(None)
        assert len(errors) == 1
        assert errors[0].id == "django_milvus.E001"

    def test_unconfigured_reports_nothing(self):
        from django_milvus.apps import check_cache_settings
        with override_settings(MILVUS_CACHE=None):
            assert check_cache_settings(None) == []


class TestSemanticOptions:
    def test_replace_returns_a_new_config(self):
        with override_settings(MILVUS_CACHE=cache_settings()):
            config = load_config("default")
        tightened = config.semantic.replace(threshold=0.99)
        assert tightened.threshold == 0.99
        assert config.semantic.threshold == 0.95

    def test_repr_summarises_the_topology(self):
        with override_settings(MILVUS_CACHE=cache_settings()):
            config = load_config("default")
        assert "L1" in repr(config)
        assert "w-tinylfu" in repr(config)
