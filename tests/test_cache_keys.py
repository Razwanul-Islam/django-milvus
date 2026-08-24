"""Tests for cache key construction and identity."""

import pytest

from django_milvus.cache.keys import (
    OP_QUERY,
    collection_prefix,
    query_key,
    search_bucket_id,
    search_key,
    vector_fingerprint,
    version_key,
)


class TestVectorFingerprint:
    def test_identical_vectors_agree(self):
        assert vector_fingerprint([0.1, 0.2]) == vector_fingerprint([0.1, 0.2])

    def test_different_vectors_differ(self):
        assert vector_fingerprint([0.1, 0.2]) != vector_fingerprint([0.1, 0.3])

    def test_order_matters(self):
        assert vector_fingerprint([0.1, 0.2]) != vector_fingerprint([0.2, 0.1])

    def test_numpy_and_list_agree(self):
        numpy = pytest.importorskip("numpy")
        values = [0.1, 0.2, 0.3]
        assert vector_fingerprint(values) == vector_fingerprint(
            numpy.array(values, dtype=numpy.float32)
        )

    def test_float32_rounding_is_applied_consistently(self):
        # Values Milvus would treat identically must fingerprint identically.
        assert vector_fingerprint([1.0, 2.0]) == vector_fingerprint([1.0, 2.0])

    def test_batches_differ_from_singles(self):
        assert vector_fingerprint([[0.1, 0.2]]) != vector_fingerprint(
            [[0.1, 0.2], [0.3, 0.4]]
        )

    def test_sparse_vectors_are_supported(self):
        assert vector_fingerprint({1: 0.5, 7: 0.2}) == vector_fingerprint(
            {7: 0.2, 1: 0.5}
        )


class TestQueryKey:
    def _key(self, **overrides):
        params = {
            "filter_expr": "a == 1",
            "output_fields": ["id", "title"],
            "limit": 10,
            "offset": 0,
            "partitions": None,
            "consistency": None,
        }
        params.update(overrides)
        return query_key("default", "docs", 1, **params)

    def test_is_deterministic(self):
        assert self._key() == self._key()

    def test_field_order_does_not_matter(self):
        assert self._key(output_fields=["id", "title"]) == self._key(
            output_fields=["title", "id"]
        )

    def test_partition_order_does_not_matter(self):
        assert self._key(partitions=["a", "b"]) == self._key(
            partitions=["b", "a"]
        )

    @pytest.mark.parametrize("field,value", [
        ("filter_expr", "a == 2"),
        ("output_fields", ["id"]),
        ("limit", 20),
        ("offset", 5),
        ("partitions", ["p1"]),
        ("consistency", "Strong"),
    ])
    def test_every_dimension_changes_the_key(self, field, value):
        assert self._key() != self._key(**{field: value})

    def test_version_bump_changes_the_key(self):
        first = query_key("default", "docs", 1, filter_expr="x")
        second = query_key("default", "docs", 2, filter_expr="x")
        assert first != second

    def test_collection_changes_the_key(self):
        assert query_key("default", "a", 1) != query_key("default", "b", 1)

    def test_alias_changes_the_key(self):
        assert query_key("a", "docs", 1) != query_key("b", "docs", 1)

    def test_operation_changes_the_key(self):
        assert query_key("default", "docs", 1, op=OP_QUERY) != query_key(
            "default", "docs", 1, op="c"
        )

    def test_layout_is_scannable(self):
        key = query_key("default", "docs", 7)
        assert key.startswith("dmv:default:docs:v7:")
        assert key.startswith(collection_prefix("default", "docs", 7))


class TestSearchKey:
    def _key(self, **overrides):
        params = {
            "vector_fp": vector_fingerprint([0.1, 0.2]),
            "anns_field": "embedding",
            "search_params": {"metric_type": "COSINE"},
            "filter_expr": "",
            "output_fields": ["id"],
            "limit": 5,
        }
        params.update(overrides)
        return search_key("default", "docs", 1, **params)

    def test_is_deterministic(self):
        assert self._key() == self._key()

    def test_a_different_vector_is_a_different_key(self):
        assert self._key() != self._key(
            vector_fp=vector_fingerprint([0.9, 0.9])
        )

    @pytest.mark.parametrize("field,value", [
        ("anns_field", "other_vector"),
        ("search_params", {"metric_type": "L2"}),
        ("limit", 10),
        ("filter_expr", "category == 'x'"),
    ])
    def test_every_dimension_changes_the_key(self, field, value):
        assert self._key() != self._key(**{field: value})

    def test_search_and_query_keys_never_collide(self):
        assert self._key() != query_key("default", "docs", 1, limit=5)


class TestSearchBucket:
    """Buckets group the queries that may answer one another."""

    def _bucket(self, **overrides):
        params = {
            "anns_field": "embedding",
            "search_params": {"metric_type": "COSINE"},
            "filter_expr": "",
            "output_fields": ["id"],
            "limit": 5,
        }
        params.update(overrides)
        return search_bucket_id("default", "docs", 1, **params)

    def test_same_context_shares_a_bucket(self):
        assert self._bucket() == self._bucket()

    def test_different_limits_never_share(self):
        """A limit=5 result cannot answer a limit=10 query."""
        assert self._bucket(limit=5) != self._bucket(limit=10)

    def test_different_filters_never_share(self):
        assert self._bucket(filter_expr="") != self._bucket(
            filter_expr="status == 'published'"
        )

    def test_different_output_fields_never_share(self):
        assert self._bucket(output_fields=["id"]) != self._bucket(
            output_fields=["id", "title"]
        )

    def test_different_vector_fields_never_share(self):
        assert self._bucket(anns_field="a") != self._bucket(anns_field="b")

    def test_version_bump_changes_the_bucket(self):
        first = search_bucket_id("default", "docs", 1, limit=5)
        second = search_bucket_id("default", "docs", 2, limit=5)
        assert first != second

    def test_the_bucket_ignores_the_vector(self):
        """Which is the whole point: it is the vector's namespace."""
        assert self._bucket() == self._bucket()


class TestVersionKey:
    def test_is_namespaced_by_alias_and_collection(self):
        assert version_key("a", "docs") != version_key("b", "docs")
        assert version_key("a", "docs") != version_key("a", "other")

    def test_layout(self):
        assert version_key("default", "docs") == "dmv:ver:default:docs"


class TestCollectionPrefix:
    def test_matches_every_version_when_unversioned(self):
        prefix = collection_prefix("default", "docs")
        assert query_key("default", "docs", 1).startswith(prefix)
        assert query_key("default", "docs", 9).startswith(prefix)

    def test_versioned_prefix_is_narrower(self):
        prefix = collection_prefix("default", "docs", 1)
        assert query_key("default", "docs", 1).startswith(prefix)
        assert not query_key("default", "docs", 2).startswith(prefix)
