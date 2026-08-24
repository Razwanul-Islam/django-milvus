"""
End-to-end tests: does caching actually reduce Milvus traffic?

These assert on ``client.calls`` rather than on returned rows. Checking
the rows proves nothing - an uncached query returns exactly the same
data. The only observable difference a cache makes is how many times
Milvus was reached.
"""

import pickle

import pytest
from django.test import override_settings

from django_milvus import fields, indexes
from django_milvus.cache import caches, get_cache, resolve_cache_options
from django_milvus.managers import (
    MilvusSearchResult,
    normalize_query_results,
    normalize_search_results,
)
from django_milvus.models import MilvusModel

from .conftest import cache_settings
from .fakes import FakeMilvusClient, hit, patch_client

np = pytest.importorskip("numpy")


class CachedDoc(MilvusModel):
    id = fields.PrimaryKeyField(auto_id=True)
    title = fields.VarCharField(max_length=256)
    embedding = fields.FloatVectorField(dim=32)

    class MilvusMeta:
        collection_name = "cached_docs"
        cache = {"ttl": 300}

    class MilvusIndexes:
        emb = indexes.HNSW(field="embedding", metric_type="COSINE")


class PlainDoc(MilvusModel):
    id = fields.PrimaryKeyField(auto_id=True)
    title = fields.VarCharField(max_length=256)
    embedding = fields.FloatVectorField(dim=32)

    class MilvusMeta:
        collection_name = "plain_docs"


@pytest.fixture
def rng():
    return np.random.default_rng(5)


def unit(rng, dim=32):
    vector = rng.normal(size=dim).astype(np.float32)
    return (vector / np.linalg.norm(vector)).tolist()


def at_similarity(base, target, rng):
    base = np.asarray(base, dtype=np.float32)
    other = np.asarray(unit(rng, len(base)), dtype=np.float32)
    other = other - float(other @ base) * base
    other = other / np.linalg.norm(other)
    mixed = target * base + np.sqrt(max(0.0, 1 - target ** 2)) * other
    return (mixed / np.linalg.norm(mixed)).tolist()


RESULTS = [[
    hit(1, 0.95, {"title": "alpha"}),
    hit(2, 0.90, {"title": "beta"}),
    hit(3, 0.85, {"title": "gamma"}),
]]

ROWS = [
    {"id": 1, "title": "alpha", "embedding": [0.1] * 32},
    {"id": 2, "title": "beta", "embedding": [0.2] * 32},
]


@pytest.fixture
def cached(request):
    """Enable caching, with per-test overrides via ``@pytest.mark.cache``."""
    marker = request.node.get_closest_marker("cache")
    overrides = marker.kwargs if marker else {}
    with override_settings(MILVUS_CACHE=cache_settings(**overrides)):
        caches.reset()
        yield
        caches.reset()


class TestSearchCaching:
    def test_repeated_search_hits_once(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            first = list(CachedDoc.objects.search(vector, limit=3))
            second = list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 1
        assert [r.id for r in first] == [r.id for r in second] == [1, 2, 3]

    def test_cached_results_rehydrate_into_models(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
            results = list(CachedDoc.objects.search(vector, limit=3))
        assert isinstance(results[0], MilvusSearchResult)
        assert isinstance(results[0].entity, CachedDoc)
        assert results[0].title == "alpha"
        assert results[0].distance == 0.95

    def test_a_different_vector_misses(self, cached, rng):
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(unit(rng), limit=3))
            list(CachedDoc.objects.search(unit(rng), limit=3))
        assert client.calls["search"] == 2

    def test_a_different_limit_misses(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=2))
        assert client.calls["search"] == 2

    def test_a_different_filter_misses(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3).filter(title="a"))
            list(CachedDoc.objects.search(vector, limit=3).filter(title="b"))
        assert client.calls["search"] == 2

    def test_the_same_filter_hits(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3).filter(title="a"))
            list(CachedDoc.objects.search(vector, limit=3).filter(title="a"))
        assert client.calls["search"] == 1


class TestOptIn:
    def test_uncached_without_configuration(self, rng):
        """With no MILVUS_CACHE, behaviour is exactly as before."""
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with override_settings(MILVUS_CACHE=None):
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(vector, limit=3))
                list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2

    def test_a_model_that_did_not_opt_in_is_uncached(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(PlainDoc, client):
            list(PlainDoc.objects.search(vector, limit=3))
            list(PlainDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2

    def test_explicit_cache_overrides_the_model_default(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(PlainDoc, client):
            list(PlainDoc.objects.search(vector, limit=3).cache())
            list(PlainDoc.objects.search(vector, limit=3).cache())
        assert client.calls["search"] == 1

    def test_no_cache_bypasses_an_opted_in_model(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3).no_cache())
            list(CachedDoc.objects.search(vector, limit=3).no_cache())
        assert client.calls["search"] == 2

    def test_disabled_alias_caches_nothing(self, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with override_settings(MILVUS_CACHE=cache_settings(ENABLED=False)):
            caches.reset()
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(vector, limit=3))
                list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2

    def test_resolution_precedence(self):
        assert resolve_cache_options(PlainDoc, None) is None
        assert resolve_cache_options(PlainDoc, {"enabled": True}) == {
            "alias": "default"
        }
        assert resolve_cache_options(CachedDoc, None)["ttl"] == 300
        assert resolve_cache_options(
            CachedDoc, {"enabled": True, "ttl": 10}
        )["ttl"] == 10
        assert resolve_cache_options(CachedDoc, {"enabled": False}) is None


class TestInvalidation:
    def _prime(self, client, vector):
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 1

    @pytest.mark.parametrize("operation", [
        "bulk_create", "upsert", "delete", "delete_by_ids", "insert_raw",
    ])
    def test_every_write_path_invalidates(self, cached, rng, operation):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        self._prime(client, vector)

        row = {"title": "new", "embedding": unit(rng)}
        actions = {
            "bulk_create": lambda: CachedDoc.objects.bulk_create(data=[row]),
            "upsert": lambda: CachedDoc.objects.upsert(data=[row]),
            "delete": lambda: CachedDoc.objects.delete(title="new"),
            "delete_by_ids": lambda: CachedDoc.objects.delete_by_ids([1]),
            "insert_raw": lambda: CachedDoc.objects.insert_raw([row]),
        }
        with patch_client(CachedDoc, client):
            actions[operation]()
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2, (
            f"{operation} must invalidate the cache"
        )

    def test_instance_save_invalidates(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        self._prime(client, vector)
        with patch_client(CachedDoc, client):
            CachedDoc(title="x", embedding=unit(rng)).save()
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2

    def test_instance_delete_invalidates(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        self._prime(client, vector)
        with patch_client(CachedDoc, client):
            instance = CachedDoc(id=1, title="x", embedding=unit(rng))
            instance.delete()
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2

    def test_the_version_advances(self, cached, rng):
        client = FakeMilvusClient(search_results=RESULTS)
        cache = get_cache("default")
        assert cache.version("cached_docs") == 0
        with patch_client(CachedDoc, client):
            CachedDoc.objects.bulk_create(data=[{"title": "a"}])
        assert cache.version("cached_docs") == 1

    def test_other_collections_are_unaffected(self, cached, rng):
        vector = unit(rng)
        cached_client = FakeMilvusClient(search_results=RESULTS)
        plain_client = FakeMilvusClient(search_results=RESULTS)

        with patch_client(PlainDoc, plain_client):
            list(PlainDoc.objects.search(vector, limit=3).cache())
        with patch_client(CachedDoc, cached_client):
            CachedDoc.objects.bulk_create(data=[{"title": "a"}])
        with patch_client(PlainDoc, plain_client):
            list(PlainDoc.objects.search(vector, limit=3).cache())

        assert plain_client.calls["search"] == 1, (
            "writing to one collection must not invalidate another"
        )

    def test_manual_cache_clear(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        self._prime(client, vector)
        CachedDoc.objects.cache_clear()
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2


class TestQueryCaching:
    def test_repeated_filter_hits_once(self, cached):
        client = FakeMilvusClient(query_results=ROWS)
        with patch_client(CachedDoc, client):
            first = list(CachedDoc.objects.filter(title="alpha"))
            second = list(CachedDoc.objects.filter(title="alpha"))
        assert client.calls["query"] == 1
        assert [d.id for d in first] == [d.id for d in second]

    def test_offset_and_limit_are_part_of_the_key(self, cached):
        client = FakeMilvusClient(query_results=ROWS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.filter(title="a").limit(5))
            list(CachedDoc.objects.filter(title="a").limit(5).offset(5))
        assert client.calls["query"] == 2

    def test_only_changes_the_key(self, cached):
        client = FakeMilvusClient(query_results=ROWS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.filter(title="a").only("id"))
            list(CachedDoc.objects.filter(title="a").only("id", "title"))
        assert client.calls["query"] == 2

    def test_count_is_cached(self, cached):
        client = FakeMilvusClient(count=42)
        with patch_client(CachedDoc, client):
            assert CachedDoc.objects.count() == 42
            assert CachedDoc.objects.count() == 42
        assert client.calls["query"] == 1

    def test_count_can_be_excluded(self, rng):
        client = FakeMilvusClient(count=42)
        with override_settings(MILVUS_CACHE=cache_settings(CACHE_COUNT=False)):
            caches.reset()
            with patch_client(CachedDoc, client):
                CachedDoc.objects.count()
                CachedDoc.objects.count()
        assert client.calls["query"] == 2

    def test_empty_results_are_cached(self, cached):
        """Negative caching: repeatedly asking for nothing costs one query."""
        client = FakeMilvusClient(query_results=[])
        with patch_client(CachedDoc, client):
            assert list(CachedDoc.objects.filter(title="absent")) == []
            assert list(CachedDoc.objects.filter(title="absent")) == []
        assert client.calls["query"] == 1


class TestConsistency:
    def test_strong_consistency_bypasses_by_default(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3).consistency("Strong"))
            list(CachedDoc.objects.search(vector, limit=3).consistency("Strong"))
        assert client.calls["search"] == 2

    def test_strong_consistency_can_be_allowed(self, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with override_settings(MILVUS_CACHE=cache_settings(
            CACHE_STRONG_CONSISTENCY=True
        )):
            caches.reset()
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(vector, limit=3)
                     .consistency("Strong"))
                list(CachedDoc.objects.search(vector, limit=3)
                     .consistency("Strong"))
        assert client.calls["search"] == 1

    def test_other_levels_still_cache(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3)
                 .consistency("Bounded"))
            list(CachedDoc.objects.search(vector, limit=3)
                 .consistency("Bounded"))
        assert client.calls["search"] == 1


class TestSemanticIntegration:
    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95,
                                 "overfetch": 3})
    def test_a_near_duplicate_vector_hits(self, cached, rng):
        base = unit(rng)
        near = at_similarity(base, 0.99, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(base, limit=3)
                 .cache(store_vectors=True))
            list(CachedDoc.objects.search(near, limit=3)
                 .cache(store_vectors=True))
        assert client.calls["search"] == 1
        assert get_cache("default").stats_dict()["semantic_hits"] == 1

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.999})
    def test_a_strict_threshold_misses(self, cached, rng):
        base = unit(rng)
        near = at_similarity(base, 0.99, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(base, limit=3).cache())
            list(CachedDoc.objects.search(near, limit=3).cache())
        assert client.calls["search"] == 2

    @pytest.mark.cache(SEMANTIC={"enabled": False})
    def test_disabling_semantic_falls_back_to_exact_matching(self, cached, rng):
        base = unit(rng)
        near = at_similarity(base, 0.999, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(base, limit=3).cache())
            list(CachedDoc.objects.search(near, limit=3).cache())
        assert client.calls["search"] == 2

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95})
    def test_per_query_threshold_override(self, cached, rng):
        base = unit(rng)
        near = at_similarity(base, 0.99, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(base, limit=3).cache(semantic=0.9999))
            list(CachedDoc.objects.search(near, limit=3).cache(semantic=0.9999))
        assert client.calls["search"] == 2, (
            "a stricter per-query threshold must be respected"
        )

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95,
                                 "overfetch": 3})
    def test_overfetch_widens_the_milvus_request(self, cached, rng):
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(unit(rng), limit=3).cache())
        limits = [kw["limit"] for name, kw in client.history if name == "search"]
        assert limits == [9], "3 * overfetch 3 should be requested"

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95,
                                 "overfetch": 3})
    def test_results_are_still_truncated_to_the_limit(self, cached, rng):
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            results = list(CachedDoc.objects.search(unit(rng), limit=2).cache())
        assert len(results) == 2, "over-fetching must not leak extra rows"

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95})
    def test_a_write_drops_the_semantic_index(self, cached, rng):
        base = unit(rng)
        near = at_similarity(base, 0.99, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(base, limit=3).cache())
            CachedDoc.objects.bulk_create(data=[{"title": "new"}])
            list(CachedDoc.objects.search(near, limit=3).cache())
        assert client.calls["search"] == 2

    @pytest.mark.cache(SEMANTIC={"enabled": True, "threshold": 0.95})
    def test_store_vectors_strips_embeddings_from_results(self, cached, rng):
        vector = unit(rng)
        results = [[hit(1, 0.9, {"title": "a", "embedding": [0.1] * 32})]]
        client = FakeMilvusClient(search_results=results)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=1)
                 .cache(store_vectors=True))
            cached_rows = list(CachedDoc.objects.search(vector, limit=1)
                               .cache(store_vectors=True))
        # The embedding was hoisted into the vector cache, not kept in
        # every cached result set.
        assert client.calls["search"] == 1
        assert cached_rows[0].title == "a"
        cache = get_cache("default")
        assert cache.vectors.for_field("cached_docs", "embedding") is not None


class TestCacheKeyIntrospection:
    def test_identical_queries_share_a_key(self, cached, rng):
        vector = unit(rng)
        first = CachedDoc.objects.search(vector, limit=3).cache_key()
        second = CachedDoc.objects.search(vector, limit=3).cache_key()
        assert first == second
        assert first.startswith("dmv:default:cached_docs:v0:s:")

    def test_uncached_queries_have_no_key(self, cached, rng):
        assert CachedDoc.objects.search(
            unit(rng), limit=3
        ).no_cache().cache_key() is None

    def test_no_key_without_configuration(self, rng):
        with override_settings(MILVUS_CACHE=None):
            assert CachedDoc.objects.filter(title="a").cache_key() is None


class TestRefreshAndWarm:
    def test_refresh_cache_repopulates(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=3).refresh_cache())
            assert client.calls["search"] == 2
            list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 2, "refresh must repopulate, not bypass"

    def test_warm_populates_the_cache(self, cached, rng):
        vectors = [unit(rng) for _ in range(3)]
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            result = CachedDoc.objects.cache_warm(vectors=vectors, limit=3)
            assert result.warmed == 3
            assert client.calls["search"] == 3
            for vector in vectors:
                list(CachedDoc.objects.search(vector, limit=3))
        assert client.calls["search"] == 3, "warmed vectors must all hit"

    def test_warm_collects_errors_rather_than_raising(self, cached, rng):
        client = FakeMilvusClient(raises=RuntimeError("down"))
        with patch_client(CachedDoc, client):
            result = CachedDoc.objects.cache_warm(vectors=[unit(rng)], limit=3)
        assert result.warmed == 0
        assert len(result.errors) == 1

    def test_warm_without_a_cache_is_a_no_op(self, rng):
        with override_settings(MILVUS_CACHE=None):
            result = CachedDoc.objects.cache_warm(vectors=[unit(rng)])
        assert result.warmed == 0


class TestFailureBehaviour:
    def test_query_errors_propagate(self, cached, rng):
        client = FakeMilvusClient(raises=RuntimeError("milvus exploded"))
        with pytest.raises(RuntimeError, match="exploded"):
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(unit(rng), limit=3))

    def test_a_broken_cache_backend_still_serves(self, cached, rng, monkeypatch):
        """Fail-open: a cache fault costs a round trip, not a request."""
        vector = unit(rng)
        cache = get_cache("default")

        def explode(*args, **kwargs):
            raise RuntimeError("cache is broken")

        monkeypatch.setattr(cache.backend, "get", explode)
        monkeypatch.setattr(cache.backend, "set", explode)

        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            results = list(CachedDoc.objects.search(vector, limit=3))
        assert [r.id for r in results] == [1, 2, 3]
        assert client.calls["search"] == 1


class TestStats:
    def test_stats_track_hits_and_misses(self, cached, rng):
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=3))
            list(CachedDoc.objects.search(vector, limit=3))
        stats = CachedDoc.objects.cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 2
        assert stats["hit_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["collection"] == "cached_docs"

    def test_stats_are_empty_without_configuration(self):
        with override_settings(MILVUS_CACHE=None):
            assert CachedDoc.objects.cache_stats() == {}

    def test_tier_counts_reconcile_with_total_hits(self, cached, rng):
        """l1_hits + l2_hits must equal hits, not double it."""
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            for _ in range(4):
                list(CachedDoc.objects.search(vector, limit=3))
        stats = CachedDoc.objects.cache_stats()
        assert stats["hits"] == 3
        assert stats["l1_hits"] + stats["l2_hits"] == stats["hits"]

    def test_tier_counts_reconcile_with_a_shared_tier(self, rng):
        """The double-count only ever showed up with an L2 configured."""
        vector = unit(rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with override_settings(MILVUS_CACHE=cache_settings(L2={
            "BACKEND": "django_milvus.cache.backends.djangocache."
                       "DjangoCacheBackend",
            "LOCATION": "default",
        })):
            caches.reset()
            with patch_client(CachedDoc, client):
                for _ in range(4):
                    list(CachedDoc.objects.search(vector, limit=3))
            stats = CachedDoc.objects.cache_stats()
        assert stats["hits"] == 3
        assert stats["l1_hits"] + stats["l2_hits"] == stats["hits"]
        assert stats["hit_rate"] <= 1.0

    def test_semantic_hits_are_counted_once(self, rng):
        vector = unit(rng)
        near = at_similarity(vector, 0.999, rng)
        client = FakeMilvusClient(search_results=RESULTS)
        with override_settings(MILVUS_CACHE=cache_settings(
            SEMANTIC={"enabled": True, "threshold": 0.95}
        )):
            caches.reset()
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(vector, limit=3))
                list(CachedDoc.objects.search(near, limit=3))
            stats = CachedDoc.objects.cache_stats()
        assert stats["hits"] == 1
        assert stats["semantic_hits"] == 1
        assert stats["hit_rate"] <= 1.0

    def test_prometheus_export(self, cached, rng):
        client = FakeMilvusClient(search_results=RESULTS)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(unit(rng), limit=3))
        output = caches.prometheus()
        assert "milvus_cache_hits_total" in output
        assert 'alias="default"' in output


class TestSignals:
    def test_hit_and_miss_signals_fire(self, cached, rng):
        from django_milvus.cache.signals import cache_hit, cache_miss

        hits, misses = [], []

        def on_hit(**kwargs):
            hits.append(kwargs)

        def on_miss(**kwargs):
            misses.append(kwargs)

        cache_hit.connect(on_hit)
        cache_miss.connect(on_miss)
        try:
            vector = unit(rng)
            client = FakeMilvusClient(search_results=RESULTS)
            with patch_client(CachedDoc, client):
                list(CachedDoc.objects.search(vector, limit=3))
                list(CachedDoc.objects.search(vector, limit=3))
        finally:
            cache_hit.disconnect(on_hit)
            cache_miss.disconnect(on_miss)

        assert len(misses) == 1
        assert len(hits) == 1
        assert hits[0]["collection"] == "cached_docs"
        assert hits[0]["sender"] is CachedDoc

    def test_a_broken_receiver_does_not_break_the_query(self, cached, rng):
        from django_milvus.cache.signals import cache_miss

        def explode(**kwargs):
            raise RuntimeError("receiver bug")

        cache_miss.connect(explode)
        try:
            client = FakeMilvusClient(search_results=RESULTS)
            with patch_client(CachedDoc, client):
                results = list(CachedDoc.objects.search(unit(rng), limit=3))
            assert len(results) == 3
        finally:
            cache_miss.disconnect(explode)


class TestNormalization:
    def test_search_results_keep_their_grouping(self):
        raw = [[{"id": 1, "distance": 0.9, "entity": {"title": "a"}}]]
        assert normalize_search_results(raw) == raw

    def test_entities_become_plain_dicts(self):
        normalized = normalize_search_results(
            [[{"id": 1, "distance": 0.5, "entity": {"a": 1}}]]
        )
        assert type(normalized[0][0]["entity"]) is dict

    def test_handles_empty_input(self):
        assert normalize_search_results(None) == []
        assert normalize_search_results([]) == []
        assert normalize_query_results(None) == []

    def test_query_rows_become_plain_dicts(self):
        rows = normalize_query_results([{"id": 1}])
        assert rows == [{"id": 1}]
        assert type(rows[0]) is dict

    def test_normalized_payloads_survive_pickling(self):
        payload = normalize_search_results(
            [[{"id": 1, "distance": 0.9, "entity": {"title": "a"}}]]
        )
        assert pickle.loads(pickle.dumps(payload)) == payload


class TestSearchResultPickling:
    def test_round_trips(self):
        """__getattr__ used to recurse forever during unpickling."""
        result = MilvusSearchResult(
            entity=CachedDoc(title="p", embedding=[0.1] * 32),
            distance=0.5, id=9,
        )
        restored = pickle.loads(pickle.dumps(result))
        assert restored.id == 9
        assert restored.distance == 0.5
        assert restored.title == "p"

    def test_unknown_attributes_still_raise(self):
        result = MilvusSearchResult(
            entity=CachedDoc(title="p", embedding=[0.1] * 32), id=1
        )
        with pytest.raises(AttributeError):
            result.definitely_not_a_field


class TestLoadState:
    def test_indexes_are_built_once(self, rng):
        from django_milvus.cache.loadstate import load_state

        load_state.clear()
        client = FakeMilvusClient(search_results=RESULTS, loaded=False)
        with patch_client(CachedDoc, client):
            list(CachedDoc.objects.search(unit(rng), limit=3).no_cache())
            client.loaded = False
            list(CachedDoc.objects.search(unit(rng), limit=3).no_cache())
        assert client.index_calls == 1
        assert client.load_calls == 2

    def test_release_forgets_the_load_state(self):
        from django_milvus.cache.loadstate import load_state

        load_state.mark_loaded("cached_docs", "default")
        assert load_state.is_loaded("cached_docs", "default")
        client = FakeMilvusClient()
        with patch_client(CachedDoc, client):
            CachedDoc.objects.release_collection()
        assert not load_state.is_loaded("cached_docs", "default")

    def test_state_expires(self):
        from django_milvus.cache.loadstate import LoadStateCache

        cache = LoadStateCache(ttl=0.05)
        cache.mark_loaded("docs")
        assert cache.is_loaded("docs")
        import time
        time.sleep(0.08)
        assert not cache.is_loaded("docs")
