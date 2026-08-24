"""Tests for the semantic (nearest-vector) cache, vector cache and versions."""

import pytest
from django.test import override_settings

from django_milvus.cache.config import load_config
from django_milvus.cache.semantic import (
    SemanticCache,
    SemanticIndex,
    create_index,
)
from django_milvus.cache.vectors import VectorCache, VectorCacheRegistry
from django_milvus.cache.versions import VersionRegistry

from .conftest import cache_settings

np = pytest.importorskip("numpy")


@pytest.fixture
def rng():
    return np.random.default_rng(17)


def unit(rng, dim=64):
    vector = rng.normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def at_similarity(base, target, rng):
    """Build a unit vector at exactly ``target`` cosine similarity to base."""
    other = unit(rng, len(base))
    other = other - float(other @ base) * base
    other = other / np.linalg.norm(other)
    mixed = target * base + np.sqrt(max(0.0, 1 - target ** 2)) * other
    return mixed / np.linalg.norm(mixed)


class TestSemanticIndex:
    def test_exact_vector_hits(self, rng):
        index = SemanticIndex(dim=64, capacity=50)
        vector = unit(rng)
        index.add("k", vector)
        key, score = index.probe(vector, 0.99)
        assert key == "k"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_empty_index_misses(self, rng):
        assert SemanticIndex(dim=64).probe(unit(rng), 0.9) == (None, 0.0)

    def test_near_neighbour_hits_below_the_threshold(self, rng):
        index = SemanticIndex(dim=64, capacity=50)
        base = unit(rng)
        index.add("base", base)
        near = at_similarity(base, 0.99, rng)
        key, score = index.probe(near, 0.97)
        assert key == "base"
        assert score == pytest.approx(0.99, abs=0.01)

    def test_near_neighbour_misses_above_the_threshold(self, rng):
        index = SemanticIndex(dim=64, capacity=50)
        base = unit(rng)
        index.add("base", base)
        key, score = index.probe(at_similarity(base, 0.99, rng), 0.995)
        assert key is None
        assert score == pytest.approx(0.99, abs=0.01)

    def test_returns_the_nearest_of_many(self, rng):
        index = SemanticIndex(dim=64, capacity=100)
        base = unit(rng)
        index.add("closest", base)
        for i in range(50):
            index.add(f"noise{i}", unit(rng))
        key, _ = index.probe(at_similarity(base, 0.995, rng), 0.9)
        assert key == "closest"

    def test_removed_entries_stop_answering(self, rng):
        index = SemanticIndex(dim=64, capacity=50)
        vector = unit(rng)
        index.add("k", vector)
        assert index.remove("k") is True
        assert index.probe(vector, 0.5)[0] is None
        assert index.remove("k") is False

    def test_removal_keeps_rows_packed(self, rng):
        """The probe reads a contiguous view; holes would corrupt it."""
        index = SemanticIndex(dim=16, capacity=20)
        vectors = {}
        for i in range(10):
            vectors[f"k{i}"] = unit(rng, 16)
            index.add(f"k{i}", vectors[f"k{i}"])
        index.remove("k3")
        index.remove("k0")
        assert len(index) == 8
        for row in range(8):
            assert index._keys[row] is not None
            assert index._rows[index._keys[row]] == row
        # Survivors must still be findable.
        for key in ["k1", "k5", "k9"]:
            assert index.probe(vectors[key], 0.99)[0] == key

    def test_capacity_evicts_least_recently_used(self, rng):
        index = SemanticIndex(dim=16, capacity=5)
        for i in range(20):
            index.add(f"k{i}", unit(rng, 16))
        assert len(index) == 5
        assert index.evictions == 15

    def test_rejects_wrong_dimensionality(self, rng):
        index = SemanticIndex(dim=64, capacity=10)
        assert index.add("bad", [0.1, 0.2]) is False
        assert len(index) == 0

    def test_readding_a_key_reuses_its_row(self, rng):
        index = SemanticIndex(dim=16, capacity=10)
        index.add("k", unit(rng, 16))
        replacement = unit(rng, 16)
        index.add("k", replacement)
        assert len(index) == 1
        assert index.probe(replacement, 0.99)[0] == "k"

    def test_l2_metric_uses_distance_not_similarity(self):
        index = SemanticIndex(dim=4, capacity=10, metric="L2")
        index.add("origin", np.array([1, 0, 0, 0], dtype=np.float32))
        near = np.array([1.01, 0, 0, 0], dtype=np.float32)
        assert index.probe(near, 0.01)[0] == "origin"
        far = np.array([9, 9, 0, 0], dtype=np.float32)
        assert index.probe(far, 0.01)[0] is None

    def test_zero_vectors_are_rejected_for_cosine(self):
        index = SemanticIndex(dim=4, capacity=10, metric="COSINE")
        assert index.add("zero", np.zeros(4, dtype=np.float32)) is False

    def test_clear(self, rng):
        index = SemanticIndex(dim=16, capacity=10)
        for i in range(5):
            index.add(f"k{i}", unit(rng, 16))
        assert index.clear() == 5
        assert len(index) == 0

    def test_stats(self, rng):
        index = SemanticIndex(dim=16, capacity=10, name="t")
        index.add("k", unit(rng, 16))
        index.probe(unit(rng, 16), 0.99)
        stats = index.stats_dict()
        assert stats["entries"] == 1
        assert stats["dim"] == 16
        assert stats["probes"] == 1
        assert stats["bytes"] == 10 * 16 * 4


class TestCreateIndex:
    def test_auto_uses_numpy_at_cache_scale(self):
        index = create_index(dim=32, capacity=1000, kind="auto")
        assert isinstance(index, SemanticIndex)

    def test_explicit_numpy(self):
        assert type(create_index(dim=32, capacity=10, kind="numpy")) is SemanticIndex

    def test_hnswlib_falls_back_when_missing(self):
        # Must not raise even when hnswlib is not installed.
        index = create_index(dim=32, capacity=10, kind="hnswlib")
        assert index is not None
        assert index.dim == 32


class TestVectorCache:
    def test_stores_and_normalizes(self, rng):
        cache = VectorCache(dim=64, capacity=10)
        cache.put(1, unit(rng) * 5)          # deliberately not unit length
        stored = cache.get(1)
        assert float(np.linalg.norm(stored)) == pytest.approx(1.0, abs=1e-5)

    def test_missing_pk_returns_none(self):
        assert VectorCache(dim=8, capacity=5).get(99) is None

    def test_capacity_evicts(self, rng):
        cache = VectorCache(dim=16, capacity=5)
        for i in range(20):
            cache.put(i, unit(rng, 16))
        assert len(cache) == 5
        assert cache.evictions == 15

    def test_get_matrix_reports_what_it_found(self, rng):
        cache = VectorCache(dim=16, capacity=10)
        for i in range(3):
            cache.put(i, unit(rng, 16))
        matrix, found = cache.get_matrix([0, 1, 2, 999])
        assert matrix.shape == (3, 16)
        assert found == [0, 1, 2]

    def test_get_matrix_with_nothing_found(self):
        matrix, found = VectorCache(dim=8, capacity=5).get_matrix([1, 2])
        assert matrix is None
        assert found == []

    def test_rejects_wrong_dimensionality(self):
        assert VectorCache(dim=64, capacity=5).put(1, [0.1, 0.2]) is False

    def test_delete_and_clear(self, rng):
        cache = VectorCache(dim=16, capacity=10)
        cache.put(1, unit(rng, 16))
        assert cache.delete(1) is True
        assert cache.delete(1) is False
        cache.put(2, unit(rng, 16))
        assert cache.clear() == 1

    def test_memory_cost_is_predictable(self):
        cache = VectorCache(dim=768, capacity=1000)
        assert cache.nbytes == 1000 * 768 * 4
        assert cache.stats_dict()["bytes_per_vector"] == 3072


class TestVectorCacheRegistry:
    def test_creates_lazily_and_reuses(self):
        registry = VectorCacheRegistry(capacity=100)
        assert registry.for_field("docs", "emb") is None
        first = registry.for_field("docs", "emb", dim=64)
        assert registry.for_field("docs", "emb") is first

    def test_fields_are_separate(self):
        registry = VectorCacheRegistry(capacity=100)
        a = registry.for_field("docs", "title_vec", dim=64)
        b = registry.for_field("docs", "body_vec", dim=128)
        assert a is not b

    def test_drop_collection(self):
        registry = VectorCacheRegistry(capacity=100)
        registry.for_field("docs", "emb", dim=64)
        registry.for_field("other", "emb", dim=64)
        assert registry.drop_collection("docs") == 1
        assert registry.for_field("docs", "emb") is None
        assert registry.for_field("other", "emb") is not None


class TestRerank:
    """Reranking is what makes serving a neighbour's results safe."""

    def _cache(self, **overrides):
        with override_settings(MILVUS_CACHE=cache_settings(**overrides)):
            config = load_config("default")
        registry = VectorCacheRegistry(capacity=1000)
        return SemanticCache(config.semantic, registry), registry

    def _scenario(self, rng):
        """Three docs whose true order differs from the cached order."""
        semantic, registry = self._cache()
        query = unit(rng)
        store = registry.for_field("docs", "emb", dim=64)

        docs = {}
        for pk, similarity in [(1, 0.20), (2, 0.60), (3, 0.95)]:
            vector = at_similarity(query, similarity, rng)
            docs[pk] = vector
            store.put(pk, vector)

        # The neighbour cached them in the opposite order.
        hits = [
            {"id": 1, "distance": 0.9, "entity": {"id": 1, "title": "a"}},
            {"id": 2, "distance": 0.8, "entity": {"id": 2, "title": "b"}},
            {"id": 3, "distance": 0.7, "entity": {"id": 3, "title": "c"}},
        ]
        return semantic, query, docs, hits

    def test_reorders_for_the_real_query(self, rng):
        semantic, query, docs, hits = self._scenario(rng)
        ranked = semantic.rerank(hits, query, "docs", "emb", limit=3)
        expected = sorted(docs, key=lambda pk: float(docs[pk] @ query),
                          reverse=True)
        assert [h["id"] for h in ranked] == expected
        assert expected != [1, 2, 3], "the fixture must actually need reordering"

    def test_scores_descend_after_reranking(self, rng):
        semantic, query, _, hits = self._scenario(rng)
        ranked = semantic.rerank(hits, query, "docs", "emb", limit=3)
        scores = [h["distance"] for h in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_truncates_to_the_limit(self, rng):
        semantic, query, _, hits = self._scenario(rng)
        assert len(semantic.rerank(hits, query, "docs", "emb", limit=2)) == 2

    def test_keeps_hits_with_unknown_embeddings(self, rng):
        semantic, query, _, hits = self._scenario(rng)
        extended = hits + [{"id": 99, "distance": 0.5, "entity": {"id": 99}}]
        ranked = semantic.rerank(extended, query, "docs", "emb", limit=4)
        assert len(ranked) == 4, "an unknown embedding must not drop the hit"

    def test_uses_embeddings_from_the_payload_when_present(self, rng):
        semantic, registry = self._cache()
        query = unit(rng)
        good = at_similarity(query, 0.99, rng)
        poor = at_similarity(query, 0.10, rng)
        hits = [
            {"id": 1, "distance": 0.9,
             "entity": {"id": 1, "emb": poor.tolist()}},
            {"id": 2, "distance": 0.5,
             "entity": {"id": 2, "emb": good.tolist()}},
        ]
        ranked = semantic.rerank(hits, query, "docs", "emb", limit=2)
        assert [h["id"] for h in ranked] == [2, 1]

    def test_no_embeddings_returns_the_cached_order(self, rng):
        semantic, _ = self._cache()
        query = unit(rng)
        hits = [{"id": i, "distance": 1 - i / 10, "entity": {"id": i}}
                for i in range(3)]
        assert semantic.rerank(hits, query, "docs", "emb", limit=3) == hits

    def test_handles_empty_and_missing_inputs(self, rng):
        semantic, _ = self._cache()
        assert semantic.rerank([], unit(rng), "docs", "emb", limit=3) == []
        hits = [{"id": 1, "distance": 0.9, "entity": {"id": 1}}]
        assert semantic.rerank(hits, None, "docs", "emb", limit=1) == hits

    def test_l2_metric_ranks_by_proximity(self, rng):
        semantic, registry = self._cache(
            SEMANTIC={"metric": "L2", "threshold": 0.5}
        )
        store = registry.for_field("docs", "emb", dim=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        store.put(1, np.array([9, 9, 0, 0], dtype=np.float32))
        store.put(2, np.array([1, 0.01, 0, 0], dtype=np.float32))
        hits = [
            {"id": 1, "distance": 0.1, "entity": {"id": 1}},
            {"id": 2, "distance": 9.0, "entity": {"id": 2}},
        ]
        ranked = semantic.rerank(hits, query, "docs", "emb", limit=2,
                                 metric="L2")
        assert ranked[0]["id"] == 2, "the closer vector must rank first"


class TestSemanticCache:
    def _cache(self, **overrides):
        with override_settings(MILVUS_CACHE=cache_settings(**overrides)):
            config = load_config("default")
        return SemanticCache(config.semantic, VectorCacheRegistry(capacity=100))

    def test_remember_then_lookup(self, rng):
        semantic = self._cache()
        vector = unit(rng)
        semantic.remember("bucket", "key", vector, collection="docs")
        found, score = semantic.lookup("bucket", vector, threshold=0.9)
        assert found == "key"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_buckets_are_isolated(self, rng):
        semantic = self._cache()
        vector = unit(rng)
        semantic.remember("bucket-a", "key", vector, collection="docs")
        assert semantic.lookup("bucket-b", vector, threshold=0.5)[0] is None

    def test_unknown_bucket_misses_cleanly(self, rng):
        assert self._cache().lookup("nope", unit(rng), threshold=0.5) == (
            None, 0.0
        )

    def test_drop_collection_removes_its_buckets(self, rng):
        semantic = self._cache()
        vector = unit(rng)
        semantic.remember("b1", "k1", vector, collection="docs")
        semantic.remember("b2", "k2", unit(rng), collection="other")
        assert semantic.drop_collection("docs") == 1
        assert semantic.lookup("b1", vector, threshold=0.5)[0] is None

    def test_forget_one_key(self, rng):
        semantic = self._cache()
        vector = unit(rng)
        semantic.remember("b", "k", vector, collection="docs")
        assert semantic.forget("b", "k") is True
        assert semantic.lookup("b", vector, threshold=0.5)[0] is None

    def test_stats(self, rng):
        semantic = self._cache()
        semantic.remember("b", "k", unit(rng), collection="docs")
        semantic.lookup("b", unit(rng), threshold=0.99)
        stats = semantic.stats_dict()
        assert stats["buckets"] == 1
        assert stats["vectors"] == 1
        assert stats["misses"] == 1


class TestVersionRegistry:
    def test_starts_at_zero(self):
        assert VersionRegistry(shared=False).get("docs") == 0

    def test_bump_increments(self):
        registry = VersionRegistry(shared=False)
        assert registry.bump("docs") == 1
        assert registry.bump("docs") == 2
        assert registry.get("docs") == 2

    def test_collections_are_independent(self):
        registry = VersionRegistry(shared=False)
        registry.bump("docs")
        assert registry.get("other") == 0

    def test_disabled_registry_is_a_constant(self):
        registry = VersionRegistry(enabled=False)
        assert registry.bump("docs") == 0
        assert registry.get("docs") == 0

    def test_reset_forgets(self):
        registry = VersionRegistry(shared=False)
        registry.bump("docs")
        registry.reset()
        assert registry.get("docs") == 0

    def test_adopts_a_higher_shared_stamp(self):
        """Another worker's bump must be picked up."""
        class SharedBackend:
            shared = True

            def __init__(self):
                self.value = 7

            def get_version(self, key, default=0):
                return self.value

            def incr_version(self, key, delta=1):
                self.value += delta
                return self.value

        registry = VersionRegistry(
            backend=SharedBackend(), shared=True, refresh_interval=0
        )
        assert registry.get("docs") == 7

    def test_survives_an_unreachable_shared_tier(self):
        class Broken:
            shared = True

            def get_version(self, key, default=0):
                raise RuntimeError("down")

            def incr_version(self, key, delta=1):
                raise RuntimeError("down")

        registry = VersionRegistry(
            backend=Broken(), shared=True, refresh_interval=0
        )
        assert registry.get("docs") == 0
        assert registry.bump("docs") == 1, "must fall back to a local stamp"

    def test_bump_emits_a_signal(self):
        from django_milvus.cache.signals import cache_invalidated

        received = []

        # A named function held in a local, not a lambda: Django's signals
        # hold receivers weakly, and a lambda with no other reference is
        # collected before the signal fires.
        def receiver(**kwargs):
            received.append(kwargs)

        cache_invalidated.connect(receiver)
        try:
            VersionRegistry(shared=False).bump("docs", reason="write")
        finally:
            cache_invalidated.disconnect(receiver)

        assert received
        assert received[0]["collection"] == "docs"
        assert received[0]["reason"] == "write"
        assert received[0]["version"] == 1
