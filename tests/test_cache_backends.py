"""Tests for the cache backends, memory accounting and window control."""

import time

import pytest
from django.test import override_settings

from django_milvus.cache.backends.base import (
    MISSING,
    BaseCacheBackend,
    CacheEntry,
    unwrap_payload,
    wrap_payload,
)
from django_milvus.cache.backends.djangocache import DjangoCacheBackend
from django_milvus.cache.backends.dummy import DummyBackend
from django_milvus.cache.backends.local import LocalRAMBackend
from django_milvus.cache.backends.tiered import CircuitBreaker, TieredCache
from django_milvus.cache.config import load_config
from django_milvus.cache.memory import MemoryGovernor, estimate_size
from django_milvus.cache.registry import build_backend
from django_milvus.cache.stats import CacheStats
from django_milvus.cache.window import WindowController

from .conftest import cache_settings


def build_config(**overrides):
    with override_settings(MILVUS_CACHE=cache_settings(**overrides)):
        return load_config("default")


def build_local(**l1_overrides):
    l1 = {"MAX_MEMORY": "4MB", "MAX_ENTRIES": 500, "SHARDS": 2,
          "JANITOR": False}
    l1.update(l1_overrides)
    config = build_config(L1=l1)
    return LocalRAMBackend(
        config=config, stats=CacheStats(window=60), l1_config=config.l1
    )


# ─────────────────────────────────────────────────────────
# Conformance: the same suite against every backend
# ─────────────────────────────────────────────────────────

def local_backend():
    return build_local()


def django_backend():
    config = build_config()
    return DjangoCacheBackend(config=config, location="default",
                              prefix=f"t{time.time_ns()}")


DJANGO_L2 = "django_milvus.cache.backends.djangocache.DjangoCacheBackend"


def build_tiered(stats=None):
    """Build an L1+L2 stack through the real construction path.

    Going through ``build_backend`` rather than assembling the tiers by
    hand matters: how statistics are wired between the tiers is part of
    what these tests check, and a hand-built stack would not reproduce it.
    """
    config = build_config(L2={"BACKEND": DJANGO_L2, "LOCATION": "default",
                              "PREFIX": f"t{time.time_ns()}"})
    return build_backend(config, stats if stats is not None
                         else CacheStats(window=60))


def tiered_backend():
    return build_tiered()


BACKENDS = [local_backend, django_backend, tiered_backend]


@pytest.fixture(params=BACKENDS, ids=["local", "django", "tiered"])
def backend(request):
    instance = request.param()
    yield instance
    instance.close()


class TestBackendContract:
    def test_set_then_get(self, backend):
        backend.set("k", [{"id": 1}])
        assert backend.get("k") == [{"id": 1}]

    def test_missing_key_returns_the_sentinel(self, backend):
        assert backend.get("absent") is MISSING

    def test_missing_key_honours_an_explicit_default(self, backend):
        assert backend.get("absent", None) is None
        assert backend.get("absent", []) == []

    def test_cached_none_is_not_a_miss(self, backend):
        """Negative caching depends on telling these two apart."""
        backend.set("empty", None)
        assert backend.get("empty") is None
        assert backend.get("empty") is not MISSING
        assert backend.has("empty")

    def test_cached_empty_list_round_trips(self, backend):
        backend.set("nothing", [])
        assert backend.get("nothing") == []
        assert backend.has("nothing")

    def test_delete(self, backend):
        backend.set("k", 1)
        assert backend.delete("k") is True
        assert backend.get("k") is MISSING
        assert backend.delete("k") is False

    def test_overwrite(self, backend):
        backend.set("k", "first")
        backend.set("k", "second")
        assert backend.get("k") == "second"

    def test_ttl_expires(self, backend):
        backend.set("k", "v", ttl=0.05)
        assert backend.get("k") == "v"
        time.sleep(0.08)
        assert backend.get("k") is MISSING

    def test_no_ttl_persists(self, backend):
        backend.set("k", "v", ttl=None)
        time.sleep(0.05)
        assert backend.get("k") == "v"

    def test_clear(self, backend):
        for i in range(5):
            backend.set(f"k{i}", i)
        backend.clear()
        assert backend.get("k0") is MISSING

    def test_get_many(self, backend):
        backend.set("a", 1)
        backend.set("b", 2)
        found = backend.get_many(["a", "b", "missing"])
        assert found == {"a": 1, "b": 2}

    def test_set_many_and_delete_many(self, backend):
        backend.set_many({"a": 1, "b": 2})
        assert backend.get("a") == 1
        backend.delete_many(["a", "b"])
        assert backend.get("a") is MISSING

    def test_incr_version_is_monotonic(self, backend):
        first = backend.incr_version("ver:x")
        second = backend.incr_version("ver:x")
        assert second == first + 1

    def test_delete_prefix(self, backend):
        backend.set("pre:a", 1)
        backend.set("pre:b", 2)
        backend.set("other", 3)
        backend.delete_prefix("pre:")
        assert backend.get("pre:a") is MISSING
        assert backend.get("other") == 3


class TestDummyBackend:
    def test_stores_nothing(self):
        backend = DummyBackend()
        assert backend.set("k", "v") is False
        assert backend.get("k") is MISSING
        assert len(backend) == 0
        assert backend.clear() == 0


class TestEnvelope:
    def test_round_trips_value_and_expiry(self):
        payload = wrap_payload([{"id": 1}], ttl=10)
        value, created, expires = unwrap_payload(payload)
        assert value == [{"id": 1}]
        assert expires - created == pytest.approx(10, abs=0.1)

    def test_no_ttl_means_no_deadline(self):
        assert unwrap_payload(wrap_payload("v", None))[2] is None

    def test_bare_values_are_tolerated(self):
        """Data written before the envelope existed must still read."""
        assert unwrap_payload([{"id": 1}]) == ([{"id": 1}], None, None)

    def test_wraps_none(self):
        assert unwrap_payload(wrap_payload(None))[0] is None


class TestCacheEntry:
    def test_expiry(self):
        entry = CacheEntry("v", expires_at=time.time() - 1)
        assert entry.is_expired()

    def test_no_expiry_never_expires(self):
        assert CacheEntry("v").is_expired() is False

    def test_stale_window(self):
        entry = CacheEntry("v", expires_at=time.time() - 1)
        assert entry.is_stale_usable(grace=10) is True
        assert entry.is_stale_usable(grace=0.5) is False

    def test_fresh_entry_is_not_stale(self):
        entry = CacheEntry("v", expires_at=time.time() + 10)
        assert entry.is_stale_usable(grace=10) is False


class TestLocalMemory:
    def test_bounded_by_bytes(self):
        backend = build_local(MAX_MEMORY="256KB", MAX_ENTRIES=None)
        payload = [{"id": i, "v": [0.5] * 64} for i in range(10)]
        for i in range(500):
            backend.set(f"k{i}", payload)
        assert backend.bytes_used <= backend.max_memory
        assert len(backend) > 0
        backend.close()

    def test_bounded_by_entries(self):
        backend = build_local(MAX_MEMORY=None, MAX_ENTRIES=40, SHARDS=1)
        for i in range(500):
            backend.set(f"k{i}", i)
        assert len(backend) <= 40
        backend.close()

    def test_oversized_payloads_are_refused(self):
        backend = build_local()
        with override_settings(MILVUS_CACHE=cache_settings(
            MAX_ENTRY_BYTES="1KB",
            L1={"MAX_MEMORY": "4MB", "JANITOR": False},
        )):
            config = load_config("default")
        backend.config = config
        assert backend.set("small", [{"id": 1}]) is True
        assert backend.set("huge", [{"v": [0.5] * 10000}]) is False
        assert backend.get("huge") is MISSING
        backend.close()

    def test_watermark_evicts_in_a_batch(self):
        """Crossing `high` must drop to `low`, not to the limit."""
        backend = build_local(
            MAX_MEMORY="256KB", MAX_ENTRIES=None, SHARDS=1,
            WATERMARK={"high": 0.9, "low": 0.5},
        )
        payload = [{"id": i, "v": [0.5] * 32} for i in range(4)]
        for i in range(400):
            backend.set(f"k{i}", payload)
        assert backend.bytes_used <= backend.max_memory * 0.9
        backend.close()

    def test_capacity_shrink_evicts(self):
        backend = build_local(MAX_MEMORY="1MB", MAX_ENTRIES=None, SHARDS=1)
        payload = [{"v": [0.5] * 64}]
        for i in range(300):
            backend.set(f"k{i}", payload)
        before = backend.bytes_used
        backend.set_capacity(128 * 1024)
        assert backend.bytes_used < before
        assert backend.bytes_used <= 128 * 1024
        backend.close()

    def test_purge_expired(self):
        backend = build_local()
        for i in range(10):
            backend.set(f"k{i}", i, ttl=0.05)
        backend.set("keep", "v", ttl=100)
        time.sleep(0.08)
        assert backend.purge_expired() == 10
        assert backend.get("keep") == "v"
        backend.close()

    def test_stats_report_utilization(self):
        backend = build_local(MAX_MEMORY="1MB", MAX_ENTRIES=None)
        backend.set("k", [{"id": 1}])
        stats = backend.stats_dict()
        assert stats["entries"] == 1
        assert stats["bytes"] > 0
        assert 0 <= stats["utilization"] <= 1
        assert stats["algorithm"] == "w-tinylfu"
        backend.close()

    def test_sharding_spreads_keys(self):
        backend = build_local(SHARDS=4)
        for i in range(400):
            backend.set(f"key{i}", i)
        used = sum(1 for shard in backend._shards if shard.entries)
        assert used > 1, "keys should not all land in one shard"
        backend.close()

    def test_admission_control_engages_once_full(self):
        """W-TinyLFU must actually filter, not silently act like LRU."""
        backend = build_local(MAX_ENTRIES=40, MAX_MEMORY=None, SHARDS=1)
        for i in range(40):
            backend.set(f"warm{i}", i)
        for _ in range(20):
            for i in range(10):
                backend.get(f"warm{i}")
        for i in range(2000):
            backend.set(f"tail{i}", i)
        survivors = sum(1 for i in range(10) if backend.get(f"warm{i}") is not MISSING)
        assert survivors >= 5, (
            f"only {survivors}/10 hot keys survived a long tail of one-off "
            f"keys; admission control is not engaging"
        )
        backend.close()


class TestEstimateSize:
    def test_scales_with_vector_length(self):
        small = estimate_size([0.1] * 64)
        large = estimate_size([0.1] * 1024)
        assert large > small * 10

    def test_float_vector_is_in_the_right_ballpark(self):
        # 768 floats: ~24 bytes each plus list slots. Anything within a
        # small factor is fine - this drives eviction, not accounting.
        size = estimate_size([0.1] * 768)
        assert 15_000 < size < 45_000

    def test_realistic_payload(self):
        payload = [
            {"id": i, "distance": 0.9, "entity": {"id": i, "title": "x" * 50}}
            for i in range(10)
        ]
        assert 1_000 < estimate_size(payload) < 60_000

    def test_shared_objects_counted_once(self):
        shared = {"a": [0.1] * 500}
        assert estimate_size([shared, shared]) < estimate_size(
            [shared, {"a": [0.1] * 500}]
        )

    def test_handles_cycles(self):
        cyclic = {}
        cyclic["self"] = cyclic
        assert estimate_size(cyclic) > 0

    @pytest.mark.parametrize("value", [None, 1, 1.5, True, "text", b"bytes",
                                       [], {}, set()])
    def test_handles_every_basic_type(self, value):
        assert estimate_size(value) >= 0


class TestMemoryGovernor:
    def test_tick_sweeps_expired_entries(self):
        backend = build_local()
        governor = MemoryGovernor(backend, backend.l1)
        for i in range(5):
            backend.set(f"k{i}", i, ttl=0.05)
        time.sleep(0.08)
        governor.tick()
        assert governor.expired_swept == 5
        assert len(backend) == 0
        backend.close()

    def test_pressure_shrinks_and_recovers(self, monkeypatch):
        backend = build_local(
            MAX_MEMORY="1MB", MAX_ENTRIES=None,
            MEMORY_PRESSURE={"enabled": True, "process_rss_limit": "100MB",
                             "floor_ratio": 0.25},
        )
        governor = MemoryGovernor(backend, backend.l1)

        rss = [200 * 1024 * 1024]      # over the limit
        monkeypatch.setattr(
            "django_milvus.cache.memory.get_process_rss", lambda: rss[0]
        )
        governor.check_pressure()
        assert governor.shrink_factor < 1.0
        assert governor.effective_max_memory < governor.base_max_memory

        rss[0] = 10 * 1024 * 1024      # comfortably under
        for _ in range(10):
            governor.check_pressure()
        assert governor.shrink_factor == 1.0
        backend.close()

    def test_pressure_respects_the_floor(self, monkeypatch):
        backend = build_local(
            MAX_MEMORY="1MB", MAX_ENTRIES=None,
            MEMORY_PRESSURE={"enabled": True, "process_rss_limit": "10MB",
                             "floor_ratio": 0.25},
        )
        governor = MemoryGovernor(backend, backend.l1)
        monkeypatch.setattr(
            "django_milvus.cache.memory.get_process_rss",
            lambda: 500 * 1024 * 1024,
        )
        for _ in range(20):
            governor.check_pressure()
        assert governor.shrink_factor == pytest.approx(0.25)
        backend.close()

    def test_inert_without_psutil(self, monkeypatch):
        backend = build_local()
        governor = MemoryGovernor(backend, backend.l1)
        monkeypatch.setattr(
            "django_milvus.cache.memory.get_process_rss", lambda: None
        )
        governor.check_pressure()
        assert governor.shrink_factor == 1.0
        backend.close()


class TestWindowController:
    def _controller(self, **window):
        settings = {"adaptive": True, "admission_ratio": 0.1, "step": 0.05}
        settings.update(window)
        config = build_config(L1={"MAX_ENTRIES": 100, "WINDOW": settings})
        return WindowController(config.l1)

    def test_grows_the_window_while_the_hit_rate_improves(self):
        controller = self._controller()
        rate = 0.30
        for i in range(10):
            rate += 0.02 if controller.direction > 0 else -0.02
            controller.tick(hit_rate=rate, lookups=(i + 1) * 1000)
        assert controller.admission_ratio > 0.1
        assert controller.adjustments > 0

    def test_reverses_when_the_hit_rate_falls(self):
        controller = self._controller()
        controller.tick(hit_rate=0.5, lookups=1000)
        before = controller.direction
        controller.tick(hit_rate=0.4, lookups=2000)
        assert controller.direction == -before

    def test_stays_within_bounds(self):
        controller = self._controller(step=0.4)
        for i in range(60):
            controller.tick(hit_rate=1.0 if i % 2 else 0.0,
                            lookups=(i + 1) * 1000)
            assert 0.0 <= controller.admission_ratio <= 0.8

    def test_ignores_low_traffic(self):
        controller = self._controller()
        for i in range(20):
            controller.tick(hit_rate=i / 20, lookups=(i + 1) * 5)
        assert controller.adjustments == 0, "must not chase noise"

    def test_non_adaptive_never_moves(self):
        controller = self._controller(adaptive=False, admission_ratio=0.25)
        for i in range(20):
            controller.tick(hit_rate=i / 20, lookups=(i + 1) * 1000)
        assert controller.admission_ratio == 0.25

    def test_notifies_subscribers(self):
        controller = self._controller()
        seen = []
        controller.subscribe(seen.append)
        rate = 0.3
        for i in range(6):
            rate += 0.05
            controller.tick(hit_rate=rate, lookups=(i + 1) * 1000)
        assert seen

    def test_reset_restores_the_configured_ratio(self):
        controller = self._controller()
        for i in range(6):
            controller.tick(hit_rate=0.3 + i * 0.05, lookups=(i + 1) * 1000)
        controller.reset()
        assert controller.admission_ratio == 0.1


class TestTiering:
    def _tiered(self):
        stats = CacheStats(window=60)
        tier = build_tiered(stats)
        return tier, tier.l1, tier.l2, stats

    def test_l1_answers_first(self):
        tier, _, _, stats = self._tiered()
        tier.set("k", "v")
        assert tier.get("k") == "v"
        assert stats.l1_hits == 1
        assert stats.l2_hits == 0

    def test_l2_answers_and_promotes(self):
        tier, l1, _, stats = self._tiered()
        tier.set("k", "v", ttl=60)
        l1.delete("k")                       # a cold worker
        assert tier.get("k") == "v"
        assert stats.l2_hits == 1
        assert tier.promotions == 1
        assert l1.get("k") == "v", "an L2 hit must land in L1"

    def test_tiers_are_not_double_counted(self):
        """l1_hits + l2_hits must equal the tier hits, counted once each."""
        tier, l1, _, stats = self._tiered()
        tier.set("a", 1, ttl=60)
        tier.set("b", 2, ttl=60)

        tier.get("a")                        # L1
        tier.get("a")                        # L1
        l1.delete("b")
        tier.get("b")                        # L2, then promoted
        tier.get("b")                        # L1 again

        assert stats.l1_hits == 3
        assert stats.l2_hits == 1
        # The backend records the tier; the orchestrator records the hit.
        # A tiered backend must not do both.
        assert stats.hits == 0, (
            "the backend must not increment the aggregate hit counter"
        )

    def test_promotion_preserves_the_original_deadline(self):
        tier, l1, _, _ = self._tiered()
        tier.set("k", "v", ttl=100)
        l1.delete("k")
        tier.get("k")
        entry = l1.get_entry("k")
        remaining = entry.expires_at - time.time()
        assert remaining < 100, "a promoted entry must not gain extra life"

    def test_writes_reach_both_tiers(self):
        tier, l1, l2, _ = self._tiered()
        tier.set("k", "v")
        assert l1.get("k") == "v"
        assert l2.get("k") == "v"

    def test_delete_reaches_both_tiers(self):
        tier, l1, l2, _ = self._tiered()
        tier.set("k", "v")
        tier.delete("k")
        assert l1.get("k") is MISSING
        assert l2.get("k") is MISSING

    def test_l1_only_when_no_l2(self):
        config = build_config()
        l1 = LocalRAMBackend(config=config, l1_config=config.l1)
        tier = TieredCache(l1=l1, l2=None, config=config, stats=CacheStats())
        tier.set("k", "v")
        assert tier.get("k") == "v"
        assert tier.shared is False


class TestFailOpen:
    """A broken shared tier must never break reads."""

    class Broken:
        name = "broken"
        shared = True

        def __getattr__(self, name):
            def explode(*args, **kwargs):
                raise RuntimeError("shared tier is down")
            return explode

    def _tiered(self, failures=3, reset_after=30):
        config = build_config()
        stats = CacheStats(window=60)
        l1 = LocalRAMBackend(config=config, stats=stats, l1_config=config.l1)
        return TieredCache(
            l1=l1, l2=self.Broken(), config=config, stats=stats,
            breaker=CircuitBreaker(failures=failures, reset_after=reset_after),
        )

    def test_reads_still_work(self):
        tier = self._tiered()
        for i in range(10):
            tier.set(f"k{i}", i)
            assert tier.get(f"k{i}") == i

    def test_breaker_stops_calling_the_dead_tier(self):
        tier = self._tiered(failures=3)
        for i in range(20):
            tier.set(f"k{i}", i)
        assert tier.l2_errors == 3
        assert tier.breaker.state == "open"

    def test_breaker_retries_after_the_cool_off(self):
        tier = self._tiered(failures=2, reset_after=0.1)
        for i in range(5):
            tier.set(f"k{i}", i)
        assert tier.breaker.state == "open"
        time.sleep(0.15)
        assert tier.breaker.state == "half"

    def test_a_healthy_call_resets_the_count(self):
        breaker = CircuitBreaker(failures=3)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.state == "closed"

    def test_stats_survive_a_dead_tier(self):
        tier = self._tiered()
        tier.set("k", "v")
        assert tier.stats_dict()["l2"]["reachable"] is False


class TestCustomBackend:
    """A backend need only match the interface, not the base class."""

    def test_duck_typed_backend_works_as_l2(self):
        class Minimal:
            name = "minimal"
            shared = False

            def __init__(self):
                self.store = {}

            def get(self, key, default=MISSING):
                return self.store.get(key, default)

            def set(self, key, value, ttl=None, size=None):
                self.store[key] = value
                return True

            def delete(self, key):
                return self.store.pop(key, MISSING) is not MISSING

            def clear(self):
                count = len(self.store)
                self.store.clear()
                return count

            def close(self):
                pass

        config = build_config()
        l1 = LocalRAMBackend(config=config, l1_config=config.l1)
        minimal = Minimal()
        tier = TieredCache(l1=l1, l2=minimal, config=config,
                           stats=CacheStats())
        tier.set("k", "v", ttl=60)
        l1.delete("k")
        # Falls back to plain get() because Minimal has no get_entry.
        assert tier.get("k") == "v"

    def test_base_class_defaults_are_usable(self):
        assert BaseCacheBackend().delete_prefix("x") == 0
        assert BaseCacheBackend().purge_expired() == 0
