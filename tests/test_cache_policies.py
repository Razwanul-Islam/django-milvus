"""Tests for the eviction policies."""

import random

import pytest

from django_milvus.cache.policies import (
    ARCPolicy,
    CountMinSketch,
    FIFOPolicy,
    LFUPolicy,
    LRUPolicy,
    RandomPolicy,
    SLRUPolicy,
    TTLPolicy,
    TwoQPolicy,
    WTinyLFUPolicy,
    available_policies,
    create_policy,
)
from django_milvus.cache.policies.base import EvictionPolicy, register
from django_milvus.exceptions import CacheConfigurationError

ALL = [
    "lru", "lfu", "fifo", "random", "ttl", "slru", "2q", "arc", "w-tinylfu",
]


class TestRegistry:
    def test_all_policies_registered(self):
        assert sorted(available_policies()) == sorted(ALL)

    def test_create_by_name(self):
        assert isinstance(create_policy("lru", capacity=10), LRUPolicy)
        assert isinstance(create_policy("w-tinylfu", capacity=10), WTinyLFUPolicy)

    def test_unknown_algorithm_names_the_alternatives(self):
        with pytest.raises(CacheConfigurationError) as info:
            create_policy("nope", capacity=10)
        assert "nope" in str(info.value)
        assert "lru" in str(info.value)

    def test_custom_policy_can_be_registered(self):
        class Custom(EvictionPolicy):
            name = "test-custom"

            def __init__(self, capacity=10, **options):
                super().__init__(capacity, **options)
                self.keys_seen = []

            def on_admit(self, key, size=0, expires_at=None):
                self.keys_seen.append(key)

            def on_hit(self, key):
                pass

            def on_remove(self, key):
                if key in self.keys_seen:
                    self.keys_seen.remove(key)

            def select_victim(self):
                return self.keys_seen[0] if self.keys_seen else None

            def clear(self):
                self.keys_seen.clear()

            def keys(self):
                return iter(self.keys_seen)

            def __len__(self):
                return len(self.keys_seen)

            def __contains__(self, key):
                return key in self.keys_seen

        register(Custom)
        policy = create_policy("test-custom", capacity=5)
        policy.on_admit("a")
        assert policy.select_victim() == "a"

    def test_policy_without_name_is_rejected(self):
        class Nameless(EvictionPolicy):
            pass

        with pytest.raises(CacheConfigurationError):
            register(Nameless)


@pytest.mark.parametrize("name", ALL)
class TestContract:
    """Every policy must honour the same contract."""

    def test_tracks_admitted_keys(self, name):
        policy = create_policy(name, capacity=10)
        for key in "abcd":
            policy.on_admit(key)
        assert len(policy) == 4
        assert all(key in policy for key in "abcd")

    def test_remove_forgets(self, name):
        policy = create_policy(name, capacity=10)
        policy.on_admit("a")
        policy.on_remove("a")
        assert len(policy) == 0
        assert "a" not in policy

    def test_victim_is_a_tracked_key(self, name):
        policy = create_policy(name, capacity=4)
        for key in "abcd":
            policy.on_admit(key)
        victim = policy.select_victim()
        assert victim in policy

    def test_empty_policy_has_no_victim(self, name):
        assert create_policy(name, capacity=4).select_victim() is None

    def test_clear_empties(self, name):
        policy = create_policy(name, capacity=10)
        for key in "abcd":
            policy.on_admit(key)
        policy.clear()
        assert len(policy) == 0
        assert policy.select_victim() is None

    def test_readmitting_does_not_double_count(self, name):
        policy = create_policy(name, capacity=10)
        policy.on_admit("a")
        policy.on_admit("a")
        assert len(policy) == 1

    def test_hit_on_unknown_key_is_safe(self, name):
        policy = create_policy(name, capacity=10)
        policy.on_hit("never-seen")
        policy.on_remove("never-seen")

    def test_age_and_resize_are_safe(self, name):
        policy = create_policy(name, capacity=10)
        for key in "abcd":
            policy.on_admit(key)
            policy.on_hit(key)
        policy.age()
        policy.set_capacity(50)
        assert policy.capacity == 50
        policy.set_capacity(2)
        assert policy.select_victim() in policy

    def test_stats_reports_algorithm(self, name):
        policy = create_policy(name, capacity=10)
        assert policy.stats()["algorithm"] == name


class TestOrdering:
    """Each policy must evict the key its name promises."""

    def test_lru_evicts_least_recently_used(self):
        policy = LRUPolicy(capacity=3)
        for key in "abc":
            policy.on_admit(key)
        policy.on_hit("a")
        assert policy.select_victim() == "b"

    def test_lfu_evicts_least_frequently_used(self):
        policy = LFUPolicy(capacity=3)
        for key in "abc":
            policy.on_admit(key)
        policy.on_hit("a")
        policy.on_hit("a")
        policy.on_hit("b")
        assert policy.select_victim() == "c"

    def test_lfu_age_halves_frequencies(self):
        policy = LFUPolicy(capacity=3)
        policy.on_admit("hot")
        for _ in range(10):
            policy.on_hit("hot")
        assert policy._freq["hot"] == 11
        policy.age()
        assert policy._freq["hot"] == 5

    def test_fifo_ignores_hits(self):
        policy = FIFOPolicy(capacity=3)
        for key in "abc":
            policy.on_admit(key)
        policy.on_hit("a")
        policy.on_hit("a")
        assert policy.select_victim() == "a"

    def test_ttl_evicts_nearest_deadline(self):
        policy = TTLPolicy(capacity=5)
        policy.on_admit("late", expires_at=1000)
        policy.on_admit("soon", expires_at=10)
        policy.on_admit("never", expires_at=None)
        assert policy.select_victim() == "soon"
        policy.on_remove("soon")
        assert policy.select_victim() == "late"

    def test_slru_protects_after_a_second_hit(self):
        policy = SLRUPolicy(capacity=10, probation_ratio=0.5)
        policy.on_admit("proven")
        policy.on_hit("proven")
        policy.on_admit("unproven")
        assert policy.select_victim() == "unproven"

    def test_slru_demotes_rather_than_dropping(self):
        policy = SLRUPolicy(capacity=4, probation_ratio=0.5)
        for key in "abcd":
            policy.on_admit(key)
            policy.on_hit(key)
        assert len(policy) == 4
        assert policy.stats()["probation"] >= 1

    def test_twoq_promotes_a_ghost_straight_to_main(self):
        policy = TwoQPolicy(capacity=10)
        policy.on_admit("x")
        policy.on_remove("x")          # evicted from A1in, becomes a ghost
        assert "x" in policy._a1out
        policy.on_admit("x")           # requested again
        assert "x" in policy._am, "a ghost hit must go straight to the main queue"

    def test_arc_adapts_toward_recency_on_a_b1_hit(self):
        policy = ARCPolicy(capacity=10)
        policy.on_admit("x")
        policy.on_remove("x")          # into B1
        before = policy._p
        policy.on_admit("x")           # B1 hit
        assert policy._p > before, "a recency-ghost hit must grow p"

    def test_arc_adapts_toward_frequency_on_a_b2_hit(self):
        policy = ARCPolicy(capacity=10)
        policy.on_admit("x")
        policy.on_hit("x")             # now in T2
        policy._p = 5
        policy.on_remove("x")          # into B2
        policy.on_admit("x")           # B2 hit
        assert policy._p < 5, "a frequency-ghost hit must shrink p"

    def test_random_victim_is_always_present(self):
        policy = RandomPolicy(capacity=10, seed=1)
        for i in range(10):
            policy.on_admit(f"k{i}")
        for _ in range(5):
            victim = policy.select_victim()
            assert victim in policy
            policy.on_remove(victim)


class TestCountMinSketch:
    def test_frequency_rises_with_use(self):
        sketch = CountMinSketch(capacity=128)
        for _ in range(5):
            sketch.increment("hot")
        sketch.increment("cold")
        assert sketch.frequency("hot") > sketch.frequency("cold")

    def test_never_underestimates(self):
        sketch = CountMinSketch(capacity=512)
        for _ in range(7):
            sketch.increment("k")
        # Count-Min may overestimate on collision, never underestimate.
        assert sketch.frequency("k") >= 7

    def test_saturates_at_the_counter_ceiling(self):
        sketch = CountMinSketch(capacity=64)
        for _ in range(500):
            sketch.increment("k")
        assert sketch.frequency("k") <= CountMinSketch.MAX_COUNT + 1

    def test_aging_halves_counts(self):
        sketch = CountMinSketch(capacity=128)
        for _ in range(8):
            sketch.increment("k")
        before = sketch.frequency("k")
        sketch.age()
        assert sketch.frequency("k") < before

    def test_unseen_key_has_no_frequency(self):
        assert CountMinSketch(capacity=128).frequency("never") == 0

    def test_ages_automatically(self):
        sketch = CountMinSketch(capacity=16)
        for i in range(500):
            sketch.increment(f"k{i}")
        assert sketch.age_count > 0


class TestScanResistance:
    """The property that justifies the defaults."""

    @staticmethod
    def _run(policy, trace):
        """Simulate a cache of ``policy.capacity`` entries over a trace."""
        stored = set()
        hits = 0
        for key in trace:
            if key in stored:
                hits += 1
                policy.on_hit(key)
                continue
            if len(stored) >= policy.capacity:
                if not policy.should_admit(key):
                    policy.on_reject(key)
                    continue
                victim = policy.select_victim()
                if victim is not None:
                    stored.discard(victim)
                    policy.on_remove(victim)
            stored.add(key)
            policy.on_admit(key)
        return hits / len(trace)

    @staticmethod
    def _zipfian(seed=7, universe=2000, length=20000):
        rng = random.Random(seed)
        weights = [1.0 / (i + 1) ** 1.1 for i in range(universe)]
        return rng.choices(range(universe), weights=weights, k=length)

    def test_wtinylfu_beats_lru_on_skewed_traffic(self):
        trace = self._zipfian()
        lru = self._run(LRUPolicy(capacity=100), trace)
        wtiny = self._run(WTinyLFUPolicy(capacity=100), trace)
        assert wtiny > lru, (
            f"W-TinyLFU ({wtiny:.4f}) should beat LRU ({lru:.4f}) on a "
            f"skewed trace; that is why it is the default"
        )

    def test_wtinylfu_beats_fifo_on_skewed_traffic(self):
        trace = self._zipfian()
        assert self._run(WTinyLFUPolicy(capacity=100), trace) > self._run(
            FIFOPolicy(capacity=100), trace
        )

    @pytest.mark.parametrize("policy_class", [WTinyLFUPolicy, TwoQPolicy])
    def test_scan_does_not_flush_the_hot_set(self, policy_class):
        """A one-pass scan must not evict entries that have proven useful."""
        hot = [f"hot{i}" for i in range(20)]
        warmup = hot * 15
        scan = [f"scan{i}" for i in range(3000)]

        policy = policy_class(capacity=50)
        stored = set()

        def feed(keys):
            for key in keys:
                if key in stored:
                    policy.on_hit(key)
                    continue
                if len(stored) >= policy.capacity:
                    if not policy.should_admit(key):
                        policy.on_reject(key)
                        continue
                    victim = policy.select_victim()
                    if victim is not None:
                        stored.discard(victim)
                        policy.on_remove(victim)
                stored.add(key)
                policy.on_admit(key)

        feed(warmup)
        feed(scan)
        survivors = sum(1 for key in hot if key in stored)
        assert survivors >= 15, (
            f"{policy_class.name} kept only {survivors}/20 hot keys through a "
            f"scan; it is meant to be scan-resistant"
        )

    def test_lru_is_flushed_by_a_scan(self):
        """The baseline this exists to improve on."""
        hot = [f"hot{i}" for i in range(20)]
        policy = LRUPolicy(capacity=50)
        stored = set()
        for key in hot * 15 + [f"scan{i}" for i in range(3000)]:
            if key in stored:
                policy.on_hit(key)
                continue
            if len(stored) >= policy.capacity:
                victim = policy.select_victim()
                stored.discard(victim)
                policy.on_remove(victim)
            stored.add(key)
            policy.on_admit(key)
        assert sum(1 for key in hot if key in stored) == 0


class TestWTinyLFUInternals:
    def test_admission_happens_at_the_window_boundary(self):
        policy = WTinyLFUPolicy(capacity=100, admission_ratio=0.1)
        # should_admit always says yes: a key must reach the window before
        # it can build the frequency it needs to win a duel.
        assert policy.should_admit("brand-new") is True

    def test_window_overflow_drains_into_the_main_region(self):
        policy = WTinyLFUPolicy(capacity=100, admission_ratio=0.05)
        for i in range(60):
            policy.on_admit(f"k{i}")
        stats = policy.stats()
        assert stats["window"] <= stats["window_capacity"]
        assert stats["probation"] + stats["protected"] > 0

    def test_a_popular_candidate_wins_its_duel(self):
        policy = WTinyLFUPolicy(capacity=20, admission_ratio=0.1)
        for i in range(20):
            policy.on_admit(f"filler{i}")
        for _ in range(10):
            policy.on_admit("popular")
            policy.on_hit("popular")
        assert policy.frequency("popular") > policy.frequency("filler0")

    def test_admission_ratio_is_retunable(self):
        policy = WTinyLFUPolicy(capacity=1000, admission_ratio=0.01)
        assert policy._window_capacity == 10
        policy.set_admission_ratio(0.25)
        assert policy._window_capacity == 250

    def test_admission_ratio_is_bounded(self):
        policy = WTinyLFUPolicy(capacity=100)
        policy.set_admission_ratio(5.0)
        assert policy.admission_ratio <= 0.8
        policy.set_admission_ratio(-1.0)
        assert policy.admission_ratio >= 0.0
