"""
In-process RAM cache backend - the recommended L1 tier.

Stores live Python objects in this process's heap. A hit costs a dict
lookup and a policy update: no serialization, no socket, no copy. That is
what makes it worth having in front of Redis rather than instead of it.

Three things distinguish it from a plain dict:

*Byte accounting.* Bounded by memory, not entry count. Vector payloads
vary by orders of magnitude - a 5-row search over a 1536-dim collection
dwarfs a 5-row scalar query - so a count-based bound gives no real
control over footprint.

*Sharding.* The keyspace is striped across N independently locked shards,
so concurrent requests rarely contend. Each shard runs its own policy
instance over its own slice of the budget.

*Watermarked batch eviction.* Crossing the high watermark triggers one
eviction pass down to the low watermark, rather than evicting one entry
per insert forever at the boundary.

Caveat worth stating plainly: this tier is per-process. Four Gunicorn
workers means four independent caches, each with its own copy of a hot
entry. That is usually the right trade - RAM is cheap next to a Milvus
round trip - but if you need one shared cache, configure an L2.
"""

import logging
import threading
import time

from ..memory import MemoryGovernor, estimate_size
from ..policies.base import create_policy
from ..window import WindowController
from .base import MISSING, BaseCacheBackend, CacheEntry

logger = logging.getLogger("django_milvus.cache")


class _Shard:
    """One independently locked slice of the keyspace."""

    __slots__ = (
        "lock", "entries", "bytes_used", "policy", "byte_budget",
        "entry_budget", "evictions", "bytes_evicted", "expirations",
        "rejections", "saturated",
    )

    def __init__(self, policy, byte_budget, entry_budget):
        self.lock = threading.RLock()
        self.entries = {}
        self.bytes_used = 0
        self.policy = policy
        self.byte_budget = byte_budget
        self.entry_budget = entry_budget
        self.evictions = 0
        self.bytes_evicted = 0
        self.expirations = 0
        self.rejections = 0
        # Set once the shard has had to evict. Admission control only
        # applies from then on - see _is_full_locked.
        self.saturated = False


class LocalRAMBackend(BaseCacheBackend):
    """Sharded, byte-bounded, policy-driven in-process cache."""

    name = "local"
    shared = False

    #: Fall back to this when no per-entry size estimate is available.
    DEFAULT_ENTRY_SIZE = 4096

    def __init__(self, config=None, stats=None, l1_config=None, **options):
        super().__init__(config=config, stats=stats, **options)
        self.l1 = l1_config if l1_config is not None else config
        if self.l1 is None:
            raise ValueError("LocalRAMBackend requires an L1 configuration")

        self.shard_count = max(1, self.l1.shards)
        self.max_memory = self.l1.max_memory
        self.max_entries = self.l1.max_entries
        self.high_watermark = self.l1.high_watermark
        self.low_watermark = self.l1.low_watermark
        self.algorithm = self.l1.algorithm

        self.window = WindowController(self.l1, stats=stats)
        self.window.subscribe(self._on_window_change)

        self._shards = [
            _Shard(
                policy=self._make_policy(),
                byte_budget=self._per_shard(self.max_memory),
                entry_budget=self._per_shard(self.max_entries),
            )
            for _ in range(self.shard_count)
        ]

        self.governor = MemoryGovernor(self, self.l1)
        if self.l1.janitor:
            self.governor.start()

    # ── construction helpers ─────────────────────────────

    def _make_policy(self):
        return create_policy(
            self.algorithm,
            capacity=self._initial_policy_capacity(),
            admission_ratio=self.l1.admission_ratio,
            probation_ratio=self.l1.probation_ratio,
        )

    def _per_shard(self, total):
        if total is None:
            return None
        return max(1, total // self.shard_count)

    def _initial_policy_capacity(self):
        """Entry capacity for one shard's policy.

        Policies size their internal segments in entries, but this cache
        is bounded in bytes. Derive a starting estimate from the byte
        budget and refine it as real payload sizes are observed.
        """
        per_shard_entries = self._per_shard(self.max_entries)
        if per_shard_entries is not None:
            return per_shard_entries
        budget = self._per_shard(self.max_memory) or 0
        return max(16, budget // self.DEFAULT_ENTRY_SIZE)

    def _shard_for(self, key):
        return self._shards[hash(key) % self.shard_count]

    def _on_window_change(self, ratio):
        """Propagate an adaptive window change to every shard's policy."""
        for shard in self._shards:
            setter = getattr(shard.policy, "set_admission_ratio", None)
            if setter is not None:
                with shard.lock:
                    setter(ratio)

    # ── reads ────────────────────────────────────────────

    def get(self, key, default=MISSING):
        shard = self._shard_for(key)
        with shard.lock:
            entry = shard.entries.get(key)
            if entry is None:
                return default
            if entry.is_expired():
                self._remove_locked(shard, key, entry)
                shard.expirations += 1
                if self.stats:
                    self.stats.record_expiration()
                return default
            entry.touch()
            shard.policy.on_hit(key)
            if self.stats:
                # Only the tier, not the hit itself: the orchestrator
                # counts the lookup. See CacheStats.record_hit.
                self.stats.record_tier("l1")
            return entry.value

    def get_entry(self, key):
        """Return the entry even when expired, for stale-serving."""
        shard = self._shard_for(key)
        with shard.lock:
            entry = shard.entries.get(key)
            return entry if entry is not None else MISSING

    def has(self, key):
        return self.get(key) is not MISSING

    # ── writes ───────────────────────────────────────────

    def set(self, key, value, ttl=None, size=None):
        if size is None:
            size = estimate_size(value)

        # Oversized payloads are refused outright: admitting one would
        # evict a large share of everything useful to hold a single row.
        max_entry = getattr(self.config, "max_entry_bytes", None)
        if max_entry and size > max_entry:
            if self.stats:
                self.stats.record_rejected()
            return False

        expires_at = (time.time() + ttl) if ttl else None
        shard = self._shard_for(key)

        with shard.lock:
            existing = shard.entries.get(key)
            if existing is not None:
                # Overwrite in place; the key already holds a slot, so no
                # admission decision is needed.
                shard.bytes_used += size - existing.size
                existing.value = value
                existing.size = size
                existing.expires_at = expires_at
                existing.created_at = time.time()
                existing.last_access = existing.created_at
                shard.policy.on_admit(key, size=size, expires_at=expires_at)
                self._enforce_locked(shard)
                if self.stats:
                    self.stats.record_set(size)
                return True

            # Admission control (W-TinyLFU) only bites once full: below
            # capacity, rejecting a key would leave space unused.
            if self._is_full_locked(shard) and not shard.policy.should_admit(key):
                shard.policy.on_reject(key)
                shard.rejections += 1
                if self.stats:
                    self.stats.record_rejected()
                return False

            entry = CacheEntry(value, size=size, expires_at=expires_at)
            shard.entries[key] = entry
            shard.bytes_used += size
            shard.policy.on_admit(key, size=size, expires_at=expires_at)
            self._enforce_locked(shard)

        if self.stats:
            self.stats.record_set(size)
        return True

    def touch(self, key, ttl=None):
        shard = self._shard_for(key)
        with shard.lock:
            entry = shard.entries.get(key)
            if entry is None:
                return False
            entry.expires_at = (time.time() + ttl) if ttl else None
            return True

    def delete(self, key):
        shard = self._shard_for(key)
        with shard.lock:
            entry = shard.entries.get(key)
            if entry is None:
                return False
            self._remove_locked(shard, key, entry)
            return True

    def delete_prefix(self, prefix):
        removed = 0
        for shard in self._shards:
            with shard.lock:
                doomed = [k for k in shard.entries if k.startswith(prefix)]
                for key in doomed:
                    self._remove_locked(shard, key, shard.entries[key])
                removed += len(doomed)
        return removed

    def clear(self):
        removed = 0
        for shard in self._shards:
            with shard.lock:
                removed += len(shard.entries)
                shard.entries.clear()
                shard.policy.clear()
                shard.bytes_used = 0
                shard.saturated = False
        return removed

    # ── eviction ─────────────────────────────────────────

    def _remove_locked(self, shard, key, entry):
        """Drop one entry. Caller holds the shard lock."""
        shard.entries.pop(key, None)
        shard.bytes_used -= entry.size
        if shard.bytes_used < 0:
            shard.bytes_used = 0
        shard.policy.on_remove(key)

    def _is_full_locked(self, shard):
        """Whether admission control should apply to a new key.

        Deliberately *not* "is usage at the byte limit right now". Batch
        eviction leaves usage at the low watermark, so that test would
        read as "not full" immediately after every eviction and the
        admission filter would never fire - silently reducing W-TinyLFU
        to plain LRU. What matters is that the shard has reached its
        capacity at all, which is what `saturated` records.
        """
        if shard.saturated:
            return True
        if shard.entry_budget is not None and len(shard.entries) >= shard.entry_budget:
            return True
        if shard.byte_budget is not None and shard.bytes_used >= shard.byte_budget:
            return True
        return False

    def _enforce_locked(self, shard):
        """Bring a shard back within budget. Caller holds the shard lock."""
        evicted = []
        freed = 0
        # Bound the loops: a policy whose bookkeeping has drifted out of
        # sync with storage must not be able to spin here.
        budget = len(shard.entries) + 1

        # Entry ceiling is a hard limit; evict one-for-one past it.
        if shard.entry_budget is not None:
            while len(shard.entries) > shard.entry_budget and budget > 0:
                budget -= 1
                victim = self._evict_one_locked(shard)
                if victim is None:
                    break
                evicted.append(victim[0])
                freed += victim[1]

        # Byte budget uses watermarks: only start evicting once usage
        # crosses `high`, then go all the way down to `low` in one pass.
        # Evicting to exactly the limit would put us right back at the
        # boundary on the very next insert.
        if shard.byte_budget is not None:
            trigger = shard.byte_budget * self.high_watermark
            if shard.bytes_used > trigger:
                target = shard.byte_budget * self.low_watermark
                while shard.bytes_used > target and shard.entries and budget > 0:
                    budget -= 1
                    victim = self._evict_one_locked(shard)
                    if victim is None:
                        break
                    evicted.append(victim[0])
                    freed += victim[1]

        if evicted:
            shard.saturated = True
            shard.evictions += len(evicted)
            shard.bytes_evicted += freed
            if self.stats:
                self.stats.record_eviction(len(evicted), freed)
            self._announce_eviction(evicted, freed, "capacity")
        return evicted

    def _evict_one_locked(self, shard):
        key = shard.policy.select_victim()
        if key is None:
            # Policy and storage disagree; fall back to any key so the
            # loop cannot spin forever.
            key = next(iter(shard.entries), None)
            if key is None:
                return None
        entry = shard.entries.get(key)
        if entry is None:
            # Stale policy bookkeeping: drop it and let the caller retry.
            shard.policy.on_remove(key)
            return (key, 0)
        size = entry.size
        self._remove_locked(shard, key, entry)
        return (key, size)

    def _announce_eviction(self, keys, freed, reason):
        from ..signals import _send, cache_evicted
        _send(cache_evicted, keys=keys, freed=freed, reason=reason,
              collection=None, alias=getattr(self.config, "alias", None))

    # ── maintenance (driven by MemoryGovernor) ───────────

    def purge_expired(self):
        """Drop every expired entry across all shards."""
        now = time.time()
        removed = 0
        for shard in self._shards:
            with shard.lock:
                doomed = [
                    key for key, entry in shard.entries.items()
                    if entry.is_expired(now)
                ]
                for key in doomed:
                    self._remove_locked(shard, key, shard.entries[key])
                shard.expirations += len(doomed)
                removed += len(doomed)
        if removed and self.stats:
            self.stats.record_expiration(removed)
        return removed

    def age_policies(self):
        """Age frequency information so old popularity decays."""
        for shard in self._shards:
            with shard.lock:
                shard.policy.age()

    def tick_window(self):
        """Advance the adaptive window controller one interval."""
        self.window.tick()
        self._retune_policy_capacity()

    def _retune_policy_capacity(self):
        """Re-derive entry capacity from observed payload sizes.

        Policies segment themselves by entry count, so when the cache is
        bounded by bytes their notion of capacity must track the average
        payload size actually being stored - which is only knowable at
        runtime and drifts as query shapes change.
        """
        if self.max_entries is not None or self.max_memory is None:
            return
        for shard in self._shards:
            with shard.lock:
                count = len(shard.entries)
                if count < 8:
                    continue
                average = max(1, shard.bytes_used // count)
                estimate = max(16, (shard.byte_budget or 0) // average)
                if abs(estimate - shard.policy.capacity) > shard.policy.capacity * 0.25:
                    shard.policy.set_capacity(estimate)

    def set_capacity(self, max_memory):
        """Resize the byte budget (used by memory-pressure handling)."""
        self.max_memory = max_memory
        per_shard = self._per_shard(max_memory)
        for shard in self._shards:
            with shard.lock:
                grew = (
                    per_shard is not None
                    and shard.byte_budget is not None
                    and per_shard > shard.byte_budget
                )
                shard.byte_budget = per_shard
                if grew:
                    # More room available: let new keys back in rather
                    # than holding the admission filter closed forever.
                    shard.saturated = False
                self._enforce_locked(shard)

    def close(self):
        self.governor.stop()
        self.clear()

    # ── introspection ────────────────────────────────────

    def __len__(self):
        return sum(len(shard.entries) for shard in self._shards)

    @property
    def bytes_used(self):
        return sum(shard.bytes_used for shard in self._shards)

    def stats_dict(self):
        data = super().stats_dict()
        entries = len(self)
        used = self.bytes_used
        data.update({
            "algorithm": self.algorithm,
            "entries": entries,
            "bytes": used,
            "max_memory": self.max_memory,
            "max_entries": self.max_entries,
            "utilization": round(used / self.max_memory, 4)
            if self.max_memory else None,
            "avg_entry_bytes": (used // entries) if entries else 0,
            "shards": self.shard_count,
            "evictions": sum(s.evictions for s in self._shards),
            "bytes_evicted": sum(s.bytes_evicted for s in self._shards),
            "expirations": sum(s.expirations for s in self._shards),
            "rejections": sum(s.rejections for s in self._shards),
            "saturated": sum(1 for s in self._shards if s.saturated),
            "window": self.window.stats_dict(),
            "governor": self.governor.stats(),
        })
        if self._shards:
            data["policy"] = self._shards[0].policy.stats()
        return data
