"""Least Frequently Used eviction, with O(1) increments."""

from collections import OrderedDict, defaultdict

from .base import EvictionPolicy, register


@register
class LFUPolicy(EvictionPolicy):
    """Evict the key accessed fewest times.

    Implemented with frequency buckets rather than a heap, so recording a
    hit is O(1) instead of O(log n): each frequency maps to an ordered
    set of keys, and a hit moves the key one bucket up. Ties inside a
    bucket break by recency, which avoids the pathological case where a
    once-hot key is preserved forever.

    Suits workloads with a genuinely stable popularity distribution. It
    is slow to forget - use ``w-tinylfu`` if popularity drifts.
    """

    name = "lfu"

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._freq = {}
        self._buckets = defaultdict(OrderedDict)
        self._min_freq = 0

    def on_admit(self, key, size=0, expires_at=None):
        if key in self._freq:
            self.on_hit(key)
            return
        self._freq[key] = 1
        self._buckets[1][key] = True
        self._min_freq = 1

    def on_hit(self, key):
        current = self._freq.get(key)
        if current is None:
            return
        bucket = self._buckets[current]
        bucket.pop(key, None)
        if not bucket:
            del self._buckets[current]
            if self._min_freq == current:
                self._min_freq = current + 1
        promoted = current + 1
        self._freq[key] = promoted
        self._buckets[promoted][key] = True

    def on_remove(self, key):
        current = self._freq.pop(key, None)
        if current is None:
            return
        bucket = self._buckets.get(current)
        if bucket is not None:
            bucket.pop(key, None)
            if not bucket:
                del self._buckets[current]
                if self._min_freq == current:
                    self._min_freq = min(self._buckets, default=0)

    def select_victim(self):
        if not self._freq:
            return None
        if self._min_freq not in self._buckets:
            self._min_freq = min(self._buckets, default=0)
            if not self._min_freq:
                return None
        for key in self._buckets[self._min_freq]:
            return key
        return None

    def age(self):
        """Halve every frequency so old popularity decays."""
        if not self._freq:
            return
        rebuilt = defaultdict(OrderedDict)
        for key, freq in self._freq.items():
            halved = max(1, freq // 2)
            self._freq[key] = halved
            rebuilt[halved][key] = True
        self._buckets = rebuilt
        self._min_freq = min(rebuilt, default=0)

    def clear(self):
        self._freq.clear()
        self._buckets.clear()
        self._min_freq = 0

    def keys(self):
        return iter(self._freq)

    def stats(self):
        data = super().stats()
        data["min_frequency"] = self._min_freq
        return data

    def __len__(self):
        return len(self._freq)

    def __contains__(self, key):
        return key in self._freq
