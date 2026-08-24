"""Expiry-ordered eviction."""

import heapq
import itertools

from .base import EvictionPolicy, register


@register
class TTLPolicy(EvictionPolicy):
    """Evict whichever key expires soonest.

    Ignores recency and frequency: the entry closest to its deadline goes
    first, on the reasoning that it was about to disappear anyway. Choose
    it when freshness dominates value - short-TTL entries where evicting
    a nearly-expired row costs almost nothing.

    A lazily-cleaned min-heap keeps admission O(log n) and victim
    selection amortised O(1). Entries with no expiry sort last, so they
    are only evicted once everything with a deadline has gone.
    """

    name = "ttl"

    #: Entries without a TTL sort behind everything that has one.
    NEVER = float("inf")

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._heap = []
        self._entries = {}
        self._counter = itertools.count()

    def on_admit(self, key, size=0, expires_at=None):
        deadline = self.NEVER if expires_at is None else expires_at
        # Monotonic tiebreaker keeps the heap total-ordered without
        # comparing keys, which may not be orderable.
        record = (deadline, next(self._counter), key)
        self._entries[key] = record
        heapq.heappush(self._heap, record)

    def on_hit(self, key):
        """No-op: a hit does not move a deadline."""

    def on_remove(self, key):
        # Leave the heap record in place; select_victim discards stale
        # records on the way past, which is cheaper than an O(n) removal.
        self._entries.pop(key, None)

    def select_victim(self):
        while self._heap:
            record = self._heap[0]
            key = record[2]
            if self._entries.get(key) is record:
                return key
            heapq.heappop(self._heap)
        return None

    def clear(self):
        self._heap.clear()
        self._entries.clear()

    def keys(self):
        return iter(self._entries)

    def stats(self):
        data = super().stats()
        data["heap_size"] = len(self._heap)
        return data

    def __len__(self):
        return len(self._entries)

    def __contains__(self, key):
        return key in self._entries
