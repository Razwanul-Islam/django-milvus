"""Random eviction."""

import random

from .base import EvictionPolicy, register


@register
class RandomPolicy(EvictionPolicy):
    """Evict an arbitrary key.

    Keeps no ordering at all, so both admission and hits are free and the
    memory overhead is one set. Surprisingly competitive when access is
    close to uniform, and immune to scan pollution by construction. Use
    it as a control when benchmarking the others.
    """

    name = "random"

    def __init__(self, capacity=1024, seed=None, **options):
        super().__init__(capacity, **options)
        self._keys = set()
        self._sample = []
        self._random = random.Random(seed)

    def on_admit(self, key, size=0, expires_at=None):
        if key not in self._keys:
            self._keys.add(key)
            self._sample.append(key)

    def on_hit(self, key):
        """No-op: random eviction ignores access history."""

    def on_remove(self, key):
        self._keys.discard(key)

    def select_victim(self):
        # The sample list is allowed to hold removed keys; skip them
        # lazily rather than paying O(n) removal on every delete.
        while self._sample:
            index = self._random.randrange(len(self._sample))
            self._sample[index], self._sample[-1] = (
                self._sample[-1], self._sample[index]
            )
            candidate = self._sample.pop()
            if candidate in self._keys:
                self._sample.append(candidate)
                return candidate
        return next(iter(self._keys), None)

    def clear(self):
        self._keys.clear()
        self._sample.clear()

    def keys(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys
