"""Least Recently Used eviction."""

from collections import OrderedDict

from .base import EvictionPolicy, register


@register
class LRUPolicy(EvictionPolicy):
    """Evict the key untouched for the longest time.

    Backed by an ``OrderedDict``: ``move_to_end`` and ``popitem(last=False)``
    are both O(1), and the C implementation keeps the constant small.

    Good default when the working set is stable and fits. Its weakness is
    scans - one pass over a large result set flushes every hot key, which
    is exactly what ``2q``, ``arc`` and ``w-tinylfu`` exist to fix.
    """

    name = "lru"

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._order = OrderedDict()

    def on_admit(self, key, size=0, expires_at=None):
        self._order[key] = True
        self._order.move_to_end(key)

    def on_hit(self, key):
        if key in self._order:
            self._order.move_to_end(key)

    def on_remove(self, key):
        self._order.pop(key, None)

    def select_victim(self):
        for key in self._order:
            return key
        return None

    def clear(self):
        self._order.clear()

    def keys(self):
        return iter(self._order)

    def __len__(self):
        return len(self._order)

    def __contains__(self, key):
        return key in self._order
