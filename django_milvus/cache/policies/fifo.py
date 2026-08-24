"""First In, First Out eviction."""

from collections import OrderedDict

from .base import EvictionPolicy, register


@register
class FIFOPolicy(EvictionPolicy):
    """Evict in insertion order, ignoring access entirely.

    The cheapest policy there is - a hit costs nothing because nothing is
    reordered. Useful when entries have uniform value and a natural
    lifetime (a rolling window of recent queries), or as a baseline when
    measuring what a smarter policy actually buys you.
    """

    name = "fifo"

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._order = OrderedDict()

    def on_admit(self, key, size=0, expires_at=None):
        if key not in self._order:
            self._order[key] = True

    def on_hit(self, key):
        """No-op: insertion order is the only order FIFO knows."""

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
