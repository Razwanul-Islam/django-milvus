"""Segmented LRU eviction."""

from collections import OrderedDict

from .base import EvictionPolicy, register


@register
class SLRUPolicy(EvictionPolicy):
    """Two-segment LRU: new keys must earn their place.

    Everything enters *probation*. Only a second access promotes a key to
    *protected*, and eviction always takes the oldest probation key. So a
    one-off query can never displace an entry that has proven itself -
    the scan-resistance plain LRU lacks, at almost the same cost.

    When protected overflows, its coldest key is demoted to probation
    rather than dropped, giving it one more chance to be re-accessed.

    ``probation_ratio`` (from ``L1.WINDOW``) sets the split; 0.2 means 20%
    probation, 80% protected.
    """

    name = "slru"

    def __init__(self, capacity=1024, probation_ratio=0.2, **options):
        super().__init__(capacity, **options)
        self.probation_ratio = probation_ratio
        self._probation = OrderedDict()
        self._protected = OrderedDict()
        self._resize_segments()

    def _resize_segments(self):
        self._probation_capacity = max(
            1, int(self.capacity * self.probation_ratio)
        )
        self._protected_capacity = max(
            1, self.capacity - self._probation_capacity
        )

    def set_capacity(self, capacity):
        super().set_capacity(capacity)
        self._resize_segments()
        self._enforce_protected()

    def on_admit(self, key, size=0, expires_at=None):
        if key in self._protected:
            self._protected.move_to_end(key)
            return
        self._probation[key] = True
        self._probation.move_to_end(key)

    def on_hit(self, key):
        if key in self._protected:
            self._protected.move_to_end(key)
            return
        if key in self._probation:
            # Proven itself: promote.
            del self._probation[key]
            self._protected[key] = True
            self._enforce_protected()

    def _enforce_protected(self):
        """Demote the coldest protected keys back to probation."""
        while len(self._protected) > self._protected_capacity:
            demoted, _ = self._protected.popitem(last=False)
            self._probation[demoted] = True
            self._probation.move_to_end(demoted, last=False)

    def on_remove(self, key):
        self._probation.pop(key, None)
        self._protected.pop(key, None)

    def select_victim(self):
        # Probation first: unproven entries are always cheaper to lose.
        for key in self._probation:
            return key
        for key in self._protected:
            return key
        return None

    def clear(self):
        self._probation.clear()
        self._protected.clear()

    def keys(self):
        yield from self._probation
        yield from self._protected

    def stats(self):
        data = super().stats()
        data.update({
            "probation": len(self._probation),
            "protected": len(self._protected),
            "probation_capacity": self._probation_capacity,
            "protected_capacity": self._protected_capacity,
        })
        return data

    def __len__(self):
        return len(self._probation) + len(self._protected)

    def __contains__(self, key):
        return key in self._probation or key in self._protected
