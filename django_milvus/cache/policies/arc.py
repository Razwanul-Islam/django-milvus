"""Adaptive Replacement Cache eviction (Megiddo & Modha)."""

from collections import OrderedDict

from .base import EvictionPolicy, register


@register
class ARCPolicy(EvictionPolicy):
    """Self-balancing between recency and frequency.

    Four lists, two real and two ghost:

    ``T1`` keys seen once (recency half)      ``B1`` ghosts evicted from T1
    ``T2`` keys seen twice or more (frequency) ``B2`` ghosts evicted from T2

    The adaptation is the whole point. A hit in ``B1`` means "I evicted
    something from the recency half too soon", so the target size ``p``
    grows and T1 gets more room. A hit in ``B2`` argues the opposite and
    shrinks ``p``. The split therefore tracks the workload continuously
    with no tuning knob and no sampling interval.

    Pick ARC when the access pattern shifts between recency-driven and
    frequency-driven phases and you would rather not think about it. Its
    cost is bookkeeping: four ordered dicts and up to 2x capacity in
    ghost keys.
    """

    name = "arc"

    def __init__(self, capacity=1024, **options):
        super().__init__(capacity, **options)
        self._t1 = OrderedDict()
        self._t2 = OrderedDict()
        self._b1 = OrderedDict()
        self._b2 = OrderedDict()
        # Target size for T1. Starts balanced.
        self._p = 0

    def on_admit(self, key, size=0, expires_at=None):
        if key in self._t1:
            # Second sighting: move to the frequency half.
            del self._t1[key]
            self._t2[key] = True
            return
        if key in self._t2:
            self._t2.move_to_end(key)
            return

        if key in self._b1:
            # Recency ghost hit: recency was under-provisioned.
            delta = 1 if len(self._b1) >= len(self._b2) else (
                len(self._b2) // max(1, len(self._b1))
            )
            self._p = min(self.capacity, self._p + delta)
            del self._b1[key]
            self._t2[key] = True
            return

        if key in self._b2:
            # Frequency ghost hit: frequency was under-provisioned.
            delta = 1 if len(self._b2) >= len(self._b1) else (
                len(self._b1) // max(1, len(self._b2))
            )
            self._p = max(0, self._p - delta)
            del self._b2[key]
            self._t2[key] = True
            return

        self._t1[key] = True
        self._trim_ghosts()

    def on_hit(self, key):
        if key in self._t1:
            del self._t1[key]
            self._t2[key] = True
        elif key in self._t2:
            self._t2.move_to_end(key)

    def on_remove(self, key):
        if self._t1.pop(key, None) is not None:
            self._b1[key] = True
        elif self._t2.pop(key, None) is not None:
            self._b2[key] = True
        else:
            self._b1.pop(key, None)
            self._b2.pop(key, None)
        self._trim_ghosts()

    def _trim_ghosts(self):
        # Ghost lists are bounded so their memory stays proportional to
        # capacity; they hold keys only, never payloads.
        while len(self._b1) > self.capacity:
            self._b1.popitem(last=False)
        while len(self._b2) > self.capacity:
            self._b2.popitem(last=False)

    def select_victim(self):
        # Evict from whichever half is over its adaptive target.
        if self._t1 and (len(self._t1) > self._p or not self._t2):
            for key in self._t1:
                return key
        for key in self._t2:
            return key
        for key in self._t1:
            return key
        return None

    def set_capacity(self, capacity):
        super().set_capacity(capacity)
        self._p = min(self._p, self.capacity)
        self._trim_ghosts()

    def clear(self):
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._p = 0

    def keys(self):
        yield from self._t1
        yield from self._t2

    def stats(self):
        data = super().stats()
        data.update({
            "t1_recency": len(self._t1),
            "t2_frequency": len(self._t2),
            "b1_ghosts": len(self._b1),
            "b2_ghosts": len(self._b2),
            "target_p": self._p,
        })
        return data

    def __len__(self):
        return len(self._t1) + len(self._t2)

    def __contains__(self, key):
        return key in self._t1 or key in self._t2
