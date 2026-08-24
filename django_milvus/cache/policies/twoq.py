"""2Q eviction (Johnson & Shasha)."""

from collections import OrderedDict

from .base import EvictionPolicy, register


@register
class TwoQPolicy(EvictionPolicy):
    """Scan-resistant LRU using an admission queue and a ghost list.

    Three structures:

    ``A1in``   FIFO queue every new key enters. Evicted from here first.
    ``A1out``  Ghost list of keys recently evicted from A1in. Holds keys
               only, no payloads, so it is nearly free.
    ``Am``     Main LRU, holding keys that proved themselves.

    Promotion needs a *second* access, and crucially a key re-requested
    after falling out of A1in (that is, found in A1out) jumps straight to
    Am. That is the distinguishing move: it recognises keys whose reuse
    distance is longer than the admission queue, which plain LRU and SLRU
    both miss.

    Strong choice for mixed workloads where a big scan runs alongside a
    steady hot set - exactly a periodic re-index next to live traffic.
    """

    name = "2q"

    def __init__(self, capacity=1024, kin_ratio=0.25, kout_ratio=0.5,
                 **options):
        super().__init__(capacity, **options)
        self.kin_ratio = kin_ratio
        self.kout_ratio = kout_ratio
        self._a1in = OrderedDict()
        self._a1out = OrderedDict()
        self._am = OrderedDict()
        self._resize_segments()

    def _resize_segments(self):
        self._kin = max(1, int(self.capacity * self.kin_ratio))
        self._kout = max(1, int(self.capacity * self.kout_ratio))
        self._main_capacity = max(1, self.capacity - self._kin)

    def set_capacity(self, capacity):
        super().set_capacity(capacity)
        self._resize_segments()
        self._trim_ghosts()

    def on_admit(self, key, size=0, expires_at=None):
        if key in self._am:
            self._am.move_to_end(key)
            return
        if key in self._a1in:
            self._a1in.move_to_end(key)
            return
        if key in self._a1out:
            # Seen before and asked for again: it belongs in the main
            # queue, not back at the start of admission.
            del self._a1out[key]
            self._am[key] = True
            self._am.move_to_end(key)
            return
        self._a1in[key] = True

    def on_hit(self, key):
        if key in self._am:
            self._am.move_to_end(key)
        elif key in self._a1in:
            # A hit while still in admission promotes to the main queue.
            del self._a1in[key]
            self._am[key] = True

    def on_remove(self, key):
        if self._a1in.pop(key, None) is not None:
            # Remember it as a ghost so a prompt re-request is recognised.
            self._a1out[key] = True
            self._trim_ghosts()
            return
        self._am.pop(key, None)
        self._a1out.pop(key, None)

    def _trim_ghosts(self):
        while len(self._a1out) > self._kout:
            self._a1out.popitem(last=False)

    def select_victim(self):
        # Drain admission first, but only while it is over its share -
        # otherwise a steady stream of new keys would never let Am shrink.
        if len(self._a1in) > self._kin:
            for key in self._a1in:
                return key
        for key in self._am:
            return key
        for key in self._a1in:
            return key
        return None

    def clear(self):
        self._a1in.clear()
        self._a1out.clear()
        self._am.clear()

    def keys(self):
        yield from self._a1in
        yield from self._am

    def stats(self):
        data = super().stats()
        data.update({
            "a1in": len(self._a1in),
            "a1out_ghosts": len(self._a1out),
            "am": len(self._am),
            "kin": self._kin,
            "kout": self._kout,
        })
        return data

    def __len__(self):
        return len(self._a1in) + len(self._am)

    def __contains__(self, key):
        return key in self._a1in or key in self._am
