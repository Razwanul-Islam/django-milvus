"""Window TinyLFU eviction - the default policy."""

from collections import OrderedDict

from .base import EvictionPolicy, register
from .sketch import CountMinSketch


@register
class WTinyLFUPolicy(EvictionPolicy):
    """Frequency-gated admission in front of a segmented LRU.

    Structure::

        new key -> [ window LRU ]
                        |
                   (overflow: candidate)
                        |
                   frequency duel  --- candidate loses --> evict candidate
                        |
                   candidate wins --> [ probation | protected ]  (SLRU main)
                                            evict main victim

    Every key enters a small *window* LRU sized by ``admission_ratio``.
    When the window overflows, its oldest key becomes a *candidate* and is
    weighed against the main region's victim using estimated frequencies
    from a Count-Min Sketch. The more popular of the two survives.

    That duel is the entire idea: what earns a key a place in the main
    region is how often it has been seen, not the accident of having just
    arrived. Plain LRU admits every newcomer unconditionally, so a stream
    of one-off queries walks straight through the cache evicting proven
    entries. W-TinyLFU makes a newcomer outrank something real first, at a
    cost of a few bits per key.

    Why it is the default here: vector-search traffic is heavily skewed. A
    small set of queries repeats constantly beneath a long tail that
    arrives once and never returns. The window still absorbs genuine
    bursts of new-but-soon-popular keys, so bursty traffic is not starved.

    Note that filtering happens at the *window boundary*, not at the cache
    door - ``should_admit`` always says yes, because a key must be allowed
    into the window before it can prove anything. The filter shows up in
    :meth:`select_victim`, which surrenders the duel's loser first.
    """

    name = "w-tinylfu"

    def __init__(self, capacity=1024, admission_ratio=0.01,
                 probation_ratio=0.2, **options):
        super().__init__(capacity, **options)
        self.admission_ratio = admission_ratio
        self.probation_ratio = probation_ratio

        self._window = OrderedDict()
        self._probation = OrderedDict()
        self._protected = OrderedDict()

        self._sketch = CountMinSketch(capacity=self.capacity)
        self._resize_segments()

        # The loser of the most recent duel: first in line to be evicted.
        self._pending_victim = None

        self.duels = 0
        self.admitted = 0
        self.rejected = 0

    # ── sizing ───────────────────────────────────────────

    def _resize_segments(self):
        self._window_capacity = max(
            1, int(self.capacity * self.admission_ratio)
        )
        main = max(1, self.capacity - self._window_capacity)
        self._main_capacity = main
        self._probation_capacity = max(1, int(main * self.probation_ratio))
        self._protected_capacity = max(1, main - self._probation_capacity)

    def set_capacity(self, capacity):
        super().set_capacity(capacity)
        self._resize_segments()
        self._sketch.resize(self.capacity)
        self._enforce_protected()
        self._drain_window()

    def set_admission_ratio(self, ratio):
        """Retune the window/main split (used by the adaptive controller)."""
        self.admission_ratio = max(0.0, min(0.8, float(ratio)))
        self._resize_segments()
        self._enforce_protected()
        self._drain_window()

    # ── policy hooks ─────────────────────────────────────

    def on_admit(self, key, size=0, expires_at=None):
        self._sketch.increment(key)
        if key in self._protected:
            self._protected.move_to_end(key)
            return
        if key in self._probation:
            self._probation.move_to_end(key)
            return
        self._window[key] = True
        self._window.move_to_end(key)
        self._drain_window()

    def on_hit(self, key):
        self._sketch.increment(key)
        if key in self._window:
            self._window.move_to_end(key)
            return
        if key in self._protected:
            self._protected.move_to_end(key)
            return
        if key in self._probation:
            # Proven twice: promote out of probation.
            del self._probation[key]
            self._protected[key] = True
            if self._pending_victim == key:
                self._pending_victim = None
            self._enforce_protected()

    def _drain_window(self):
        """Move window overflow into the main region, subject to the duel."""
        while len(self._window) > self._window_capacity:
            candidate, _ = self._window.popitem(last=False)

            if not self._main_full():
                # Room to spare: no need to displace anything.
                self._probation[candidate] = True
                continue

            victim = self._main_victim()
            if victim is None:
                self._probation[candidate] = True
                continue

            self.duels += 1
            self._probation[candidate] = True
            if self._sketch.frequency(candidate) > self._sketch.frequency(victim):
                # Newcomer is genuinely more popular: the incumbent goes.
                self.admitted += 1
                self._pending_victim = victim
            else:
                # Not proven: the newcomer is first out. It keeps its place
                # in probation until the backend actually needs the room,
                # so a burst of hits can still rescue it.
                self.rejected += 1
                self._pending_victim = candidate

    def _main_full(self):
        return len(self._probation) + len(self._protected) >= self._main_capacity

    def _enforce_protected(self):
        while len(self._protected) > self._protected_capacity:
            demoted, _ = self._protected.popitem(last=False)
            self._probation[demoted] = True
            self._probation.move_to_end(demoted, last=False)

    def on_remove(self, key):
        self._window.pop(key, None)
        self._probation.pop(key, None)
        self._protected.pop(key, None)
        if self._pending_victim == key:
            self._pending_victim = None

    def should_admit(self, key):
        """Always yes - see the class docstring.

        A key cannot win a frequency duel before it has been seen, so
        refusing it at the door would mean nothing new could ever enter.
        The window exists to give it that chance; the filtering happens
        when the window overflows.
        """
        return True

    def on_reject(self, key):
        # Count the sighting anyway: a key refused repeatedly accumulates
        # frequency and eventually wins its duel.
        self._sketch.increment(key)

    def _main_victim(self):
        for key in self._probation:
            return key
        for key in self._protected:
            return key
        return None

    def select_victim(self):
        # The most recent duel already decided who should go.
        pending = self._pending_victim
        if pending is not None and pending in self:
            self._pending_victim = None
            return pending

        if len(self._window) > self._window_capacity:
            for key in self._window:
                return key
        victim = self._main_victim()
        if victim is not None:
            return victim
        for key in self._window:
            return key
        return None

    def age(self):
        self._sketch.age()

    def clear(self):
        self._window.clear()
        self._probation.clear()
        self._protected.clear()
        self._sketch.clear()
        self._pending_victim = None
        self.duels = 0
        self.admitted = 0
        self.rejected = 0

    def keys(self):
        yield from self._window
        yield from self._probation
        yield from self._protected

    def frequency(self, key):
        """Estimated access frequency, exposed for tests and debugging."""
        return self._sketch.frequency(key)

    def stats(self):
        data = super().stats()
        data.update({
            "window": len(self._window),
            "probation": len(self._probation),
            "protected": len(self._protected),
            "window_capacity": self._window_capacity,
            "admission_ratio": round(self.admission_ratio, 4),
            "duels": self.duels,
            "admitted": self.admitted,
            "rejected": self.rejected,
            "sketch": self._sketch.stats(),
        })
        return data

    def __len__(self):
        return (
            len(self._window) + len(self._probation) + len(self._protected)
        )

    def __contains__(self, key):
        return (
            key in self._window
            or key in self._probation
            or key in self._protected
        )
