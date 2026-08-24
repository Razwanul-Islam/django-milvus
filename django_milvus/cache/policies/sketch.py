"""
Count-Min Sketch with 4-bit counters, for W-TinyLFU admission.

Tracking exact access frequencies would cost more memory than the cached
payloads themselves. A Count-Min Sketch answers "roughly how often has
this key been seen?" in a fixed, tiny footprint - and admission control
only needs to compare two frequencies, so a small overestimate is
harmless.

Two details make this practical for a long-lived cache:

*Saturation at 15.* Counters are 4 bits, so frequency tops out at 15.
That is plenty: once a key is hit 15 times it is unambiguously hot.

*Periodic aging.* Every ``reset_at`` increments, all counters are halved.
Without it the sketch would record all-time popularity and yesterday's
hot keys would keep out today's. Halving keeps relative ordering while
letting the sketch forget.

A doorkeeper bloom filter absorbs the long tail of keys seen exactly
once, which in a vector-search workload is most of them.
"""


class CountMinSketch:
    """4-bit counting sketch with halving-based aging."""

    #: Counters saturate here (2**4 - 1).
    MAX_COUNT = 15

    #: Independent 64-bit mixers, applied to one base hash.
    _SEEDS = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
              0x94D049BB133111EB, 0x2545F4914F6CDD1D)

    def __init__(self, capacity=1024, depth=4):
        # Four counters per tracked entry keeps collision error low while
        # staying far smaller than the payloads themselves.
        width = 1
        while width < max(16, capacity * 4):
            width <<= 1
        self.width = width
        self.mask = width - 1
        self.depth = depth
        # Two counters per byte.
        self.table = bytearray(width * depth // 2)

        # Halve everything once we have seen this many increments.
        self.reset_at = capacity * 10
        self.additions = 0
        self.age_count = 0

        self._doorkeeper = _Doorkeeper(capacity)

    # ── internals ────────────────────────────────────────

    def _positions(self, key):
        base = hash(key) & 0xFFFFFFFFFFFFFFFF
        for row, seed in enumerate(self._SEEDS[:self.depth]):
            mixed = (base ^ seed) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF
            mixed ^= mixed >> 29
            yield row * self.width + (mixed & self.mask)

    def _read(self, index):
        byte = self.table[index >> 1]
        return (byte & 0x0F) if (index & 1) == 0 else (byte >> 4)

    def _write(self, index, value):
        slot = index >> 1
        byte = self.table[slot]
        if (index & 1) == 0:
            self.table[slot] = (byte & 0xF0) | (value & 0x0F)
        else:
            self.table[slot] = (byte & 0x0F) | ((value & 0x0F) << 4)

    # ── public API ───────────────────────────────────────

    def increment(self, key):
        """Record one access to ``key``."""
        # First sighting goes to the doorkeeper only, keeping the sketch
        # free of the one-hit tail.
        if not self._doorkeeper.test_and_set(key):
            self.additions += 1
            self._maybe_age()
            return

        for index in self._positions(key):
            current = self._read(index)
            if current < self.MAX_COUNT:
                self._write(index, current + 1)

        self.additions += 1
        self._maybe_age()

    def frequency(self, key):
        """Estimated access count for ``key`` (0-16)."""
        estimate = min(self._read(index) for index in self._positions(key))
        # A key in the doorkeeper has been seen at least once beyond what
        # the sketch recorded.
        if self._doorkeeper.contains(key):
            estimate += 1
        return estimate

    def _maybe_age(self):
        if self.additions >= self.reset_at:
            self.age()

    def age(self):
        """Halve every counter and clear the doorkeeper."""
        table = self.table
        for i in range(len(table)):
            byte = table[i]
            if byte:
                # Halve both nibbles in one pass.
                table[i] = ((byte >> 1) & 0x77)
        self.additions = 0
        self.age_count += 1
        self._doorkeeper.clear()

    def clear(self):
        self.table = bytearray(len(self.table))
        self.additions = 0
        self._doorkeeper.clear()

    def resize(self, capacity):
        """Rebuild for a new capacity, discarding history."""
        self.__init__(capacity=capacity, depth=self.depth)

    def stats(self):
        return {
            "width": self.width,
            "depth": self.depth,
            "bytes": len(self.table),
            "additions": self.additions,
            "age_count": self.age_count,
        }


class _Doorkeeper:
    """Small bloom filter that absorbs first sightings."""

    def __init__(self, capacity):
        bits = 1
        while bits < max(64, capacity * 8):
            bits <<= 1
        self.bits = bits
        self.mask = bits - 1
        self.table = bytearray(bits // 8)

    def _positions(self, key):
        base = hash(key) & 0xFFFFFFFFFFFFFFFF
        first = base & self.mask
        second = ((base >> 32) * 0x9E3779B9) & self.mask
        return first, second

    def test_and_set(self, key):
        """True when ``key`` was already present; marks it either way."""
        seen = True
        for position in self._positions(key):
            slot, bit = position >> 3, 1 << (position & 7)
            if not self.table[slot] & bit:
                seen = False
                self.table[slot] |= bit
        return seen

    def contains(self, key):
        return all(
            self.table[p >> 3] & (1 << (p & 7)) for p in self._positions(key)
        )

    def clear(self):
        self.table = bytearray(len(self.table))
