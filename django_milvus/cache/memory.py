"""
Memory accounting and automatic capacity management.

Two responsibilities:

``estimate_size``
    Approximate the retained heap size of a cached payload, so the cache
    can be bounded in bytes rather than in entry counts. Vector payloads
    vary hugely in size (a 5-row search on a 1536-dim collection is two
    orders of magnitude larger than a 5-row scalar query), which makes
    entry counts a poor proxy for memory.

``MemoryGovernor``
    The background janitor: sweeps expired entries, ages frequency
    sketches, ticks the adaptive window controller, and shrinks effective
    capacity when the process comes under memory pressure.
"""

import logging
import sys
import threading

logger = logging.getLogger("django_milvus.cache")

# Per-container overheads measured on CPython 3.11 (64-bit). Exact values
# differ slightly by version; being within a few percent is enough, since
# these numbers drive eviction decisions, not accounting.
_DICT_ENTRY_OVERHEAD = 100
_LIST_ITEM_OVERHEAD = 8
_FLOAT_SIZE = 24
_INT_SIZE = 28
_STR_OVERHEAD = 49


def estimate_size(obj, _seen=None, _depth=0):
    """Estimate the retained bytes of ``obj``.

    Uses fast arithmetic paths for the shapes actually cached - lists of
    dicts whose values are scalars, strings or float vectors - and falls
    back to a memoised recursive ``sys.getsizeof`` walk for anything else.
    Shared objects are counted once.
    """
    if obj is None:
        return 16

    # Homogeneous float vectors are the hot case: compute, do not walk.
    # A 1536-dim embedding would otherwise cost 1536 getsizeof calls.
    if type(obj) is list and obj:
        first = obj[0]
        if type(first) is float:
            return 56 + len(obj) * (_LIST_ITEM_OVERHEAD + _FLOAT_SIZE)
        if type(first) is int:
            return 56 + len(obj) * (_LIST_ITEM_OVERHEAD + _INT_SIZE)

    if type(obj) is str:
        return _STR_OVERHEAD + len(obj)
    if type(obj) in (int, float, bool):
        return sys.getsizeof(obj)
    if type(obj) is bytes:
        return 33 + len(obj)

    # NumPy arrays report their buffer via nbytes.
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None and isinstance(nbytes, int):
        return nbytes + 128

    if _depth > 12:
        return sys.getsizeof(obj)

    if _seen is None:
        _seen = set()
    marker = id(obj)
    if marker in _seen:
        return 0
    _seen.add(marker)

    if isinstance(obj, dict):
        total = 64 + len(obj) * _DICT_ENTRY_OVERHEAD
        for key, value in obj.items():
            total += estimate_size(key, _seen, _depth + 1)
            total += estimate_size(value, _seen, _depth + 1)
        return total

    if isinstance(obj, (list, tuple, set, frozenset)):
        total = 56 + len(obj) * _LIST_ITEM_OVERHEAD
        for item in obj:
            total += estimate_size(item, _seen, _depth + 1)
        return total

    try:
        total = sys.getsizeof(obj)
    except TypeError:  # pragma: no cover - exotic objects
        return 64

    slots = getattr(obj, "__dict__", None)
    if slots:
        total += estimate_size(slots, _seen, _depth + 1)
    return total


def get_process_rss():
    """Current process resident set size in bytes, or None if unavailable.

    Requires ``psutil``; without it, memory-pressure handling is inert
    rather than an error.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process().memory_info().rss
    except Exception:  # pragma: no cover - platform quirks
        return None


class MemoryGovernor:
    """Background maintenance for an L1 cache.

    Runs one daemon thread per cache (not per shard) that every
    ``sample_interval`` seconds:

    1. sweeps entries whose TTL has passed, so dead payloads do not sit
       occupying budget until they happen to be selected as victims;
    2. ages the frequency sketch, so yesterday's hot keys stop crowding
       out today's;
    3. ticks the adaptive window controller;
    4. checks process RSS and shrinks the cache's effective capacity when
       the process is under pressure, restoring it as pressure clears.

    The thread is started lazily on first use and never blocks shutdown.
    """

    def __init__(self, backend, config):
        self.backend = backend
        self.config = config
        self.base_max_memory = config.max_memory
        self.effective_max_memory = config.max_memory
        self.shrink_factor = 1.0

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.sweeps = 0
        self.expired_swept = 0
        self.pressure_events = 0

    # ── lifecycle ────────────────────────────────────────

    def start(self):
        """Start the janitor thread (idempotent)."""
        if not self.config.janitor:
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"milvus-cache-janitor-{id(self.backend):x}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout=1.0):
        """Signal the janitor to stop and wait briefly for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _run(self):
        interval = self.config.sample_interval
        while not self._stop.wait(interval):
            try:
                self.tick()
            except Exception:  # pragma: no cover - janitor must not die
                logger.warning(
                    "django-milvus cache janitor pass failed", exc_info=True
                )

    # ── one maintenance pass ─────────────────────────────

    def tick(self):
        """Run a single maintenance pass. Safe to call directly in tests."""
        self.sweeps += 1
        self.expired_swept += self.backend.purge_expired()
        self.backend.age_policies()
        self.backend.tick_window()
        self.check_pressure()

    def check_pressure(self):
        """Shrink or restore effective capacity based on process RSS."""
        if not self.config.pressure_enabled or not self.base_max_memory:
            return
        limit = self.config.process_rss_limit
        if not limit:
            return
        rss = get_process_rss()
        if rss is None:
            return

        floor = self.config.floor_ratio
        if rss > limit:
            # Over budget: halve the allowance, bounded by the floor.
            new_factor = max(floor, self.shrink_factor * 0.5)
            if new_factor < self.shrink_factor:
                self.pressure_events += 1
                logger.info(
                    "django-milvus cache: process RSS %.0f MB exceeds limit "
                    "%.0f MB, shrinking cache capacity to %.0f%%",
                    rss / 1048576, limit / 1048576, new_factor * 100,
                )
                self._apply_factor(new_factor)
        elif rss < limit * 0.85 and self.shrink_factor < 1.0:
            # Comfortably back under: recover gradually, not all at once,
            # so we do not oscillate across the threshold.
            new_factor = min(1.0, self.shrink_factor * 1.5)
            self._apply_factor(new_factor)

    def _apply_factor(self, factor):
        self.shrink_factor = factor
        self.effective_max_memory = int(self.base_max_memory * factor)
        self.backend.set_capacity(self.effective_max_memory)

    def stats(self):
        return {
            "sweeps": self.sweeps,
            "expired_swept": self.expired_swept,
            "pressure_events": self.pressure_events,
            "shrink_factor": round(self.shrink_factor, 3),
            "effective_max_memory": self.effective_max_memory,
            "running": bool(self._thread and self._thread.is_alive()),
        }
