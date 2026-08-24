"""
Cache stampede protection.

The failure this prevents: a popular cached query expires, and the fifty
requests already in flight all miss simultaneously and all fire the same
expensive vector search at Milvus. The cache made the spike *worse* -
without it those fifty would have been spread out; with it they arrive
together the instant the entry lapses.

Single-flight fixes it. The first caller to miss becomes the leader and
queries Milvus; the rest wait on the leader and reuse its answer. One
query instead of fifty.

Two levels, because there are two scopes to coordinate:

*In-process* - a ``threading.Event`` per key, which is free and covers
the common case of many threads in one worker.

*Cross-process* - a Redis ``SET NX PX`` lock when a shared tier is
configured, covering many workers on many hosts. Non-leaders wait
briefly and re-check the cache rather than blocking on the lock itself.

Waiters always have an escape hatch. If the leader has not produced an
answer within ``STAMPEDE.timeout``, the waiter queries Milvus itself.
Stampede protection is an optimisation; it must never become a way for
one slow query to stall every request behind it.
"""

import logging
import threading
import time

logger = logging.getLogger("django_milvus.cache")


class _Flight:
    """One in-progress computation that other callers may wait on."""

    __slots__ = ("event", "value", "error", "waiters", "started_at")

    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None
        self.waiters = 0
        self.started_at = time.time()


class SingleFlight:
    """Deduplicates concurrent misses on the same key."""

    def __init__(self, timeout=5, stats=None):
        self.timeout = timeout
        self.stats = stats
        self._lock = threading.Lock()
        self._flights = {}

        self.leads = 0
        self.joins = 0
        self.timeouts = 0

    def run(self, key, producer):
        """Compute ``producer()`` once per key across concurrent callers.

        Returns ``(value, was_leader)``. Every caller either gets the
        leader's value or, on timeout, computes its own - never an error
        from someone else's attempt beyond the one raised by the producer
        itself, which is re-raised to all waiters so failures are visible
        rather than silently retried by fifty threads at once.
        """
        with self._lock:
            flight = self._flights.get(key)
            if flight is None:
                flight = _Flight()
                self._flights[key] = flight
                leader = True
                self.leads += 1
            else:
                flight.waiters += 1
                leader = False
                self.joins += 1

        if leader:
            try:
                flight.value = producer()
            except Exception as exc:
                flight.error = exc
            finally:
                with self._lock:
                    self._flights.pop(key, None)
                flight.event.set()
            if flight.error is not None:
                raise flight.error
            return flight.value, True

        if self.stats:
            self.stats.record_stampede_wait()

        if not flight.event.wait(timeout=self.timeout):
            # The leader is taking too long. Rather than queue behind it
            # indefinitely, do the work ourselves.
            self.timeouts += 1
            logger.debug(
                "django-milvus cache: waited %ss for in-flight query, "
                "proceeding independently", self.timeout,
            )
            return producer(), False

        if flight.error is not None:
            raise flight.error
        return flight.value, False

    def in_flight(self, key):
        with self._lock:
            return key in self._flights

    def stats_dict(self):
        with self._lock:
            active = len(self._flights)
        return {
            "active": active,
            "leads": self.leads,
            "joins": self.joins,
            "timeouts": self.timeouts,
            "timeout": self.timeout,
        }


class DistributedFlight:
    """Cross-process single-flight built on a shared tier's lock.

    Wraps :class:`SingleFlight` so both scopes apply: threads in this
    worker collapse onto one leader, and that leader then competes with
    the other workers for the shared lock.

    A worker that loses the shared lock does not block on it. It sleeps in
    short increments and re-checks the cache, so the moment the winner
    writes its answer everyone else picks it up. If nothing appears within
    the timeout, it proceeds on its own - correctness never depends on the
    lock being honoured.
    """

    #: How often a non-leader re-checks the cache while waiting.
    POLL_INTERVAL = 0.02

    def __init__(self, backend, timeout=5, stats=None):
        self.backend = backend
        self.timeout = timeout
        self.stats = stats
        self.local = SingleFlight(timeout=timeout, stats=stats)

        self.lock_acquired = 0
        self.lock_missed = 0

    def _shared_backend(self):
        """The tier that can hold a cross-process lock, if any."""
        shared = getattr(self.backend, "l2", None)
        if shared is not None and hasattr(shared, "acquire_lock"):
            breaker = getattr(self.backend, "breaker", None)
            if breaker is None or breaker.allows():
                return shared
        if hasattr(self.backend, "acquire_lock"):
            return self.backend
        return None

    def run(self, key, producer, reader=None):
        """Compute ``producer()`` once per key across the whole fleet.

        ``reader`` re-reads the cache; waiters use it to pick up the
        winner's answer.
        """
        def guarded():
            shared = self._shared_backend()
            if shared is None:
                return producer()

            try:
                won = shared.acquire_lock(key, timeout=self.timeout)
            except Exception:
                # Lock unavailable: degrade to per-process dedup, which
                # still removes most of the duplicate work.
                logger.debug(
                    "django-milvus cache: shared lock unavailable", exc_info=True
                )
                return producer()

            if won:
                self.lock_acquired += 1
                try:
                    return producer()
                finally:
                    try:
                        shared.release_lock(key)
                    except Exception:  # pragma: no cover
                        pass

            self.lock_missed += 1
            if reader is not None:
                deadline = time.time() + self.timeout
                while time.time() < deadline:
                    time.sleep(self.POLL_INTERVAL)
                    found = reader()
                    if found is not None:
                        return found
            return producer()

        return self.local.run(key, guarded)

    def stats_dict(self):
        data = self.local.stats_dict()
        data.update({
            "shared_lock_acquired": self.lock_acquired,
            "shared_lock_missed": self.lock_missed,
        })
        return data
