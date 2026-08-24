"""
Two-tier cache: a fast local L1 in front of a shared L2.

Read path::

    get(key)
      -> L1 hit?  return it                    (~microseconds)
      -> L2 hit?  promote into L1, return it   (~1ms, once per worker)
      -> miss                                  (query Milvus)

Promotion is what makes the pairing worth more than either tier alone.
The first worker to want a key pays Milvus; every other worker pays one
Redis round trip; and each worker pays that only once, because the answer
lands in its own L1 on the way past.

Writes go to both tiers. L2 gets the authoritative copy with the full
TTL; L1 gets a copy that inherits the same absolute deadline, so an entry
does not quietly gain extra life by being promoted.

**Failure handling.** Every L2 call is wrapped. A failure is recorded,
logged once at warning, and treated as a miss - the query falls through
to Milvus and the request succeeds. After ``CIRCUIT_BREAKER.failures``
consecutive failures the tier is skipped outright for
``CIRCUIT_BREAKER.reset_after`` seconds, so a dead Redis costs one failed
connection attempt every 30 seconds instead of one per request. That
matters: without a breaker, a Redis outage with a 200 ms socket timeout
would add 200 ms to *every* request and turn a cache problem into an
availability problem.
"""

import logging
import time

from .base import MISSING, BaseCacheBackend

logger = logging.getLogger("django_milvus.cache")


class CircuitBreaker:
    """Trips after repeated failures, retries after a cool-off."""

    CLOSED = "closed"      # healthy, calls go through
    OPEN = "open"          # tripped, calls are skipped
    HALF_OPEN = "half"     # cool-off elapsed, letting one call probe

    def __init__(self, failures=5, reset_after=30):
        self.threshold = max(1, failures)
        self.reset_after = reset_after
        self.failure_count = 0
        self.opened_at = None
        self.trips = 0

    @property
    def state(self):
        if self.opened_at is None:
            return self.CLOSED
        if time.time() - self.opened_at >= self.reset_after:
            return self.HALF_OPEN
        return self.OPEN

    def allows(self):
        """Whether a call should be attempted right now."""
        return self.state != self.OPEN

    def record_success(self):
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold and self.opened_at is None:
            self.opened_at = time.time()
            self.trips += 1
            logger.warning(
                "django-milvus cache: shared tier tripped after %d consecutive "
                "failures; skipping it for %ss",
                self.failure_count, self.reset_after,
            )

    def stats_dict(self):
        return {
            "state": self.state,
            "failures": self.failure_count,
            "trips": self.trips,
        }


class TieredCache(BaseCacheBackend):
    """Composes a local L1 with an optional shared L2."""

    name = "tiered"

    def __init__(self, l1, l2=None, config=None, stats=None,
                 breaker=None, **options):
        super().__init__(config=config, stats=stats, **options)
        self.l1 = l1
        self.l2 = l2
        self.breaker = breaker or CircuitBreaker(
            failures=getattr(config.l2, "breaker_failures", 5) if config
            and getattr(config, "l2", None) else 5,
            reset_after=getattr(config.l2, "breaker_reset_after", 30) if config
            and getattr(config, "l2", None) else 30,
        )
        self.promotions = 0
        self.l2_errors = 0

        # Does the shared tier report expiry, or only values? Bundled
        # backends do; a custom one need not. Checked once here rather
        # than probed per request, and it decides whether a promoted
        # entry can inherit its original deadline.
        # getattr rather than attribute access: a custom backend need not
        # subclass BaseCacheBackend, only match its interface.
        self._l2_has_entries = (
            l2 is not None
            and getattr(type(l2), "get_entry", None)
            not in (None, BaseCacheBackend.get_entry)
        )

    @property
    def shared(self):
        return self.l2 is not None and self.l2.shared

    # ── L2 call guard ────────────────────────────────────

    def _l2_call(self, operation, *args, **kwargs):
        """Run an L2 operation, failing open on any error."""
        if self.l2 is None or not self.breaker.allows():
            return MISSING
        try:
            result = getattr(self.l2, operation)(*args, **kwargs)
        except Exception:
            self.l2_errors += 1
            self.breaker.record_failure()
            if self.stats:
                self.stats.record_error()
            logger.warning(
                "django-milvus cache: shared tier %s failed; falling through",
                operation, exc_info=True,
            )
            return MISSING
        self.breaker.record_success()
        return result

    # ── reads ────────────────────────────────────────────

    def get(self, key, default=MISSING):
        value = self.l1.get(key, MISSING)
        if value is not MISSING:
            # The local tier already recorded itself; nothing to add.
            return value

        if self.l2 is None:
            return default

        if self._l2_has_entries:
            entry = self._l2_call("get_entry", key)
            if entry is MISSING or entry is None or entry.is_expired():
                return default
            value = entry.value
            size = entry.size or None
            # Promote with the original deadline rather than restarting
            # the clock - a promoted entry must not outlive its L2 twin.
            remaining = None
            if entry.expires_at is not None:
                remaining = max(0.001, entry.expires_at - time.time())
        else:
            # A backend that cannot report expiry: promote with the
            # configured TTL. Slightly generous, still bounded.
            value = self._l2_call("get", key, MISSING)
            if value is MISSING:
                return default
            size = None
            remaining = getattr(self.config, "ttl", None)

        self.l1.set(key, value, ttl=remaining, size=size)
        self.promotions += 1
        if self.stats:
            self.stats.record_tier("l2")
        return value

    def get_entry(self, key):
        entry = self.l1.get_entry(key)
        if entry is not MISSING:
            return entry
        if self.l2 is None:
            return MISSING
        return self._l2_call("get_entry", key)

    def get_many(self, keys):
        keys = list(keys)
        found = self.l1.get_many(keys)
        missing = [k for k in keys if k not in found]
        if not missing or self.l2 is None:
            return found

        from_l2 = self._l2_call("get_many", missing)
        if from_l2 is MISSING or not from_l2:
            return found
        for key, value in from_l2.items():
            self.l1.set(key, value)
            self.promotions += 1
            found[key] = value
        return found

    # ── writes ───────────────────────────────────────────

    def set(self, key, value, ttl=None, size=None):
        admitted = self.l1.set(key, value, ttl=ttl, size=size)
        # Write to L2 even when L1 refused: the payload may be too large
        # for a local budget yet perfectly fine in Redis, and another
        # worker may well want it.
        self._l2_call("set", key, value, ttl=ttl, size=size)
        return admitted

    def set_many(self, items, ttl=None):
        count = self.l1.set_many(items, ttl=ttl)
        self._l2_call("set_many", items, ttl=ttl)
        return count

    def delete(self, key):
        local = self.l1.delete(key)
        shared = self._l2_call("delete", key)
        return bool(local) or bool(shared is not MISSING and shared)

    def delete_many(self, keys):
        keys = list(keys)
        count = self.l1.delete_many(keys)
        self._l2_call("delete_many", keys)
        return count

    def delete_prefix(self, prefix):
        count = self.l1.delete_prefix(prefix)
        shared = self._l2_call("delete_prefix", prefix)
        if shared is not MISSING and isinstance(shared, int):
            count += shared
        return count

    def touch(self, key, ttl=None):
        local = self.l1.touch(key, ttl=ttl)
        self._l2_call("touch", key, ttl=ttl)
        return local

    def clear(self):
        count = self.l1.clear()
        shared = self._l2_call("clear")
        if shared is not MISSING and isinstance(shared, int):
            count += shared
        return count

    def incr_version(self, key, delta=1):
        """Bump a version stamp, preferring the shared tier.

        When L2 is present its counter is authoritative, because it is the
        only one every worker can see. If it is unreachable we fall back
        to the local counter: invalidation then applies to this process
        only, and other workers stay correct via TTL.
        """
        if self.l2 is not None:
            result = self._l2_call("incr_version", key, delta)
            if result is not MISSING:
                return result
        return self.l1.incr_version(key, delta)

    def purge_expired(self):
        return self.l1.purge_expired()

    def age_policies(self):
        self.l1.age_policies()

    def tick_window(self):
        self.l1.tick_window()

    def set_capacity(self, max_memory):
        self.l1.set_capacity(max_memory)

    def close(self):
        self.l1.close()
        if self.l2 is not None:
            try:
                self.l2.close()
            except Exception:  # pragma: no cover
                pass

    # ── introspection ────────────────────────────────────

    def __len__(self):
        return len(self.l1)

    def stats_dict(self):
        data = {
            "backend": "tiered",
            "shared": self.shared,
            "promotions": self.promotions,
            "l2_errors": self.l2_errors,
            "circuit_breaker": self.breaker.stats_dict(),
            "l1": self.l1.stats_dict(),
        }
        # Roll the local tier's numbers up to the top level so that
        # stats_dict()["backend"]["algorithm"] and friends read the same
        # whether or not a shared tier is configured. Callers should not
        # have to branch on the topology to answer basic questions. The
        # full per-tier detail stays under "l1".
        local = data["l1"]
        for field in (
            "entries", "bytes", "utilization", "evictions", "expirations",
            "rejections", "algorithm", "shards", "max_memory", "max_entries",
            "avg_entry_bytes", "policy", "window", "governor",
        ):
            if field in local:
                data[field] = local[field]

        if self.l2 is not None:
            if self.breaker.allows():
                try:
                    data["l2"] = self.l2.stats_dict()
                except Exception:
                    data["l2"] = {"backend": self.l2.name, "reachable": False}
            else:
                data["l2"] = {
                    "backend": self.l2.name,
                    "reachable": False,
                    "note": "circuit breaker open",
                }
        return data
