"""
Cache statistics: counters, sliding-window hit rate, latency percentiles.

Two views are kept because they answer different questions:

* **Lifetime counters** - "how much work has this cache saved since the
  process started?" Cheap monotonic integers.
* **Sliding window** - "is the cache working *right now*?" A ring of
  per-second buckets covering ``STATS.window`` seconds. This is what the
  adaptive window controller reads, since a lifetime average would take
  hours to reflect a workload change.

Latency is tracked as a fixed-bucket histogram rather than a reservoir,
so percentiles cost no allocation on the query path.
"""

import threading
import time

# Upper bounds in milliseconds. Chosen to straddle both ends of what this
# cache sees: L1 hits land in the first buckets, Milvus round trips in
# the tail.
LATENCY_BUCKETS_MS = (
    0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000,
)


class _Window:
    """Ring buffer of per-second hit/miss counts."""

    def __init__(self, seconds):
        self.seconds = max(1, int(seconds))
        self._hits = [0] * self.seconds
        self._misses = [0] * self.seconds
        self._stamps = [0] * self.seconds

    def _slot(self, now):
        index = int(now) % self.seconds
        # Reset the bucket when it belongs to an older revolution.
        if self._stamps[index] != int(now):
            self._stamps[index] = int(now)
            self._hits[index] = 0
            self._misses[index] = 0
        return index

    def record(self, hit, now=None):
        now = now if now is not None else time.time()
        index = self._slot(now)
        if hit:
            self._hits[index] += 1
        else:
            self._misses[index] += 1

    def totals(self, now=None):
        now = int(now if now is not None else time.time())
        oldest = now - self.seconds
        hits = misses = 0
        for i in range(self.seconds):
            if self._stamps[i] > oldest:
                hits += self._hits[i]
                misses += self._misses[i]
        return hits, misses

    def hit_rate(self, now=None):
        hits, misses = self.totals(now)
        total = hits + misses
        return (hits / total) if total else 0.0

    def reset(self):
        self._hits = [0] * self.seconds
        self._misses = [0] * self.seconds
        self._stamps = [0] * self.seconds


class CacheStats:
    """Thread-safe statistics for one cache alias."""

    def __init__(self, window=300, enabled=True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._window = _Window(window)
        self._latency = [0] * (len(LATENCY_BUCKETS_MS) + 1)
        self._latency_sum = 0.0
        self.reset()

    def reset(self):
        with self._lock:
            self.hits = 0
            self.l1_hits = 0
            self.l2_hits = 0
            self.semantic_hits = 0
            self.misses = 0
            self.negative_hits = 0
            self.stale_hits = 0
            self.sets = 0
            self.rejected = 0
            self.evictions = 0
            self.expirations = 0
            self.invalidations = 0
            self.errors = 0
            self.stampede_waits = 0
            self.bytes_written = 0
            self.bytes_evicted = 0
            self.started_at = time.time()
            self._latency = [0] * (len(LATENCY_BUCKETS_MS) + 1)
            self._latency_sum = 0.0
            self._window.reset()

    # ── recording ────────────────────────────────────────

    def record_hit(self, elapsed=None):
        """Record one lookup served from cache.

        Deliberately separate from :meth:`record_tier`. A backend knows
        which tier answered but not whether the surrounding lookup was a
        hit; the orchestrator knows the opposite. Splitting them keeps a
        tiered cache from counting the same hit twice - once on the way
        out of the backend and again on the way out of the orchestrator.
        """
        if not self.enabled:
            return
        with self._lock:
            self.hits += 1
            self._window.record(True)
            if elapsed is not None:
                self._record_latency(elapsed)

    def record_tier(self, tier):
        """Record which storage tier answered, without counting a hit.

        ``l1_hits + l2_hits`` therefore equals ``hits``: every hit came
        from exactly one tier. ``semantic_hits`` is orthogonal - it counts
        *how* the entry was found, not where it was stored.
        """
        if not self.enabled:
            return
        with self._lock:
            if tier == "l1":
                self.l1_hits += 1
            elif tier == "l2":
                self.l2_hits += 1

    def record_semantic_hit(self):
        """Record that a lookup was answered by nearest-vector matching."""
        if not self.enabled:
            return
        with self._lock:
            self.semantic_hits += 1

    def record_miss(self, elapsed=None):
        if not self.enabled:
            return
        with self._lock:
            self.misses += 1
            self._window.record(False)
            if elapsed is not None:
                self._record_latency(elapsed)

    def record_set(self, size=0):
        if not self.enabled:
            return
        with self._lock:
            self.sets += 1
            self.bytes_written += size

    def record_rejected(self):
        if not self.enabled:
            return
        with self._lock:
            self.rejected += 1

    def record_eviction(self, count=1, freed=0):
        if not self.enabled:
            return
        with self._lock:
            self.evictions += count
            self.bytes_evicted += freed

    def record_expiration(self, count=1):
        if not self.enabled:
            return
        with self._lock:
            self.expirations += count

    def record_invalidation(self, count=1):
        if not self.enabled:
            return
        with self._lock:
            self.invalidations += count

    def record_error(self):
        if not self.enabled:
            return
        with self._lock:
            self.errors += 1

    def record_stampede_wait(self):
        if not self.enabled:
            return
        with self._lock:
            self.stampede_waits += 1

    def record_negative_hit(self):
        if not self.enabled:
            return
        with self._lock:
            self.negative_hits += 1

    def record_stale_hit(self):
        if not self.enabled:
            return
        with self._lock:
            self.stale_hits += 1

    def _record_latency(self, seconds):
        """Caller must hold the lock."""
        millis = seconds * 1000.0
        self._latency_sum += millis
        for index, bound in enumerate(LATENCY_BUCKETS_MS):
            if millis <= bound:
                self._latency[index] += 1
                return
        self._latency[-1] += 1

    # ── reading ──────────────────────────────────────────

    def percentile(self, fraction):
        """Approximate latency percentile in milliseconds.

        Returns the upper bound of the bucket the percentile falls in;
        the overflow bucket reports ``inf``.
        """
        with self._lock:
            total = sum(self._latency)
            if not total:
                return 0.0
            target = total * fraction
            running = 0
            for index, count in enumerate(self._latency):
                running += count
                if running >= target:
                    if index >= len(LATENCY_BUCKETS_MS):
                        return float("inf")
                    return LATENCY_BUCKETS_MS[index]
        return 0.0

    def hit_rate(self):
        """Lifetime hit rate across all lookups."""
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def recent_hit_rate(self):
        """Hit rate over the configured sliding window."""
        with self._lock:
            return self._window.hit_rate()

    def as_dict(self):
        with self._lock:
            lookups = self.hits + self.misses
            data = {
                "enabled": self.enabled,
                "lookups": lookups,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(
                    (self.hits / lookups) if lookups else 0.0, 4
                ),
                "recent_hit_rate": round(self._window.hit_rate(), 4),
                "l1_hits": self.l1_hits,
                "l2_hits": self.l2_hits,
                "semantic_hits": self.semantic_hits,
                "negative_hits": self.negative_hits,
                "stale_hits": self.stale_hits,
                "sets": self.sets,
                "rejected": self.rejected,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "invalidations": self.invalidations,
                "errors": self.errors,
                "stampede_waits": self.stampede_waits,
                "bytes_written": self.bytes_written,
                "bytes_evicted": self.bytes_evicted,
                "uptime": round(time.time() - self.started_at, 1),
            }
        data["latency_p50_ms"] = self.percentile(0.50)
        data["latency_p95_ms"] = self.percentile(0.95)
        data["latency_p99_ms"] = self.percentile(0.99)
        return data


def prometheus_metrics(stats_by_alias):
    """Render stats in Prometheus text exposition format.

    ``stats_by_alias`` maps alias -> the dict returned by
    :meth:`CacheStats.as_dict` (optionally merged with backend stats).
    """
    counters = {
        "lookups": "Total cache lookups",
        "hits": "Total cache hits",
        "misses": "Total cache misses",
        "l1_hits": "Hits served from the in-process tier",
        "l2_hits": "Hits served from the shared tier",
        "semantic_hits": "Hits served by nearest-vector matching",
        "sets": "Payloads written to the cache",
        "rejected": "Payloads rejected as too large",
        "evictions": "Entries evicted to reclaim capacity",
        "expirations": "Entries removed after expiring",
        "invalidations": "Entries invalidated by writes",
        "errors": "Backend errors (all fail open)",
        "bytes_written": "Bytes written to the cache",
        "bytes_evicted": "Bytes reclaimed by eviction",
    }
    gauges = {
        "hit_rate": "Lifetime hit rate",
        "recent_hit_rate": "Hit rate over the recent sliding window",
        "entries": "Entries currently held",
        "bytes": "Bytes currently held",
        "latency_p50_ms": "Median lookup latency in milliseconds",
        "latency_p95_ms": "95th percentile lookup latency in milliseconds",
        "latency_p99_ms": "99th percentile lookup latency in milliseconds",
    }

    lines = []
    for field, help_text in counters.items():
        name = f"milvus_cache_{field}_total"
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        for alias, data in stats_by_alias.items():
            if field in data:
                lines.append(f'{name}{{alias="{alias}"}} {data[field]}')

    for field, help_text in gauges.items():
        name = f"milvus_cache_{field}"
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        for alias, data in stats_by_alias.items():
            value = data.get(field)
            if value is None:
                continue
            if value == float("inf"):
                value = "+Inf"
            lines.append(f'{name}{{alias="{alias}"}} {value}')

    return "\n".join(lines) + "\n"
