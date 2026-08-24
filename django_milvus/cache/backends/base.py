"""
Cache backend contract.

A backend stores payloads under string keys with an optional TTL. It does
*not* know about Milvus, models, querysets or semantics - the layer above
handles all of that. This keeps the interface small enough that writing a
custom backend is genuinely easy.

To write one::

    from django_milvus.cache.backends.base import BaseCacheBackend, MISSING

    class MyBackend(BaseCacheBackend):
        def get(self, key, default=MISSING): ...
        def set(self, key, value, ttl=None, size=None): ...
        def delete(self, key): ...
        def clear(self): ...

Then point ``BACKEND`` at its dotted path.

``MISSING`` exists because ``None`` is a legitimate cached value: an empty
result set is worth caching (see ``NEGATIVE_TTL``), and conflating it with
"absent" would defeat that.

Every method must be safe to call from multiple threads. Backends should
raise on genuine failures rather than swallowing them - the tier above
catches, records, and falls through to Milvus, so a failure here degrades
performance but never correctness.
"""

import time


class _Missing:
    """Sentinel distinguishing "no entry" from a cached ``None``."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return "<MISSING>"


MISSING = _Missing()


# ─────────────────────────────────────────────────────────
# Shared-tier envelope
# ─────────────────────────────────────────────────────────

#: Bumped if the envelope layout ever changes, so old entries can be
#: recognised and discarded rather than misread.
ENVELOPE_VERSION = 1
_ENVELOPE_TAG = "__dmv__"


def wrap_payload(value, ttl=None, now=None):
    """Package a value with the metadata a shared tier cannot infer.

    Shared backends store bytes and report nothing about when an entry was
    written or when it expires. Both matter: promoting an L2 entry into L1
    must preserve its *original* deadline, or an entry would gain a fresh
    full TTL every time a new worker touched it and could outlive its own
    invalidation.

    The envelope also settles the ``None`` problem - an empty result set
    is worth caching, and a bare ``None`` in the store is indistinguishable
    from a missing key.
    """
    now = now if now is not None else time.time()
    return {
        _ENVELOPE_TAG: ENVELOPE_VERSION,
        "v": value,
        "c": now,
        "x": (now + ttl) if ttl else None,
    }


def is_envelope(payload):
    return isinstance(payload, dict) and _ENVELOPE_TAG in payload


def unwrap_payload(payload):
    """Return ``(value, created_at, expires_at)`` for a stored payload.

    Values written without an envelope (by an older version, or directly)
    are returned as-is with no timing information.
    """
    if not is_envelope(payload):
        return payload, None, None
    return payload.get("v"), payload.get("c"), payload.get("x")


class CacheEntry:
    """One stored payload plus the bookkeeping the cache needs.

    ``expires_at`` is an absolute monotonic-ish timestamp (``time.time()``)
    rather than a duration, so an entry promoted between tiers keeps its
    original deadline instead of having its life extended.
    """

    __slots__ = (
        "value", "size", "expires_at", "created_at", "last_access", "hits",
    )

    def __init__(self, value, size=0, expires_at=None, created_at=None):
        self.value = value
        self.size = size
        self.expires_at = expires_at
        self.created_at = created_at if created_at is not None else time.time()
        self.last_access = self.created_at
        self.hits = 0

    def is_expired(self, now=None):
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    def is_stale_usable(self, grace, now=None):
        """True when expired but still within the stale-serving grace."""
        if self.expires_at is None:
            return False
        now = now if now is not None else time.time()
        return self.expires_at <= now < self.expires_at + grace

    def age(self, now=None):
        return (now if now is not None else time.time()) - self.created_at

    def touch(self, now=None):
        self.last_access = now if now is not None else time.time()
        self.hits += 1

    def __repr__(self):
        return (
            f"<CacheEntry size={self.size} hits={self.hits} "
            f"expires_at={self.expires_at}>"
        )


class BaseCacheBackend:
    """Interface every cache tier implements."""

    #: Human name used in stats and log messages.
    name = "base"

    #: Whether this tier is shared between processes. The version registry
    #: uses it to decide where stamps can live.
    shared = False

    def __init__(self, config=None, stats=None, **options):
        self.config = config
        self.stats = stats
        self.options = options

    # ── required ─────────────────────────────────────────

    def get(self, key, default=MISSING):
        """Return the value for ``key``, or ``default`` if absent/expired."""
        raise NotImplementedError

    def set(self, key, value, ttl=None, size=None):
        """Store ``value`` under ``key``. Returns True if it was admitted.

        A backend may legitimately refuse (payload too large, admission
        policy rejected it); returning False is not an error.
        """
        raise NotImplementedError

    def delete(self, key):
        """Remove ``key``. Returns True if something was removed."""
        raise NotImplementedError

    def clear(self):
        """Remove everything. Returns the number of entries removed."""
        raise NotImplementedError

    # ── optional ─────────────────────────────────────────

    def get_entry(self, key):
        """Return the raw :class:`CacheEntry`, including expired ones.

        Used for stale-while-revalidate. Backends that cannot expose
        entries return ``MISSING`` and simply forgo stale serving.
        """
        return MISSING

    def get_many(self, keys):
        """Fetch several keys. Returns a dict of the ones that were present."""
        found = {}
        for key in keys:
            value = self.get(key)
            if value is not MISSING:
                found[key] = value
        return found

    def set_many(self, items, ttl=None):
        """Store several key/value pairs."""
        return sum(bool(self.set(k, v, ttl=ttl)) for k, v in items.items())

    def delete_many(self, keys):
        """Remove several keys, returning how many existed."""
        return sum(bool(self.delete(key)) for key in keys)

    def delete_prefix(self, prefix):
        """Remove every key starting with ``prefix``.

        Version stamping means this is rarely needed on the query path;
        it exists for explicit ``cache_clear()`` calls.
        """
        return 0

    def has(self, key):
        return self.get(key) is not MISSING

    def incr_version(self, key, delta=1):
        """Atomically bump a counter, returning its new value.

        Backends shared between processes must make this atomic, since it
        underpins cross-worker invalidation.
        """
        current = self.get(key, 0) or 0
        updated = int(current) + delta
        self.set(key, updated, ttl=None)
        return updated

    def purge_expired(self):
        """Proactively drop expired entries. Returns how many went."""
        return 0

    def touch(self, key, ttl=None):
        """Extend an entry's life without rewriting its payload."""
        return False

    def close(self):
        """Release any resources (sockets, threads)."""

    # ── introspection ────────────────────────────────────

    def stats_dict(self):
        """Backend-specific numbers merged into ``cache_stats()``."""
        return {"backend": self.name, "shared": self.shared}

    def __len__(self):
        return 0

    def __contains__(self, key):
        return self.has(key)

    def __repr__(self):
        return f"<{type(self).__name__} entries={len(self)}>"
