"""
Cache backend that delegates to ``django.core.cache``.

Use this when you already run a cache Django knows about - Redis via
``django-redis``, Memcached, database, or LocMem in tests - and would
rather not configure a second connection. Whatever backend that alias
uses, this tier inherits: its eviction, its persistence, its clustering.

The trade is control. Django's cache API exposes no byte accounting, no
eviction algorithm choice and no key enumeration, so ``MAX_MEMORY``,
``ALGORITHM`` and the window settings do not apply here. Configure it as
an L2 behind the local L1, which is where those settings live.

Values are stored inside the shared envelope (see :mod:`.base`) so that
expiry survives the round trip and an empty result set stays
distinguishable from a missing key - Django's ``cache.get`` cannot tell a
stored ``None`` from an absent one, which would otherwise break negative
caching outright.
"""

import time

from django.core.cache import caches as django_caches

from .base import (
    MISSING,
    BaseCacheBackend,
    CacheEntry,
    unwrap_payload,
    wrap_payload,
)


class DjangoCacheBackend(BaseCacheBackend):
    """Thin adapter over a configured ``CACHES`` alias."""

    name = "django-cache"

    def __init__(self, config=None, stats=None, location="default",
                 prefix="dmv", **options):
        super().__init__(config=config, stats=stats, **options)
        # LOCATION names a CACHES alias here, not a network address.
        self.cache_alias = location or "default"
        self.prefix = prefix
        self._keys_seen = set()

    @property
    def cache(self):
        return django_caches[self.cache_alias]

    @property
    def shared(self):
        """Whether this tier is visible to other processes.

        LocMem is per-process. Reporting it as shared would make the
        version registry mirror stamps somewhere the other workers cannot
        see, which is worse than not mirroring them at all.
        """
        module = type(self.cache).__module__
        return "locmem" not in module and "dummy" not in module

    def _prefixed(self, key):
        return f"{self.prefix}:{key}" if self.prefix else key

    # ── reads ────────────────────────────────────────────

    def get(self, key, default=MISSING):
        payload = self.cache.get(self._prefixed(key), MISSING)
        if payload is MISSING:
            return default
        value, _, expires_at = unwrap_payload(payload)
        # Django enforces its own expiry, but an envelope written with a
        # longer server-side timeout can still be logically stale.
        if expires_at is not None and time.time() >= expires_at:
            return default
        return value

    def get_entry(self, key):
        payload = self.cache.get(self._prefixed(key), MISSING)
        if payload is MISSING:
            return MISSING
        value, created_at, expires_at = unwrap_payload(payload)
        return CacheEntry(value, expires_at=expires_at, created_at=created_at)

    def get_many(self, keys):
        mapping = {self._prefixed(k): k for k in keys}
        raw = self.cache.get_many(list(mapping))
        found = {}
        for prefixed, payload in raw.items():
            original = mapping.get(prefixed)
            if original is None:
                continue
            found[original] = unwrap_payload(payload)[0]
        return found

    # ── writes ───────────────────────────────────────────

    def set(self, key, value, ttl=None, size=None):
        max_entry = getattr(self.config, "max_entry_bytes", None)
        if max_entry and size and size > max_entry:
            if self.stats:
                self.stats.record_rejected()
            return False
        # Django reads timeout=None as "never expire", which matches a
        # ttl of None.
        self.cache.set(
            self._prefixed(key), wrap_payload(value, ttl), timeout=ttl
        )
        self._keys_seen.add(key)
        if self.stats:
            self.stats.record_set(size or 0)
        return True

    def set_many(self, items, ttl=None):
        payload = {
            self._prefixed(k): wrap_payload(v, ttl) for k, v in items.items()
        }
        self.cache.set_many(payload, timeout=ttl)
        self._keys_seen.update(items)
        return len(items)

    def delete(self, key):
        prefixed = self._prefixed(key)
        existed = self.cache.get(prefixed, MISSING) is not MISSING
        self.cache.delete(prefixed)
        self._keys_seen.discard(key)
        return existed

    def delete_many(self, keys):
        keys = list(keys)
        self.cache.delete_many([self._prefixed(k) for k in keys])
        for key in keys:
            self._keys_seen.discard(key)
        return len(keys)

    def delete_prefix(self, prefix):
        """Best-effort prefix delete.

        Django's API has no key enumeration, so this can only clear keys
        *this process* wrote. Version stamping is what actually makes
        invalidation correct across workers; this is a convenience for
        explicit ``cache_clear()`` calls.
        """
        doomed = [k for k in self._keys_seen if k.startswith(prefix)]
        if doomed:
            self.delete_many(doomed)
        return len(doomed)

    def touch(self, key, ttl=None):
        return bool(self.cache.touch(self._prefixed(key), timeout=ttl))

    # ── version stamps ───────────────────────────────────
    #
    # Stored as bare integers, not envelopes, so the backend's own atomic
    # incr can operate on them.

    def incr_version(self, key, delta=1):
        """Atomic where the underlying backend supports it.

        Redis and Memcached implement ``incr`` server-side. Others fall
        back to get/set, which can lose a concurrent bump - harmless here,
        since a missed increment costs a stale read until the TTL lapses,
        never a wrong one.
        """
        prefixed = self._prefixed(key)
        try:
            return int(self.cache.incr(prefixed, delta))
        except ValueError:
            # Django raises ValueError when the key does not exist yet.
            self.cache.set(prefixed, delta, timeout=None)
            return delta
        except NotImplementedError:  # pragma: no cover - exotic backends
            current = int(self.cache.get(prefixed) or 0)
            updated = current + delta
            self.cache.set(prefixed, updated, timeout=None)
            return updated

    def get_version(self, key, default=0):
        raw = self.cache.get(self._prefixed(key))
        try:
            return int(raw) if raw is not None else default
        except (TypeError, ValueError):
            return default

    # ── housekeeping ─────────────────────────────────────

    def clear(self):
        count = len(self._keys_seen)
        # Only clear what we wrote: the alias may be shared with the rest
        # of the application, and wiping it would be well outside our remit.
        self.delete_many(list(self._keys_seen))
        self._keys_seen.clear()
        return count

    def __len__(self):
        return len(self._keys_seen)

    def stats_dict(self):
        data = super().stats_dict()
        data.update({
            "shared": self.shared,
            "cache_alias": self.cache_alias,
            "prefix": self.prefix,
            "keys_written": len(self._keys_seen),
            "note": "django.core.cache manages its own memory and eviction",
        })
        return data
