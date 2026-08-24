"""
Collection version stamps - how writes invalidate reads.

Every collection carries a monotonic counter. That counter is baked into
every cache key for the collection::

    dmv:default:documents:v7:s:44adb3ded3ed...
                          ^^

Bump it to 8 and every existing key becomes unreachable in one integer
increment. No key scanning, no reverse index from entities to the queries
that touched them, no ``KEYS documents:*`` sweep against a live Redis.
The orphaned entries are reclaimed later by TTL or normal eviction, which
costs nothing extra because eviction was going to happen anyway.

The alternative - tracking which cached queries contain which primary
keys - cannot actually be made exact. A filter expression like
``score > 0.8`` matches rows that do not exist yet, so an insert can
change its result without touching any key the cache has ever seen. Since
precision is unattainable, this takes the cheap correct option: on a
write, assume everything for that collection is suspect.

**Scope.** A bump invalidates one collection, not the whole cache.

**Sharing.** With an L2 the counter lives there and every worker agrees,
so a write in worker 1 invalidates worker 3's local cache too. Reads of
the shared stamp are themselves cached for ``refresh_interval`` seconds -
without that, checking the version would cost a Redis round trip per
query and undo the point of having a local tier.

**Out-of-band writes.** Another service, or direct pymilvus, writes
without bumping anything. TTL is the backstop for that case, which is why
``TTL`` is not optional.
"""

import logging
import threading
import time

from .keys import version_key

logger = logging.getLogger("django_milvus.cache")


class VersionRegistry:
    """Tracks and bumps per-collection version stamps."""

    def __init__(self, alias="default", backend=None, shared=True,
                 refresh_interval=5.0, enabled=True):
        self.alias = alias
        self.backend = backend
        self.shared = shared
        self.refresh_interval = refresh_interval
        self.enabled = enabled

        self._lock = threading.RLock()
        self._versions = {}      # collection -> int
        self._checked_at = {}    # collection -> monotonic timestamp
        self.bumps = 0
        self.refreshes = 0

    # ── reading ──────────────────────────────────────────

    def get(self, collection):
        """Current version for ``collection``.

        Returns 0 when versioning is disabled, which yields a constant key
        prefix - the cache then relies on TTL alone.
        """
        if not self.enabled:
            return 0

        with self._lock:
            local = self._versions.get(collection, 0)
            if not self._should_refresh(collection):
                return local

        remote = self._read_shared(collection)
        with self._lock:
            self._checked_at[collection] = time.monotonic()
            if remote is not None and remote > self._versions.get(collection, 0):
                # Another worker wrote; adopt its stamp so our keys move
                # in step with theirs.
                self._versions[collection] = remote
            return self._versions.get(collection, 0)

    def _should_refresh(self, collection):
        """Caller holds the lock."""
        if not self.shared or self.backend is None:
            return False
        last = self._checked_at.get(collection)
        if last is None:
            return True
        return (time.monotonic() - last) >= self.refresh_interval

    def _read_shared(self, collection):
        if not self.shared or self.backend is None:
            return None
        shared = self._shared_backend()
        if shared is None:
            return None
        self.refreshes += 1
        try:
            getter = getattr(shared, "get_version", None)
            if getter is not None:
                return int(getter(version_key(self.alias, collection), 0))
            raw = shared.get(version_key(self.alias, collection), 0)
            return int(raw or 0)
        except Exception:
            # Unreachable shared tier: keep using the local stamp. Reads
            # stay correct within this worker and TTL bounds the rest.
            logger.debug(
                "django-milvus cache: could not read shared version stamp",
                exc_info=True,
            )
            return None

    def _shared_backend(self):
        backend = self.backend
        if backend is None:
            return None
        inner = getattr(backend, "l2", None)
        if inner is not None:
            breaker = getattr(backend, "breaker", None)
            if breaker is not None and not breaker.allows():
                return None
            return inner
        return backend if getattr(backend, "shared", False) else None

    # ── writing ──────────────────────────────────────────

    def bump(self, collection, reason="write", sender=None):
        """Invalidate every cached entry for ``collection``.

        Returns the new version.
        """
        if not self.enabled:
            return 0

        new_version = None
        shared = self._shared_backend()
        if shared is not None:
            try:
                new_version = int(
                    shared.incr_version(version_key(self.alias, collection))
                )
            except Exception:
                logger.warning(
                    "django-milvus cache: could not bump shared version for "
                    "%r; invalidating locally only", collection, exc_info=True,
                )
                new_version = None

        with self._lock:
            if new_version is None:
                new_version = self._versions.get(collection, 0) + 1
            self._versions[collection] = new_version
            self._checked_at[collection] = time.monotonic()
            self.bumps += 1

        from .signals import _send, cache_invalidated
        _send(cache_invalidated, sender=sender, collection=collection,
              alias=self.alias, version=new_version, reason=reason)
        return new_version

    def bump_many(self, collections, reason="write", sender=None):
        return {c: self.bump(c, reason=reason, sender=sender) for c in collections}

    def reset(self, collection=None):
        """Forget cached stamps, forcing a re-read from the shared tier."""
        with self._lock:
            if collection is None:
                self._versions.clear()
                self._checked_at.clear()
            else:
                self._versions.pop(collection, None)
                self._checked_at.pop(collection, None)

    def known_collections(self):
        with self._lock:
            return sorted(self._versions)

    def stats_dict(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "shared": bool(self.shared and self._shared_backend() is not None),
                "refresh_interval": self.refresh_interval,
                "bumps": self.bumps,
                "shared_reads": self.refreshes,
                "versions": dict(self._versions),
            }

    def __repr__(self):
        return (
            f"<VersionRegistry alias={self.alias!r} "
            f"collections={len(self._versions)} bumps={self.bumps}>"
        )
