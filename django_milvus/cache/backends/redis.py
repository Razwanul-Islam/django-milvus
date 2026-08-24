"""
Redis cache backend - the recommended L2 tier.

One cache shared by every worker and every host, so a query paid for by
one process is free for all the others, and a version bump invalidates
the fleet at once. That is what an in-process L1 cannot do.

The cost is a network round trip plus serialization on every access,
which is why this belongs *behind* the local tier rather than instead of
it: L1 answers the repeats, L2 answers everything L1 has not seen yet.

Requires ``redis``::

    pip install django-milvus[cache]

Two design notes worth knowing:

*Server-side expiry.* TTLs are handed to Redis via ``PSETEX`` rather than
being enforced in Python, so an entry expires on schedule even if no
client ever asks for it again.

*Atomic version bumps.* ``INCR`` is atomic on the server, which is what
makes cross-worker invalidation correct without a distributed lock.
"""

import logging

from ...exceptions import CacheBackendError
from ..serializers import Codec, PickleSerializer
from .base import (
    MISSING,
    BaseCacheBackend,
    CacheEntry,
    unwrap_payload,
    wrap_payload,
)

logger = logging.getLogger("django_milvus.cache")


class RedisBackend(BaseCacheBackend):
    """Shared cache tier backed by Redis."""

    name = "redis"
    shared = True

    def __init__(self, config=None, stats=None, location=None, prefix="dmv",
                 codec=None, socket_timeout=0.2, client=None, **options):
        super().__init__(config=config, stats=stats, **options)
        self.location = location or "redis://localhost:6379/0"
        self.prefix = prefix
        self.socket_timeout = socket_timeout

        if codec is not None:
            self.codec = codec
        elif config is not None and getattr(config, "l2", None) is not None:
            self.codec = Codec.from_config(config.l2)
        else:
            self.codec = Codec(PickleSerializer())

        if client is not None:
            self._client = client
        else:
            self._client = self._connect(options)

    def _connect(self, options):
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - import guard
            raise CacheBackendError(
                "The Redis cache backend requires the redis package. "
                "Install it with: pip install django-milvus[cache]"
            ) from exc

        kwargs = {
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_timeout,
            # Payloads are bytes; never let redis-py try to decode them.
            "decode_responses": False,
        }
        kwargs.update(options.get("connection_kwargs") or {})
        try:
            return redis.Redis.from_url(self.location, **kwargs)
        except Exception as exc:
            raise CacheBackendError(
                f"Could not configure Redis at {self.location}: {exc}"
            ) from exc

    @property
    def client(self):
        return self._client

    def _prefixed(self, key):
        return f"{self.prefix}:{key}".encode("utf-8") if self.prefix \
            else key.encode("utf-8")

    # ── reads ────────────────────────────────────────────

    def get(self, key, default=MISSING):
        raw = self._client.get(self._prefixed(key))
        if raw is None:
            return default
        try:
            payload = self.codec.decode(raw)
        except Exception:
            # A payload we cannot read is worse than none: drop it so the
            # next request repopulates cleanly rather than failing forever.
            logger.warning(
                "django-milvus cache: dropping undecodable Redis entry %s",
                key, exc_info=True,
            )
            self._client.delete(self._prefixed(key))
            return default
        return unwrap_payload(payload)[0]

    def get_entry(self, key):
        """Return a :class:`CacheEntry` reconstructed from Redis."""
        raw = self._client.get(self._prefixed(key))
        if raw is None:
            return MISSING
        try:
            payload = self.codec.decode(raw)
        except Exception:
            return MISSING
        value, created_at, expires_at = unwrap_payload(payload)
        return CacheEntry(
            value, size=len(raw), expires_at=expires_at, created_at=created_at
        )

    def get_many(self, keys):
        keys = list(keys)
        if not keys:
            return {}
        raws = self._client.mget([self._prefixed(k) for k in keys])
        found = {}
        for key, raw in zip(keys, raws):
            if raw is None:
                continue
            try:
                payload = self.codec.decode(raw)
            except Exception:
                continue
            found[key] = unwrap_payload(payload)[0]
        return found

    # ── writes ───────────────────────────────────────────

    def set(self, key, value, ttl=None, size=None):
        max_entry = getattr(self.config, "max_entry_bytes", None)
        if max_entry and size and size > max_entry:
            if self.stats:
                self.stats.record_rejected()
            return False

        data = self.codec.encode(wrap_payload(value, ttl))
        if max_entry and len(data) > max_entry:
            if self.stats:
                self.stats.record_rejected()
            return False

        prefixed = self._prefixed(key)
        if ttl:
            # Milliseconds, so sub-second TTLs survive the trip.
            self._client.psetex(prefixed, max(1, int(ttl * 1000)), data)
        else:
            self._client.set(prefixed, data)
        if self.stats:
            self.stats.record_set(len(data))
        return True

    def set_many(self, items, ttl=None):
        if not items:
            return 0
        pipe = self._client.pipeline(transaction=False)
        for key, value in items.items():
            data = self.codec.encode(wrap_payload(value, ttl))
            if ttl:
                pipe.psetex(self._prefixed(key), max(1, int(ttl * 1000)), data)
            else:
                pipe.set(self._prefixed(key), data)
        pipe.execute()
        return len(items)

    def delete(self, key):
        return bool(self._client.delete(self._prefixed(key)))

    def delete_many(self, keys):
        keys = list(keys)
        if not keys:
            return 0
        return int(self._client.delete(*[self._prefixed(k) for k in keys]))

    def delete_prefix(self, prefix):
        """Delete every key under ``prefix`` using SCAN.

        SCAN, never KEYS: this may run against a production instance and
        KEYS blocks the whole server for the duration of the sweep.
        """
        pattern = self._prefixed(prefix) + b"*"
        removed = 0
        cursor = 0
        while True:
            cursor, batch = self._client.scan(
                cursor=cursor, match=pattern, count=500
            )
            if batch:
                removed += int(self._client.delete(*batch))
            if cursor == 0:
                break
        return removed

    def touch(self, key, ttl=None):
        if ttl:
            return bool(
                self._client.pexpire(self._prefixed(key), max(1, int(ttl * 1000)))
            )
        return bool(self._client.persist(self._prefixed(key)))

    def incr_version(self, key, delta=1):
        """Atomic server-side increment.

        Version stamps are stored as plain integers rather than encoded
        payloads so Redis can operate on them directly.
        """
        return int(self._client.incr(self._prefixed(key), delta))

    def get_version(self, key, default=0):
        raw = self._client.get(self._prefixed(key))
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def clear(self):
        return self.delete_prefix("")

    # ── locking, for stampede protection ─────────────────

    def acquire_lock(self, key, timeout=5):
        """Try to become the single flight for ``key``.

        ``SET NX PX`` is atomic, so exactly one worker across the fleet
        wins. The expiry guarantees the lock is released even if the
        holder dies mid-query.
        """
        return bool(
            self._client.set(
                self._prefixed(f"lock:{key}"), b"1",
                nx=True, px=max(1, int(timeout * 1000)),
            )
        )

    def release_lock(self, key):
        self._client.delete(self._prefixed(f"lock:{key}"))

    # ── introspection ────────────────────────────────────

    def ping(self):
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def close(self):
        try:
            self._client.close()
        except Exception:  # pragma: no cover - already closed
            pass

    def __len__(self):
        """Approximate entry count for this prefix.

        Counted with SCAN, so it is O(keyspace) and meant for diagnostics
        and management commands - not for the query path.
        """
        pattern = self._prefixed("") + b"*"
        cursor = 0
        total = 0
        while True:
            cursor, batch = self._client.scan(
                cursor=cursor, match=pattern, count=500
            )
            total += len(batch)
            if cursor == 0:
                return total

    def stats_dict(self):
        data = super().stats_dict()
        data.update({"location": self.location, "prefix": self.prefix})
        try:
            info = self._client.info("memory")
            data["redis_used_memory"] = info.get("used_memory")
            data["redis_maxmemory"] = info.get("maxmemory")
        except Exception:
            data["reachable"] = False
        else:
            data["reachable"] = True
        return data
